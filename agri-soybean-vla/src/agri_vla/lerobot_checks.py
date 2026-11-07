"""Audit the actual upstream dataset-factory observation/action history axes."""

from .dataset import require, SPLITS, read_json
from .lerobot_io import CAMERAS, train_runtime


def factory_batch(root, repo_id):
    torch = train_runtime()
    from torch.utils.data import DataLoader
    from lerobot.datasets import LeRobotDataset, EpisodeAwareSampler
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig(chunk_size=1, n_action_steps=1, device="cpu")
    metadata = LeRobotDataset(repo_id, root=root, video_backend="pyav").meta
    deltas = resolve_delta_timestamps(config, metadata)
    require(deltas["action"] == [0.0] and all(deltas[k] == [0.0] for k in (*CAMERAS, "observation.state")), "Unexpected singleton factory deltas")
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=deltas, video_backend="pyav")
    starts, ends = list(metadata.episodes["dataset_from_index"]), list(metadata.episodes["dataset_to_index"])
    sampler = EpisodeAwareSampler(starts, ends, drop_n_last_frames=getattr(config, "drop_n_last_frames", 0), shuffle=True, seed=42)
    indices = list(iter(sampler))
    require(sorted(indices) == list(range(len(dataset))), "Factory sampler removed or repeated singleton frames")
    batch = next(iter(DataLoader(dataset, batch_size=min(4, len(dataset)), sampler=EpisodeAwareSampler(starts, ends), num_workers=0)))
    b = min(4, len(dataset))
    require(batch["action"].shape == (b, 1, 3) and batch["observation.state"].shape == (b, 1, 3), "Factory action/state time axis mismatch")
    for camera in CAMERAS:
        require(batch[camera].shape == (b, 1, 3, 256, 256), "Factory camera time axis mismatch")
    pad_keys = [key for key in batch if key.endswith("_is_pad")]
    require(pad_keys and all(batch[key].shape == (b, 1) and not batch[key].any() for key in pad_keys), "Factory masks hide real singleton observations/actions")
    return {"status": "pass", "delta_timestamps": deltas,
            "shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
            "dtypes": {key: str(value.dtype) for key, value in batch.items() if isinstance(value, torch.Tensor)},
            "padding_masks": {key: batch[key].tolist() for key in pad_keys}, "task": batch["task"],
            "sampled_epoch_frames": len(indices), "unique_epoch_frames": len(set(indices)),
            "observation_history_length": 1, "action_horizon": 1, "policy_forward_performed": False}


def audit_factory_batches(root):
    if (root / "export_info.json").is_file():
        info = read_json(root / "export_info.json")
        return factory_batch(root, info["repo_id"])
    return {split: factory_batch(root / split, read_json(root / split / "export_info.json")["repo_id"]) for split in SPLITS}
