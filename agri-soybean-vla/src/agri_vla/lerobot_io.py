"""Convert validated static scenes to genuine LeRobot singleton episodes.

ML imports are lazy and confined to conversion/round-trip, never data-only CLI.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from .dataset import (SPLITS, DatasetError, require, new_output, safe_output, read_json, read_jsonl, jsonl_bytes,
                      load_compose_config)
from .policy import all_requests, load_unique_yaml
from .preflight import LEROBOT_COMMIT, project_root
from .split import canonical_bytes, digest, relative_file
from .validate import validate_dataset

VERSION = "agri_soybean_lerobot_v0"
CAMERAS = tuple(f"observation.images.camera{i}" for i in (1, 2, 3))
MODEL_FEATURES = {*CAMERAS, "observation.state", "action"}
BUILTIN_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def load_conversion_config(path: Path) -> dict:
    raw = path.read_bytes()
    config = load_unique_yaml(raw, context="LeRobot conversion")
    keys = {"conversion_version", "base_dir", "scene_config", "dataset_lock_sha256", "manifests", "output_dir", "lerobot_commit",
            "image_size", "image_transform", "fps", "use_videos", "smoke_count", "seed"}
    require(isinstance(config, dict) and set(config) == keys, "Missing/unknown conversion config fields")
    require(isinstance(config["base_dir"], str) and not Path(config["base_dir"]).is_absolute(), "base_dir must be relative")
    require(config["conversion_version"] == VERSION and config["lerobot_commit"] == LEROBOT_COMMIT, "Unsupported converter/LeRobot revision")
    require(config["image_size"] == [256, 256] and all(type(x) is int for x in config["image_size"]), "v0 requires 256×256 RGB")
    require(config["image_transform"] == "pillow_lanczos_letterbox_black_v1", "Unknown image transform")
    require(type(config["fps"]) is int and config["fps"] == 1 and config["use_videos"] is False, "Use singleton fps=1 / use_videos=false")
    require(type(config["smoke_count"]) is int and config["smoke_count"] == 64, "Smoke must contain 64 records")
    require(type(config["seed"]) is int and config["seed"] >= 0, "seed must be a nonnegative integer")
    require(isinstance(config["dataset_lock_sha256"], str) and len(config["dataset_lock_sha256"]) == 64, "Missing dataset lock pin")
    base = (path.resolve().parent / config["base_dir"]).resolve()
    for key in ("scene_config", "manifests", "output_dir"):
        relative_file(base, config[key])
    return config | {"_base": base, "_path": path.resolve(), "_sha256": digest(raw)}


def scene_inputs(config: dict):
    base = config["_base"]
    scene_config = load_compose_config(relative_file(base, config["scene_config"]))
    require(scene_config["_base"] == base, "Conversion and scene base_dir differ")
    directory = relative_file(base, config["manifests"])
    require(digest((directory / "dataset_lock.json").read_bytes()) == config["dataset_lock_sha256"], "Input dataset lock mismatch")
    report, inputs, rows = validate_dataset(scene_config, directory)
    return report, inputs, rows


def coverage(row: dict) -> set:
    params = json.dumps(row["intent_params"], sort_keys=True)
    tokens = {("intent", row["intent"]), ("params", row["intent"], params), ("kind", row["generation_metadata"]["kind"])}
    tokens.update(("action", i, value) for i, value in enumerate(row["target_action"]))
    tokens.update(("wilt", i, value) for i, value in enumerate(row["teacher_annotations"]["wilt_scores"]))
    return tokens


def select_smoke(rows: list[dict], *, count=64, seed=42) -> list[dict]:
    require(type(count) is int and count >= 2 and count % 2 == 0, "Smoke must preserve bilingual pairs")
    groups = defaultdict(list)
    for row in rows:
        require(row["split"] == "train", "Smoke may only use train sources")
        groups[row["translation_pair_id"]].append(row)
    require(all(len(pair) == 2 and {r["language"] for r in pair} == {"vi", "en"} for pair in groups.values()), "Incomplete train translations")
    pool = sorted(groups)
    random.Random(seed).shuffle(pool)
    require(len(pool) >= count // 2, "Not enough train translation pairs")
    selected, seen = set(), set()
    # Include the controlled [0,2,4] counterfactual panel before greedy coverage.
    for pid in pool:
        if groups[pid][0]["generation_metadata"]["kind"] == "contrastive_base":
            selected.add(pid)
            seen |= coverage(groups[pid][0])
    require(len(selected) <= count // 2, "Smoke too small for contrastive panel")
    while len(selected) < count // 2:
        pid = max((p for p in pool if p not in selected), key=lambda p: len(coverage(groups[p][0]) - seen))
        selected.add(pid)
        seen |= coverage(groups[pid][0])
    required = {("params", str(r.intent), json.dumps(r.params, sort_keys=True)) for r in all_requests()}
    required |= {("action", zone, level) for zone in range(3) for level in range(4)}
    require(required <= seen, "Smoke cannot cover all intents/params/action levels per zone")
    return [r for r in rows if r["translation_pair_id"] in selected]


def selection_summary(rows):
    return {"records": len(rows), "languages": dict(Counter(r["language"] for r in rows)),
            "intents": dict(Counter(r["intent"] for r in rows)),
            "action_levels_by_zone": {f"zone{i+1}": sorted({r["target_action"][i] for r in rows}) for i in range(3)},
            "sample_ids": [r["sample_id"] for r in rows]}


def train_runtime():
    require(Path(sys.prefix).resolve() == (project_root() / ".venv-train").resolve(), "Use agri-soybean-vla/.venv-train/bin/python for conversion/verification")
    vendor = project_root() / ".vendor/lerobot"
    head = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    require(head == LEROBOT_COMMIT, "LeRobot vendor commit mismatch")
    require(not subprocess.check_output(["git", "-C", str(vendor), "diff", "--name-only"], text=True).strip(), "Vendor tracked source differs from pinned commit")
    # Local-only dataset IO; no fallback network/model downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_DATASETS_CACHE"] = str(project_root() / ".cache/hf-datasets")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import datasets
    import lerobot
    torch.set_num_threads(1)
    datasets.disable_progress_bars()
    require(Path(lerobot.__file__).resolve().is_relative_to(vendor.resolve()), "Imported LeRobot is not the pinned vendor")
    return torch


def features():
    return {**{key: {"dtype": "image", "shape": (3, 256, 256), "names": ["channels", "height", "width"]} for key in CAMERAS},
            "observation.state": {"dtype": "float32", "shape": (3,), "names": ["valve1_current", "valve2_current", "valve3_current"]},
            "action": {"dtype": "float32", "shape": (3,), "names": ["zone1_level", "zone2_level", "zone3_level"]}}


@lru_cache(maxsize=256)
def transformed_image(path: Path, source_sha256: str):
    import numpy as np
    from PIL import Image, ImageOps
    data = path.read_bytes()
    require(digest(data) == source_sha256, "Source image changed after input QA")
    from io import BytesIO
    with Image.open(BytesIO(data)) as image:
        require(image.mode == "RGB", "Source is not RGB")
        resized = ImageOps.contain(image, (256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (256, 256), (0, 0, 0))
        canvas.paste(resized, ((256 - resized.width) // 2, (256 - resized.height) // 2))
        return np.asarray(canvas).copy()


def pixel_digest(array):
    return digest(b"RGB:256x256:" + array.tobytes())


def make_frame(row, base):
    import numpy as np
    return {**{key: transformed_image(relative_file(base, row["images"][f"zone{i+1}"]), row["provenance"]["source_sha256"][i])
               for i, key in enumerate(CAMERAS)},
            "observation.state": np.zeros(3, dtype=np.float32), "action": np.asarray(row["target_action"], dtype=np.float32),
            "task": row["instruction"]}


def write_dataset(config, rows, root, *, split, profile):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    new_output(root)
    repo_id = f"agrivla/soybean_v0_{profile}_{split}"
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=1, features=features(), root=root,
                                   robot_type="synthetic_three_zone_valves", use_videos=False,
                                   video_backend="pyav", image_writer_processes=0, image_writer_threads=0)
    mapping = []
    for index, row in enumerate(rows):
        frame = make_frame(row, config["_base"])
        require(set(frame) == MODEL_FEATURES | {"task"}, "Teacher/provenance leaked into frame features")
        pixels = [pixel_digest(frame[key]) for key in CAMERAS]
        dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
        mapping.append({"episode_index": index, "frame_index": 0, "sample_id": row["sample_id"],
                        "transformed_pixel_sha256": pixels, "source_record": row})
        if (index + 1) % 100 == 0:
            print(f"Converted {split}: {index+1}/{len(rows)} singleton episodes", file=sys.stderr, flush=True)
    dataset.finalize()
    metadata_bytes = jsonl_bytes(mapping)
    with (root / "sample_metadata.jsonl").open("xb") as stream:
        stream.write(metadata_bytes)
    info = {"conversion_version": VERSION, "profile": profile, "source_split": split, "repo_id": repo_id,
            "dataset_lock_sha256": config["dataset_lock_sha256"], "conversion_config_sha256": config["_sha256"],
            "lerobot_commit": LEROBOT_COMMIT, "records": len(rows), "fps_semantics": "serialization_only",
            "image_transform": config["image_transform"], "source_base": "module/base_dir",
            "mapping_sha256": digest(metadata_bytes), "selection": selection_summary(rows),
            "normalization": {"observation.state": "IDENTITY", "action": "MEAN_STD_FROM_FULL_TRAIN_ONLY"}}
    (root / "export_info.json").write_bytes(canonical_bytes(info))
    files = sorted(p for p in root.rglob("*") if p.is_file() and (p.suffix == ".parquet" or p.name in {"info.json", "stats.json", "sample_metadata.jsonl", "export_info.json"}))
    lock = {"conversion_version": VERSION, "dataset_lock_sha256": config["dataset_lock_sha256"],
            "conversion_config_sha256": config["_sha256"], "converter_sha256": digest(Path(__file__).read_bytes()),
            "files_sha256": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in files}}
    (root / "export_lock.json").write_bytes(canonical_bytes(lock))
    return info


def batch_report(dataset):
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    cfg = SmolVLAConfig(chunk_size=1, n_action_steps=1, device="cpu")
    require(cfg.action_delta_indices == [0], "Singleton config has unexpected action delta indices")
    starts = list(dataset.meta.episodes["dataset_from_index"])
    ends = list(dataset.meta.episodes["dataset_to_index"])
    require(all(end - start == 1 for start, end in zip(starts, ends, strict=True)), "Not all episodes have one frame")
    sampler = EpisodeAwareSampler(starts, ends, drop_n_last_frames=getattr(cfg, "drop_n_last_frames", 0), shuffle=True, seed=42)
    indices = list(iter(sampler))
    require(sorted(indices) == list(range(len(dataset))), "Sampler drops/repeats singleton episodes")
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), sampler=EpisodeAwareSampler(starts, ends, shuffle=False), num_workers=0)
    batch = next(iter(loader))
    b = min(4, len(dataset))
    require(batch["action"].shape == (b, 1, 3) and batch["action_is_pad"].shape == (b, 1), "Action batch/padding shape differs from [B,1,3]/[B,1]")
    require(not batch["action_is_pad"].any(), "Singleton actions are masked as padding")
    require(batch["observation.state"].shape == (b, 3) and not batch["observation.state"].any(), "State batch is not all-valves-closed zeros")
    for key in CAMERAS:
        require(batch[key].shape == (b, 3, 256, 256), "Camera batch shape/order mismatch")
    require(all(batch[key].device.type == "cpu" for key in MODEL_FEATURES), "Conversion batch is not CPU")
    return {"batch_size": b, "shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
            "dtypes": {key: str(value.dtype) for key, value in batch.items() if isinstance(value, torch.Tensor)},
            "task": batch["task"], "action_values": batch["action"].tolist(),
            "state_values": batch["observation.state"].tolist(), "action_is_pad": batch["action_is_pad"].tolist(),
            "action_delta_indices": cfg.action_delta_indices, "drop_n_last_frames": getattr(cfg, "drop_n_last_frames", 0),
            "sampler_epoch_frames": len(indices), "sampler_unique_frames": len(set(indices)), "device": "cpu"}


def verify_export(config, root, *, validated=None):
    torch = train_runtime()
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if validated is None:
        validated = scene_inputs(config)
    _, _, all_rows = validated
    info = read_json(root / "export_info.json")
    lock = read_json(root / "export_lock.json")
    require(info["conversion_version"] == lock["conversion_version"] == VERSION, "Export version mismatch")
    require(info["dataset_lock_sha256"] == lock["dataset_lock_sha256"] == config["dataset_lock_sha256"], "Export source lock differs")
    require(info["conversion_config_sha256"] == lock["conversion_config_sha256"] == config["_sha256"], "Export conversion config changed")
    require(lock["converter_sha256"] == digest(Path(__file__).read_bytes()), "Converter implementation differs from export")
    for name, expected in lock["files_sha256"].items():
        path = relative_file(root.resolve(), name)
        require(digest(path.read_bytes()) == expected, f"Export file checksum changed: {name}")
    require(info["source_split"] in SPLITS and info["profile"] in ("smoke", "full"), "Unknown export split/profile")
    split = info["source_split"]
    expected_rows = select_smoke(all_rows["train"], count=config["smoke_count"], seed=config["seed"]) if info["profile"] == "smoke" else all_rows[split]
    require(info["profile"] != "smoke" or split == "train", "Smoke contains non-train data")
    mapping_data = (root / "sample_metadata.jsonl").read_bytes()
    require(digest(mapping_data) == info["mapping_sha256"], "Episode/sample mapping hash mismatch")
    mapping = read_jsonl(root / "sample_metadata.jsonl")
    require(len(mapping) == len(expected_rows) == info["records"], "Export/mapping/source counts differ")
    require(info["selection"] == selection_summary(expected_rows), "Export selection changed")
    require(not list(root.rglob("*.mp4")), "Image-only export unexpectedly contains MP4")
    dataset = LeRobotDataset(info["repo_id"], root=root, delta_timestamps={"action": [0.0]}, video_backend="pyav")
    require(set(dataset.features) == MODEL_FEATURES | BUILTIN_FEATURES, "Unexpected features or metadata leakage")
    require(dataset.meta.fps == 1 and not dataset.meta.video_keys, "Unexpected fps or video features")
    require(len(dataset) == dataset.num_episodes == len(expected_rows), "Export must have one frame per episode")
    for key, value in features().items():
        require(dataset.features[key]["dtype"] == value["dtype"] and tuple(dataset.features[key]["shape"]) == value["shape"], f"Feature dtype/shape mismatch: {key}")
    pixels_checked = 0
    for index, (row, link) in enumerate(zip(expected_rows, mapping, strict=True)):
        require(link["source_record"] == row and link["sample_id"] == row["sample_id"] and link["episode_index"] == index and link["frame_index"] == 0, "Episode↔sample identity/provenance mismatch")
        actual = dataset[index]
        require(actual["task"] == row["instruction"], "Task text/camera metadata does not round-trip")
        require(int(actual["episode_index"].item()) == index and int(actual["index"].item()) == index and int(actual["frame_index"].item()) == 0 and float(actual["timestamp"].item()) == 0.0, "Singleton frame/episode/time indices mismatch")
        require(actual["action"].dtype == actual["observation.state"].dtype == torch.float32, "Numeric dtype differs from float32")
        require(torch.equal(actual["action"], torch.tensor([row["target_action"]], dtype=torch.float32)), "Action values do not round-trip")
        require(actual["observation.state"].shape == (3,) and not actual["observation.state"].any(), "State not zeros")
        require(actual["action_is_pad"].shape == (1,) and not actual["action_is_pad"].any(), "Action frame incorrectly padded")
        for zone, key in enumerate(CAMERAS, 1):
            expected = transformed_image(relative_file(config["_base"], row["images"][f"zone{zone}"]), row["provenance"]["source_sha256"][zone - 1])
            image = actual[key]
            require(image.dtype == torch.float32 and tuple(image.shape) == (3, 256, 256) and torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1, "Invalid decoded camera tensor")
            restored = (image * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            require(np.array_equal(restored, expected), f"Camera pixels/order changed: {row['sample_id']}/{key}")
            require(pixel_digest(restored) == link["transformed_pixel_sha256"][zone - 1], "Transformed pixel hash differs from mapping")
            pixels_checked += 1
        if (index + 1) % 250 == 0:
            print(f"Round-trip {split}: {index+1}/{len(expected_rows)} records", file=sys.stderr, flush=True)
    stats = dataset.meta.stats
    require(MODEL_FEATURES <= set(stats), "Missing numeric/image stats")
    for key, fields in stats.items():
        for name, value in fields.items():
            require(np.isfinite(np.asarray(value, dtype=np.float64)).all(), f"Nonfinite stats: {key}/{name}")
    actions = np.asarray([r["target_action"] for r in expected_rows], dtype=np.float64)
    require(np.allclose(stats["action"]["mean"], actions.mean(0), atol=1e-6) and np.allclose(stats["action"]["std"], actions.std(0), atol=1e-6), "Action stats differ from exact source mean/std")
    require(not np.asarray(stats["observation.state"]["std"]).any(), "Constant-state stats not zero std")
    report = {"status": "pass", "conversion_version": VERSION, "profile": info["profile"], "split": split,
              "records_checked": len(expected_rows), "episodes": dataset.num_episodes, "frames": len(dataset),
              "images_pixel_exact_checked": pixels_checked, "task_text_exact": True, "episode_sample_ids_exact": True,
              "stats_finite": True, "action_stats_match_source": True, "state_zero_variance_handling": "IDENTITY required",
              "batch": batch_report(dataset), "selection": info["selection"],
              "dataset_lock_sha256": config["dataset_lock_sha256"], "export_lock_sha256": digest((root / "export_lock.json").read_bytes()),
              "stats_sha256": digest((root / "meta/stats.json").read_bytes()),
              "model_weights_downloaded": False, "training_performed": False}
    return report


def train_statistics(root: Path) -> dict:
    info = read_json(root / "export_info.json")
    require(info["source_split"] == "train" and info["profile"] == "full", "Normalization stats must come from full TRAIN, never smoke/val/test")
    return read_json(root / "meta/stats.json")


def verify_normalization(full_root):
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    stats = train_statistics(full_root / "train")
    feature_spec = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    modes = {FeatureType.STATE: NormalizationMode.IDENTITY, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessorStep(features=feature_spec, norm_map=modes, stats=stats, device="cpu")
    inverse = UnnormalizerProcessorStep(features=feature_spec, norm_map=modes, stats=stats, device="cpu")
    results = {}
    for split in SPLITS:
        info = read_json(full_root / split / "export_info.json")
        dataset = LeRobotDataset(info["repo_id"], root=full_root / split, delta_timestamps={"action": [0.]}, video_backend="pyav")
        batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
        transition = {TransitionKey.OBSERVATION: {"observation.state": batch["observation.state"]}, TransitionKey.ACTION: batch["action"]}
        normalized = normalizer(transition)
        restored = inverse(normalized)
        mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
        std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
        require(torch.allclose(normalized[TransitionKey.ACTION], (batch["action"] - mean) / (std + normalizer.eps), atol=1e-6), "Normalizer did not use TRAIN stats")
        error = (restored[TransitionKey.ACTION] - batch["action"]).abs().max().item()
        require(error < 1e-6 and torch.equal(normalized[TransitionKey.OBSERVATION]["observation.state"], batch["observation.state"]), "Normalize/inverse or identity state failed")
        results[split] = {"stats_source": "train/meta/stats.json", "max_action_roundtrip_error": error, "state_identity": True}
    return {"status": "pass", "stats_source_split": "train", "stats_path": "train/meta/stats.json",
            "stats_sha256": digest((full_root / "train/meta/stats.json").read_bytes()),
            "normalization_modes": {"observation.state": "IDENTITY", "action": "MEAN_STD"},
            "smolvla_default_normalization_must_be_overridden": True, "split_checks": results}


def convert(config, *, profile, output_dir=None, dry_run=False, smoke_root=None):
    require(profile in ("smoke", "full"), "profile must be smoke/full")
    validated = scene_inputs(config)
    qa, inputs, rows = validated
    base = relative_file(config["_base"], config["output_dir"])
    output = safe_output(inputs.config, output_dir or base / profile)
    chosen = {"train": select_smoke(rows["train"], count=config["smoke_count"], seed=config["seed"])} if profile == "smoke" else rows
    result = {"status": "pass", "dry_run": dry_run, "profile": profile, "output": str(output),
              "dataset_lock_sha256": config["dataset_lock_sha256"], "selection": {s: selection_summary(r) for s, r in chosen.items()}}
    if dry_run:
        return result
    train_runtime()
    if profile == "full":
        gate_root = smoke_root or base / "smoke"
        gate = read_json(gate_root / "roundtrip_report.json")
        require(gate["status"] == "pass" and gate["profile"] == "smoke" and gate["records_checked"] == 64 and gate["dataset_lock_sha256"] == config["dataset_lock_sha256"], "Full conversion requires a PASS smoke round-trip of 64 train samples")
        # Recheck smoke file/code/source pins rather than trusting only a PASS string.
        verify_export(config, gate_root, validated=validated)
        output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for split, selected in chosen.items():
        root = output if profile == "smoke" else output / split
        write_dataset(config, selected, root, split=split, profile=profile)
        report = verify_export(config, root, validated=validated)
        (root / "roundtrip_report.json").write_bytes(canonical_bytes(report))
        reports[split] = report
    if profile == "full":
        normalization = verify_normalization(output)
        (output / "normalization_contract.json").write_bytes(canonical_bytes(normalization))
        (output / "train_stats.json").write_bytes(canonical_bytes(train_statistics(output / "train")))
        summary = {"status": "pass", "profile": "full", "dataset_lock_sha256": config["dataset_lock_sha256"],
                   "splits": {s: {k: r[k] for k in ("records_checked", "episodes", "frames", "images_pixel_exact_checked", "export_lock_sha256")} for s, r in reports.items()},
                   "normalization": normalization, "model_weights_downloaded": False, "training_performed": False}
        (output / "roundtrip_report.json").write_bytes(canonical_bytes(summary))
        result["normalization"] = normalization
    result["roundtrip"] = {s: {k: r[k] for k in ("records_checked", "episodes", "frames", "images_pixel_exact_checked", "batch")} for s, r in reports.items()}
    return result


def verify_path(config, root):
    validated = scene_inputs(config)
    if (root / "export_info.json").is_file():
        return verify_export(config, root, validated=validated)
    require(all((root / split / "export_info.json").is_file() for split in SPLITS), "Expected one export root or a full parent with train/val/test")
    reports = {split: verify_export(config, root / split, validated=validated) for split in SPLITS}
    normal = verify_normalization(root)
    require(read_json(root / "train_stats.json") == train_statistics(root / "train"), "Saved inference stats differ from full train")
    require(read_json(root / "normalization_contract.json") == normal, "Normalization contract/stats changed")
    return {"status": "pass", "splits": reports, "normalization": normal, "dataset_lock_sha256": config["dataset_lock_sha256"]}
