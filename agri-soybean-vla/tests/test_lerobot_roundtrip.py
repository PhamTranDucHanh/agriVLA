"""Native pinned-LeRobot integration checks; run explicitly in .venv-train."""

from pathlib import Path
import sys

import pytest

if Path(sys.prefix).name != ".venv-train":
    pytest.skip("Native LeRobot integration uses .venv-train; data env stays ML-free", allow_module_level=True)

from agri_vla.lerobot_io import train_runtime, transformed_image, batch_report, features
from agri_vla.split import digest


def test_literal_rgb_letterbox_and_real_singleton_batch(tmp_path):
    torch = train_runtime()
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        path = tmp_path / f"camera{i}.png"
        Image.new("RGB", (640, 480), color).save(path)
        image = transformed_image(path, digest(path.read_bytes()))
        assert image.shape == (256, 256, 3) and image.dtype == np.uint8
        assert np.all(image[:32] == 0) and np.all(image[224:] == 0)
        assert np.all(image[32:224] == color)
        images.append(image)
    root = tmp_path / "native"
    dataset = LeRobotDataset.create("agrivla/native_fixture", fps=1, features=features(), root=root, use_videos=False, video_backend="pyav")
    tasks = ["Chỉ tưới vùng 2 nếu đạt ngưỡng héo.", "Stop watering all zones."]
    actions = [[0, 1, 0], [0, 0, 0]]
    for task, action in zip(tasks, actions, strict=True):
        dataset.add_frame({**{f"observation.images.camera{i+1}": im for i, im in enumerate(images)},
                           "observation.state": np.zeros(3, dtype=np.float32), "action": np.asarray(action, dtype=np.float32), "task": task})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    loaded = LeRobotDataset("agrivla/native_fixture", root=root, delta_timestamps={"action": [0.]}, video_backend="pyav")
    assert loaded.num_episodes == len(loaded) == 2
    for index, task in enumerate(tasks):
        row = loaded[index]
        assert row["task"] == task and row["action"].tolist() == [actions[index]]
        for i, expected in enumerate(images):
            pixels = (row[f"observation.images.camera{i+1}"] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            assert np.array_equal(pixels, expected)
    report = batch_report(loaded)
    assert report["shapes"]["action"] == [2, 1, 3] and report["action_is_pad"] == [[False], [False]]
    assert report["sampler_epoch_frames"] == report["sampler_unique_frames"] == 2
    from agri_vla.lerobot_checks import factory_batch
    factory = factory_batch(root, "agrivla/native_fixture")
    assert factory["shapes"]["action"] == [2, 1, 3]
    assert factory["shapes"]["observation.state"] == [2, 1, 3]
    assert factory["shapes"]["observation.images.camera1"] == [2, 1, 3, 256, 256]
    assert all(not any(mask) for masks in factory["padding_masks"].values() for mask in masks)


def test_native_normalizer_preserves_zero_state_and_inverts_action_chunk():
    torch = train_runtime()
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    spec = {"observation.state": PolicyFeature(FeatureType.STATE, (3,)), "action": PolicyFeature(FeatureType.ACTION, (3,))}
    modes = {FeatureType.STATE: NormalizationMode.IDENTITY, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {"observation.state": {"mean": [0., 0., 0.], "std": [0., 0., 0.]},
             "action": {"mean": [1., 1.5, 2.], "std": [.8, 1.1, .7]}}
    normalizer = NormalizerProcessorStep(features=spec, norm_map=modes, stats=stats, device="cpu")
    inverse = UnnormalizerProcessorStep(features=spec, norm_map=modes, stats=stats, device="cpu")
    actions = torch.tensor([[[0., 1., 3.]], [[3., 0., 0.]]])
    state = torch.zeros(2, 3)
    transition = {TransitionKey.OBSERVATION: {"observation.state": state}, TransitionKey.ACTION: actions}
    normalized = normalizer(transition)
    restored = inverse(normalized)
    assert torch.equal(normalized[TransitionKey.OBSERVATION]["observation.state"], state)
    assert torch.isfinite(normalized[TransitionKey.ACTION]).all()
    assert torch.allclose(normalized[TransitionKey.ACTION], (actions - torch.tensor([1., 1.5, 2.])) / torch.tensor([.8, 1.1, .7]), atol=1e-6)
    assert torch.allclose(restored[TransitionKey.ACTION], actions, atol=1e-6)
