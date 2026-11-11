"""Input contract tests for the local interactive inference server."""

import base64
import importlib.util
from io import BytesIO

import pytest
from PIL import Image

from agri_vla.dataset import DatasetError
from agri_vla.demo_server import DemoRuntime, decode_image, validate_request


needs_numpy = pytest.mark.skipif(importlib.util.find_spec("numpy") is None,
                                 reason="Uploaded-image inference uses the ML .venv-train environment")


def image_url(mode="RGB", size=(17, 9)):
    stream = BytesIO()
    Image.new(mode, size, (20, 80, 30) if mode == "RGB" else 128).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


@needs_numpy
def test_uploaded_image_uses_training_shape_and_rgb():
    array = decode_image(image_url())
    assert array.shape == (256, 256, 3)
    assert str(array.dtype) == "uint8"
    # 17:9 is letterboxed vertically, so the first row is black.
    assert not array[0].any()


@needs_numpy
def test_request_requires_three_images_and_annotations_are_metadata_only():
    payload = {"instruction": "Dừng tưới tất cả khu vực.",
               "images": [image_url(), image_url(), image_url()],
               "wilt_notes": [0, 2, None]}
    instruction, arrays, notes = validate_request(payload)
    assert instruction == payload["instruction"] and len(arrays) == 3 and notes == [0, 2, None]
    for bad in (
        payload | {"images": payload["images"][:2]},
        payload | {"wilt_notes": [0, 5, None]},
        payload | {"instruction": ""},
        payload | {"teacher_annotations": {}},
    ):
        with pytest.raises(DatasetError):
            validate_request(bad)


def test_runtime_status_warns_about_checkpoint_quality(tmp_path):
    runtime = DemoRuntime(tmp_path / "checkpoint", device="cuda")
    status = runtime.status_payload()
    assert status["status"] == "loading"
    assert "chưa đạt baseline" in status["quality_warning"]
    assert status["state_semantics"].startswith("Ba van đầu vào")
