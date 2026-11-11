"""Loopback-only interactive SmolVLA demo for three uploaded zone images."""

from __future__ import annotations

import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .dataset import DatasetError, require
from .preflight import project_root


MAX_REQUEST_BYTES = 30 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
LEVEL_NAMES = ("OFF", "LOW", "MEDIUM", "HIGH")
DEFAULT_CHECKPOINT = (
    project_root()
    / "outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3"
    / "checkpoints/006000/pretrained_model"
)


def decode_image(data_url: str):
    """Decode one browser upload and apply the training letterbox transform."""
    require(isinstance(data_url, str) and data_url.startswith("data:image/"), "Mỗi khu vực cần một ảnh hợp lệ")
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as error:
        raise DatasetError("Ảnh upload thiếu phần dữ liệu") from error
    require(";base64" in header and len(encoded) <= 4 * MAX_IMAGE_BYTES // 3 + 8, "Ảnh upload không hợp lệ hoặc quá lớn")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DatasetError("Ảnh upload không phải base64 hợp lệ") from error
    require(0 < len(raw) <= MAX_IMAGE_BYTES, "Mỗi ảnh phải nhỏ hơn hoặc bằng 8 MiB")

    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(raw)) as source:
            require(source.width * source.height <= MAX_IMAGE_PIXELS, "Ảnh có quá nhiều pixel")
            image = source.convert("RGB")
            resized = ImageOps.contain(image, (256, 256), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (256, 256), (0, 0, 0))
            canvas.paste(resized, ((256 - resized.width) // 2, (256 - resized.height) // 2))
            return np.asarray(canvas).copy()
    except (UnidentifiedImageError, OSError) as error:
        raise DatasetError("Không đọc được định dạng ảnh upload") from error


def validate_request(payload: object):
    require(isinstance(payload, dict), "Request phải là JSON object")
    require(set(payload) == {"instruction", "images", "wilt_notes"}, "Request có field thiếu hoặc không được hỗ trợ")
    instruction = payload["instruction"]
    require(isinstance(instruction, str) and instruction.strip(), "Hãy nhập instruction Việt hoặc Anh")
    require(len(instruction) <= 1000, "Instruction quá dài")
    images = payload["images"]
    require(isinstance(images, list) and len(images) == 3, "Cần đúng ba ảnh theo thứ tự khu vực 1, 2, 3")
    arrays = [decode_image(item) for item in images]
    notes = payload["wilt_notes"]
    require(isinstance(notes, list) and len(notes) == 3, "Cần ba ghi chú mức héo")
    require(all(value is None or type(value) is int and 0 <= value <= 4 for value in notes),
            "Mức héo chú thích phải từ 0 đến 4 hoặc không xác định")
    return instruction.strip(), arrays, notes


class DemoRuntime:
    def __init__(self, checkpoint: Path, *, device: str):
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.status = "loading"
        self.error = None
        self.loaded_at = None
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        try:
            from .evaluation import load_model
            from .training import load_training_config

            config = load_training_config(project_root() / "configs/training_v0.yaml")
            bundle = config["_base"] / config["bundle"]
            policy, pre, post, params = load_model(
                config, bundle, checkpoint=self.checkpoint, device=self.device
            )
            self._model = (config, policy, pre, post)
            self.params = params
            self.loaded_at = time.time()
            self.status = "ready"
        except Exception as error:  # Keep the UI available for a readable diagnostic.
            self.error = f"{type(error).__name__}: {error}"
            self.status = "error"

    def status_payload(self):
        result = {
            "status": self.status,
            "device": self.device,
            "checkpoint": str(self.checkpoint),
            "quality_warning": (
                "Checkpoint step 6000 chạy được nhưng chưa đạt baseline; kết quả chỉ dùng để quan sát mô hình."
            ),
            "state_semantics": "Ba van đầu vào luôn là [0,0,0], giống dữ liệu training.",
        }
        if self.error:
            result["error"] = self.error
        if self.status == "ready":
            result["trainable_parameters"] = self.params["trainable"]
        return result

    def predict(self, payload):
        require(self.status == "ready" and self._model is not None,
                "Model chưa sẵn sàng; hãy đợi trạng thái READY")
        instruction, arrays, wilt_notes = validate_request(payload)
        config, policy, pre, post = self._model

        import torch
        from .evaluation import predict_batch
        from .language import normalize_text
        from .training import CAMERAS, prompt_text

        ambiguous = {normalize_text(value) for value in (
            "Tưới cả 3 vườn", "Tưới cả ba vườn", "Tưới cả 3 vùng",
            "Water all 3 gardens", "Water all three gardens",
        )}
        require(normalize_text(instruction) not in ambiguous,
                "WATER_ALL cần ghi rõ mức tưới 1/2/3; model không được tự đoán nhãn")
        tokenizer = next(step for step in pre.steps if type(step).__name__ == "TokenizerProcessorStep")
        token_count = len(tokenizer.input_tokenizer(prompt_text(instruction), truncation=False)["input_ids"])
        require(token_count <= tokenizer.max_length,
                f"Instruction có {token_count} tokens, vượt giới hạn {tokenizer.max_length}")

        batch = {
            camera: torch.from_numpy(array).permute(2, 0, 1).float()[None, None] / 255.0
            for camera, array in zip(CAMERAS, arrays, strict=True)
        }
        batch.update({
            "observation.state": torch.zeros(1, 1, 3, dtype=torch.float32),
            "task": [instruction],
        })
        with self._lock:
            prediction = predict_batch(policy, pre, post, batch, seed=config["seed"])
        actions = prediction["quantized_action"]
        return {
            "status": "ok",
            "instruction": instruction,
            "token_count": token_count,
            "wilt_notes": wilt_notes,
            "wilt_notes_are_model_inputs": False,
            "input_state": [0, 0, 0],
            "zones": [
                {
                    "zone": index + 1,
                    "wilt_note": wilt_notes[index],
                    "continuous": prediction["continuous_action"][index],
                    "clipped": prediction["clipped_action"][index],
                    "action": action,
                    "level": LEVEL_NAMES[action],
                    "valve_open": action > 0,
                    "out_of_range": prediction["out_of_range"][index],
                }
                for index, action in enumerate(actions)
            ],
            **prediction,
            "checkpoint": str(self.checkpoint),
            "quality_warning": self.status_payload()["quality_warning"],
        }


def handler_for(runtime: DemoRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgriVLADemo/1"

        def log_message(self, format, *args):
            print(f"[demo-web] {self.address_string()} {format % args}")

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self.send_json(200, runtime.status_payload())
            assets = {
                "/": ("demo.html", "text/html; charset=utf-8"),
                "/demo.css": ("demo.css", "text/css; charset=utf-8"),
                "/demo.js": ("demo.js", "text/javascript; charset=utf-8"),
            }
            if path in assets:
                name, content_type = assets[path]
                return self.send_bytes(200, files("agri_vla").joinpath("web", name).read_bytes(), content_type)
            self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/infer":
                return self.send_json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(0 < length <= MAX_REQUEST_BYTES, "Request rỗng hoặc lớn hơn 30 MiB")
                require(self.headers.get_content_type() == "application/json", "Content-Type phải là application/json")
                payload = json.loads(self.rfile.read(length))
                self.send_json(200, runtime.predict(payload))
            except (DatasetError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": f"{type(error).__name__}: {error}"})

        def do_PUT(self):
            self.send_json(405, {"error": "Read/infer-only demo"})

        do_DELETE = do_PUT
        do_PATCH = do_PUT

    return Handler


def serve_demo(*, checkpoint=DEFAULT_CHECKPOINT, device="cuda", port=8000):
    require(type(port) is int and 1024 <= port <= 65535, "Port phải trong khoảng 1024..65535")
    checkpoint = Path(checkpoint)
    require(checkpoint.is_dir(), f"Không tìm thấy checkpoint: {checkpoint}")
    runtime = DemoRuntime(checkpoint, device=device)
    threading.Thread(target=runtime.load, daemon=True, name="agrivla-model-loader").start()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(runtime))
    print(f"AgriVLA interactive demo: http://127.0.0.1:{port}")
    print(f"Checkpoint: {checkpoint.resolve()}")
    print("Model đang tải trong background; theo dõi trạng thái trên giao diện. Dừng bằng Ctrl+C.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

