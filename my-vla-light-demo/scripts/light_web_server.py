"""Serve the local light UI and the existing SmolVLA inference pipeline."""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from esp32_client import ESP32Error, get_light_state, set_light

WEB = Path(__file__).resolve().parent.parent / "web"


class Controller:
    def __init__(self, hardware=False, validate=False):
        self.hardware = hardware
        self.validate = validate
        self.infer = None
        self.error = None
        self.lock = threading.Lock()

    def load(self):
        try:
            from demo_light_vla import infer
            self.infer = infer
        except Exception as error:
            self.error = str(error)
            print(f"Model loading failed: {error}", flush=True)

    def predict(self, prompt):
        started = time.monotonic()
        action, value, semantic, desired = self.infer(prompt, control_hardware=False)
        if not math.isfinite(value) or not all(math.isfinite(float(x)) for x in action):
            raise ValueError("Model returned a non-finite action; command was not sent")
        result = dict(prompt=prompt, action=[float(x) for x in action], light_value=value,
                      semantic_action=semantic, desired_on=desired, hardware_enabled=self.hardware,
                      final_action=semantic, hardware_state=None, hardware_error=None)
        if self.hardware:
            try:
                if self.validate and get_light_state()["light_on"] == desired:
                    result.update(final_action="NO_ACTION", hardware_state=desired)
                else:
                    response = set_light(desired)
                    result.update(hardware_state=response["light_on"], changed=response["changed"])
            except ESP32Error as error:
                result.update(final_action="COMMUNICATION_FAILED", hardware_error=str(error))
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        return result


def handler_for(controller):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                self.reply(200, dict(model="error" if controller.error else "ready" if controller.infer else "loading",
                                     error=controller.error, hardware_enabled=controller.hardware,
                                     validate_state=controller.validate))
            elif self.path == "/api/state":
                if not controller.hardware:
                    self.reply(200, dict(light_on=None, hardware_enabled=False))
                    return
                try:
                    self.reply(200, get_light_state())
                except ESP32Error as error:
                    self.reply(502, dict(error=str(error)))
            elif self.path in ("/", "/index.html", "/style.css", "/app.js"):
                filename = "index.html" if self.path == "/" else self.path[1:]
                body = (WEB / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}[Path(filename).suffix])
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.reply(404, dict(error="Not found"))

        def do_POST(self):
            if self.path != "/api/infer":
                self.reply(404, dict(error="Not found"))
                return
            # Browser requests must originate from this local UI.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.reply(403, dict(error="Origin rejected"))
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, dict(error="Expected application/json"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                prompt = data.get("prompt") if isinstance(data, dict) else None
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
                    raise ValueError("Prompt must contain 1–1000 characters")
            except (ValueError, UnicodeError) as error:
                self.reply(400, dict(error=str(error)))
                return
            if controller.infer is None:
                self.reply(503, dict(error=controller.error or "Model is loading"))
                return
            if not controller.lock.acquire(blocking=False):
                self.reply(409, dict(error="Inference is busy; please wait"))
                return
            try:
                self.reply(200, controller.predict(prompt.strip()))
            except Exception as error:
                self.reply(500, dict(error=str(error)))
            finally:
                controller.lock.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--esp32", action="store_true")
    parser.add_argument("--validate-state", action="store_true")
    args = parser.parse_args()
    if args.validate_state and not args.esp32:
        parser.error("--validate-state requires --esp32")
    controller = Controller(args.esp32, args.validate_state)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(controller))
    threading.Thread(target=controller.load, daemon=True).start()
    print(f"Open http://localhost:{args.port} — model is loading", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
