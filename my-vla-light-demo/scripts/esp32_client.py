"""HTTP transport for the local ESP32 LED controller; no model dependencies."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ESP32_URL = os.getenv("ESP32_URL", "http://192.168.1.50").rstrip("/")


class ESP32Error(RuntimeError):
    """Transport failure or invalid controller response."""


def _request(path: str, command: str | None = None) -> dict:
    parsed = urlsplit(ESP32_URL)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
        raise ESP32Error("ESP32_URL must be an HTTP(S) base URL without query or fragment")
    request = Request(
        f"{ESP32_URL}{path}",
        data=command.encode("utf-8") if command is not None else None,
        headers={"Content-Type": "text/plain", "Accept": "application/json"},
        method="POST" if command is not None else "GET",
    )
    try:
        with urlopen(request, timeout=3) as response:
            result = json.load(response)
    except HTTPError as error:
        raise ESP32Error(f"{request.method} {path}: HTTP {error.code} {error.reason}") from error
    except (URLError, OSError, ValueError) as error:
        raise ESP32Error(f"{request.method} {path}: {error}") from error
    if not isinstance(result, dict) or type(result.get("light_on")) is not bool:
        raise ESP32Error(f"{path}: response must contain boolean light_on")
    if command is not None:
        if type(result.get("changed")) is not bool:
            raise ESP32Error(f"{path}: response must contain boolean changed")
        if result["light_on"] != (command == "ON"):
            raise ESP32Error("Controller did not acknowledge the requested light state")
    return result


def get_light_state() -> dict:
    """Read the controller's GPIO state (not an optical LED measurement)."""
    return _request("/state")


def set_light(turn_on: bool) -> dict:
    """Send an idempotent desired state and validate its acknowledgement."""
    if type(turn_on) is not bool:
        raise TypeError("turn_on must be a bool")
    return _request("/light", "ON" if turn_on else "OFF")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", type=str.lower, choices=("on", "off", "state"))
    args = parser.parse_args()
    try:
        result = get_light_state() if args.command == "state" else set_light(args.command == "on")
        print("ESP32 response:", json.dumps(result))
        return 0
    except ESP32Error as error:
        print(f"ESP32 communication failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
