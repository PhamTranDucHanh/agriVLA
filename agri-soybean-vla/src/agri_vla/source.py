"""Source configuration and checksum-verified, no-overwrite local acquisition."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen


class SourceError(ValueError):
    """Invalid configuration/source or a non-destructive acquisition failure."""


def load_config(path: Path) -> dict:
    import yaml

    path = path.resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SourceError(f"Invalid YAML config: {error}") from error
    if not isinstance(config, dict) or config.get("config_version") != 1:
        raise SourceError("Expected config_version: 1")
    if not isinstance(config.get("base_dir", ".."), str):
        raise SourceError("base_dir must be a relative directory string")
    for key in ("paths", "files", "inspection", "expected"):
        if not isinstance(config.get(key), dict):
            raise SourceError(f"Missing or malformed config section: {key}")
    base = (path.parent / config.get("base_dir", "..")).resolve()
    config["_config_path"] = path
    config["_base"] = base
    config["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        for key in ("csv", "zip"):
            spec = config["files"][key]
            name = spec["name"]
            if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} or "\\" in name:
                raise SourceError(f"Unsafe file name for {key}")
            if not re.fullmatch(r"[a-f0-9]{32}", str(spec["md5"])):
                raise SourceError(f"Invalid MD5 for {key}")
            if not str(spec["url"]).startswith("https://"):
                raise SourceError("Source download URLs must use HTTPS")
            if type(spec["approximate_bytes"]) is not int or spec["approximate_bytes"] <= 0:
                raise SourceError(f"approximate_bytes for {key} must be positive")
        for key in ("raw_dir", "images_dir", "source_output_dir"):
            relative = Path(config["paths"][key])
            if relative.is_absolute() or ".." in relative.parts:
                raise SourceError(f"paths.{key} must be relative to base_dir without '..'")
            resolved = (base / relative).resolve()
            if not resolved.is_relative_to(base) or resolved == base:
                raise SourceError(f"paths.{key} escapes or equals base_dir")
        limits = config["inspection"]
        for key in ("max_member_bytes", "max_total_uncompressed_bytes", "contact_sheet_per_label", "contact_sheet_columns"):
            if type(limits[key]) is not int or limits[key] <= 0:
                raise SourceError(f"inspection.{key} must be a positive integer")
        if len(limits["thumbnail_size"]) != 2 or any(type(v) is not int or v <= 0 for v in limits["thumbnail_size"]):
            raise SourceError("thumbnail_size must contain two positive integers")
    except (KeyError, TypeError) as error:
        raise SourceError(f"Missing or malformed source config: {error}") from error
    if local_path(config, "source_output_dir").is_relative_to(local_path(config, "raw_dir")):
        raise SourceError("Inspection output must be outside raw_dir")
    if not local_path(config, "images_dir").is_relative_to(local_path(config, "raw_dir")) or local_path(config, "images_dir") == local_path(config, "raw_dir"):
        raise SourceError("images_dir must be a child of raw_dir")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise SourceError("seed must be a nonnegative integer")
    return config


def local_path(config: dict, key: str) -> Path:
    return (config["_base"] / config["paths"][key]).resolve()


def raw_path(config: dict, kind: str) -> Path:
    return local_path(config, "raw_dir") / config["files"][kind]["name"]


def checksums(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def verify_file(path: Path, expected_md5: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise SourceError(f"Expected a regular, non-symlink source file: {path}")
    digest = checksums(path)
    if digest["md5"] != expected_md5:
        raise SourceError(f"Checksum mismatch for {path}: expected {expected_md5}, got {digest['md5']}; file left untouched")
    return digest


def publish_new_file(temporary: Path, destination: Path) -> None:
    """Atomic publish without os.replace/rename overwriting an existing raw file."""
    os.link(temporary, destination)
    temporary.unlink()  # Only the temporary created by this operation, after publish.


def acquire_file(config: dict, kind: str, *, local: Path | None = None, dry_run: bool = False) -> dict:
    spec = config["files"][kind]
    destination = raw_path(config, kind)
    result = {"kind": kind, "path": str(destination), "url": spec["url"], "expected_md5": spec["md5"]}
    if destination.exists() or destination.is_symlink():
        return result | {"action": "reuse", **verify_file(destination, spec["md5"])}
    if local is not None:
        local = local.resolve()
        digest = verify_file(local, spec["md5"])
        result |= {"action": "import_local", "local_source": str(local), **digest}
    else:
        result |= {"action": "download", "approximate_bytes": spec["approximate_bytes"]}
    if dry_run:
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    if free < int(result.get("bytes", spec["approximate_bytes"])) + 16 * 1024 * 1024:
        raise SourceError(f"Insufficient free disk space for {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".part-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if local is not None:
                with local.open("rb") as source:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            else:
                request = Request(spec["url"], headers={"User-Agent": "AgriVLA-source-audit/0.1"})
                with urlopen(request, timeout=60) as response:
                    expected_length = response.headers.get("Content-Length")
                    size, next_progress = 0, 32 * 1024 * 1024
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                        if size >= next_progress:
                            print(f"{kind}: {size / 1024**2:.0f} MiB received", file=sys.stderr, flush=True)
                            next_progress += 32 * 1024 * 1024
                    if expected_length is not None and size != int(expected_length):
                        raise SourceError(f"Incomplete HTTP body: expected {expected_length}, got {size} bytes")
            output.flush()
            os.fsync(output.fileno())
        digest = verify_file(temporary, spec["md5"])
        publish_new_file(temporary, destination)
    except Exception as error:
        raise SourceError(f"Acquisition failed; destination not overwritten; partial file retained at {temporary}: {error}") from error
    return result | digest


def fetch_source(config: dict, *, local_csv: Path | None = None, local_zip: Path | None = None, dry_run: bool = False) -> dict:
    # Preflight existing/local files before performing any download or copy.
    plan = [
        acquire_file(config, "csv", local=local_csv, dry_run=True),
        acquire_file(config, "zip", local=local_zip, dry_run=True),
    ]
    if dry_run:
        return {"dry_run": True, "files": plan, "approximate_download_bytes": sum(p.get("approximate_bytes", 0) for p in plan if p["action"] == "download")}
    files = [acquire_file(config, "csv", local=local_csv), acquire_file(config, "zip", local=local_zip)]
    return {"dry_run": False, "status": "pass", "files": files}
