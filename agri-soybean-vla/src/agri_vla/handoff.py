"""Portable source + locked data/model-cache handoff, without venv/vendor/Git."""
import json
import os
from pathlib import Path
import tarfile

from .dataset import require, new_output
from .training import file_sha, write_json


def selected_files(base):
    repo = base.parent
    selected = [repo / n for n in ("AGENTS.md", "README.md", ".gitignore", "AGRIVLA_CODEX_CONTEXT_2026-11-05.md", "AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md", "AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md", "AGRIVLA_IMPLEMENTATION_STATUS.md")]
    roots = {"src","configs","schemas","tests","examples","scripts","data","reports"}
    for directory, dirs, names in os.walk(base):
        parent = Path(directory)
        if parent == base:
            dirs[:] = [d for d in dirs if d in roots]
        else:
            dirs[:] = [d for d in dirs if d not in {"__pycache__",".pytest_cache"} and not d.endswith(".egg-info")]
        for name in sorted(names):
            p = parent/name
            # Browser remains live while packaging; structured reports are stable,
            # request logs can change between hashing and adding an archive member.
            if parent.is_relative_to(base/"reports") and p.suffix == ".log":
                continue
            if parent == base and not (name.endswith(".md") or name == "pyproject.toml" or name.startswith("requirements-")):
                continue
            require(not p.is_symlink(), f"Unexpected handoff symlink: {p}")
            if p.is_file():
                selected.append(p)
    model_cache = base / ".cache/models"
    require((model_cache / "assets_lock.json").is_file(), "Prepare pinned model cache first")
    for p in sorted(model_cache.rglob("*")):
        if p.is_file() and ".cache" not in p.relative_to(model_cache).parts:
            require(not p.is_symlink(), "Expected regular downloaded model files")
            selected.append(p)
    return repo, sorted(set(selected))


def package_handoff(base, output, *, dry_run=False):
    output = Path(output).resolve()
    require(output.suffix == ".tar", "Use .tar: weights/images are already compressed")
    new_output(output)
    new_output(Path(str(output)+".sha256"))
    new_output(Path(str(output)+".manifest.json"))
    repo, paths = selected_files(base)
    require(output not in paths, "Archive cannot include itself")
    summary = {"status": "dry_run" if dry_run else "pass", "files": len(paths), "bytes": sum(p.stat().st_size for p in paths), "excluded": ["Git metadata", "venvs", "vendor", "prepared bundle with host paths", "outputs", "pip caches", "HF download metadata", "volatile .log files"]}
    if dry_run:
        return summary
    manifest = {"format": "agri_soybean_handoff_v0", "archive_root": "agriVLA", "files": {p.relative_to(repo).as_posix(): {"sha256":file_sha(p), "bytes":p.stat().st_size} for p in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x") as archive:
        for p in paths:
            archive.add(p, arcname="agriVLA/"+p.relative_to(repo).as_posix(), recursive=False)
    checksum = file_sha(output)
    Path(str(output)+".sha256").write_text(f"{checksum}  {output.name}\n")
    write_json(Path(str(output)+".manifest.json"), manifest)
    return summary | {"archive":str(output), "sha256":checksum}


def verify_handoff(repo, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    require(manifest["format"] == "agri_soybean_handoff_v0", "Unknown manifest format")
    for name, item in manifest["files"].items():
        p = (Path(repo)/name).resolve()
        require(p.is_relative_to(Path(repo).resolve()), "Unsafe manifest path")
        require(p.stat().st_size == item["bytes"] and file_sha(p) == item["sha256"], f"Transfer file differs: {name}")
    return {"status":"pass", "verified_files":len(manifest["files"])}
