"""Environment checks; ML imports are confined to explicit train-mode checks."""

from importlib import metadata
import importlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

LEROBOT_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
EXPECTED_BRANCH = "feat/compose-dataset"
DATA_DISTRIBUTIONS = ("agri-vla-soybean", "Pillow", "PyYAML", "jsonschema", "pytest")
TRAIN_DISTRIBUTIONS = ("lerobot", "torch", "torchvision", "transformers", "peft", "accelerate", "datasets")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def collect_report(mode: str, *, require_cuda: bool = False) -> dict:
    if mode not in {"data", "train"}:
        raise ValueError("mode must be data or train")
    if require_cuda and mode != "train":
        raise ValueError("--require-cuda is only valid with --mode train")

    root = project_root()
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add("python", "pass" if sys.version_info[:2] == (3, 12) else "fail", platform.python_version())
    expected_venv = (root / (".venv-train" if mode == "train" else ".venv")).resolve()
    isolated = sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == expected_venv
    add("project_virtualenv", "pass" if isolated else "fail", str(Path(sys.prefix)))
    add("linux", "pass" if platform.system() == "Linux" else "fail", platform.system())
    add("project_writable", "pass" if os.access(root, os.W_OK) else "fail", str(root))
    free_gib = shutil.disk_usage(root).free / 1024**3
    threshold = 2 if mode == "data" else 15
    add("free_disk", "pass" if free_gib >= threshold else "warn", f"{free_gib:.2f} GiB free; planning threshold {threshold} GiB")

    try:
        branch = subprocess.run(
            ["git", "-C", str(root.parent), "branch", "--show-current"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        add("git_branch", "pass" if branch == EXPECTED_BRANCH else "fail", branch or "detached HEAD")
    except (OSError, subprocess.SubprocessError) as error:
        add("git_branch", "fail", str(error))

    versions = {}
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = declared["project"]["optional-dependencies"]
    expected_versions = {
        requirement.split("==")[0].lower(): requirement.split("==")[1]
        for requirement in optional["data"] + optional["dev"]
    }
    if mode == "train":
        expected_versions["lerobot"] = "0.6.1"
        for line in (root / "requirements-train.constraints.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                name, version = line.strip().split("==", 1)
                expected_versions[name.lower()] = version
    distributions = DATA_DISTRIBUTIONS + (TRAIN_DISTRIBUTIONS if mode == "train" else ())
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
            expected = expected_versions.get(name.lower())
            # CUDA wheels may append a local version such as +cu126.
            matches = expected is None or versions[name].partition("+")[0] == expected
            add(f"package:{name}", "pass" if matches else "fail", versions[name] + (f"; expected {expected}" if expected else ""))
        except metadata.PackageNotFoundError:
            add(f"package:{name}", "fail", "not installed; run setup.sh for this mode")

    if mode == "train":
        vendor = root / ".vendor" / "lerobot"
        try:
            head = subprocess.run(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
            add("lerobot_commit", "pass" if head == LEROBOT_COMMIT else "fail", head)
        except (OSError, subprocess.SubprocessError):
            add("lerobot_commit", "fail", f"missing verified checkout at {vendor}")
        if "torch" in versions:
            try:
                import torch  # The only heavy import, never reached in data mode.

                available = torch.cuda.is_available()
                detail = torch.cuda.get_device_name(0) if available else "CUDA unavailable"
                add("cuda", "pass" if available else "fail" if require_cuda else "warn", detail)
            except Exception as error:
                add("cuda", "fail", f"torch import/CUDA check failed: {error}")
        else:
            add("cuda", "fail" if require_cuda else "warn", "not checked: torch is not installed")
        for module in ("torchvision", "datasets", "peft", "lerobot.datasets.lerobot_dataset", "lerobot.policies.smolvla.modeling_smolvla"):
            try:
                imported = importlib.import_module(module)
                add(f"import:{module}", "pass", str(getattr(imported, "__file__", "imported")))
            except Exception as error:
                add(f"import:{module}", "fail", f"{type(error).__name__}: {error}")

    return {
        "report_version": 1,
        "mode": mode,
        "status": "fail" if any(c["status"] == "fail" for c in checks) else "pass",
        "scope": "environment_only",
        "checks": checks,
        "versions": versions,
        "paths": {"project": str(root), "venv": str(expected_venv)},
        "not_checked": ["raw_data", "generated_dataset", "model_weights", "training", "4gb_training_fit"]
        + (["CUDA", "LeRobot"] if mode == "data" else ["model_forward_backward", "checkpoint_resume"]),
    }
