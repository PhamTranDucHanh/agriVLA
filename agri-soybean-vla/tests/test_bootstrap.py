from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

from agri_vla.preflight import LEROBOT_COMMIT, collect_report, project_root

ROOT = project_root()


def test_data_commands_cannot_import_ml_frameworks():
    # Reject both installed and missing ML modules, so an accidental import fails.
    code = r'''
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchvision", "lerobot", "transformers", "peft"}:
            raise AssertionError("Data command tried to import " + fullname)
sys.meta_path.insert(0, BlockML())
from agri_vla.cli import main
assert main(["contract"]) == 0
assert main(["preflight", "--mode", "data", "--json"]) == 0
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_refuses_to_overwrite_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("keep this existing artifact", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "agri_vla.cli", "preflight", "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "keep this existing artifact"


def test_data_mode_rejects_cuda_flag():
    result = subprocess.run(
        [sys.executable, "-m", "agri_vla.cli", "preflight", "--mode", "data", "--require-cuda"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires --mode train" in result.stderr


def test_missing_dependency_fails_preflight(monkeypatch):
    from importlib import metadata

    original = metadata.version

    def version(name):
        if name == "Pillow":
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(metadata, "version", version)
    report = collect_report("data")
    assert report["status"] == "fail"
    assert next(c for c in report["checks"] if c["name"] == "package:Pillow")["status"] == "fail"


def test_train_setup_dry_run_has_no_side_effects():
    before = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if ".cache" not in p.parts)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/setup.sh"), "--mode", "train", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    after = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if ".cache" not in p.parts)
    assert before == after
    assert LEROBOT_COMMIT in result.stdout
    assert "smolvla\\,peft\\,training" in result.stdout
    assert "cu126" in result.stdout
    assert "hf download" not in result.stdout
    assert "lerobot-train" not in result.stdout


def test_model_contract_separates_teacher_from_features():
    contract = json.loads(files("agri_vla").joinpath("model_contract.json").read_text(encoding="utf-8"))
    assert set(contract["model_inputs"]) == {
        "observation.images.camera1", "observation.images.camera2", "observation.images.camera3",
        "observation.state", "task",
    }
    assert not set(contract["metadata_only"]) & set(contract["model_inputs"])
    assert contract["model_inputs"]["observation.state"]["normalization"] == "IDENTITY"
    assert contract["model_inputs"]["observation.state"]["value"] == [0, 0, 0]
    assert contract["physical_irrigation_validated"] is False
