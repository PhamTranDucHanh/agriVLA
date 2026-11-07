import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agri_vla.language import load_registry
from agri_vla.policy import PolicyError, load_policy
from agri_vla.policy_audit import audit_policy
from agri_vla.preflight import project_root

ROOT = project_root()


def test_audit_is_complete_reproducible_and_refuses_overwrite(tmp_path):
    policy = load_policy(ROOT / "configs/policy_v0.yaml")
    registry = load_registry(ROOT / "configs/language_v0.yaml")
    before = sorted(tmp_path.rglob("*"))
    dry = audit_policy(policy, registry, output_dir=tmp_path / "one", dry_run=True)
    assert sorted(tmp_path.rglob("*")) == before
    first = audit_policy(policy, registry, output_dir=tmp_path / "one")
    second = audit_policy(policy, registry, output_dir=tmp_path / "two")
    assert dry["policy_lock_sha256"] == first["policy_lock_sha256"] == second["policy_lock_sha256"]
    assert first["wilt_combinations"] == 125 and first["intent_parameter_cases"] == 11 and first["policy_evaluations"] == 1375
    assert first["artifact_kind"] == "policy_audit_not_dataset"
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()
    lock_path = tmp_path / "one/policy_lock.json"
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == first["policy_lock_sha256"]
    for name, expected in first["lock"]["artifacts_sha256"].items():
        assert hashlib.sha256((tmp_path / "one" / name).read_bytes()).hexdigest() == expected
    with (tmp_path / "one/policy_truth_table.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1375
    with (tmp_path / "one/template_catalog.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 220
    with pytest.raises(PolicyError, match="already exists"):
        audit_policy(policy, registry, output_dir=tmp_path / "one")


@pytest.mark.parametrize("arguments", [
    ["--wilt", "0", "2", "5"],
    ["--wilt", "0", "2", "4", "--zone", "2"],
    ["--wilt", "0", "2", "4", "--intent", "STOP_ALL", "--zone", "2"],
    ["--wilt", "0", "2", "4", "--intent", "ZONE_ONLY", "--zone", "2", "--cap", "1"],
    ["--wilt", "0", "2", "4", "--intent", "AUTO+STOP_ALL"],
    ["--wilt", "0", "2", "4", "--split", "train", "--wording", "unseen"],
])
def test_explain_cli_rejects_invalid_requests_without_tracebacks(arguments):
    result = subprocess.run([sys.executable, "-m", "agri_vla.cli", "explain-policy", *arguments], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Policy error:" in result.stderr and "Traceback" not in result.stderr


def test_cli_examples_and_heldout_wording_do_not_import_ml():
    code = r'''
import importlib.abc
import sys
attempts = []
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchvision", "lerobot", "transformers", "peft"}:
            attempts.append(fullname)
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockML())
from agri_vla.cli import main
assert main(["explain-policy", "--wilt", "0", "2", "4", "--json"]) == 0
assert main(["explain-policy", "--wilt", "4", "4", "1", "--intent", "WORST_ONLY", "--split", "test", "--wording", "unseen"]) == 0
assert main(["audit-policy", "--dry-run"]) == 0
assert not attempts, attempts
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_explain_json_anchor_targets_are_exact():
    result = subprocess.run([sys.executable, "-m", "agri_vla.cli", "explain-policy", "--wilt", "0", "2", "4", "--json"], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert [item["target_action"] for item in data["examples"]] == [[0, 1, 3], [0, 0, 3], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
    assert all({x["language"] for x in item["instructions"]} == {"vi", "en"} for item in data["examples"])
