from pathlib import Path
import json
import tarfile

import pytest

from agri_vla.dataset import DatasetError
from agri_vla.handoff import package_handoff,verify_handoff


def fixture_tree(tmp_path):
    repo=tmp_path/"project";base=repo/"agri-soybean-vla";base.mkdir(parents=True)
    for name in ("AGENTS.md","README.md",".gitignore","AGRIVLA_CODEX_CONTEXT_2026-11-05.md","AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md","AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md","AGRIVLA_IMPLEMENTATION_STATUS.md"):
        (repo/name).write_text("fixture\n")
    for name in ("README.md","pyproject.toml","src/agri_vla/cli.py","data/raw/a.jpg","data/manifests/train.jsonl","reports/live.log",".cache/models/assets_lock.json",".cache/models/repo/revision/model.safetensors",".cache/models/repo/revision/.cache/download/metadata",".venv-train/secret",".vendor/secret","outputs/secret","models/prepared/host_path","private.env"):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("fixture\n")
    return repo,base


def test_handoff_roundtrip_and_exclusions(tmp_path):
    repo,base=fixture_tree(tmp_path)
    out=tmp_path/"transfer.tar"
    result=package_handoff(base,out)
    assert result["status"]=="pass"
    with tarfile.open(out) as archive:
        names=archive.getnames()
        assert any(n.endswith("data/raw/a.jpg") for n in names)
        assert any(n.endswith("model.safetensors") for n in names)
        assert not any("secret" in n or "host_path" in n or "private.env" in n or "download/metadata" in n or n.endswith("live.log") for n in names)
        fresh=tmp_path/"fresh";archive.extractall(fresh,filter="data")
    assert verify_handoff(fresh/"agriVLA",Path(str(out)+".manifest.json"))["verified_files"]==result["files"]
    (fresh/"agriVLA/agri-soybean-vla/README.md").write_text("changed")
    with pytest.raises(DatasetError,match="differs"):
        verify_handoff(fresh/"agriVLA",Path(str(out)+".manifest.json"))
    with pytest.raises(DatasetError,match="exists"):
        package_handoff(base,out)


def test_handoff_dry_run_and_path_guard(tmp_path):
    repo,base=fixture_tree(tmp_path)
    out=tmp_path/"transfer.tar"
    assert package_handoff(base,out,dry_run=True)["status"]=="dry_run"
    assert not out.exists()
    manifest=tmp_path/"unsafe.json";manifest.write_text(json.dumps({"format":"agri_soybean_handoff_v0","files":{"../outside":{"bytes":0,"sha256":"x"}}}))
    with pytest.raises(DatasetError,match="Unsafe"):
        verify_handoff(repo,manifest)
