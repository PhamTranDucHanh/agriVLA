from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from agri_vla.dataset import (DatasetError, SPLITS, load_compose_config, load_inputs, read_jsonl,
                              jsonl_bytes, safe_output)
from agri_vla.language import load_registry
from agri_vla.policy import load_policy
from agri_vla.policy_audit import audit_policy
from agri_vla.preflight import project_root
from agri_vla.preview import preview_dataset, select_samples
from agri_vla.scenes import SourceSampler, compose, generate_split
from agri_vla.source import SourceError
from agri_vla.split import canonical_bytes, digest, load_split_config, split_sources
from agri_vla.validate import semantic_digest, validate_dataset, validate_records
from test_split import fixture_split, save_inputs, save_image, pattern


@pytest.fixture(scope="module")
def fixture_project(tmp_path_factory):
    base = tmp_path_factory.mktemp("compose-inputs")
    split_config, initial = fixture_split(base)
    rows = []
    for row in initial:
        if not row["eligible_v0"]:
            rows.append(row)
            continue
        for variant in range(6):
            clone = deepcopy(row)
            clone["source_image_id"] = f"{row['split_group']}_18_09_2019_06_{row['wilt_score']}{variant}_00.jpg"
            clone["original_path"] = "images/" + clone["source_image_id"]
            save_image(base, clone, pattern(len(rows) * 79 + 99))
            rows.append(clone)
    split_config["expected_counts"] = {split: 30 for split in SPLITS}
    split_config["output_dir"] = "source_splits"
    save_inputs(base, split_config, rows)
    assert split_sources(load_split_config(base / "split.yaml"))["status"] == "pass"
    (base / "configs").mkdir()
    (base / "schemas").mkdir()
    for name in ("policy_v0.yaml", "language_v0.yaml"):
        shutil.copyfile(project_root() / "configs" / name, base / "configs" / name)
    shutil.copyfile(project_root() / "schemas/scene_v0.schema.json", base / "schemas/scene_v0.schema.json")
    audit_dir = base / "policy_audit"
    audit_policy(load_policy(base / "configs/policy_v0.yaml"), load_registry(base / "configs/language_v0.yaml"), output_dir=audit_dir)
    config = yaml.safe_load((project_root() / "configs/dataset_v0.yaml").read_text())
    config["base_dir"] = "."
    config["paths"]["raw_dir"] = "images"
    g = config["generation"]
    g.update({"source_split_config": "split.yaml", "source_split_dir": "source_splits", "policy_audit_dir": "policy_audit",
              "output_dir": "generated", "record_counts": {"train": 1200, "val": 600, "test": 600},
              "split_lock_sha256": digest((base / "source_splits/split_lock.json").read_bytes()),
              "policy_lock_sha256": digest((audit_dir / "policy_lock.json").read_bytes())})
    (base / "dataset.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    return base


@pytest.fixture(scope="module")
def inputs(fixture_project):
    return load_inputs(load_compose_config(fixture_project / "dataset.yaml"))


@pytest.fixture(scope="module")
def records(inputs):
    return {split: generate_split(inputs, split) for split in SPLITS}


@pytest.fixture(scope="module")
def published(fixture_project):
    config = load_compose_config(fixture_project / "dataset.yaml")
    result = compose(config)
    assert result["status"] == "pass"
    return fixture_project / "generated"


def test_generation_has_exact_quotas_paired_wording_and_actual_template_coverage(inputs, records):
    report = validate_records(inputs, records)
    assert [report["splits"][s]["records"] for s in SPLITS] == [1200, 600, 600]
    assert [report["splits"][s]["cases"] for s in SPLITS] == [600, 150, 150]
    assert report["splits"]["train"]["intent_cases"] == {"AUTO": 180, "WORST_ONLY": 120, "ZONE_ONLY": 90, "CAP_LEVEL": 90, "EXCLUDE_ZONE": 60, "STOP_ALL": 60}
    assert all(n == 11 for n in report["permutations_checked"].values())
    assert all(not v for fields in report["source_intersections"].values() for v in fields.values())
    assert len(report["family_coverage_actual"]["train"]["seen"]) == 36
    assert all(len(v["seen"]) == 6 for v in report["splits"]["train"]["parameter_family_coverage"].values())
    for split in ("val", "test"):
        assert report["splits"][split]["languages"] == {"vi": 300, "en": 300}
        assert report["splits"][split]["wording"] == {"seen": 300, "unseen": 300}
        assert len(report["family_coverage_actual"][split]["unseen"]) == 12


def test_contrastive_targets_and_permutation_examples_independently(records):
    for split, rows in records.items():
        base = [r for r in rows if r["generation_metadata"]["kind"] == "contrastive_base" and r["language"] == "vi" and r["wording_partition"] == "seen"]
        assert len({r["scene_id"] for r in base}) == 1
        expected = {("AUTO", "{}"): [0, 1, 3], ("WORST_ONLY", "{}"): [0, 0, 3],
                    ("ZONE_ONLY", '{"zone": 2}'): [0, 1, 0], ("CAP_LEVEL", '{"cap": 2}'): [0, 1, 2],
                    ("EXCLUDE_ZONE", '{"zone": 3}'): [0, 1, 0], ("STOP_ALL", "{}"): [0, 0, 0]}
        for row in base:
            key = row["intent"], json.dumps(row["intent_params"])
            if key in expected:
                assert row["target_action"] == expected[key]
        child = next(r for r in rows if r["generation_metadata"]["kind"] == "permutation" and r["intent"] == "ZONE_ONLY" and r["intent_params"] == {"zone": 3})
        assert child["teacher_annotations"]["wilt_scores"] == [4, 0, 2]
        assert child["target_action"] == [0, 0, 1]
        same_instruction = defaultdict(set)
        for row in rows:
            same_instruction[row["instruction"]].add(tuple(row["target_action"]))
        assert any(len(targets) >= 3 for targets in same_instruction.values())


@pytest.mark.parametrize("mutation", ["target", "teacher", "path", "duplicate_image", "cross_split", "instruction", "extra_feature",
    "provenance", "pair_id", "sample_id", "case_id", "seed", "heldout", "edge_kind", "permutation", "translation_missing"])
def test_qa_rejects_tampered_scenes(inputs, records, mutation):
    broken = deepcopy(records)
    row = broken["train"][0]
    if mutation == "target":
        row["target_action"] = [3, 3, 3]
    elif mutation == "teacher":
        row["teacher_annotations"]["wilt_scores"] = [4, 4, 4]
    elif mutation == "path":
        row["images"]["zone1"] = "../escape.jpg"
    elif mutation == "duplicate_image":
        row["source_image_ids"][1] = row["source_image_ids"][0]
    elif mutation == "cross_split":
        row["source_image_ids"][0] = inputs.rows["val"][0]["source_image_id"]
    elif mutation == "instruction":
        row["instruction"] = "Water the driest soil."
    elif mutation == "extra_feature":
        row["observation.state"] = [0, 2, 4]
    elif mutation == "provenance":
        row["provenance"]["source_sha256"][0] = "0" * 64
    elif mutation in {"pair_id", "sample_id", "case_id"}:
        row[{"pair_id": "translation_pair_id"}.get(mutation, mutation)] = "tampered"
    elif mutation == "seed":
        row["seed"] += 1
    elif mutation == "heldout":
        row["template_family_id"] = "auto_val_01"
    elif mutation == "edge_kind":
        row["generation_metadata"]["kind"] = "all_high"
    elif mutation == "permutation":
        child = next(r for r in broken["train"] if r["generation_metadata"]["kind"] == "permutation")
        child["generation_metadata"]["permutation"] = [1, 2, 3]
    else:
        broken["train"].pop()
    with pytest.raises((DatasetError, ValueError)):
        validate_records(inputs, broken)


@pytest.mark.parametrize("duplicate_kind", ["source_image_id", "sha256", "pixel_sha256", "duplicate_component_id", "near", "uncertain"])
def test_within_scene_duplicates_are_rejected(inputs, monkeypatch, duplicate_kind):
    test_inputs = deepcopy(inputs)
    sources = [dict(r) for r in test_inputs.rows["train"][:3]]
    test_inputs.pair_cache.clear()
    if duplicate_kind in ("near", "uncertain"):
        a, b = [r["source_image_id"] for r in sources[:2]]
        test_inputs.features[b] = deepcopy(test_inputs.features[a])
        if duplicate_kind == "uncertain":
            monkeypatch.setattr("agri_vla.dataset.classify_pair", lambda *args: "needs_review")
    else:
        sources[1][duplicate_kind] = sources[0][duplicate_kind]
    assert not test_inputs.distinct(sources)


def test_sampler_refuses_impossible_three_distinct_sources_and_preserves_rng(inputs):
    copy = deepcopy(inputs)
    copy.rows["val"] = [r for r in copy.rows["val"] if r["wilt_score"] != 4] + [r for r in copy.rows["val"] if r["wilt_score"] == 4][:2]
    before = random.getstate()
    with pytest.raises(DatasetError, match="No three"):
        SourceSampler(copy, "val", random.Random(7)).take([4, 4, 4])
    assert random.getstate() == before


def test_dry_run_and_reproducibility_no_overwrite(fixture_project, tmp_path, published):
    config = load_compose_config(fixture_project / "dataset.yaml")
    before = sorted(tmp_path.rglob("*"))
    dry = compose(config, output_dir=tmp_path / "dry", dry_run=True)
    assert sorted(tmp_path.rglob("*")) == before
    real = compose(config, output_dir=tmp_path / "real")
    assert dry["dataset_lock_sha256"] == real["dataset_lock_sha256"]
    for path in published.iterdir():
        assert path.read_bytes() == (tmp_path / "real" / path.name).read_bytes()
    with pytest.raises(DatasetError, match="already exists"):
        compose(config)
    with pytest.raises(DatasetError, match="outside raw"):
        compose(config, output_dir=fixture_project / "images/generated")


def test_standalone_validation_detects_manifest_byte_tampering(fixture_project, published, tmp_path):
    config = load_compose_config(fixture_project / "dataset.yaml")
    report, _, _ = validate_dataset(config, published)
    assert report["status"] == "pass"
    shutil.copytree(published, tmp_path / "copy")
    path = tmp_path / "copy/train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DatasetError, match="lock"):
        validate_dataset(config, tmp_path / "copy")


@pytest.mark.parametrize("target", ["split_lock", "source_manifest", "image", "policy_config", "language_config", "policy_artifact"])
def test_pinned_inputs_detect_source_or_config_tampering(fixture_project, tmp_path, target):
    base = tmp_path / "copy"
    shutil.copytree(fixture_project, base, ignore=shutil.ignore_patterns("generated"))
    paths = {"split_lock": "source_splits/split_lock.json", "source_manifest": "source_splits/train.sources.jsonl",
             "policy_config": "configs/policy_v0.yaml", "language_config": "configs/language_v0.yaml", "policy_artifact": "policy_audit/template_catalog.csv"}
    if target == "image":
        row = read_jsonl(base / "source_splits/train.sources.jsonl")[0]
        path = base / row["original_path"]
    else:
        path = base / paths[target]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((DatasetError, SourceError, ValueError)):
        load_inputs(load_compose_config(base / "dataset.yaml"))


@pytest.mark.parametrize("mutation", ["seed", "odd_records", "wrong_quota", "unknown", "unsafe_path", "relax_duplicates", "fractional_quota"])
def test_generation_config_rejects_invalid_values(fixture_project, tmp_path, mutation):
    config = yaml.safe_load((fixture_project / "dataset.yaml").read_text())
    g = config["generation"]
    if mutation == "seed":
        config["seed"] = True
    elif mutation == "odd_records":
        g["record_counts"]["train"] = 601
    elif mutation == "wrong_quota":
        g["train_intent_percent"]["AUTO"] += 1
    elif mutation == "unknown":
        g["made_up"] = 1
    elif mutation == "unsafe_path":
        g["source_split_dir"] = "../outside"
    elif mutation == "relax_duplicates":
        g["within_scene_near_policy"] = "allow"
    else:
        g["record_counts"]["train"] = 602
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises((DatasetError, SourceError, ValueError)):
        load_compose_config(path)


def test_preview_strata_html_images_dry_run_and_refusal(fixture_project, published, tmp_path, records):
    config = load_compose_config(fixture_project / "dataset.yaml")
    selected = select_samples(records, 72)
    assert len(selected) == 72
    assert {r["language"] for r in selected} == {"vi", "en"}
    assert {r["generation_metadata"]["kind"] for r in selected} == {"all_off", "all_high", "tie", "below_threshold", "contrastive_base", "permutation", "balanced"}
    high = [r for r in selected if r["split"] == "test" and r["intent"] in {"AUTO", "STOP_ALL"}
            and r["generation_metadata"]["kind"] == "all_high"]
    assert len({r["scene_id"] for r in high}) == 1
    assert all(r["teacher_annotations"]["wilt_scores"] == [4, 4, 4] for r in high)
    assert {tuple(r["target_action"]) for r in high} == {(3, 3, 3), (0, 0, 0)}
    before = sorted(tmp_path.rglob("*"))
    preview_dataset(config, published, output_dir=tmp_path / "preview", count=72, dry_run=True)
    assert sorted(tmp_path.rglob("*")) == before
    result = preview_dataset(config, published, output_dir=tmp_path / "preview", count=72)
    page = (tmp_path / "preview/index.html").read_text()
    assert page.count("<article>") == 72 and page.count("<img ") == 216
    assert "<script" not in page and "target_action" in page and "source_image_ids" in page
    assert len(result["contact_sheets"]) == 12
    assert result["visual_review"].startswith("not_performed")
    for name in result["contact_sheets"]:
        with Image.open(tmp_path / "preview" / name) as image:
            assert image.mode == "RGB" and image.width == 1180
    with pytest.raises(DatasetError, match="already exists"):
        preview_dataset(config, published, output_dir=tmp_path / "preview", count=72)


def test_semantic_hash_ignores_order_but_not_content(records):
    rows = records["train"]
    assert semantic_digest(rows) == semantic_digest(list(reversed(rows)))
    altered = deepcopy(rows)
    altered[0]["instruction"] += " changed"
    assert semantic_digest(rows) != semantic_digest(altered)
    assert len(jsonl_bytes(rows).splitlines()) == len(rows)


def test_preview_can_select_every_record_once(records):
    total = sum(len(rows) for rows in records.values())
    selected = select_samples(records, total)
    assert len(selected) == total == len({r["sample_id"] for r in selected})
    with pytest.raises(DatasetError):
        select_samples(records, 59)


def test_new_cli_commands_never_import_ml(fixture_project, published):
    code = r'''
import importlib.abc, sys
attempts=[]
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','lerobot','transformers','peft'}:
            attempts.append(fullname)
            raise AssertionError(fullname)
sys.meta_path.insert(0,BlockML())
from agri_vla.cli import main
base=sys.argv[1]
assert main(['compose','--config',base+'/dataset.yaml','--output-dir',base+'/cli-dry','--dry-run'])==0
assert main(['validate','--config',base+'/dataset.yaml','--manifests',base+'/generated','--dry-run'])==0
assert main(['preview','--config',base+'/dataset.yaml','--manifests',base+'/generated','--count','60','--dry-run'])==0
assert not attempts,attempts
'''
    result = subprocess.run([sys.executable, "-c", code, str(fixture_project)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr
