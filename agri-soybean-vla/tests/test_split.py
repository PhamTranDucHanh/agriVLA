from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
import sys

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest
import yaml

from agri_vla.ingest import dhash64
from agri_vla.preflight import project_root
from agri_vla.source import SourceError
from agri_vla.split import Components, load_split_config, split_sources


def pattern(seed):
    rng = random.Random(seed)
    image = Image.new("RGB", (40, 30))
    image.putdata([tuple(rng.randrange(20, 220) for _ in range(3)) for _ in range(40 * 30)])
    return image


def fixture_split(tmp_path, *, groups=3):
    config = yaml.safe_load((project_root() / "configs/split_v0.yaml").read_text())
    config["base_dir"] = "."
    config["output_dir"] = "splits"
    config["source"]["manifest"] = "sources.jsonl"
    config["source"]["report"] = "source_report.json"
    group_names = [f"{i}-{i * 10}" for i in range(1, groups + 1)]
    config["assignments"] = {"train": group_names[:-2], "val": [group_names[-2]], "test": [group_names[-1]]}
    config["expected_counts"] = {"train": (groups - 2) * 5, "val": 5, "test": 5}
    rows = []
    for index, group in enumerate(group_names):
        camera, plot = group.split("-")
        for score in range(5):
            image_id = f"{group}_18_09_2019_06_3{score}_00.jpg"
            rows.append({"schema_version": "agri-soybean-source-v0", "source_image_id": image_id,
                         "original_path": f"images/{image_id}", "split_group": group,
                         "treatment_camera": group, "camera_id": camera, "plot_id": plot,
                         "annotation": score, "annotation_raw": str(score), "wilt_score": score,
                         "eligible_v0": True, "qa_status": "valid", "qa_issues": [], "exclusion_reasons": [],
                         "readable": True, "mode": "RGB", "width": 40, "height": 30})
            save_image(tmp_path, rows[-1], pattern(index * 100 + score))
    excluded = deepcopy(rows[0]) | {"source_image_id": "excluded-label-5.jpg", "eligible_v0": False,
                                    "annotation": 5, "annotation_raw": "5", "wilt_score": None,
                                    "qa_status": "quarantined", "exclusion_reasons": ["annotation_outside_v0_domain"]}
    rows.append(excluded)
    save_inputs(tmp_path, config, rows)
    return config, rows


def save_image(base, row, image):
    output = BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    path = base / row["original_path"]
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    row["sha256"] = hashlib.sha256(data).hexdigest()
    row["pixel_sha256"] = hashlib.sha256(f"RGB:{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    row["dhash64"] = dhash64(image)


def save_inputs(base, config, rows):
    content = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (base / "sources.jsonl").write_bytes(content)
    count = sum(row["eligible_v0"] for row in rows)
    report = json.dumps({"status": "pass", "row_accounting": {"total": len(rows), "eligible_v0": count, "excluded": len(rows) - count}}).encode()
    (base / "source_report.json").write_bytes(report)
    config["source"]["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    config["source"]["report_sha256"] = hashlib.sha256(report).hexdigest()
    save_config(base, config)


def save_config(base, config):
    path = base / "split.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_split_config(path)


def run_fixture(base, config, rows, **kwargs):
    save_inputs(base, config, rows)
    return split_sources(load_split_config(base / "split.yaml"), **kwargs)


def test_group_split_filters_excluded_and_has_empty_intersections(tmp_path):
    config, rows = fixture_split(tmp_path)
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "pass" and result["excluded_images"] == 1
    report = json.loads((tmp_path / "splits/split_report.json").read_text())
    assert all(not values for fields in report["intersections"].values() for values in fields.values())
    for split, group in (("train", "1-10"), ("val", "2-20"), ("test", "3-30")):
        selected = [json.loads(line) for line in (tmp_path / f"splits/{split}.sources.jsonl").read_text().splitlines()]
        assert len(selected) == 5
        assert {r["split_group"] for r in selected} == {group}
        assert all(r["eligible_v0"] is True and r["wilt_score"] in range(5) for r in selected)
    assert (tmp_path / "splits/split_lock.json").is_file()


def test_dry_run_makes_no_files_and_matches_real_lock(tmp_path):
    config, rows = fixture_split(tmp_path)
    loaded = load_split_config(tmp_path / "split.yaml")
    before = sorted(tmp_path.rglob("*"))
    dry = split_sources(loaded, dry_run=True)
    assert sorted(tmp_path.rglob("*")) == before
    real = split_sources(loaded)
    assert dry["lock_sha256"] == real["lock_sha256"]
    assert dry["split_stats"] == real["split_stats"]


def test_reproducibility_hashes_and_no_overwrite(tmp_path):
    config, rows = fixture_split(tmp_path)
    loaded = load_split_config(tmp_path / "split.yaml")
    first = split_sources(loaded)
    second = split_sources(loaded, output_dir=tmp_path / "rerun")
    assert first["lock_sha256"] == second["lock_sha256"]
    for path in (tmp_path / "splits").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (tmp_path / "rerun" / path.relative_to(tmp_path / "splits")).read_bytes()
    with pytest.raises(SourceError, match="already exists"):
        split_sources(loaded)


def test_dhash_collision_is_not_automatically_a_duplicate(tmp_path):
    config, rows = fixture_split(tmp_path)
    save_image(tmp_path, rows[0], Image.new("RGB", (40, 30), "red"))
    save_image(tmp_path, rows[5], Image.new("RGB", (40, 30), "blue"))
    assert rows[0]["dhash64"] == rows[5]["dhash64"]
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "pass"
    pairs = [json.loads(line) for line in (tmp_path / "splits/near_duplicate_pairs.jsonl").read_text().splitlines()]
    pair = next(p for p in pairs if set(p["source_image_ids"]) == {rows[0]["source_image_id"], rows[5]["source_image_id"]})
    assert pair["decision"] == "distinct_under_policy"


@pytest.mark.parametrize("duplicate_kind", ["exact", "near"])
def test_cross_split_duplicate_blocks_lock_and_manifests(tmp_path, duplicate_kind):
    config, rows = fixture_split(tmp_path)
    with Image.open(tmp_path / rows[0]["original_path"]) as original:
        image = original.copy() if duplicate_kind == "exact" else original.point(lambda p: p + 1)
    save_image(tmp_path, rows[5], image)
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "fail"
    assert any(f.get("field") == "duplicate_component_id" for f in result["failures"])
    assert not (tmp_path / "splits/split_lock.json").exists()
    assert not (tmp_path / "splits/train.sources.jsonl").exists()
    assert (tmp_path / "splits/split_report.json").exists()


def test_same_split_exact_duplicates_are_kept_without_relabeling(tmp_path):
    config, rows = fixture_split(tmp_path, groups=4)
    with Image.open(tmp_path / rows[0]["original_path"]) as image:
        save_image(tmp_path, rows[5], image.copy())
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "pass"
    assert result["split_stats"]["train"]["images"] == 10
    assert result["near_duplicates"]["decisions"]["exact_duplicate"] >= 1


def test_exact_duplicate_conflicting_labels_block_even_within_train(tmp_path):
    config, rows = fixture_split(tmp_path, groups=4)
    with Image.open(tmp_path / rows[0]["original_path"]) as image:
        save_image(tmp_path, rows[6], image.copy())
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "fail"
    assert any(f["type"] == "exact_duplicate_label_conflict" for f in result["failures"])


def test_uncertain_cross_split_pair_requires_manual_review(tmp_path):
    config, rows = fixture_split(tmp_path)
    save_image(tmp_path, rows[0], Image.new("RGB", (40, 30), (100, 100, 100)))
    save_image(tmp_path, rows[5], Image.new("RGB", (40, 30), (101, 101, 101)))
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "fail"
    assert any(f["type"] == "unresolved_cross_split_candidate" for f in result["failures"])


def test_camera_isolation_and_expected_counts_are_checked(tmp_path):
    config, rows = fixture_split(tmp_path)
    for row in rows[5:10]:
        old_path = tmp_path / row["original_path"]
        row |= {"camera_id": "1", "split_group": "1-20", "treatment_camera": "1-20",
                "source_image_id": row["source_image_id"].replace("2-20_", "1-20_")}
        row["original_path"] = "images/" + row["source_image_id"]
        with Image.open(old_path) as image:
            save_image(tmp_path, row, image.copy())
    config["assignments"]["val"] = ["1-20"]
    config["expected_counts"]["train"] = 100
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "fail"
    assert any(f.get("field") == "camera_id" for f in result["failures"])
    assert any(f["type"] == "expected_count_mismatch" for f in result["failures"])


def test_pinned_manifest_and_actual_image_mutation_are_detected(tmp_path):
    config, rows = fixture_split(tmp_path)
    loaded = load_split_config(tmp_path / "split.yaml")
    manifest = tmp_path / "sources.jsonl"
    content = manifest.read_bytes()
    manifest.write_bytes(content + b"\n")
    with pytest.raises(SourceError, match="SHA256 mismatch"):
        split_sources(loaded)
    manifest.write_bytes(content)
    (tmp_path / rows[0]["original_path"]).write_bytes(b"changed image")
    with pytest.raises(SourceError, match="Image SHA256 changed"):
        split_sources(loaded)


@pytest.mark.parametrize("mutation", ["duplicate_group", "unknown_group", "null_wilt", "group_mismatch", "camera_type"])
def test_bad_assignment_or_eligible_metadata_is_rejected(tmp_path, mutation):
    config, rows = fixture_split(tmp_path)
    if mutation == "duplicate_group":
        config["assignments"]["val"].append("1-10")
        with pytest.raises(SourceError, match="more than once"):
            save_config(tmp_path, config)
        return
    if mutation == "unknown_group":
        config["assignments"]["val"] = ["9-99"]
    elif mutation == "null_wilt":
        rows[0]["wilt_score"] = None
    elif mutation == "camera_type":
        rows[0]["camera_id"] = 1
    else:
        rows[0]["camera_id"] = "99"
    with pytest.raises(SourceError):
        run_fixture(tmp_path, config, rows)


def test_components_preserve_transitive_connections():
    components = Components(4)
    components.join(1, 2)
    components.join(2, 3)
    assert components.find(1) == components.find(3)
    assert components.find(0) != components.find(3)


def test_split_cli_does_not_import_ml(tmp_path):
    fixture_split(tmp_path)
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
assert main(["split", "--config", sys.argv[1], "--dry-run"]) == 0
assert not attempts, attempts
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "split.yaml")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manual_review_can_resolve_an_uncertain_pair_with_a_pinned_reason(tmp_path):
    config, rows = fixture_split(tmp_path)
    save_image(tmp_path, rows[0], Image.new("RGB", (40, 30), (100, 100, 100)))
    save_image(tmp_path, rows[5], Image.new("RGB", (40, 30), (101, 101, 101)))
    review = [{"source_image_ids": [rows[0]["source_image_id"], rows[5]["source_image_id"]],
               "decision": "distinct", "reason": "Fixture provenance identifies independent synthetic scenes."}]
    data = json.dumps(review).encode()
    (tmp_path / "reviews.json").write_bytes(data)
    config["near_duplicates"]["manual_reviews"] = "reviews.json"
    config["near_duplicates"]["manual_reviews_sha256"] = hashlib.sha256(data).hexdigest()
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "pass"
    assert result["near_duplicates"]["decisions"]["distinct_by_review"] == 1


@pytest.mark.parametrize("failure", ["wrong_hash", "exact_override", "missing_reason"])
def test_invalid_manual_review_never_bypasses_duplicate_checks(tmp_path, failure):
    config, rows = fixture_split(tmp_path)
    with Image.open(tmp_path / rows[0]["original_path"]) as image:
        save_image(tmp_path, rows[5], image.copy())
    item = {"source_image_ids": [rows[0]["source_image_id"], rows[5]["source_image_id"]], "decision": "distinct", "reason": "test"}
    if failure == "missing_reason":
        item.pop("reason")
    data = json.dumps([item]).encode()
    (tmp_path / "reviews.json").write_bytes(data)
    config["near_duplicates"]["manual_reviews"] = "reviews.json"
    config["near_duplicates"]["manual_reviews_sha256"] = "0" * 64 if failure == "wrong_hash" else hashlib.sha256(data).hexdigest()
    with pytest.raises(SourceError):
        run_fixture(tmp_path, config, rows)


def test_identical_pixels_with_different_encoded_bytes_still_block_cross_split(tmp_path):
    config, rows = fixture_split(tmp_path)
    target = tmp_path / rows[5]["original_path"]
    with Image.open(tmp_path / rows[0]["original_path"]) as image:
        metadata = PngInfo()
        metadata.add_text("fixture", "different encoding metadata, same pixels")
        image.save(target, format="PNG", pnginfo=metadata)
    rows[5]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    rows[5]["pixel_sha256"] = rows[0]["pixel_sha256"]
    rows[5]["dhash64"] = rows[0]["dhash64"]
    assert rows[0]["sha256"] != rows[5]["sha256"]
    result = run_fixture(tmp_path, config, rows)
    assert result["status"] == "fail"
    assert any(f.get("field") == "pixel_sha256" for f in result["failures"])
