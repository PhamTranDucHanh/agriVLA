from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

from PIL import Image
import pytest
import yaml

from agri_vla.ingest import archive_inventory, inspect_source, parse_annotations
from agri_vla.source import SourceError, acquire_file, fetch_source, load_config, raw_path


def image_bytes(color=(10, 30, 50), mode="RGB"):
    output = BytesIO()
    Image.new(mode, (40, 30), color if mode == "RGB" else 100).save(output, format="JPEG")
    return output.getvalue()


def fixture_source(tmp_path, *, rows=None, entries=None):
    if rows is None:
        rows = [
            ("1-10", "1-10_18_09_2019_06_30_00.jpg", "0"),
            ("1-10", "1-10_18_09_2019_06_31_00.jpg", "2"),
            ("1-10", "1-10_18_09_2019_06_32_00.jpg", "4"),
            ("11-46", "11-46_18_09_2019_06_33_00.jpg", "5"),
        ]
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    csv_path, zip_path = raw / "annotations.csv", raw / "images.zip"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["", "treatment_camera", "fullID", "", "Annotation"])
        for index, (group, name, annotation) in enumerate(rows):
            writer.writerow([index, group, name, "", annotation])
    if entries is None:
        entries = {f"nested/{name}": image_bytes((index * 40, 30, 50)) for index, (_, name, _) in enumerate(rows)}
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    config = {
        "config_version": 1, "base_dir": ".", "source_id": "fixture", "source_url": "https://example.invalid/fixture", "seed": 42,
        "paths": {"raw_dir": "data/raw", "images_dir": "data/raw/images", "source_output_dir": "reports/source"},
        "files": {key: {"name": path.name, "url": f"https://example.invalid/{path.name}",
                        "md5": hashlib.md5(path.read_bytes()).hexdigest(), "approximate_bytes": path.stat().st_size}
                  for key, path in (("csv", csv_path), ("zip", zip_path))},
        "expected": {"rows": len(rows), "unique_full_ids": len({r[1] for r in rows}), "groups": len({r[0] for r in rows}),
                     "labels": dict(Counter(r[2] for r in rows)), "out_of_domain_group": "11-46"},
        "inspection": {"max_member_bytes": 1024**2, "max_total_uncompressed_bytes": 10 * 1024**2,
                       "contact_sheet_per_label": 2, "contact_sheet_columns": 2, "thumbnail_size": [80, 60]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_config(path)


def report_and_rows(result):
    output = Path(result["output_dir"])
    report = json.loads((output / "source_report.json").read_text())
    rows = [json.loads(line) for line in (output / "sources.jsonl").read_text().splitlines()]
    return report, rows


def test_csv_blank_headers_day_first_and_quarantine(tmp_path):
    config = fixture_source(tmp_path)
    result = inspect_source(config)
    report, rows = report_and_rows(result)
    assert result["status"] == "pass"
    assert report["csv"]["blank_header_positions"] == [0, 3]
    assert report["row_accounting"] == {"total": 4, "eligible_v0": 3, "excluded": 1, "qa_invalid": 0, "label_only_quarantine": 1}
    assert rows[0]["timestamp"] == "2019-09-18T06:30:00"
    assert rows[0]["timestamp_timezone"] == "unknown"
    quarantined = rows[-1]
    assert quarantined["annotation_raw"] == "5" and quarantined["annotation"] == 5
    assert quarantined["wilt_score"] is None and quarantined["eligible_v0"] is False
    assert quarantined["exclusion_reasons"] == ["annotation_outside_v0_domain"]
    with (Path(result["output_dir"]) / "quarantine.csv").open() as stream:
        assert next(csv.DictReader(stream))["annotation_raw"] == "5"
    with zipfile.ZipFile(raw_path(config, "zip")) as archive:
        for row in rows:
            assert (config["_base"] / row["original_path"]).read_bytes() == archive.read(row["archive_member"])
    assert len(report["contact_sheets"]) == 4


@pytest.mark.parametrize("header", [[], ["fullID", "Annotation"], ["treatment_camera", "fullID", "Annotation", "Annotation"]])
def test_bad_csv_header_is_rejected(tmp_path, header):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerow(header)
    with pytest.raises(SourceError, match="header"):
        parse_annotations(path)


def test_missing_extra_corrupt_nonrgb_and_bad_timestamp_are_reported(tmp_path):
    rows = [
        ("1-10", "1-10_18_09_2019_06_30_00.jpg", "0"),
        ("1-10", "1-10_18_09_2019_06_31_00.jpg", "2"),
        ("1-10", "1-10_18_09_2019_06_32_00.jpg", "4"),
        ("11-46", "11-46_32_09_2019_06_33_00.jpg", "5"),
    ]
    config = fixture_source(tmp_path, rows=rows, entries={
        rows[1][1]: b"not a JPEG",
        rows[2][1]: image_bytes(mode="L"),
        rows[3][1]: image_bytes(),
        "extra.jpg": image_bytes((80, 100, 110)),
    })
    result = inspect_source(config)
    report, source = report_and_rows(result)
    assert result["status"] == "fail"
    assert report["archive"]["missing_images"] == [rows[0][1]]
    assert report["archive"]["unreadable_images"] == [rows[1][1]]
    assert report["archive"]["non_rgb_images"] == [rows[2][1]]
    assert report["archive"]["extra_images"] == ["extra.jpg"]
    assert "invalid_timestamp" in source[3]["qa_issues"]
    assert report["row_accounting"]["eligible_v0"] == 0


@pytest.mark.parametrize("unsafe", ["../outside.jpg", "/absolute.jpg", "C:/drive.jpg", "folder\\windows.jpg"])
def test_unsafe_zip_never_extracts(tmp_path, unsafe):
    config = fixture_source(tmp_path, entries={unsafe: image_bytes()})
    with pytest.raises(SourceError, match="Unsafe archive"):
        inspect_source(config)
    assert not (tmp_path / "data/raw/images").exists()
    assert not (tmp_path / "reports/source").exists()


def test_zip_symlink_and_duplicate_paths_are_rejected(tmp_path):
    config = fixture_source(tmp_path)
    path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("link.jpg")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with zipfile.ZipFile(path) as archive, pytest.raises(SourceError, match="Unsafe archive"):
        archive_inventory(archive, config["inspection"])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("same.jpg", image_bytes())
        with pytest.warns(UserWarning):
            archive.writestr("same.jpg", image_bytes())
    with zipfile.ZipFile(path) as archive, pytest.raises(SourceError, match="Duplicate archive path"):
        archive_inventory(archive, config["inspection"])


def test_ambiguous_basename_and_size_limits(tmp_path):
    config = fixture_source(tmp_path, entries={"a/same.jpg": image_bytes(), "b/same.jpg": image_bytes()})
    with pytest.raises(SourceError, match="ambiguous image basename"):
        inspect_source(config)
    config["inspection"]["max_member_bytes"] = 1
    with pytest.raises(SourceError, match="size limit"):
        inspect_source(config)


def test_exact_duplicate_content_is_reported_not_silently_deduplicated(tmp_path):
    rows = [("1-10", f"1-10_18_09_2019_06_3{i}_00.jpg", str(i)) for i in range(2)]
    data = image_bytes()
    config = fixture_source(tmp_path, rows=rows, entries={row[1]: data for row in rows})
    result = inspect_source(config)
    report, source = report_and_rows(result)
    assert len(source) == 2
    assert len(report["exact_duplicates"]["encoded_sha256"]) == 1
    assert report["exact_duplicates"]["encoded_sha256"][0]["annotations"] == [0, 1]
    assert report["warnings"]


def test_expected_counts_mismatch_is_a_failure(tmp_path):
    config = fixture_source(tmp_path)
    config["expected"]["rows"] = 100
    result = inspect_source(config)
    report, _ = report_and_rows(result)
    assert result["status"] == "fail"
    assert next(c for c in report["expectation_checks"] if c["field"] == "rows")["matches"] is False


def test_raw_conflict_and_existing_report_are_never_overwritten(tmp_path):
    config = fixture_source(tmp_path)
    destination = tmp_path / "data/raw/images/1-10_18_09_2019_06_30_00.jpg"
    destination.parent.mkdir()
    destination.write_bytes(b"user existing content")
    result = inspect_source(config)
    report, source = report_and_rows(result)
    assert result["status"] == "fail" and destination.read_bytes() == b"user existing content"
    assert "existing_extracted_image_conflict" in source[0]["qa_issues"]
    original_report = (Path(result["output_dir"]) / "source_report.json").read_bytes()
    with pytest.raises(SourceError, match="Output already exists"):
        inspect_source(config)
    assert (Path(result["output_dir"]) / "source_report.json").read_bytes() == original_report


def test_reinspection_reuses_images_and_preserves_manifest(tmp_path):
    config = fixture_source(tmp_path)
    first = inspect_source(config)
    second = inspect_source(config, output_dir=tmp_path / "reports/recheck")
    report, _ = report_and_rows(second)
    assert report["materialized_images"] == {"reused": 4}
    assert (Path(first["output_dir"]) / "sources.jsonl").read_bytes() == (Path(second["output_dir"]) / "sources.jsonl").read_bytes()


def test_fetch_and_inspect_dry_runs_do_not_write_or_use_network(tmp_path, monkeypatch):
    config = fixture_source(tmp_path)
    from agri_vla import source
    monkeypatch.setattr(source, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    config["paths"]["raw_dir"] = "new_raw"
    before = sorted(tmp_path.rglob("*"))
    assert fetch_source(config, dry_run=True)["approximate_download_bytes"] > 0
    assert inspect_source(config, dry_run=True)["dry_run"] is True
    assert sorted(tmp_path.rglob("*")) == before


def test_fetch_imports_local_files_and_reuses_verified_raw(tmp_path, monkeypatch):
    config = fixture_source(tmp_path)
    csv_path, zip_path = raw_path(config, "csv"), raw_path(config, "zip")
    config["paths"]["raw_dir"] = "imported_raw"
    from agri_vla import source
    monkeypatch.setattr(source, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    result = fetch_source(config, local_csv=csv_path, local_zip=zip_path)
    assert all(f["action"] == "import_local" for f in result["files"])
    assert raw_path(config, "csv").read_bytes() == csv_path.read_bytes()
    assert all(f["action"] == "reuse" for f in fetch_source(config)["files"])
    raw_path(config, "csv").write_bytes(b"existing wrong file")
    with pytest.raises(SourceError, match="Checksum mismatch"):
        fetch_source(config)
    assert raw_path(config, "csv").read_bytes() == b"existing wrong file"


@pytest.mark.parametrize("failure", ["checksum", "truncated"])
def test_failed_download_retains_partial_without_publishing(tmp_path, monkeypatch, failure):
    config = fixture_source(tmp_path)
    csv_data = raw_path(config, "csv").read_bytes()
    config["paths"]["raw_dir"] = "downloads"
    data = b"bad checksum" if failure == "checksum" else csv_data

    class Response(BytesIO):
        headers = {"Content-Length": str(len(data) + (1 if failure == "truncated" else 0))}

    from agri_vla import source
    monkeypatch.setattr(source, "urlopen", lambda *a, **k: Response(data))
    with pytest.raises(SourceError, match="partial file retained"):
        acquire_file(config, "csv")
    assert not raw_path(config, "csv").exists()
    assert len(list((tmp_path / "downloads").glob("*.part-*"))) == 1


def test_csv_duplicate_id_and_group_mismatch_are_not_eligible(tmp_path):
    name = "1-10_18_09_2019_06_30_00.jpg"
    config = fixture_source(tmp_path, rows=[("1-10", name, "0"), ("9-99", name, "2")])
    result = inspect_source(config)
    report, rows = report_and_rows(result)
    assert result["status"] == "fail"
    assert report["csv"]["unique_full_ids"] == 1
    assert all("duplicate_csv_full_id" in row["qa_issues"] for row in rows)
    assert "group_filename_mismatch" in rows[1]["qa_issues"]
    assert report["row_accounting"]["eligible_v0"] == 0


def test_invalid_annotation_is_preserved_not_coerced(tmp_path):
    config = fixture_source(tmp_path, rows=[("1-10", "1-10_18_09_2019_06_30_00.jpg", "2.0")])
    result = inspect_source(config)
    _, rows = report_and_rows(result)
    assert result["status"] == "fail"
    assert rows[0]["annotation_raw"] == "2.0" and rows[0]["annotation"] is None
    assert rows[0]["eligible_v0"] is False


def test_successful_download_is_published_only_with_matching_hash(tmp_path, monkeypatch):
    config = fixture_source(tmp_path)
    data = raw_path(config, "csv").read_bytes()
    config["paths"]["raw_dir"] = "new_downloads"

    class Response(BytesIO):
        headers = {"Content-Length": str(len(data))}

    from agri_vla import source
    monkeypatch.setattr(source, "urlopen", lambda *a, **k: Response(data))
    result = acquire_file(config, "csv")
    assert raw_path(config, "csv").read_bytes() == data
    assert result["md5"] == config["files"]["csv"]["md5"]
    assert not list((tmp_path / "new_downloads").glob("*.part-*"))


def test_all_source_commands_work_when_ml_imports_are_blocked(tmp_path):
    fixture_source(tmp_path)
    code = r'''
import importlib.abc
import sys
attempted = []
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchvision", "lerobot", "transformers", "peft"}:
            attempted.append(fullname)
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockML())
from agri_vla.cli import main
assert main(["fetch-source", "--config", sys.argv[1], "--dry-run"]) == 0
assert main(["inspect-source", "--config", sys.argv[1], "--dry-run"]) == 0
assert main(["inspect-source", "--config", sys.argv[1]]) == 0
assert not attempted, attempted
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "config.yaml")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_corrupted_zip_member_crc_is_reported(tmp_path):
    config = fixture_source(tmp_path)
    path = raw_path(config, "zip")
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()[0]
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + 5
    content = bytearray(path.read_bytes())
    content[offset] ^= 1
    path.write_bytes(content)
    config["files"]["zip"]["md5"] = hashlib.md5(content).hexdigest()
    result = inspect_source(config)
    report, rows = report_and_rows(result)
    assert result["status"] == "fail"
    assert len(report["archive"]["unreadable_images"]) == 1
    assert "unreadable_image_or_crc_error" in rows[0]["qa_issues"]
