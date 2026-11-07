"""CSV/ZIP audit and original-byte image materialization. No scene generation."""

from collections import Counter, defaultdict
import csv
from datetime import datetime
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import stat
import statistics
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .source import SourceError, local_path, publish_new_file, raw_path, verify_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ID_PATTERN = re.compile(r"^(?P<camera>\d+)-(?P<plot>\d+)_(?P<stamp>\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def parse_annotations(path: Path) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise SourceError("CSV is empty") from error
        columns = {}
        for name in ("treatment_camera", "fullID", "Annotation"):
            if header.count(name) != 1:
                raise SourceError(f"CSV must have exactly one {name!r} header; got {header}")
            columns[name] = header.index(name)
        records = []
        blank_rows = 0
        for line_number, values in enumerate(reader, 2):
            if not values or all(not value.strip() for value in values):
                blank_rows += 1
                continue
            if len(values) != len(header):
                raise SourceError(f"CSV line {line_number}: expected {len(header)} columns, got {len(values)}")
            record = {name: values[index].strip() for name, index in columns.items()}
            record |= {"csv_line": line_number, "csv_values": values, "issues": []}
            raw_annotation = record["Annotation"]
            record["annotation"] = int(raw_annotation) if re.fullmatch(r"\d+", raw_annotation) else None
            if record["annotation"] is None:
                record["issues"].append("invalid_annotation")
            match = ID_PATTERN.fullmatch(record["fullID"])
            record |= {"camera_id": None, "plot_id": None, "timestamp": None}
            if not match:
                record["issues"].append("invalid_image_id")
            else:
                record["camera_id"], record["plot_id"] = match["camera"], match["plot"]
                if record["treatment_camera"] != f"{match['camera']}-{match['plot']}":
                    record["issues"].append("group_filename_mismatch")
                try:
                    record["timestamp"] = datetime.strptime(match["stamp"], "%d_%m_%Y_%H_%M_%S").isoformat()
                except ValueError:
                    record["issues"].append("invalid_timestamp")
            records.append(record)
    counts = Counter(row["fullID"] for row in records)
    if not records:
        raise SourceError("CSV has no annotation records")
    for row in records:
        if counts[row["fullID"]] > 1:
            row["issues"].append("duplicate_csv_full_id")
    return records, {"header": header, "blank_header_positions": [i for i, value in enumerate(header) if not value], "blank_rows_skipped": blank_rows}


def archive_inventory(archive: zipfile.ZipFile, limits: dict) -> tuple[list[dict], dict]:
    entries, images = [], {}
    names, basenames = set(), set()
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename or re.match(r"^[A-Za-z]:", info.filename) or stat.S_ISLNK(mode):
            raise SourceError(f"Unsafe archive member (nothing extracted): {info.filename}")
        if info.filename in names:
            raise SourceError(f"Duplicate archive path (ambiguous member): {info.filename}")
        names.add(info.filename)
        if info.is_dir():
            continue
        total += info.file_size
        if info.file_size > limits["max_member_bytes"] or total > limits["max_total_uncompressed_bytes"]:
            raise SourceError(f"Archive size limit exceeded at {info.filename}")
        entry = {"member": info.filename, "basename": path.name, "bytes": info.file_size,
                 "compressed_bytes": info.compress_size, "is_image": path.suffix.lower() in IMAGE_SUFFIXES}
        entries.append(entry)
        if entry["is_image"]:
            if path.name in basenames:
                raise SourceError(f"Duplicate/ambiguous image basename in ZIP: {path.name}")
            basenames.add(path.name)
            images[path.name] = info
    return entries, images


def materialize_original(destination: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise SourceError(f"Existing image differs or is not a regular file; left untouched: {destination}")
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".part-", dir=destination.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        temporary_path = Path(temporary.name)
    publish_new_file(temporary_path, destination)
    return "created"


def dhash64(image: Image.Image) -> str:
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def duplicate_groups(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row)
    return [{"hash": digest, "source_image_ids": [r["source_image_id"] for r in group],
             "annotations": sorted({r["annotation"] for r in group if r["annotation"] is not None}),
             "source_groups": sorted({r["split_group"] for r in group})}
            for digest, group in sorted(groups.items()) if len(group) > 1]


def create_contact_sheets(output: Path, previews: dict, config: dict) -> list[str]:
    limits = config["inspection"]
    width, height = limits["thumbnail_size"]
    columns = limits["contact_sheet_columns"]
    paths = []
    for annotation, samples in sorted(previews.items()):
        if not samples:
            continue
        samples = sorted(samples, key=lambda item: item[0])
        sheet = Image.new("RGB", (columns * width, 44 + ((len(samples) + columns - 1) // columns) * (height + 42)), "white")
        draw = ImageDraw.Draw(sheet)
        suffix = "QUARANTINE: outside v0 label domain" if annotation not in range(5) else "source wilt label"
        draw.text((8, 10), f"Annotation {annotation}: {suffix}", fill="black")
        for index, (image_id, thumbnail) in enumerate(samples):
            x, y = index % columns * width, 44 + index // columns * (height + 42)
            sheet.paste(thumbnail, (x + (width - thumbnail.width) // 2, y))
            draw.text((x + 4, y + height + 2), image_id[:32], fill="black")
            draw.text((x + 4, y + height + 17), image_id[32:], fill="black")
        relative = f"contact_sheets/annotation_{annotation}.jpg"
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        sheet.save(path, quality=90)
        paths.append(relative)
    return paths


def inspect_source(config: dict, *, output_dir: Path | None = None, dry_run: bool = False) -> dict:
    output = output_dir.resolve() if output_dir is not None else local_path(config, "source_output_dir")
    raw_dir, images_dir = local_path(config, "raw_dir"), local_path(config, "images_dir")
    if output == config["_base"] or output.is_relative_to(raw_dir) or raw_dir.is_relative_to(output):
        raise SourceError("Inspection output must be a separate directory outside the raw tree")
    plan = {"dry_run": dry_run, "csv": str(raw_path(config, "csv")), "zip": str(raw_path(config, "zip")),
            "images_dir": str(images_dir), "output_dir": str(output),
            "writes": ["sources.jsonl", "quarantine.csv", "source_report.json", "archive_inventory.jsonl", "group_stats.csv", "contact_sheets/*.jpg"],
            "image_policy": "write original bytes only to new files; reuse existing only if SHA256 matches"}
    if output.exists() or output.is_symlink():
        raise SourceError(f"Output already exists; choose a NEW --output-dir, no files changed: {output}")
    if dry_run:
        return plan
    files = {kind: verify_file(raw_path(config, kind), config["files"][kind]["md5"]) for kind in ("csv", "zip")}
    csv_rows, csv_details = parse_annotations(raw_path(config, "csv"))
    rng = random.Random(config["seed"])
    candidates = defaultdict(set)
    for row in csv_rows:
        if row["annotation"] is not None:
            candidates[row["annotation"]].add(row["fullID"])
    preview_ids = set()
    for annotation, identifiers in sorted(candidates.items()):
        preview_ids.update(rng.sample(sorted(identifiers), min(len(identifiers), config["inspection"]["contact_sheet_per_label"])))
    rows_by_id = defaultdict(list)
    for row in csv_rows:
        rows_by_id[row["fullID"]].append(row)
    image_details, previews = {}, defaultdict(list)
    materialized = Counter()
    with zipfile.ZipFile(raw_path(config, "zip")) as archive:
        inventory, members = archive_inventory(archive, config["inspection"])
        # Inventory/path/size checks finish before any raw images are written.
        output.mkdir(parents=True, exist_ok=False)
        for index, (image_id, info) in enumerate(sorted(members.items()), 1):
            details = {"archive_member": info.filename, "image_bytes": info.file_size, "issues": []}
            try:
                data = archive.read(info)  # Also verifies the member's CRC.
                details["sha256"] = hashlib.sha256(data).hexdigest()
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    details |= {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format,
                                "exif_orientation": image.getexif().get(274, 1), "readable": True}
                    details["pixel_sha256"] = hashlib.sha256(f"{image.mode}:{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
                    details["dhash64"] = dhash64(image)
                    if image.mode != "RGB":
                        details["issues"].append("non_rgb_image")
                    if image_id in preview_ids:
                        thumbnail = ImageOps.contain(image.convert("RGB"), tuple(config["inspection"]["thumbnail_size"]))
                        for annotation in {r["annotation"] for r in rows_by_id[image_id]}:
                            previews[annotation].append((image_id, thumbnail.copy()))
                if image_id in rows_by_id:
                    destination = images_dir / image_id
                    try:
                        materialized[materialize_original(destination, data)] += 1
                        details["original_path"] = destination.relative_to(config["_base"]).as_posix()
                    except (SourceError, OSError) as error:
                        details["issues"].append("existing_extracted_image_conflict" if isinstance(error, SourceError) else "image_materialization_error")
                        details["error"] = str(error)
            except (OSError, ValueError, UnidentifiedImageError, zipfile.BadZipFile, Image.DecompressionBombError) as error:
                details["readable"] = False
                details["issues"].append("unreadable_image_or_crc_error")
                details["error"] = str(error)
            image_details[image_id] = details
            if index % 200 == 0:
                print(f"Inspected {index}/{len(members)} archive images", file=sys.stderr, flush=True)

    sources, quarantine = [], []
    for row in csv_rows:
        details = image_details.get(row["fullID"], {})
        issues = row["issues"] + details.get("issues", [])
        if not details:
            issues.append("missing_image")
        in_domain = row["annotation"] is not None and 0 <= row["annotation"] <= 4
        reasons = issues + ([] if in_domain else ["annotation_outside_v0_domain"])
        record = {
            "schema_version": "agri-soybean-source-v0", "source_image_id": row["fullID"],
            "csv_line": row["csv_line"], "csv_values": row["csv_values"],
            "treatment_camera": row["treatment_camera"], "split_group": row["treatment_camera"],
            "camera_id": row["camera_id"], "plot_id": row["plot_id"],
            "timestamp": row["timestamp"], "timestamp_timezone": "unknown",
            "annotation_raw": row["Annotation"], "annotation": row["annotation"],
            "wilt_score": row["annotation"] if in_domain else None,
            "eligible_v0": not reasons, "qa_status": "invalid" if issues else "valid" if in_domain else "quarantined",
            "qa_issues": issues, "exclusion_reasons": reasons,
            "original_path": details.get("original_path"), "archive_member": details.get("archive_member"),
            "source_url": config["source_url"], "source_archive": raw_path(config, "zip").relative_to(config["_base"]).as_posix(),
            **{key: details.get(key) for key in ("sha256", "pixel_sha256", "dhash64", "image_bytes", "width", "height", "mode", "format", "readable", "exif_orientation")},
        }
        sources.append(record)
        if reasons:
            quarantine.append({"csv_line": row["csv_line"], "fullID": row["fullID"], "treatment_camera": row["treatment_camera"],
                               "annotation_raw": row["Annotation"], "reasons": ";".join(reasons)})

    label_counts = Counter(str(row["annotation"]) if row["annotation"] is not None else "invalid" for row in csv_rows)
    groups = sorted({row["treatment_camera"] for row in csv_rows})
    stats = []
    for group in groups:
        group_rows = [r for r in sources if r["split_group"] == group]
        labels = Counter(str(r["annotation"]) for r in group_rows)
        group_dates = sorted(r["timestamp"] for r in group_rows if r["timestamp"])
        stats.append({"split_group": group, "rows": len(group_rows), "eligible_v0": sum(r["eligible_v0"] for r in group_rows),
                      "timestamp_min": group_dates[0] if group_dates else None, "timestamp_max": group_dates[-1] if group_dates else None,
                      **{f"label_{i}": labels[str(i)] for i in range(6)}})
    actual = {"rows": len(sources), "unique_full_ids": len(rows_by_id), "groups": len(groups), "labels": dict(sorted(label_counts.items()))}
    expected = config["expected"]
    comparisons = [{"field": key, "expected": expected[key], "actual": actual[key], "matches": expected[key] == actual[key]}
                   for key in ("rows", "unique_full_ids", "groups")]
    expected_labels = {str(k): v for k, v in expected["labels"].items()}
    comparisons.append({"field": "labels", "expected": expected_labels, "actual": actual["labels"], "matches": expected_labels == actual["labels"]})
    outside_groups = sorted({r["split_group"] for r in sources if r["annotation"] == 5})
    comparisons.append({"field": "annotation_5_groups", "expected": [expected["out_of_domain_group"]], "actual": outside_groups,
                        "matches": outside_groups == [expected["out_of_domain_group"]] if label_counts["5"] else True})
    duplicates = {"encoded_sha256": duplicate_groups(sources, "sha256"), "decoded_pixel_sha256": duplicate_groups(sources, "pixel_sha256")}
    missing = sorted(set(rows_by_id) - set(image_details))
    extra = sorted(set(image_details) - set(rows_by_id))
    unreadable = sorted(name for name, details in image_details.items() if not details.get("readable"))
    non_rgb = sorted(name for name, details in image_details.items() if details.get("readable") and details["mode"] != "RGB")
    dates = sorted(r["timestamp"] for r in sources if r["timestamp"])
    sizes = [d["image_bytes"] for d in image_details.values()]
    contact_sheets = create_contact_sheets(output, previews, config)
    report = {
        "report_version": 1, "source_id": config["source_id"], "config_sha256": config["_config_sha256"], "seed": config["seed"],
        "status": "fail" if missing or unreadable or non_rgb or any(r["qa_issues"] for r in sources) or not all(c["matches"] for c in comparisons) else "pass",
        "files": files, "csv": csv_details | actual, "expectation_checks": comparisons,
        "row_accounting": {"total": len(sources), "eligible_v0": sum(r["eligible_v0"] for r in sources),
                           "excluded": len(quarantine), "qa_invalid": sum(r["qa_status"] == "invalid" for r in sources),
                           "label_only_quarantine": sum(r["qa_status"] == "quarantined" for r in sources)},
        "eligible_label_counts": dict(sorted(Counter(str(r["wilt_score"]) for r in sources if r["eligible_v0"]).items())),
        "exclusion_reason_counts": dict(Counter(reason for r in sources for reason in r["exclusion_reasons"])),
        "archive": {"files": len(inventory), "images": len(image_details), "uncompressed_bytes": sum(e["bytes"] for e in inventory),
                    "missing_images": missing, "extra_images": extra, "unreadable_images": unreadable, "non_rgb_images": non_rgb},
        "image_dimensions": dict(sorted(Counter(f"{d['width']}x{d['height']}" for d in image_details.values() if d.get("readable")).items())),
        "image_modes": dict(Counter(d["mode"] for d in image_details.values() if d.get("readable"))),
        "image_size_bytes": {"min": min(sizes, default=None), "max": max(sizes, default=None), "total": sum(sizes),
                             "mean": statistics.mean(sizes) if sizes else None, "median": statistics.median(sizes) if sizes else None},
        "timestamps": {"format": "DD_MM_YYYY_HH_MM_SS", "timezone": "unknown", "min": dates[0] if dates else None, "max": dates[-1] if dates else None,
                       "per_date": dict(sorted(Counter(value[:10] for value in dates).items()))},
        "per_group": stats, "exact_duplicates": duplicates,
        "warnings": (["Exact image duplicates require resolution/grouping before source split"] if any(duplicates.values()) else [])
                    + (["Unannotated extra archive images are excluded from v0"] if extra else [])
                    + (["Annotation 5 retained outside v0; meaning not inferred"] if label_counts["5"] else []),
        "materialized_images": dict(materialized), "contact_sheets": contact_sheets,
        "not_checked": ["near_duplicate_grouping", "source_split", "scene_generation", "action_labels", "training"],
    }
    for filename, records in (("sources.jsonl", sources), ("archive_inventory.jsonl", inventory)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    for filename, records, fields in (
        ("quarantine.csv", quarantine, ["csv_line", "fullID", "treatment_camera", "annotation_raw", "reasons"]),
        ("group_stats.csv", stats, ["split_group", "rows", "eligible_v0", "timestamp_min", "timestamp_max"] + [f"label_{i}" for i in range(6)]),
    ):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    with (output / "source_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"status": report["status"], "output_dir": str(output), "row_accounting": report["row_accounting"],
            "archive": report["archive"], "warnings": report["warnings"], "contact_sheets": contact_sheets}
