"""Lock source-group splits and detect exact/near duplicate leakage on CPU."""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageDraw, ImageOps

from .ingest import dhash64
from .source import SourceError

SPLITS = ("train", "val", "test")
ALGORITHM = "dhash64_rgb_mae_gray_pearson_v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def relative_file(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceError("Expected a nonempty relative POSIX path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", value):
        raise SourceError(f"Nonportable or escaping path: {value}")
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise SourceError(f"Path escapes base_dir: {value}")
    return resolved


def load_split_config(path: Path) -> dict:
    import yaml

    path = path.resolve()
    content = path.read_bytes()
    try:
        config = yaml.safe_load(content)
        if not isinstance(config, dict) or config.get("config_version") != 1:
            raise SourceError("Expected split config_version: 1")
        if not isinstance(config.get("base_dir"), str) or Path(config["base_dir"]).is_absolute():
            raise SourceError("base_dir must be relative to split config")
        config["_base"] = (path.parent / config["base_dir"]).resolve()
        config["_config_sha256"] = digest(content)
        if config["assignment_policy"] != "explicit_source_groups" or config["split_version"] != "soybean_source_split_v0":
            raise SourceError("Unsupported assignment policy or split version")
        if type(config["seed"]) is not int or config["seed"] < 0:
            raise SourceError("seed must be a nonnegative integer")
        if set(config["assignments"]) != set(SPLITS) or set(config["expected_counts"]) != set(SPLITS):
            raise SourceError("assignments/expected_counts must contain train, val, test")
        seen = set()
        for split in SPLITS:
            groups = config["assignments"][split]
            if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or not re.fullmatch(r"\d+-\d+", g) for g in groups):
                raise SourceError(f"Invalid source group list for {split}")
            if len(set(groups)) != len(groups) or seen.intersection(groups):
                raise SourceError("A source group is assigned more than once")
            seen.update(groups)
            if type(config["expected_counts"][split]) is not int or config["expected_counts"][split] <= 0:
                raise SourceError("Expected split counts must be positive integers")
        for field in ("manifest", "report"):
            relative_file(config["_base"], config["source"][field])
            if not re.fullmatch(r"[a-f0-9]{64}", config["source"][field + "_sha256"]):
                raise SourceError(f"Missing/invalid pinned {field} SHA256")
        relative_file(config["_base"], config["output_dir"])
        for field in ("camera", "plot"):
            if type(config["isolation"][field]) is not bool:
                raise SourceError("isolation.camera/plot must be booleans")
        policy = config["near_duplicates"]
        if policy["algorithm"] != ALGORITHM:
            raise SourceError("Unsupported near-duplicate algorithm")
        if type(policy["hamming_max"]) is not int or not 0 <= policy["hamming_max"] <= 64:
            raise SourceError("hamming_max must be an integer in [0,64]")
        if len(policy["thumbnail_size"]) != 2 or any(type(n) is not int or not 8 <= n <= 256 for n in policy["thumbnail_size"]):
            raise SourceError("thumbnail_size dimensions must be integers in [8,256]")
        for key in ("confirmed_rgb_mae_max", "confirmed_gray_correlation_min", "min_gray_std", "review_rgb_mae_max", "review_gray_correlation_min"):
            if type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or not 0 <= policy[key] <= 1:
                raise SourceError(f"Invalid threshold: {key}")
        if policy["review_rgb_mae_max"] < policy["confirmed_rgb_mae_max"] or policy["review_gray_correlation_min"] > policy["confirmed_gray_correlation_min"]:
            raise SourceError("Review region must include the confirmed region")
        if type(policy["diagnostic_pair_count"]) is not int or not 0 <= policy["diagnostic_pair_count"] <= 100:
            raise SourceError("diagnostic_pair_count must be in [0,100]")
        if policy["manual_reviews"] is not None:
            relative_file(config["_base"], policy["manual_reviews"])
            if not re.fullmatch(r"[a-f0-9]{64}", policy["manual_reviews_sha256"] or ""):
                raise SourceError("Manual reviews must have a pinned SHA256")
        elif policy["manual_reviews_sha256"] is not None:
            raise SourceError("Review SHA256 supplied without a review file")
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise SourceError(f"Malformed split config: {error}") from error
    return config


def read_pinned(base: Path, spec: dict, field: str) -> bytes:
    path = relative_file(base, spec[field])
    data = path.read_bytes()
    if digest(data) != spec[field + "_sha256"]:
        raise SourceError(f"Pinned {field} SHA256 mismatch: {path}; source unchanged by this command")
    return data


def load_sources(config: dict) -> tuple[list[dict], list[dict], dict]:
    base = config["_base"]
    try:
        report = json.loads(read_pinned(base, config["source"], "report"))
        rows = [json.loads(line) for line in read_pinned(base, config["source"], "manifest").splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SourceError(f"Invalid source JSON: {error}") from error
    if report.get("status") != "pass":
        raise SourceError("Source report must be PASS before splitting")
    identifiers = set()
    eligible, excluded = [], []
    for row in rows:
        image_id = row.get("source_image_id")
        if not isinstance(image_id, str) or not image_id or image_id in identifiers:
            raise SourceError(f"Missing/duplicate source ID: {image_id}")
        identifiers.add(image_id)
        if type(row.get("eligible_v0")) is not bool:
            raise SourceError(f"eligible_v0 must be a boolean: {image_id}")
        if not row["eligible_v0"]:
            excluded.append(row)
            continue
        score = row.get("wilt_score")
        if type(score) is not int or score not in range(5) or row.get("annotation") != score:
            raise SourceError(f"Eligible source has invalid wilt/annotation: {image_id}")
        if row.get("qa_status") != "valid" or row.get("qa_issues") or row.get("exclusion_reasons") or row.get("readable") is not True or row.get("mode") != "RGB":
            raise SourceError(f"Eligible source did not pass image QA: {image_id}")
        if any(type(row.get(key)) is not int or row[key] <= 0 for key in ("width", "height")):
            raise SourceError(f"Invalid source image dimensions: {image_id}")
        if any(not isinstance(row.get(key), str) or not row[key].isdigit() for key in ("camera_id", "plot_id")):
            raise SourceError(f"Camera/plot IDs must be digit strings: {image_id}")
        group = row.get("split_group")
        if group != f"{row.get('camera_id')}-{row.get('plot_id')}" or group != row.get("treatment_camera") or not image_id.startswith(group + "_"):
            raise SourceError(f"Camera/plot/source-group mismatch: {image_id}")
        for key, length in (("sha256", 64), ("pixel_sha256", 64), ("dhash64", 16)):
            if not re.fullmatch(f"[a-f0-9]{{{length}}}", row.get(key, "")):
                raise SourceError(f"Invalid {key}: {image_id}")
        relative_file(base, row["original_path"])
        eligible.append(row)
    accounting = report.get("row_accounting", {})
    if (len(rows), len(eligible), len(excluded)) != (accounting.get("total"), accounting.get("eligible_v0"), accounting.get("excluded")):
        raise SourceError("Source manifest and report row accounting differ")
    return sorted(eligible, key=lambda r: r["source_image_id"]), sorted(excluded, key=lambda r: r["source_image_id"]), report


def image_features(row: dict, base: Path, size: tuple[int, int]) -> dict:
    path = relative_file(base, row["original_path"])
    if (base / row["original_path"]).is_symlink():
        raise SourceError(f"Source image must not be a symlink: {path}")
    data = path.read_bytes()
    if digest(data) != row["sha256"]:
        raise SourceError(f"Image SHA256 changed since source audit: {path}")
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB" or [image.width, image.height] != [row["width"], row["height"]]:
            raise SourceError(f"Image mode/dimensions changed: {path}")
        pixel_hash = digest(f"{image.mode}:{image.width}x{image.height}:".encode() + image.tobytes())
        if pixel_hash != row["pixel_sha256"] or dhash64(image) != row["dhash64"]:
            raise SourceError(f"Image pixel/dHash metadata mismatch: {path}")
        thumbnail = image.resize(size, Image.Resampling.LANCZOS)
        gray = thumbnail.convert("L").tobytes()
        total, squares, n = sum(gray), sum(p * p for p in gray), len(gray)
        variance_term = n * squares - total * total
        return {"rgb": thumbnail.tobytes(), "gray": gray, "sum": total, "variance_term": variance_term,
                "std": math.sqrt(max(variance_term, 0)) / n / 255, "dhash": int(row["dhash64"], 16)}


def pair_metrics(a: dict, b: dict) -> dict:
    mae = sum(abs(x - y) for x, y in zip(a["rgb"], b["rgb"], strict=True)) / len(a["rgb"]) / 255
    denominator = math.sqrt(a["variance_term"] * b["variance_term"])
    correlation = None if denominator == 0 else (
        len(a["gray"]) * sum(x * y for x, y in zip(a["gray"], b["gray"], strict=True)) - a["sum"] * b["sum"]
    ) / denominator
    return {"rgb_mae": mae, "gray_correlation": correlation, "min_gray_std": min(a["std"], b["std"])}


def classify_pair(metrics: dict, policy: dict) -> str:
    correlation = metrics["gray_correlation"]
    textured = metrics["min_gray_std"] >= policy["min_gray_std"]
    if textured and correlation is not None and correlation >= policy["confirmed_gray_correlation_min"] and metrics["rgb_mae"] <= policy["confirmed_rgb_mae_max"]:
        return "near_duplicate"
    if ((correlation is not None and correlation >= policy["review_gray_correlation_min"] and metrics["rgb_mae"] <= policy["review_rgb_mae_max"])
            or (not textured and metrics["rgb_mae"] <= policy["confirmed_rgb_mae_max"])):
        return "needs_review"
    return "distinct_under_policy"


class Components:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def manual_reviews(config: dict, ids: set[str]) -> dict:
    policy = config["near_duplicates"]
    if policy["manual_reviews"] is None:
        return {}
    data = relative_file(config["_base"], policy["manual_reviews"]).read_bytes()
    if digest(data) != policy["manual_reviews_sha256"]:
        raise SourceError("Manual review SHA256 mismatch")
    reviews = {}
    items = json.loads(data)
    if not isinstance(items, list):
        raise SourceError("Manual reviews must be a JSON array")
    for item in items:
        identifiers = item.get("source_image_ids") if isinstance(item, dict) else None
        if not isinstance(identifiers, list) or len(identifiers) != 2 or any(not isinstance(value, str) for value in identifiers):
            raise SourceError("Manual review source_image_ids must contain two strings")
        pair = tuple(sorted(identifiers))
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= ids or pair in reviews:
            raise SourceError("Invalid/duplicate manual review pair")
        if item.get("decision") not in {"near_duplicate", "distinct"} or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise SourceError("Manual reviews require an explicit decision and reason")
        reviews[pair] = item
    return reviews


def audit_duplicates(rows: list[dict], assignments: dict, config: dict) -> tuple[list[dict], list[str], dict, list[dict]]:
    policy = config["near_duplicates"]
    features = []
    for index, row in enumerate(rows, 1):
        features.append(image_features(row, config["_base"], tuple(policy["thumbnail_size"])))
        if index % 300 == 0:
            print(f"Verified source images: {index}/{len(rows)}", file=sys.stderr, flush=True)
    components = Components(len(rows))
    reviews = manual_reviews(config, {r["source_image_id"] for r in rows})
    used_reviews, pairs, nearest = set(), [], []
    exact_keys = ("sha256", "pixel_sha256")
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            distance = (features[i]["dhash"] ^ features[j]["dhash"]).bit_count()
            exact = [key for key in exact_keys if a[key] == b[key]]
            cross_split = assignments[a["split_group"]] != assignments[b["split_group"]]
            if cross_split:
                nearest.append((distance, i, j))
            if not exact and distance > policy["hamming_max"]:
                continue
            metrics = pair_metrics(features[i], features[j])
            automatic = "exact_duplicate" if exact else classify_pair(metrics, policy)
            decision = automatic
            key = (a["source_image_id"], b["source_image_id"])
            review = reviews.get(key)
            if review:
                used_reviews.add(key)
                if exact and review["decision"] == "distinct":
                    raise SourceError("An exact duplicate cannot be overridden as distinct")
                decision = "exact_duplicate" if exact else "near_duplicate" if review["decision"] == "near_duplicate" else "distinct_by_review"
            if decision in {"exact_duplicate", "near_duplicate"}:
                components.join(i, j)
            pairs.append({"source_image_ids": list(key), "source_groups": [a["split_group"], b["split_group"]],
                          "splits": [assignments[a["split_group"]], assignments[b["split_group"]]],
                          "dhash_hamming": distance, "automatic_decision": automatic, "decision": decision,
                          **metrics, "exact_hash_fields": exact, "manual_review": review,
                          "conflicting_exact_labels": bool(exact and a["wilt_score"] != b["wilt_score"])})
    if set(reviews) != used_reviews:
        raise SourceError("Manual review contains a pair outside this candidate policy")
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[components.find(index)].append(row["source_image_id"])
    names = {root: "dup_" + digest(canonical_bytes(sorted(ids))) for root, ids in members.items()}
    component_ids = [names[components.find(i)] for i in range(len(rows))]
    diagnostics = [p for p in pairs if p["source_groups"][0] != p["source_groups"][1]]
    # Also inspect closest cross-split pairs beyond the dHash candidate radius.
    # These are diagnostics, not silently reclassified near-duplicate edges.
    chosen = sorted(nearest)[:policy["diagnostic_pair_count"]]
    for distance, i, j in chosen:
        a, b = rows[i], rows[j]
        diagnostics.append({"source_image_ids": [a["source_image_id"], b["source_image_id"]],
                            "source_groups": [a["split_group"], b["split_group"]],
                            "splits": [assignments[a["split_group"]], assignments[b["split_group"]]],
                            "dhash_hamming": distance, **pair_metrics(features[i], features[j]),
                            "decision": "closest_cross_split_diagnostic"})
    unique = {tuple(p["source_image_ids"]): p for p in diagnostics}
    diagnostics = sorted(unique.values(), key=lambda p: (p["dhash_hamming"], p["source_image_ids"]))
    # Seed is used only for representative within-group diagnostic examples.
    within = [p for p in pairs if p["source_groups"][0] == p["source_groups"][1] and p["decision"] == "near_duplicate"]
    diagnostics += random.Random(config["seed"]).sample(within, min(4, len(within)))
    summary = {"candidate_pairs": len(pairs), "decisions": dict(Counter(p["decision"] for p in pairs)),
               "cross_group_candidates": sum(p["source_groups"][0] != p["source_groups"][1] for p in pairs),
               "cross_split_candidates": sum(p["splits"][0] != p["splits"][1] for p in pairs),
               "components": len(members), "multi_image_components": sum(len(ids) > 1 for ids in members.values()),
               "largest_component": max(map(len, members.values()), default=0), "images_verified": len(rows),
               "method_limitations": ["heuristic thresholds, not a universal duplicate detector", "no crop/rotation/registration search", "manual review panels are samples, not all pairs"]}
    return pairs, component_ids, summary, diagnostics


def intersection_report(rows: list[dict]) -> dict:
    fields = ("source_image_id", "sha256", "pixel_sha256", "split_group", "camera_id", "plot_id", "duplicate_component_id")
    result = {}
    for a, b in itertools.combinations(SPLITS, 2):
        result[f"{a}__{b}"] = {field: sorted({r[field] for r in rows if r["split"] == a} & {r[field] for r in rows if r["split"] == b}) for field in fields}
    return result


def review_panels(output: Path, pairs: list[dict], rows: list[dict], base: Path) -> list[str]:
    by_id = {r["source_image_id"]: r for r in rows}
    paths = []
    for start in range(0, len(pairs), 4):
        page = pairs[start:start + 4]
        sheet = Image.new("RGB", (640, 292 * len(page)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, pair in enumerate(page):
            y = row_index * 292
            for column, image_id in enumerate(pair["source_image_ids"]):
                record = by_id[image_id]
                with Image.open(relative_file(base, record["original_path"])) as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"), (320, 240))
                    sheet.paste(thumbnail, (column * 320, y))
                draw.text((column * 320 + 3, y + 240), image_id, fill="black")
                draw.text((column * 320 + 3, y + 254), f"{pair['splits'][column]} / group {record['split_group']}", fill="black")
            correlation = pair["gray_correlation"]
            draw.text((3, y + 270), f"Hamming={pair['dhash_hamming']} RGB_MAE={pair['rgb_mae']:.4f} corr={correlation if correlation is None else round(correlation, 4)} / {pair['decision']}", fill="black")
        relative = f"review/near_pairs_{start // 4 + 1:02d}.jpg"
        (output / "review").mkdir(exist_ok=True)
        sheet.save(output / relative, quality=90)
        paths.append(relative)
    return paths


def split_sources(config: dict, *, output_dir: Path | None = None, dry_run: bool = False) -> dict:
    rows, excluded, source_report = load_sources(config)
    output = output_dir.resolve() if output_dir else relative_file(config["_base"], config["output_dir"])
    if output.exists() or output.is_symlink():
        raise SourceError(f"Split output already exists; choose a NEW --output-dir: {output}")
    raw_paths = [(config["_base"] / row["original_path"]).parent.resolve() for row in rows]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in raw_paths):
        raise SourceError("Split output must not overlap original image directories")
    assignments = {group: split for split, groups in config["assignments"].items() for group in groups}
    actual_groups = {row["split_group"] for row in rows}
    if actual_groups != set(assignments):
        raise SourceError(f"Configured groups differ from eligible groups: missing={sorted(actual_groups - set(assignments))}, unknown={sorted(set(assignments) - actual_groups)}")
    pairs, component_ids, duplicate_summary, diagnostics = audit_duplicates(rows, assignments, config)
    split_rows = [row | {"split": assignments[row["split_group"]], "duplicate_component_id": component,
                         "source_split_version": config["split_version"]} for row, component in zip(rows, component_ids, strict=True)]
    intersections = intersection_report(split_rows)
    failures = []
    for pair_name, fields in intersections.items():
        for field, values in fields.items():
            enabled = field not in {"camera_id", "plot_id"} or config["isolation"]["camera" if field == "camera_id" else "plot"]
            if values and enabled:
                failures.append({"type": "cross_split_intersection", "pair": pair_name, "field": field, "values": values})
    for pair in pairs:
        if pair["conflicting_exact_labels"]:
            failures.append({"type": "exact_duplicate_label_conflict", "source_image_ids": pair["source_image_ids"]})
        if pair["decision"] == "needs_review" and pair["splits"][0] != pair["splits"][1]:
            failures.append({"type": "unresolved_cross_split_candidate", "source_image_ids": pair["source_image_ids"]})
    split_stats = {}
    for split in SPLITS:
        selected = [r for r in split_rows if r["split"] == split]
        labels = Counter(str(r["wilt_score"]) for r in selected)
        split_stats[split] = {"images": len(selected), "groups": sorted({r["split_group"] for r in selected}),
                              "labels": {str(i): labels[str(i)] for i in range(5)}}
        if len(selected) != config["expected_counts"][split]:
            failures.append({"type": "expected_count_mismatch", "split": split, "expected": config["expected_counts"][split], "actual": len(selected)})
        if any(labels[str(i)] == 0 for i in range(5)):
            failures.append({"type": "missing_wilt_class", "split": split})
    lock = {
        "split_version": config["split_version"], "algorithm_version": ALGORITHM,
        "seed": config["seed"], "assignment_policy": config["assignment_policy"],
        "config_sha256": config["_config_sha256"], "source_manifest_sha256": config["source"]["manifest_sha256"],
        "source_report_sha256": config["source"]["report_sha256"],
        "assignments": config["assignments"], "isolation": config["isolation"], "near_duplicate_policy": config["near_duplicates"],
    }
    artifact_bytes = {f"{split}.sources.jsonl": b"".join(canonical_bytes(r) for r in split_rows if r["split"] == split) for split in SPLITS}
    artifact_bytes["excluded.sources.jsonl"] = b"".join(canonical_bytes(r) for r in excluded)
    lock["manifest_sha256"] = {name: digest(data) for name, data in artifact_bytes.items()}
    lock_id = digest(canonical_bytes(lock))
    group_stats = []
    for group in sorted(actual_groups):
        selected = [r for r in split_rows if r["split_group"] == group]
        labels = Counter(r["wilt_score"] for r in selected)
        group_stats.append({"group": group, "camera_id": selected[0]["camera_id"], "plot_id": selected[0]["plot_id"],
                            "split": assignments[group], "images": len(selected), **{f"label_{i}": labels[i] for i in range(5)}})
    report = {
        "status": "fail" if failures else "pass", "split_version": config["split_version"], "seed": config["seed"],
        "lock_sha256": lock_id if not failures else None, "config_sha256": config["_config_sha256"],
        "source_manifest_sha256": config["source"]["manifest_sha256"], "source_report_sha256": config["source"]["report_sha256"],
        "source_row_accounting": source_report["row_accounting"], "eligible_images": len(rows), "excluded_images": len(excluded),
        "split_stats": split_stats, "per_group": group_stats, "intersections": intersections,
        "near_duplicates": duplicate_summary, "failures": failures,
        "policy": lock, "diagnostic_pairs": diagnostics,
        "warnings": ["Validation has only four independent wilt-4 source images in the current split", "One held-out test plot is not evidence of generalization to all fields", "Near-duplicate clearance applies to the declared heuristic; source groups retain complete plot sequences"],
        "not_performed": ["scene_generation", "action_label_generation", "training"],
    }
    summary = {"status": report["status"], "dry_run": dry_run, "output_dir": str(output), "lock_sha256": report["lock_sha256"],
               "split_stats": split_stats, "eligible_images": len(rows), "excluded_images": len(excluded),
               "near_duplicates": duplicate_summary, "failures": failures}
    if dry_run:
        return summary
    output.mkdir(parents=True, exist_ok=False)
    report["review_panels"] = review_panels(output, diagnostics, rows, config["_base"])
    with (output / "near_duplicate_pairs.jsonl").open("xb") as stream:
        stream.write(b"".join(canonical_bytes(pair) for pair in pairs))
    with (output / "group_stats.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "camera_id", "plot_id", "split", "images"] + [f"label_{i}" for i in range(5)])
        writer.writeheader()
        writer.writerows(group_stats)
    with (output / "split_report.json").open("xb") as stream:
        stream.write(canonical_bytes(report))
    if not failures:
        for name, data in artifact_bytes.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        with (output / "split_lock.json").open("xb") as stream:
            stream.write(canonical_bytes(lock))
    return summary
