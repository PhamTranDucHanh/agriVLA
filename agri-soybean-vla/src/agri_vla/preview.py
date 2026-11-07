"""Static local HTML and contact sheets; labels stay outside training images."""

from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path
import textwrap
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

from .dataset import SPLITS, safe_output, require, jsonl_bytes
from .split import canonical_bytes, digest, relative_file
from .validate import validate_dataset


def select_samples(records: dict, count: int) -> list[dict]:
    total = sum(len(rows) for rows in records.values())
    require(type(count) is int and 60 <= count <= total, f"Preview count must be 60–{total}")
    selected, seen = [], set()
    kinds = {"AUTO": "all_off", "WORST_ONLY": "tie", "ZONE_ONLY": "permutation",
             "CAP_LEVEL": "all_high", "EXCLUDE_ZONE": "contrastive_base", "STOP_ALL": "below_threshold"}
    # Full bilingual/wording case for each split/intent: 12 train + 24 val + 24 test.
    for split in SPLITS:
        by_case = defaultdict(list)
        for row in records[split]:
            by_case[row["case_id"]].append(row)
        for intent, kind in kinds.items():
            if split == "test" and intent in ("AUTO", "STOP_ALL"):
                kind = "all_high"  # Same high-wilt images: AUTO [3,3,3] versus STOP [0,0,0].
            if intent == "AUTO" and split == "train":
                candidates = [group for group in by_case.values() if group[0]["intent"] == intent and
                              group[0]["generation_metadata"]["kind"] == "balanced" and 3 in group[0]["teacher_annotations"]["wilt_scores"]]
            else:
                candidates = [group for group in by_case.values() if group[0]["intent"] == intent and group[0]["generation_metadata"]["kind"] == kind]
            require(candidates, f"Missing preview stratum: {split}/{intent}/{kind}")
            chosen = min(candidates, key=lambda group: group[0]["case_id"])
            selected.extend(chosen)
            seen.update(r["sample_id"] for r in chosen)
    # Fill with additional balanced cases, cycling splits; selection never mutates data.
    queues = {split: sorted((r for r in rows if r["sample_id"] not in seen),
                            key=lambda r: (r["generation_metadata"]["kind"] != "balanced", r["sample_id"])) for split, rows in records.items()}
    while len(selected) < count:
        progress = False
        for split in SPLITS:
            if queues[split] and len(selected) < count:
                selected.append(queues[split].pop(0))
                progress = True
        require(progress, "Preview selection exhausted the dataset")
    return selected


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def write_panels(output: Path, selected: list[dict], thumbnails: dict) -> list[str]:
    names = []
    for start in range(0, len(selected), 6):
        batch = selected[start:start + 6]
        sheet = Image.new("RGB", (1180, 36 + 190 * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 7), f"AgriVLA synthetic scene review | records {start+1}–{start+len(batch)} | teacher metadata outside images", font=font(15), fill="black")
        for index, row in enumerate(batch):
            y = 36 + 190 * index
            draw.line((0, y, 1180, y), fill="#999999")
            for zone in range(3):
                thumb = thumbnails[row["source_image_ids"][zone]]
                sheet.paste(thumb, (8 + zone * 188, y + 22))
                draw.text((8 + zone * 188, y + 3), f"zone{zone+1} | wilt {row['teacher_annotations']['wilt_scores'][zone]}", font=font(12), fill="black")
                source = row["source_image_ids"][zone]
                for n, line in enumerate(textwrap.wrap(source, 26)):
                    draw.text((8 + zone * 188, y + 160 + n * 12), line, font=font(10), fill="black")
            info = [f"#{start+index+1} {row['split']} / {row['language']} / {row['wording_partition']}",
                    f"{row['intent']} {row['intent_params']} | {row['generation_metadata']['kind']}",
                    f"wilt {row['teacher_annotations']['wilt_scores']} → target {row['target_action']}",
                    row["template_family_id"], *textwrap.wrap(row["instruction"], 69), row["case_id"]]
            for n, line in enumerate(info):
                draw.text((578, y + 5 + 21 * n), line, font=font(14 if n != 6 else 12), fill="black")
        name = f"review_{len(names)+1:02d}.jpg"
        sheet.save(output / name, quality=92)
        names.append(name)
    return names


def preview_dataset(config: dict, directory: Path, *, output_dir: Path | None = None, count=60, dry_run=False) -> dict:
    output = safe_output(config, output_dir or directory / "preview")
    report, inputs, records = validate_dataset(config, directory)
    selected = select_samples(records, count)
    coverage = Counter(f"{r['split']}|{r['language']}|{r['wording_partition']}|{r['intent']}" for r in selected)
    result = {"status": "pass", "dry_run": dry_run, "records": len(selected), "cases": len({r["case_id"] for r in selected}),
              "ordered_scenes": len({r["scene_id"] for r in selected}), "source_images": len({s for r in selected for s in r["source_image_ids"]}),
              "dataset_lock_sha256": report["dataset_lock_sha256"], "coverage": dict(sorted(coverage.items())),
              "generation_kinds": dict(Counter(r["generation_metadata"]["kind"] for r in selected)),
              "selected_sample_ids": [r["sample_id"] for r in selected],
              "visual_review": "not_performed: file generation alone is not visual inspection"}
    if dry_run:
        return result
    output.mkdir(parents=True, exist_ok=False)
    (output / "thumbnails").mkdir()
    thumbnails, thumbnail_names = {}, {}
    source_map = {r["source_image_id"]: r for rows in inputs.rows.values() for r in rows}
    for source in sorted({s for r in selected for s in r["source_image_ids"]}):
        with Image.open(relative_file(inputs.base, source_map[source]["original_path"])) as image:
            thumb = image.resize((180, 135), Image.Resampling.LANCZOS)
        name = "thumbnails/" + digest(source.encode())[:24] + ".jpg"
        thumb.save(output / name, quality=92)
        thumbnails[source], thumbnail_names[source] = thumb, name
    panels = write_panels(output, selected, thumbnails)
    cards = []
    escape = html.escape
    for number, row in enumerate(selected, 1):
        images = []
        for zone, source in enumerate(row["source_image_ids"], 1):
            original = relative_file(inputs.base, row["images"][f"zone{zone}"])
            original_link = quote(Path(os.path.relpath(original, output)).as_posix())
            images.append(f'<figure><a href="{escape(original_link, quote=True)}"><img src="{thumbnail_names[source]}" alt="zone {zone}"></a><figcaption>Zone {zone}: {escape(source)}<br>Teacher wilt: {row["teacher_annotations"]["wilt_scores"][zone-1]}</figcaption></figure>')
        cards.append(f'<article><h2>#{number} · {row["split"]} · {row["language"]} · {row["wording_partition"]} · {row["intent"]}</h2><div class="images">{"".join(images)}</div><p>{escape(row["instruction"])}</p><p>Target: <strong>{row["target_action"]}</strong> · Params: {escape(str(row["intent_params"]))} · {row["generation_metadata"]["kind"]}</p><details><summary>Full record / provenance / teacher metadata</summary><pre>{escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></article>')
    page = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgriVLA dataset preview</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#f4f7f3;color:#172a21}article{background:white;padding:18px;margin:18px 0;border:1px solid #cad6c8;border-radius:10px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}figcaption{font-size:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:18px}</style><h1>AgriVLA-Soybean-v0 — local review</h1><p>Synthetic three-zone scenes. Teacher labels and proxy targets appear only in this review; training images retain original bytes. Samples are correlated translations/paraphrases, not independent observations.</p>' + ''.join(cards) + '</html>'
    for name, content in (("index.html", page.encode()), ("preview.samples.jsonl", jsonl_bytes(selected))):
        with (output / name).open("xb") as stream:
            stream.write(content)
    result["contact_sheets"] = panels
    result["preview_samples_sha256"] = digest((output / "preview.samples.jsonl").read_bytes())
    with (output / "preview_report.json").open("xb") as stream:
        stream.write(canonical_bytes(result))
    return result | {"html": str(output / "index.html")}
