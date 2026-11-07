"""Deterministic, quota-based synthetic three-zone scene generation."""

from collections import Counter
import itertools
from pathlib import Path
import random

from .dataset import (Inputs, SPLITS, GENERATOR_VERSION, DatasetError, require, new_output,
                      load_inputs, case_quotas, scene_id, case_id, pair_id, jsonl_bytes, safe_output)
from .policy import all_requests, validate_request
from .split import canonical_bytes, digest, relative_file


class SourceSampler:
    def __init__(self, inputs: Inputs, split: str, rng: random.Random):
        self.inputs = inputs
        self.labels = {score: sorted((r for r in inputs.rows[split] if r["wilt_score"] == score), key=lambda r: r["source_image_id"]) for score in range(5)}
        self.rank = {}
        for rows in self.labels.values():
            rng.shuffle(rows)
            self.rank.update({r["source_image_id"]: n for n, r in enumerate(rows)})
        self.uses, self.zone_uses = Counter(), Counter()

    def take(self, scores) -> list[dict]:
        selected = []
        # Backtracking also handles rare labels with only a few mutually distinct images.
        def pick(zone):
            if zone == 3:
                return self.inputs.distinct(selected)
            choices = sorted(self.labels[scores[zone]], key=lambda r: (self.uses[r["source_image_id"]],
                             self.zone_uses[r["source_image_id"], zone], self.rank[r["source_image_id"]]))
            for row in choices:
                if any(row[key] == other[key] for other in selected for key in ("source_image_id", "sha256", "pixel_sha256", "duplicate_component_id")):
                    continue
                selected.append(row)
                if pick(zone + 1):
                    return True
                selected.pop()
            return False
        require(pick(0), f"No three distinct/non-near source images satisfy wilt pattern {list(scores)}")
        return list(selected)

    def record(self, sources):
        for zone, row in enumerate(sources):
            self.uses[row["source_image_id"]] += 1
            self.zone_uses[row["source_image_id"], zone] += 1


def generate_split(inputs: Inputs, split: str) -> list[dict]:
    config, registry = inputs.config, inputs.registry
    seed = config["seed"]
    # Stable split-local RNG: changing execution order does not change a split.
    rng = random.Random(int(digest(canonical_bytes([GENERATOR_VERSION, seed, split])), 16))
    sampler = SourceSampler(inputs, split, rng)
    quotas = case_quotas(config, split)
    counts, param_counts, family_counts = Counter(), Counter(), Counter()
    records, used_cases = [], set()

    def add(sources, request, kind, *, parent=None, permutation=None):
        intent, params = str(request.intent), request.params
        require(counts[intent] < quotas[intent], f"Quota too small for required edge panel: {intent}")
        ids = [r["source_image_id"] for r in sources]
        sid = scene_id(split, ids)
        cid = case_id(sid, intent, params)
        if cid in used_cases:
            return None
        used_cases.add(cid)
        counts[intent] += 1
        param_counts[intent, tuple(params.items())] += 1
        sampler.record(sources)
        target = inputs.policy.action([r["wilt_score"] for r in sources], intent, params)
        for wording in (("seen",) if split == "train" else ("seen", "unseen")):
            pool = registry.pool(intent, split=split, wording=wording)
            # Rotate within each parameter value to avoid family↔zone/cap correlation.
            key = (intent, tuple(params.items()), wording)
            # Fixed wording for the controlled panel gives same-text/different-image cases.
            index = 0 if kind != "balanced" else (1 + family_counts[key]) % len(pool)
            if kind == "balanced":
                family_counts[key] += 1
            family = pool[index]
            pid = pair_id(cid, family.family_id)
            for rendered in registry.render_pair(family.family_id, params, split=split):
                language = rendered["language"]
                records.append({"schema_version": "agri-soybean-scene-v0", "sample_id": pid + "_" + language,
                                "scene_id": sid, "case_id": cid, "scene_type": "synthetic_three_zone", "split": split,
                                "images": {f"zone{n}": row["original_path"] for n, row in enumerate(sources, 1)},
                                "source_image_ids": ids, "source_groups": [r["split_group"] for r in sources],
                                **rendered, "translation_pair_id": pid,
                                "teacher_annotations": {"wilt_scores": [r["wilt_score"] for r in sources]},
                                "target_action": target, "policy_version": inputs.policy.version,
                                "label_source": inputs.policy.version, "label_type": "synthetic_policy_action", "seed": seed,
                                "provenance": inputs.provenance(sources),
                                "generation_metadata": {"kind": kind, "permutation_of": parent, "permutation": permutation}})
        return cid

    requests = all_requests()
    base_sources = sampler.take([0, 2, 4])
    parents = [add(base_sources, request, "contrastive_base") for request in requests]
    permutation = [3, 1, 2]  # New zone i takes old zone permutation[i], one-based.
    permuted = [base_sources[index - 1] for index in permutation]
    for request, parent in zip(requests, parents, strict=True):
        params = request.params
        if "zone" in params:
            params["zone"] = permutation.index(params["zone"]) + 1
        add(permuted, validate_request(request.intent, params), "permutation", parent=parent, permutation=permutation)
    for pattern, kind in (([0, 0, 0], "all_off"), ([4, 4, 4], "all_high"),
                          ([4, 4, 1], "tie"), ([1, 1, 0], "below_threshold")):
        sources = sampler.take(pattern)
        for request in requests:
            add(sources, request, kind)

    remaining = [intent for intent, quota in quotas.items() for _ in range(quota - counts[intent])]
    rng.shuffle(remaining)
    patterns = list(itertools.product(range(5), repeat=3))
    rng.shuffle(patterns)
    pattern_index = 0
    for intent in remaining:
        candidates = [r for r in requests if r.intent == intent]
        request = min(candidates, key=lambda r: param_counts[intent, tuple(r.params.items())])
        for _ in range(2000):
            scores = patterns[pattern_index % len(patterns)]
            pattern_index += 1
            sources = sampler.take(scores)
            if add(sources, request, "balanced") is not None:
                break
        else:
            raise DatasetError("Could not meet case quota without duplicate cases")
    require(dict(counts) == quotas, f"Case quotas not met: {split}")
    require(len(records) == config["generation"]["record_counts"][split], f"Record count not met: {split}")
    return records


def compose(config: dict, *, output_dir: Path | None = None, dry_run=False) -> dict:
    from .validate import validate_records, build_lock, dataset_card

    output = safe_output(config, output_dir or relative_file(config["_base"], config["generation"]["output_dir"]))
    inputs = load_inputs(config)
    records = {split: generate_split(inputs, split) for split in SPLITS}
    report = validate_records(inputs, records)
    payloads = {f"{split}.jsonl": jsonl_bytes(records[split]) for split in SPLITS}
    lock = build_lock(inputs, payloads)
    lock_bytes = canonical_bytes(lock)
    report["dataset_lock_sha256"] = digest(lock_bytes)
    report["manifest_sha256"] = lock["manifest_sha256"]
    report["manifest_semantic_sha256"] = lock["manifest_semantic_sha256"]
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        for name, data in (("dataset_lock.json", lock_bytes), ("qa_report.json", canonical_bytes(report)),
                           ("DATASET_CARD.md", dataset_card(report).encode("utf-8"))):
            with (output / name).open("xb") as stream:
                stream.write(data)
    return {"status": "pass", "dry_run": dry_run, "output_dir": str(output),
            "dataset_lock_sha256": report["dataset_lock_sha256"], "manifest_sha256": report["manifest_sha256"],
            "manifest_semantic_sha256": report["manifest_semantic_sha256"], "splits": report["splits"],
            "visual_review": "not_performed: run preview and inspect images"}
