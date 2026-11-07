"""Full scene QA against pinned sources and a deterministic teacher policy."""

from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import statistics

from jsonschema import Draft202012Validator

from .dataset import (Inputs, SPLITS, GENERATOR_VERSION, require, load_inputs, read_json, read_jsonl,
                      config_snapshot, case_quotas, scene_id, case_id, pair_id)
from .policy import all_requests, validate_request
from .split import canonical_bytes, digest, intersection_report, relative_file


def semantic_digest(records: list[dict]) -> str:
    return digest(b"".join(canonical_bytes(row) for row in sorted(records, key=lambda r: r["sample_id"])))


def build_lock(inputs: Inputs, payloads: dict[str, bytes]) -> dict:
    config = inputs.config
    source_dir = Path(__file__).parent
    return {"dataset_version": "agri-soybean-scene-v0", "generator_version": GENERATOR_VERSION,
            "seed": config["seed"], "config": config_snapshot(config), "config_sha256": config["_config_sha256"],
            "split_lock_sha256": config["generation"]["split_lock_sha256"],
            "policy_lock_sha256": config["generation"]["policy_lock_sha256"],
            "schema_sha256": digest(relative_file(inputs.base, config["generation"]["schema"]).read_bytes()),
            "model_contract_sha256": digest((source_dir / "model_contract.json").read_bytes()),
            "implementation_sha256": {name: digest((source_dir / name).read_bytes()) for name in
                                      ("dataset.py", "scenes.py", "validate.py", "policy.py", "language.py")},
            "manifest_sha256": {name: digest(data) for name, data in payloads.items()},
            "manifest_semantic_sha256": {name: semantic_digest([json.loads(line) for line in data.splitlines()]) for name, data in payloads.items()}}


def reuse_stats(counts: Counter, available: set[str]) -> dict:
    values = [counts[source] for source in sorted(available)]
    return {"min": min(values), "median": statistics.median(values), "max": max(values),
            "total_appearances": sum(values), "unused_sources": sum(v == 0 for v in values)}


def validate_records(inputs: Inputs, records: dict[str, list[dict]]) -> dict:
    validator = Draft202012Validator(inputs.schema)
    Draft202012Validator.check_schema(inputs.schema)
    require(set(records) == set(SPLITS), "Need all three dataset splits for leakage QA")
    all_sample_ids, all_pair_ids = set(), set()
    used_source_rows, family_sets, train_parameter_families = [], {}, {}
    report = {"status": "pass", "dataset_version": "agri-soybean-scene-v0", "generator_version": GENERATOR_VERSION,
              "seed": inputs.config["seed"], "splits": {}, "contrasts": {}, "permutations_checked": {},
              "source_paths_relative_to": "module base_dir (not manifest directory)",
              "visual_review": "not_performed: inspect preview separately",
              "label_type": "synthetic_policy_action", "physical_irrigation_validated": False,
              "not_performed": ["LeRobot conversion/round-trip", "tokenizer length/truncation check", "model training", "4GB GPU smoke"]}
    for split in SPLITS:
        rows = records[split]
        require(len(rows) == inputs.config["generation"]["record_counts"][split], f"Record quota mismatch: {split}")
        source_map = {r["source_image_id"]: r for r in inputs.rows[split]}
        pairs, cases, scenes = defaultdict(list), defaultdict(list), {}
        local_families = {"seen": set(), "unseen": set()}
        for number, row in enumerate(rows, 1):
            errors = list(validator.iter_errors(row))
            require(not errors, f"Schema failure {split}:{number}: {errors[0].message if errors else ''}")
            require(row["split"] == split and row["seed"] == inputs.config["seed"], f"Split/seed mismatch: {split}:{number}")
            require(row["sample_id"] not in all_sample_ids, "Duplicate sample ID")
            all_sample_ids.add(row["sample_id"])
            sources = [source_map.get(source) for source in row["source_image_ids"]]
            require(all(sources), "Source ID is excluded, unknown or from a different split")
            require(inputs.distinct(sources), "Repeated/exact/near/uncertain duplicate images within scene")
            require(row["images"] == {f"zone{i}": r["original_path"] for i, r in enumerate(sources, 1)}, "Image paths/order differ from pinned source")
            for path in row["images"].values():
                relative_file(inputs.base, path)
            require(row["source_groups"] == [r["split_group"] for r in sources], "Source group provenance mismatch")
            scores = [r["wilt_score"] for r in sources]
            require(row["teacher_annotations"]["wilt_scores"] == scores, "Teacher labels differ from pinned source")
            request = validate_request(row["intent"], row["intent_params"])
            require(row["target_action"] == inputs.policy.action(scores, request.intent, request.params), "Target differs from recomputed reference policy")
            require(row.get("provenance") == inputs.provenance(sources), "Missing/tampered source/config/lock provenance")
            expected = inputs.registry.render_pair(row["template_family_id"], request.params, split=split)
            expected = next(item for item in expected if item["language"] == row["language"])
            require(all(row[key] == value for key, value in expected.items()), "Instruction/intent/language/template semantics mismatch")
            sid = scene_id(split, row["source_image_ids"])
            cid = case_id(sid, row["intent"], row["intent_params"])
            pid = pair_id(cid, row["template_family_id"])
            require(row["scene_id"] == sid and row.get("case_id") == cid and row["translation_pair_id"] == pid and row["sample_id"] == pid + "_" + row["language"], "Scene/case/pair/sample ID mismatch")
            require("generation_metadata" in row, "Missing generation relation metadata")
            declared_patterns = {"contrastive_base": [0, 2, 4], "all_off": [0, 0, 0], "all_high": [4, 4, 4],
                                 "tie": [4, 4, 1], "below_threshold": [1, 1, 0]}
            kind = row["generation_metadata"]["kind"]
            if kind in declared_patterns:
                require(scores == declared_patterns[kind], "Declared edge pattern differs from source wilt")
            pairs[pid].append(row)
            cases[cid].append(row)
            scenes[sid] = sources
            local_families[row["wording_partition"]].add(row["template_family_id"])

        for pid, translations in pairs.items():
            require(pid not in all_pair_ids, "Translation pair crosses splits")
            all_pair_ids.add(pid)
            require(len(translations) == 2 and {r["language"] for r in translations} == {"vi", "en"}, "Translation pair must contain exactly vi/en")
            excluded_keys = {"language", "instruction", "template_id", "sample_id"}
            require({k: v for k, v in translations[0].items() if k not in excluded_keys} ==
                    {k: v for k, v in translations[1].items() if k not in excluded_keys}, "Translations differ in teacher/target/provenance")
        case_rows = [group[0] for _, group in sorted(cases.items())]
        for group in cases.values():
            require(len(group) == (2 if split == "train" else 4), "Case must have two train or four eval records")
            require(Counter((r["language"], r["wording_partition"]) for r in group) ==
                    Counter(itertools.product(("vi", "en"), ("seen",) if split == "train" else ("seen", "unseen"))), "Case language/wording pairing mismatch")
            for row in group:
                for key in ("scene_id", "source_image_ids", "images", "intent", "intent_params", "target_action", "teacher_annotations", "provenance", "generation_metadata"):
                    require(row[key] == group[0][key], f"Seen/unseen case differs in {key}")
        case_counts = Counter(row["intent"] for row in case_rows)
        require(dict(case_counts) == case_quotas(inputs.config, split), f"Intent case quota mismatch: {split}")
        parameter_counts = defaultdict(Counter)
        for row in case_rows:
            parameter_counts[row["intent"]][json.dumps(row["intent_params"], sort_keys=True)] += 1
        for intent, values in parameter_counts.items():
            require(max(values.values()) - min(values.values()) <= 1, f"Unbalanced zone/cap parameters: {split}/{intent}")
        used = {r["source_image_id"] for sources in scenes.values() for r in sources}
        used_source_rows.extend(source_map[source] for source in sorted(used))
        source_case_uses, source_record_uses, source_scene_uses = Counter(), Counter(), Counter()
        zone_cases = defaultdict(lambda: [0, 0, 0])
        for row in case_rows:
            source_case_uses.update(row["source_image_ids"])
            for zone, source in enumerate(row["source_image_ids"]):
                zone_cases[source][zone] += 1
        for row in rows:
            source_record_uses.update(row["source_image_ids"])
        for sources in scenes.values():
            source_scene_uses.update(r["source_image_id"] for r in sources)
        action_vectors = Counter(json.dumps(row["target_action"], separators=(",", ":")) for row in rows)
        labels_by_zone = {f"zone{zone+1}": dict(sorted(Counter(r["teacher_annotations"]["wilt_scores"][zone] for r in case_rows).items())) for zone in range(3)}
        actions_by_zone = {f"zone{zone+1}": dict(sorted(Counter(r["target_action"][zone] for r in rows).items())) for zone in range(3)}
        edge = {
            "all_wilt_below_2": sum(max(r["teacher_annotations"]["wilt_scores"]) < 2 for r in case_rows),
            "all_wilt4": sum(r["teacher_annotations"]["wilt_scores"] == [4, 4, 4] for r in case_rows),
            "all_action3": sum(r["target_action"] == [3, 3, 3] for r in case_rows),
            "actionable_max_tie": sum(max(r["teacher_annotations"]["wilt_scores"]) >= 2 and r["teacher_annotations"]["wilt_scores"].count(max(r["teacher_annotations"]["wilt_scores"])) > 1 for r in case_rows),
            "stop": case_counts["STOP_ALL"], "exclusion": case_counts["EXCLUDE_ZONE"],
        }
        require(all(edge.values()), f"Missing required edge cases: {split}")
        by_scene, by_instruction = defaultdict(list), defaultdict(list)
        for row in case_rows:
            by_scene[row["scene_id"]].append(row)
        for row in rows:
            by_instruction[row["instruction"]].append(row)
        image_contrasts = [r for group in by_scene.values() if len({tuple(r["target_action"]) for r in group}) > 1 for r in group[:1]]
        text_contrasts = [group[:2] for group in by_instruction.values() if len({r["scene_id"] for r in group}) > 1 and len({tuple(r["target_action"]) for r in group}) > 1]
        require(image_contrasts and text_contrasts, f"Missing image/language contrastive cases: {split}")
        report["contrasts"][split] = {"same_scene_multiple_targets": len(image_contrasts), "same_instruction_multiple_scenes_and_targets": len(text_contrasts)}
        permutations = 0
        for row in case_rows:
            relation = row["generation_metadata"]
            if relation["kind"] != "permutation":
                require(relation["permutation_of"] is None and relation["permutation"] is None, "Unexpected permutation metadata")
                continue
            parent_group = cases.get(relation["permutation_of"])
            require(parent_group is not None, "Permutation parent missing or in another split")
            parent = parent_group[0]
            permutation = relation["permutation"]
            require(isinstance(permutation, list) and all(type(i) is int for i in permutation) and sorted(permutation) == [1, 2, 3], "Invalid zone permutation")
            require(parent["generation_metadata"]["kind"] == "contrastive_base" and row["intent"] == parent["intent"], "Invalid permutation parent intent/kind")
            for key in ("source_image_ids", "target_action"):
                require(row[key] == [parent[key][i - 1] for i in permutation], f"Permutation mismatch: {key}")
            params = dict(parent["intent_params"])
            if "zone" in params:
                params["zone"] = permutation.index(params["zone"]) + 1
            require(row["intent_params"] == params, "Permutation zone parameter not remapped")
            permutations += 1
        require(permutations == 11, f"Need all 11 intent/params permutation cases: {split}")
        report["permutations_checked"][split] = permutations
        family_sets[split] = local_families
        parameter_family_coverage = {}
        for request in all_requests():
            key = str(request.intent) + "|" + json.dumps(request.params, sort_keys=True)
            selected_rows = [r for r in rows if r["intent"] == request.intent and r["intent_params"] == request.params]
            parameter_family_coverage[key] = {wording: sorted({r["template_family_id"] for r in selected_rows if r["wording_partition"] == wording}) for wording in ("seen", "unseen")}
            if split == "train":
                expected = {f.family_id for f in inputs.registry.pool(request.intent, split="train", wording="seen")}
                require(set(parameter_family_coverage[key]["seen"]) == expected, f"Incomplete train family coverage for parameters: {key}")
                train_parameter_families[key] = expected
            else:
                require(set(parameter_family_coverage[key]["seen"]) <= train_parameter_families[key], "Seen family/parameter combination absent from train")
        record_cross = Counter("|".join((r["language"], r["wording_partition"], r["intent"], json.dumps(r["intent_params"], sort_keys=True))) for r in rows)
        available = set(source_map)
        zeros = sum(r["target_action"] == [0, 0, 0] for r in rows)
        report["splits"][split] = {
            "records": len(rows), "translation_pairs": len(pairs), "cases": len(cases), "ordered_scenes": len(scenes),
            "unordered_source_triplets": len({tuple(sorted(r["source_image_ids"])) for r in case_rows}),
            "available_source_images": len(source_map), "used_source_images": len(used),
            "source_groups": sorted({source_map[s]["split_group"] for s in used}),
            "source_wilt_available": dict(sorted(Counter(r["wilt_score"] for r in source_map.values()).items())),
            "source_wilt_used": dict(sorted(Counter(source_map[s]["wilt_score"] for s in used).items())),
            "languages": dict(Counter(r["language"] for r in rows)),
            "wording": dict(Counter(r["wording_partition"] for r in rows)),
            "intent_cases": dict(sorted(case_counts.items())), "language_wording_intent_params": dict(sorted(record_cross.items())),
            "action_vectors_records": dict(sorted(action_vectors.items())), "action_levels_records_by_zone": actions_by_zone,
            "teacher_wilt_cases_by_zone": labels_by_zone, "zero_action_records": zeros, "zero_action_fraction": zeros / len(rows),
            "edge_cases": edge, "generation_kinds": dict(Counter(r["generation_metadata"]["kind"] for r in case_rows)),
            "family_ids": {wording: sorted(values) for wording, values in local_families.items()},
            "parameter_family_coverage": parameter_family_coverage,
            "template_records": dict(sorted(Counter(r["template_id"] for r in rows).items())),
            "reuse_cases": reuse_stats(source_case_uses, available), "reuse_records": reuse_stats(source_record_uses, available),
            "reuse_ordered_scenes": reuse_stats(source_scene_uses, available),
            "reuse_by_source": {s: {"case_appearances": source_case_uses[s], "record_appearances": source_record_uses[s],
                                    "ordered_scene_appearances": source_scene_uses[s], "zone_case_appearances": zone_cases[s]} for s in sorted(available)},
        }
    train_families = family_sets["train"]["seen"]
    expected_train = {f.family_id for f in inputs.registry.families if f.partition == "train"}
    require(train_families == expected_train and not family_sets["train"]["unseen"], "Train template-family coverage incomplete")
    for split in ("val", "test"):
        expected_unseen = {f.family_id for f in inputs.registry.families if f.partition == "unseen-" + split}
        require(family_sets[split]["seen"] <= train_families, "Seen family was never used in train")
        require(family_sets[split]["unseen"] == expected_unseen, "Unseen family coverage/isolation mismatch")
    intersections = intersection_report(used_source_rows)
    require(not any(values for fields in intersections.values() for values in fields.values()), "Generated dataset leaks source/hash/group/camera/plot/component across splits")
    report["source_intersections"] = intersections
    report["family_coverage_actual"] = {s: {w: sorted(v) for w, v in families.items()} for s, families in family_sets.items()}
    report["model_input_contract"] = {"images": ["zone1", "zone2", "zone3"], "task": "instruction only",
                                      "state": [0, 0, 0], "state_semantics": "assumed_all_valves_initially_closed",
                                      "teacher_and_structured_intent_in_features": False,
                                      "image_verification": "original encoded/pixel checksums; no annotation overlay or re-encoding"}
    return report


def validate_dataset(config: dict, directory: Path) -> tuple[dict, Inputs, dict]:
    inputs = load_inputs(config)
    lock = read_json(directory / "dataset_lock.json")
    payloads = {f"{split}.jsonl": (directory / f"{split}.jsonl").read_bytes() for split in SPLITS}
    require(isinstance(lock, dict) and lock.get("manifest_sha256") == {name: digest(data) for name, data in payloads.items()},
            "Dataset lock manifest hash mismatch")
    require(lock == build_lock(inputs, payloads), "Dataset lock/config/code/schema/manifest hash mismatch")
    records = {split: read_jsonl(directory / f"{split}.jsonl") for split in SPLITS}
    report = validate_records(inputs, records)
    report.update({"dataset_lock_sha256": digest((directory / "dataset_lock.json").read_bytes()),
                   "manifest_sha256": lock["manifest_sha256"], "manifest_semantic_sha256": lock["manifest_semantic_sha256"]})
    return report, inputs, records


def dataset_card(report: dict) -> str:
    lines = ["# AgriVLA-Soybean-v0 — Dataset card", "", "Dataset tổng hợp ba zone, song ngữ Việt–Anh; QA tự động PASS.", "",
             "Ảnh nguồn: Study of Leaf Wilt in Soybean Plants, Zenodo 8256382 (https://zenodo.org/records/8256382).",
             "Ảnh giữ nguyên bytes RGB; ba ảnh không phải quan sát đồng thời. Raw wilt 0–4 do nguồn gán, action do wilt_proxy_policy_v0 sinh.",
             "104 ảnh annotation=5 vẫn excluded. License/phạm vi phân phối raw chưa được xác minh đầy đủ; không đóng gói raw vào Git.", "",
             "| Split | Ảnh nguồn có sẵn/đã dùng | Ordered scenes | Cases | Records | All-zero records |",
             "|---|---:|---:|---:|---:|---:|"]
    for split, stats in report["splits"].items():
        lines.append(f"| {split} | {stats['available_source_images']}/{stats['used_source_images']} | {stats['ordered_scenes']} | {stats['cases']} | {stats['records']} | {stats['zero_action_records']} ({stats['zero_action_fraction']:.1%}) |")
    lines += ["", "Một ordered scene là một bộ ba ảnh theo thứ tự zone. Một case là scene + intent/params; train có hai bản dịch, val/test có bốn record (vi/en × seen/unseen).",
              "Các bản dịch, paraphrase, tổ hợp và ảnh cùng plot/thời điểm không phải quan sát thống kê độc lập.",
              "Val có bốn ảnh gốc wilt 4; test chỉ một plot. Reuse/zone/action/quota/template coverage chi tiết ở qa_report.json.", "",
              "Model nhận ba RGB, task=instruction và state [0,0,0] giả định mọi van ban đầu đóng; teacher wilt/intent/provenance chỉ ở metadata.",
              "Action 0/1/2/3 là OFF/LOW/MEDIUM/HIGH proxy; ngưỡng wilt 2. Không phải lượng nước/nông học đã hiệu chuẩn, không đo đất khô nhất.",
              "Policy mapping {0:0,1:0,2:1,3:2,4:3}; WORST_ONLY giữ mọi zone hòa max nếu max≥2, STOP luôn zeros với input hợp lệ.", "",
              "Tái lập: compose theo configs/dataset_v0.yaml, seed và hai upstream locks; dataset_lock.json pin config, implementation, schema, manifests và semantic hashes.",
              "Paths ảnh tương đối với module/base_dir, không tương đối với thư mục manifests. Source IDs tra trong source_splits/v0/*.sources.jsonl đã khóa.",
              "Train quota AUTO/WORST/ZONE/CAP/EXCLUDE/STOP = 30/20/15/15/10/10%; eval là panel thiết kế, không phải tần suất vận hành thực.",
              "60 family song ngữ; unseen-val/test tách train và nhau; seen coverage được kiểm trên record train thật.", "",
              "QA ảnh/hash/schema/policy/pairs/leakage/permutation đã chạy. Visual review lưu riêng ở preview/visual_review.json khi người/agent thực sự xem ảnh.",
              "Chưa convert LeRobot, kiểm tokenizer, train model, đo GPU 4 GB hoặc chứng minh visual/language grounding/hiệu quả tưới.", ""]
    return "\n".join(lines)
