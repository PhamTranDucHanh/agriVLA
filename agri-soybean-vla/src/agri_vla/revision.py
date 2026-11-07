"""Versioned wording + explicit WATER_ALL extension; immutable v0 remains valid.

No new source images/scenes/splits: reuse QA-approved source triplets from v0.
Inherited six-intent targets stay byte-identical. Explicit operator override is
separate from the wilt proxy and must include level=1/2/3, never guessed by LLM.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import itertools
from pathlib import Path
import random

from .dataset import (require, new_output, read_json, read_jsonl, jsonl_bytes,
                      load_compose_config, case_id, pair_id, SPLITS)
from .language import load_registry, normalize_text
from .policy import load_unique_yaml, reference_action, validate_wilt
from .split import digest, canonical_bytes, relative_file
from .training import INPUT_LOCK, file_sha, write_json

VERSION = "agri_soybean_revision_v1"
POLICY = "wilt_proxy_with_operator_override_v1"


def revision_action(wilt, intent, params=None):
    validate_wilt(wilt)
    params = {} if params is None else params
    if intent == "WATER_ALL":
        require(isinstance(params, dict) and set(params) == {"level"} and type(params["level"]) is int and params["level"] in (1,2,3), "WATER_ALL requires explicit integer level=1/2/3")
        return [params["level"]] * 3
    return reference_action(wilt, intent, params)


def load_revision(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    c = load_unique_yaml(raw, context="dataset revision")
    require(set(c) == {"version","base_dir","parent_config","parent_manifests","parent_lock_sha256","language_config","extension_config","schema","output_dir","seed","counts","force_cases_per_level"}, "Invalid revision config fields")
    require(c["version"] == VERSION and c["parent_lock_sha256"] == INPUT_LOCK and c["seed"] == 42, "Revision must descend from locked v0")
    require(c["counts"] == {"train":3600,"val":720,"test":720} and c["force_cases_per_level"] == {"train":100,"val":10,"test":10}, "Unexpected v1 quotas")
    require(not Path(c["base_dir"]).is_absolute(), "Relative base required")
    base = (path.parent/c["base_dir"]).resolve()
    for key in ("parent_config","parent_manifests","language_config","extension_config","schema","output_dir"):
        relative_file(base,c[key])
    return c | {"_base":base,"_path":path,"_sha256":digest(raw)}


def extension_templates(c):
    e = load_unique_yaml((c["_base"]/c["extension_config"]).read_bytes(), context="operator override")
    require(e["policy_version"] == POLICY and e["intent"] == "WATER_ALL" and e["allowed_levels"] == [1,2,3] and e["target_rule"] == ["level"]*3, "Invalid override policy")
    require(e["meaning"] == "explicit_operator_override_independent_of_wilt" and e["missing_level"] == "reject_ambiguous_command", "Override ambiguity")
    require(set(e["templates"]) == {"train","unseen-val","unseen-test"}, "Invalid force partitions")
    ids, texts = set(), set()
    for partition, pool in e["templates"].items():
        require(len(pool) == (6 if partition == "train" else 2), "Expected 6/2/2 force families")
        for family in pool:
            require(set(family) == {"id","vi","en"} and family["id"] not in ids, "Invalid/duplicate force family")
            ids.add(family["id"])
            for lang in ("vi","en"):
                text = family[lang]
                lower = text.casefold()
                all_three = ("cả ba" in lower or "cả 3" in lower) if lang == "vi" else (("all three" in lower or "all 3" in lower) or "every one of the three" in lower)
                require(text.count("{level}") == 1 and "{" not in text.replace("{level}","") and all_three, "Force template must explicitly mention all gardens and level")
                norm = (lang, normalize_text(text))
                require(norm not in texts, "Force wording leaks across pools")
                texts.add(norm)
    return e


def parent_inputs(c):
    from .validate import validate_dataset
    parent = c["_base"]/c["parent_manifests"]
    require(file_sha(parent/"dataset_lock.json") == INPUT_LOCK, "Parent v0 lock changed")
    return validate_dataset(load_compose_config(c["_base"]/c["parent_config"]),parent)


def generate_revision(c, parents):
    registry = load_registry(c["_base"]/c["language_config"])
    extension = extension_templates(c)
    old_ids = {f.family_id for f in registry.families}
    old_texts = {(lang,normalize_text(getattr(f,lang))) for f in registry.families for lang in ("vi","en")}
    require(all(f["id"] not in old_ids and all((lang,normalize_text(f[lang])) not in old_texts for lang in ("vi","en")) for pool in extension["templates"].values() for f in pool), "Force template overlaps inherited family/text")
    output = {}
    for split, rows in parents.items():
        records = []
        for original in rows:
            row = deepcopy(original)
            rendered = registry.render_pair(row["template_family_id"],row["intent_params"],split=split)
            row["instruction"] = next(r["instruction"] for r in rendered if r["language"] == row["language"])
            row.update(schema_version="agri-soybean-scene-v1",policy_version=POLICY,label_source=POLICY,
                       revision_metadata={"parent_dataset_lock_sha256":INPUT_LOCK,"parent_sample_id":original["sample_id"],"target_rule":"inherited_v0"})
            records.append(row)
        # Choose existing ordered triplets, without sampling or reassigning sources.
        scenes = {r["scene_id"]:r for r in rows}
        pool = sorted(scenes)
        random.Random(c["seed"]+SPLITS.index(split)).shuffle(pool)
        n = c["force_cases_per_level"][split]
        require(len(pool) >= n, "Not enough independent parent triplets")
        # Ensure even the all-OFF scene has explicit override examples.
        off = next(r["scene_id"] for r in rows if max(r["teacher_annotations"]["wilt_scores"]) < 2)
        pool = [off] + [s for s in pool if s != off]
        for level in (1,2,3):
            for i, sid in enumerate(pool[:n]):
                parent = scenes[sid]
                cid = case_id(sid,"WATER_ALL",{"level":level})
                for wording in (("seen",) if split == "train" else ("seen","unseen")):
                    families = extension["templates"]["train" if wording == "seen" else "unseen-"+split]
                    family = families[i % len(families)]
                    pid = pair_id(cid,family["id"])
                    for language in ("vi","en"):
                        row = deepcopy(parent)
                        row.update(schema_version="agri-soybean-scene-v1",sample_id=pid+"_"+language,case_id=cid,
                            instruction=family[language].format(level=level),language=language,intent="WATER_ALL",intent_params={"level":level},
                            template_family_id=family["id"],template_id=family["id"]+"_"+language,translation_pair_id=pid,wording_partition=wording,
                            target_action=revision_action(parent["teacher_annotations"]["wilt_scores"],"WATER_ALL",{"level":level}),
                            policy_version=POLICY,label_source=POLICY,label_type="synthetic_operator_override_action",
                            generation_metadata={"kind":"operator_override_all","permutation_of":None,"permutation":None},
                            revision_metadata={"parent_dataset_lock_sha256":INPUT_LOCK,"parent_sample_id":parent["sample_id"],"target_rule":"operator_override_all"})
                        records.append(row)
        output[split] = records
    return output


def qa_revision(c, rows, parents):
    from jsonschema import Draft202012Validator
    schema = Draft202012Validator(read_json(c["_base"]/c["schema"]))
    expected = generate_revision(c,parents)
    require(rows == expected, "Revision differs from deterministic parent/wording/policy generation")
    seen, sources, groups, cases = set(),{}, {}, {}
    result = {"status":"pass","parent_dataset_lock_sha256":INPUT_LOCK,"policy_version":POLICY,"splits":{},"source_triplets_changed":False,"inherited_targets_changed":False,"new_intent":"WATER_ALL","physical_irrigation_validated":False}
    for split in SPLITS:
        require(len(rows[split]) == c["counts"][split], "Revision quota mismatch")
        parent_by_id = {r["sample_id"]:r for r in parents[split]}
        sources[split] = {s for r in rows[split] for s in r["source_image_ids"]}
        groups[split] = {s for r in rows[split] for s in r["source_groups"]}
        inherited, changed = 0, 0
        pairs = defaultdict(list)
        for row in rows[split]:
            schema.validate(row)
            require(row["sample_id"] not in seen, "Duplicate revision sample ID")
            seen.add(row["sample_id"])
            require(row["target_action"] == revision_action(row["teacher_annotations"]["wilt_scores"],row["intent"],row["intent_params"]), "Incorrect v1 target")
            parent = parent_by_id[row["revision_metadata"]["parent_sample_id"]]
            for key in ("images","source_image_ids","source_groups","teacher_annotations","provenance","scene_id"):
                require(row[key] == parent[key], f"Source structure changed: {key}")
            if row["intent"] != "WATER_ALL":
                inherited += 1
                require(row["target_action"] == parent["target_action"], "Inherited v0 label changed")
                changed += row["instruction"] != parent["instruction"]
            pairs[row["translation_pair_id"]].append(row)
        require(all(len(p)==2 and {r["language"] for r in p}=={"vi","en"} and p[0]["target_action"]==p[1]["target_action"] for p in pairs.values()), "Broken translation pair")
        force = [r for r in rows[split] if r["intent"] == "WATER_ALL"]
        result["splits"][split] = {"records":len(rows[split]),"inherited_records":inherited,"wording_changed_records":changed,"water_all_records":len(force),"water_all_levels":dict(Counter(r["intent_params"]["level"] for r in force)),"languages":dict(Counter(r["language"] for r in rows[split])),"intents":dict(Counter(r["intent"] for r in rows[split])),"zero_action_records":sum(r["target_action"] == [0,0,0] for r in rows[split]),"source_images":len(sources[split])}
    for a,b in itertools.combinations(SPLITS,2):
        require(not sources[a]&sources[b] and not groups[a]&groups[b], "Revision source/group leakage")
    result["source_intersections"] = {a+"/"+b:sorted(sources[a]&sources[b]) for a,b in itertools.combinations(SPLITS,2)}
    return result


def revision_lock(c, rows):
    return {"version":VERSION,"parent_lock_sha256":INPUT_LOCK,"config_sha256":c["_sha256"],
            "files_sha256":{name:file_sha(c["_base"]/c[name]) for name in ("language_config","extension_config","schema")},
            "implementation_sha256":file_sha(Path(__file__)),
            "manifest_sha256":{s+".jsonl":digest(jsonl_bytes(rows[s])) for s in SPLITS}}


def compose_revision(c, *, dry_run=False):
    output = c["_base"]/c["output_dir"]
    new_output(output)
    _,inputs,parents = parent_inputs(c)
    rows = generate_revision(c,parents)
    report = qa_revision(c,rows,parents)
    lock = revision_lock(c,rows)
    report["dataset_lock_sha256"] = digest(canonical_bytes(lock))
    if not dry_run:
        output.mkdir(parents=True)
        for split in SPLITS:
            (output/(split+".jsonl")).write_bytes(jsonl_bytes(rows[split]))
        (output/"dataset_lock.json").write_bytes(canonical_bytes(lock))
        write_json(output/"qa_report.json",report)
        (output/"DATASET_CARD.md").write_text("# Soybean wording/override revision v1\n\nParent v0 is immutable. 3600/720/720 bilingual records. WATER_ALL(level=1/2/3) always yields [level]*3, independent of wilt. Missing level is ambiguous and not a target. STOP_ALL stays [0,0,0]. Original six-intent targets/source triplets/splits and family partitions preserved; some formal wording replaced by casual paired wording. New WATER_ALL families are 6/2/2 train/unseen-val/unseen-test. All source images already reviewed in v0; new language is reviewed in configs, not a new field observation.\n",encoding="utf-8")
    return report


def validate_revision(c):
    _,inputs,parents = parent_inputs(c)
    output = c["_base"]/c["output_dir"]
    rows = {s:read_jsonl(output/(s+".jsonl")) for s in SPLITS}
    lock = read_json(output/"dataset_lock.json")
    require(lock == revision_lock(c,rows), "Revision config/code/schema/manifest hashes changed")
    report = qa_revision(c,rows,parents)
    report["dataset_lock_sha256"] = file_sha(output/"dataset_lock.json")
    return report,inputs,rows
