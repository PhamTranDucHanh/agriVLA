"""Pinned inputs and portable paths shared by scene generation and QA; no ML."""

from collections import Counter
from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path
import re

from .language import load_registry, TemplateRegistry
from .policy import Intent, load_policy, load_unique_yaml, Policy, PolicyError
from .source import SourceError
from .split import (canonical_bytes, digest, relative_file, load_sources, load_split_config,
                    intersection_report, image_features, pair_metrics, classify_pair)

SPLITS = ("train", "val", "test")
GENERATOR_VERSION = "soybean_scene_generator_v0"


class DatasetError(ValueError):
    """Dataset/config/QA contract violation; existing inputs remain untouched."""


def require(condition, message):
    if not condition:
        raise DatasetError(message)


def new_output(path: Path) -> Path:
    require(not path.exists() and not path.is_symlink(), f"Output already exists; choose a NEW path: {path}")
    return path


def safe_output(config: dict, path: Path) -> Path:
    raw = relative_file(config["_base"], config["paths"]["raw_dir"])
    require(not path.resolve().is_relative_to(raw), "Generated output must be outside raw data")
    return new_output(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_bytes(rows) -> bytes:
    return b"".join(canonical_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL line: {path}:{number}")
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL row must be an object: {path}:{number}")
        rows.append(value)
    return rows


def load_compose_config(path: Path) -> dict:
    path = path.resolve()
    raw = path.read_bytes()
    config = load_unique_yaml(raw, context="dataset")
    try:
        require(type(config["config_version"]) is int and config["config_version"] == 1, "Unsupported dataset config version")
        require(type(config["seed"]) is int and config["seed"] >= 0, "seed must be a nonnegative integer")
        require(isinstance(config["base_dir"], str) and not Path(config["base_dir"]).is_absolute(), "base_dir must be relative to config")
        base = (path.parent / config["base_dir"]).resolve()
        g = config["generation"]
        expected = {"generator_version", "source_split_config", "source_split_dir", "split_lock_sha256", "policy_config",
                    "language_config", "policy_audit_dir", "policy_lock_sha256", "schema", "output_dir", "record_counts",
                    "train_intent_percent", "eval_intent_cases", "within_scene_near_policy"}
        require(isinstance(g, dict) and set(g) == expected, "Missing/unknown generation config fields")
        require(g["generator_version"] == GENERATOR_VERSION, "Unsupported generator version")
        require(g["within_scene_near_policy"] == "reject_near_and_uncertain", "Do not relax within-scene duplicate policy")
        for key in ("source_split_config", "source_split_dir", "policy_config", "language_config", "policy_audit_dir", "schema", "output_dir"):
            relative_file(base, g[key])
        for key in ("split_lock_sha256", "policy_lock_sha256"):
            require(isinstance(g[key], str) and bool(re.fullmatch(r"[a-f0-9]{64}", g[key])), f"Invalid {key}")
        require(set(g["record_counts"]) == set(SPLITS), "record_counts needs train/val/test")
        for split in SPLITS:
            n = g["record_counts"][split]
            require(type(n) is int and n > 0 and n % (2 if split == "train" else 4) == 0, f"Invalid bilingual/wording record count: {split}")
        for key, total in (("train_intent_percent", 100), ("eval_intent_cases", g["record_counts"]["val"] // 4)):
            require(isinstance(g[key], dict) and set(g[key]) == {str(i) for i in Intent}, f"Invalid {key} intents")
            require(all(type(v) is int and v > 0 for v in g[key].values()) and sum(g[key].values()) == total, f"Invalid {key} quota total")
        require(g["record_counts"]["val"] == g["record_counts"]["test"], "v0 uses equal-sized val/test panels")
        train_cases = g["record_counts"]["train"] // 2
        require(all(train_cases * p % 100 == 0 for p in g["train_intent_percent"].values()), "Train intent quota must yield integer case counts")
    except (KeyError, TypeError) as error:
        raise DatasetError(f"Malformed compose config: {error}") from error
    return config | {"_base": base, "_config_path": path, "_config_sha256": digest(raw)}


def config_snapshot(config: dict) -> dict:
    # YAML label maps have integer keys; a lock must compare in JSON's key domain.
    raw = {key: value for key, value in config.items() if not key.startswith("_")}
    return json.loads(json.dumps(raw, ensure_ascii=False))


def case_quotas(config: dict, split: str) -> dict:
    g = config["generation"]
    if split == "train":
        return {intent: g["record_counts"]["train"] // 2 * percent // 100 for intent, percent in g["train_intent_percent"].items()}
    return dict(g["eval_intent_cases"])


def scene_id(split: str, sources: list[str]) -> str:
    return f"scene_{split}_" + digest(canonical_bytes(sources))[:24]


def case_id(scene: str, intent: str, params: dict) -> str:
    return "case_" + digest(canonical_bytes([scene, intent, params]))[:24]


def pair_id(case: str, family: str) -> str:
    return "pair_" + digest(canonical_bytes([case, family]))[:24]


@dataclass
class Inputs:
    config: dict
    rows: dict[str, list[dict]]
    split_lock: dict
    policy_lock: dict
    policy: Policy
    registry: TemplateRegistry
    schema: dict
    features: dict = field(default_factory=dict)
    pair_cache: dict = field(default_factory=dict)

    @property
    def base(self):
        return self.config["_base"]

    def provenance(self, sources: list[dict]) -> dict:
        g = self.config["generation"]
        return {"generator_version": GENERATOR_VERSION, "compose_config_sha256": self.config["_config_sha256"],
                "split_lock_sha256": g["split_lock_sha256"], "policy_lock_sha256": g["policy_lock_sha256"],
                "source_manifest_sha256": self.split_lock["manifest_sha256"][sources[0]["split"] + ".sources.jsonl"],
                "source_sha256": [r["sha256"] for r in sources], "source_pixel_sha256": [r["pixel_sha256"] for r in sources],
                "source_duplicate_components": [r["duplicate_component_id"] for r in sources]}

    def distinct(self, sources: list[dict]) -> bool:
        for key in ("source_image_id", "sha256", "pixel_sha256", "duplicate_component_id"):
            if len({r[key] for r in sources}) != 3:
                return False
        for a, b in itertools.combinations(sources, 2):
            key = tuple(sorted((a["source_image_id"], b["source_image_id"])))
            if key not in self.pair_cache:
                fa, fb = self.features[key[0]], self.features[key[1]]
                policy = self.config["_near_policy"]
                self.pair_cache[key] = ((fa["dhash"] ^ fb["dhash"]).bit_count() > policy["hamming_max"] or
                                        classify_pair(pair_metrics(fa, fb), policy) == "distinct_under_policy")
            if not self.pair_cache[key]:
                return False
        return True


def load_inputs(config: dict) -> Inputs:
    """Verify locks, provenance, all eligible source files and duplicate isolation."""
    base, g = config["_base"], config["generation"]
    split_dir = relative_file(base, g["source_split_dir"])
    policy_dir = relative_file(base, g["policy_audit_dir"])
    for path, expected in ((split_dir / "split_lock.json", g["split_lock_sha256"]),
                           (policy_dir / "policy_lock.json", g["policy_lock_sha256"])):
        require(digest(path.read_bytes()) == expected, f"Pinned lock SHA256 mismatch: {path}")
    sl, pl = read_json(split_dir / "split_lock.json"), read_json(policy_dir / "policy_lock.json")
    sc = load_split_config(relative_file(base, g["source_split_config"]))
    require(sc["_base"] == base and sc["_config_sha256"] == sl["config_sha256"], "Split config/base differs from lock")
    require(sl["assignments"] == sc["assignments"] and sl["isolation"] == sc["isolation"], "Split lock/config assignments differ")
    require(sc["source"]["manifest_sha256"] == sl["source_manifest_sha256"] and sc["source"]["report_sha256"] == sl["source_report_sha256"], "Source provenance differs from lock")
    eligible, excluded, _ = load_sources(sc)
    original = {r["source_image_id"]: r for r in eligible}
    for name, expected in sl["manifest_sha256"].items():
        require(Path(name).name == name, "Unsafe split manifest name")
        require(digest((split_dir / name).read_bytes()) == expected, f"Source split manifest hash mismatch: {name}")
    rows, all_rows = {}, []
    for split in SPLITS:
        rows[split] = read_jsonl(split_dir / f"{split}.sources.jsonl")
        require(len(rows[split]) == sc["expected_counts"][split], f"Unexpected source count: {split}")
        for row in rows[split]:
            raw = original.get(row.get("source_image_id"))
            require(raw is not None and all(row.get(k) == v for k, v in raw.items()), "Source metadata changed or excluded row entered split")
            require(row.get("split") == split and row.get("split_group") in sl["assignments"][split], "Source split/group mismatch")
            require(row.get("source_split_version") == sl["split_version"] and isinstance(row.get("duplicate_component_id"), str), "Missing split/component provenance")
        all_rows.extend(rows[split])
    require(len({r["source_image_id"] for r in all_rows}) == len(eligible) == len(all_rows), "Source split accounting/IDs mismatch")
    require(len(read_jsonl(split_dir / "excluded.sources.jsonl")) == len(excluded), "Excluded accounting mismatch")
    intersections = intersection_report(all_rows)
    require(not any(values for per_field in intersections.values() for values in per_field.values()), "Cross-split source/hash/group/camera/plot/component leakage")
    sr, pr = read_json(split_dir / "split_report.json"), read_json(policy_dir / "policy_report.json")
    require(sr["status"] == pr["status"] == "pass" and sr["lock_sha256"] == g["split_lock_sha256"] and pr["policy_lock_sha256"] == g["policy_lock_sha256"], "Prior step audit not PASS or lock mismatch")
    policy = load_policy(relative_file(base, g["policy_config"]))
    registry = load_registry(relative_file(base, g["language_config"]))
    require(policy.config_sha256 == pl["policy_config_sha256"] and policy.semantic_sha256 == pl["policy_semantic_sha256"], "Policy config/semantic hash mismatch")
    require(registry.config_sha256 == pl["language_config_sha256"] and registry.semantic_sha256 == pl["language_semantic_sha256"], "Language config/semantic hash mismatch")
    for name, expected in pl["artifacts_sha256"].items():
        require(Path(name).name == name and digest((policy_dir / name).read_bytes()) == expected, "Policy audit artifact hash mismatch")
    config = config | {"_near_policy": sc["near_duplicates"]}
    inputs = Inputs(config, rows, sl, pl, policy, registry, read_json(relative_file(base, g["schema"])))
    # Reuse step-2 decoding/checksum/dHash checks without rerunning its O(n²) split audit.
    for row in all_rows:
        inputs.features[row["source_image_id"]] = image_features(row, base, tuple(sc["near_duplicates"]["thumbnail_size"]))
    return inputs
