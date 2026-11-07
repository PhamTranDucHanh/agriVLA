"""Controlled paired VI/EN templates; rendering never accepts teacher wilt scores."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from string import Formatter
import unicodedata

from .policy import Intent, POLICY_VERSION, PolicyError, all_requests, load_unique_yaml, required_parameters, validate_request

PARTITIONS = ("train", "unseen-val", "unseen-test")
LANGUAGES = ("vi", "en")
MIN_FAMILIES = {"train": 6, "unseen-val": 2, "unseen-test": 2}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def check_text(text: str, intent: Intent, language: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise PolicyError("Templates must be nonempty strings")
    if text != unicodedata.normalize("NFC", text) or "\n" in text or "\r" in text:
        raise PolicyError("Use a single-line NFC-normalized template")
    fields = set()
    try:
        for _, field, format_spec, conversion in Formatter().parse(text):
            if field is not None:
                if format_spec or conversion:
                    raise PolicyError("Template formatting/conversion expressions are forbidden")
                fields.add(field)
    except ValueError as error:
        raise PolicyError(f"Malformed template format: {error}") from error
    if fields != set(required_parameters(intent)):
        raise PolicyError(f"{intent}/{language} must use exactly slots {required_parameters(intent)}; got {fields}")
    lower = text.casefold()
    if any(token in lower for token in ("đất khô", "driest soil", "soil moisture", "độ ẩm đất", "wilt_scores", "target_action")):
        raise PolicyError("Template invents soil measurements or exposes teacher/target fields")
    if any(re.search(r"\b" + str(value).casefold() + r"\b", lower) for value in Intent):
        raise PolicyError("Structured intent names belong in metadata, not model instructions")
    if intent != Intent.STOP_ALL:
        threshold = "ngưỡng" in lower if language == "vi" else bool(re.search(r"\b(threshold|eligible)\b", lower))
        if not threshold:
            raise PolicyError(f"{intent}/{language} must state threshold eligibility (wilt 0–1 stay OFF)")
    if intent == Intent.WORST_ONLY:
        if language == "vi":
            explicit_all = "tất cả" in lower or "toàn bộ" in lower
            explicit_tie = any(term in lower for term in ("hòa", "đồng hạng", "bằng nhau", "cùng đứng đầu"))
        else:
            explicit_all = bool(re.search(r"\b(all|every)\b", lower))
            explicit_tie = bool(re.search(r"\b(tie|ties|tied|equal)\b", lower))
        if not explicit_all or not explicit_tie:
            raise PolicyError("WORST_ONLY must explicitly include all tied zones in both languages")


@dataclass(frozen=True)
class TemplateFamily:
    family_id: str
    intent: Intent
    partition: str
    vi: str
    en: str


@dataclass(frozen=True)
class TemplateRegistry:
    families: tuple[TemplateFamily, ...]
    seed: int
    config_sha256: str
    semantic_sha256: str
    version: str = "bilingual_templates_v0"

    def family(self, family_id: str) -> TemplateFamily:
        for family in self.families:
            if family.family_id == family_id:
                return family
        raise PolicyError(f"Unknown template family: {family_id}")

    def pool(self, intent: str, *, split: str, wording: str) -> tuple[TemplateFamily, ...]:
        if not isinstance(intent, str):
            raise PolicyError("intent must be a supported string")
        try:
            selected = Intent(intent)
        except ValueError as error:
            raise PolicyError(f"Unsupported intent: {intent!r}") from error
        if split not in ("train", "val", "test") or wording not in ("seen", "unseen"):
            raise PolicyError("split must be train/val/test and wording must be seen/unseen")
        if split == "train" and wording != "seen":
            raise PolicyError("Train cannot use held-out template families")
        partition = "train" if wording == "seen" else f"unseen-{split}"
        return tuple(f for f in self.families if f.intent == selected and f.partition == partition)

    def render_pair(self, family_id: str, params=None, *, split: str) -> list[dict]:
        family = self.family(family_id)
        request = validate_request(family.intent, params)
        wording = "seen" if family.partition == "train" else "unseen"
        if family not in self.pool(family.intent, split=split, wording=wording):
            raise PolicyError(f"Family {family_id} belongs to {family.partition}, not {split}/{wording}")
        return [{"instruction": getattr(family, language).format(**request.params), "language": language,
                 "template_id": family.family_id + "_" + language, "template_family_id": family.family_id,
                 "wording_partition": wording, "intent": str(family.intent), "intent_params": request.params,
                 "policy_version": POLICY_VERSION} for language in LANGUAGES]

    def choose_pair(self, intent: str, params=None, *, split: str, wording: str, seed: int | None = None) -> list[dict]:
        request = validate_request(intent, params)
        if seed is None:
            seed = self.seed
        if type(seed) is not int or seed < 0:
            raise PolicyError("Template seed must be a nonnegative integer")
        choices = self.pool(request.intent, split=split, wording=wording)
        family = random.Random(seed).choice(choices)
        return self.render_pair(family.family_id, request.params, split=split)

    def catalog(self) -> list[dict]:
        records = []
        for family in self.families:
            split = "train" if family.partition == "train" else family.partition.removeprefix("unseen-")
            for request in all_requests():
                if request.intent == family.intent:
                    for rendered in self.render_pair(family.family_id, request.params, split=split):
                        records.append(rendered | {"template_partition": family.partition})
        return records


def load_registry(path: Path) -> TemplateRegistry:
    content = path.read_bytes()
    config = load_unique_yaml(content, context="language")
    expected_keys = {"config_version", "registry_version", "policy_version", "seed", "languages", "families"}
    if not isinstance(config, dict) or set(config) != expected_keys or type(config["config_version"]) is not int or config["config_version"] != 1:
        raise PolicyError("Malformed language config or unsupported config_version")
    if config["registry_version"] != "bilingual_templates_v0" or config["policy_version"] != POLICY_VERSION or config["languages"] != list(LANGUAGES):
        raise PolicyError("Language registry must target policy v0 and both vi/en languages")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise PolicyError("Language seed must be a nonnegative integer")
    if not isinstance(config["families"], dict) or set(config["families"]) != {str(i) for i in Intent}:
        raise PolicyError("Template registry must cover all six intents")
    families, identifiers = [], set()
    for intent in Intent:
        pools = config["families"][str(intent)]
        if not isinstance(pools, dict) or set(pools) != set(PARTITIONS):
            raise PolicyError(f"{intent} must have train/unseen-val/unseen-test families")
        for partition in PARTITIONS:
            items = pools[partition]
            if not isinstance(items, list) or len(items) < MIN_FAMILIES[partition]:
                raise PolicyError(f"{intent}/{partition} needs at least {MIN_FAMILIES[partition]} bilingual families")
            for item in items:
                if not isinstance(item, dict) or set(item) != {"id", "vi", "en"}:
                    raise PolicyError("Each family needs exactly id, vi, en")
                family_id = item["id"]
                if not isinstance(family_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", family_id) or family_id in identifiers:
                    raise PolicyError(f"Invalid/duplicate family ID: {family_id}")
                for language in LANGUAGES:
                    check_text(item[language], intent, language)
                identifiers.add(family_id)
                families.append(TemplateFamily(family_id, intent, partition, item["vi"], item["en"]))
    semantic = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    registry = TemplateRegistry(tuple(sorted(families, key=lambda f: f.family_id)), config["seed"],
                                hashlib.sha256(content).hexdigest(), hashlib.sha256(semantic).hexdigest())
    seen_text = {}
    for item in registry.catalog():
        key = (item["language"], normalize_text(item["instruction"]))
        if key in seen_text:
            raise PolicyError(f"Duplicate/held-out wording leakage: {item['template_id']} repeats {seen_text[key]}")
        seen_text[key] = item["template_id"]
    return registry


def registry_summary(registry: TemplateRegistry) -> dict:
    catalog = registry.catalog()
    pairs = (("train", "unseen-val"), ("train", "unseen-test"), ("unseen-val", "unseen-test"))
    families = {partition: {f.family_id for f in registry.families if f.partition == partition} for partition in PARTITIONS}
    texts = {(language, partition): {normalize_text(r["instruction"]) for r in catalog if r["language"] == language and r["template_partition"] == partition}
             for language in LANGUAGES for partition in PARTITIONS}
    return {"families": len(registry.families), "languages": list(LANGUAGES), "rendered_instructions": len(catalog),
            "partitions": {partition: {"families": sum(f.partition == partition for f in registry.families),
                                       "rendered_by_language": dict(Counter(r["language"] for r in catalog if r["template_partition"] == partition))}
                           for partition in PARTITIONS},
            "family_intersections": {a + "__" + b: sorted(families[a] & families[b]) for a, b in pairs},
            "normalized_wording_intersections": {language: {a + "__" + b: sorted(texts[language, a] & texts[language, b]) for a, b in pairs} for language in LANGUAGES}}
