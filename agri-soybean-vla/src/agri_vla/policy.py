"""Versioned deterministic teacher policy. No language parsing or model imports."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path

POLICY_VERSION = "wilt_proxy_policy_v0"
WILT_TO_LEVEL = (0, 0, 1, 2, 3)


class PolicyError(ValueError):
    """Request or configuration is outside the supported v0 contract."""


class Intent(StrEnum):
    AUTO = "AUTO"
    WORST_ONLY = "WORST_ONLY"
    ZONE_ONLY = "ZONE_ONLY"
    CAP_LEVEL = "CAP_LEVEL"
    EXCLUDE_ZONE = "EXCLUDE_ZONE"
    STOP_ALL = "STOP_ALL"


def required_parameters(intent: Intent) -> tuple[str, ...]:
    if intent in (Intent.ZONE_ONLY, Intent.EXCLUDE_ZONE):
        return ("zone",)
    if intent == Intent.CAP_LEVEL:
        return ("cap",)
    return ()


@dataclass(frozen=True)
class Request:
    intent: Intent
    zone: int | None = None
    cap: int | None = None

    def __post_init__(self):
        if not isinstance(self.intent, Intent):
            raise PolicyError("Construct requests through validate_request or use an Intent enum")
        present = {key for key in ("zone", "cap") if getattr(self, key) is not None}
        if present != set(required_parameters(self.intent)):
            raise PolicyError(f"Contradictory/missing parameters for {self.intent}")
        if self.zone is not None and (type(self.zone) is not int or self.zone not in (1, 2, 3)):
            raise PolicyError("zone must be integer 1, 2 or 3")
        if self.cap is not None and (type(self.cap) is not int or self.cap not in (1, 2)):
            raise PolicyError("cap must be integer 1 or 2")

    @property
    def params(self) -> dict:
        return {key: getattr(self, key) for key in required_parameters(self.intent)}


def validate_request(intent: str, params: Mapping | None = None) -> Request:
    if not isinstance(intent, str):
        raise PolicyError("intent must be one supported string, not a list/dict/multiple intents")
    try:
        selected = Intent(intent)
    except ValueError as error:
        raise PolicyError(f"Unsupported intent: {intent!r}") from error
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise PolicyError("intent_params must be a mapping")
    required = set(required_parameters(selected))
    if set(params) != required:
        raise PolicyError(f"{selected} requires exactly {sorted(required)}; got keys {[repr(k) for k in params]}")
    if "zone" in params and (type(params["zone"]) is not int or params["zone"] not in (1, 2, 3)):
        raise PolicyError("zone must be an integer 1, 2 or 3 (one-based; booleans/floats are rejected)")
    if "cap" in params and (type(params["cap"]) is not int or params["cap"] not in (1, 2)):
        raise PolicyError("cap must be integer 1 or 2")
    return Request(selected, params.get("zone"), params.get("cap"))


def validate_wilt(wilt_scores) -> tuple[int, int, int]:
    if not isinstance(wilt_scores, (list, tuple)) or len(wilt_scores) != 3:
        raise PolicyError("wilt_scores must be a list/tuple of exactly three integers")
    if any(type(value) is not int or value not in range(5) for value in wilt_scores):
        raise PolicyError("wilt_scores must contain only integers 0–4; label 5, floats and booleans are rejected")
    return tuple(wilt_scores)


def reference_action(wilt_scores, intent: str, params: Mapping | None = None) -> list[int]:
    """Return synthetic levels, never a physically calibrated irrigation quantity."""
    scores = validate_wilt(wilt_scores)
    request = validate_request(intent, params)
    levels = [WILT_TO_LEVEL[score] for score in scores]
    if request.intent == Intent.STOP_ALL:
        return [0, 0, 0]
    if request.intent == Intent.AUTO:
        return levels
    if request.intent == Intent.WORST_ONLY:
        highest = max(scores)
        return [level if score == highest and highest >= 2 else 0 for score, level in zip(scores, levels, strict=True)]
    if request.intent == Intent.ZONE_ONLY:
        return [level if index == request.zone else 0 for index, level in enumerate(levels, 1)]
    if request.intent == Intent.CAP_LEVEL:
        return [min(level, request.cap) for level in levels]
    if request.intent == Intent.EXCLUDE_ZONE:
        return [0 if index == request.zone else level for index, level in enumerate(levels, 1)]
    raise PolicyError(f"No implementation for intent {request.intent}")


def all_requests() -> tuple[Request, ...]:
    result = []
    for intent in Intent:
        choices = ({"zone": zone} for zone in (1, 2, 3)) if "zone" in required_parameters(intent) else (
            ({"cap": cap} for cap in (1, 2)) if intent == Intent.CAP_LEVEL else ({},)
        )
        result.extend(validate_request(intent, params) for params in choices)
    return tuple(result)


def expected_config() -> dict:
    return {
        "config_version": 1, "policy_version": POLICY_VERSION, "label_source": POLICY_VERSION,
        "label_type": "synthetic_policy_action", "wilt_to_level": dict(enumerate(WILT_TO_LEVEL)),
        "action_levels": {0: "OFF", 1: "LOW", 2: "MEDIUM", 3: "HIGH"},
        "actionable_wilt_min": 2, "worst_tie_policy": "all_max", "zones": [1, 2, 3], "caps": [1, 2],
        "intent_parameters": {str(intent): list(required_parameters(intent)) for intent in Intent},
        "multiple_intents": "reject",
    }


def strict_equal(a, b) -> bool:
    """Config equality must not treat True as integer 1 or silently ignore keys."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if {(type(k), k) for k in a} != {(type(k), k) for k in b}:
            return False
        return all(strict_equal(a[key], value) for key, value in b.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@dataclass(frozen=True)
class Policy:
    config_sha256: str
    semantic_sha256: str
    version: str = POLICY_VERSION

    def action(self, wilt_scores, intent: str, params: Mapping | None = None) -> list[int]:
        return reference_action(wilt_scores, intent, params)


def load_policy(path: Path) -> Policy:
    content = path.read_bytes()
    config = load_unique_yaml(content, context="policy")
    expected = expected_config()
    if not strict_equal(config, expected):
        raise PolicyError("Policy config is malformed or contradicts wilt_proxy_policy_v0; semantic changes require a new explicit policy version")
    semantic = json.dumps(expected, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return Policy(hashlib.sha256(content).hexdigest(), hashlib.sha256(semantic).hexdigest())


def load_unique_yaml(content: bytes, *, context: str):
    """Reject ambiguous duplicate keys instead of accepting last-value-wins YAML."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        loader.flatten_mapping(node)
        result = {}
        for key, value in loader.construct_pairs(node, deep=True):
            try:
                if key in result:
                    raise PolicyError(f"Duplicate YAML key in {context}: {key!r}")
                result[key] = value
            except TypeError as error:
                raise PolicyError(f"Unhashable YAML key in {context}") from error
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    try:
        return yaml.load(content, Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise PolicyError(f"Invalid {context} YAML: {error}") from error
