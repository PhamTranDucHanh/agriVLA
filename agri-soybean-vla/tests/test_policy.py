from copy import deepcopy
import itertools
from pathlib import Path
import random

import pytest
import yaml

from agri_vla.policy import Intent, PolicyError, Request, all_requests, load_policy, reference_action, validate_request
from agri_vla.preflight import project_root

# Test-side specification: index masks + a literal score table, independent of
# the production branch implementation and its WILT_TO_LEVEL constant.
SCORE_TRUTH = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
REQUESTS = [("AUTO", {}), ("WORST_ONLY", {})] + [("ZONE_ONLY", {"zone": z}) for z in (1, 2, 3)] + \
    [("CAP_LEVEL", {"cap": c}) for c in (1, 2)] + [("EXCLUDE_ZONE", {"zone": z}) for z in (1, 2, 3)] + [("STOP_ALL", {})]


def specification(scores, intent, params):
    allowed = set(range(3))
    if intent == "STOP_ALL":
        allowed.clear()
    elif intent == "ZONE_ONLY":
        allowed = {params["zone"] - 1}
    elif intent == "EXCLUDE_ZONE":
        allowed.remove(params["zone"] - 1)
    elif intent == "WORST_ONLY":
        allowed = {i for i, score in enumerate(scores) if score >= 2 and not any(other > score for other in scores)}
    ceiling = params["cap"] if intent == "CAP_LEVEL" else 3
    return [min(SCORE_TRUTH[score], ceiling) if i in allowed else 0 for i, score in enumerate(scores)]


@pytest.mark.parametrize("scores", list(itertools.product(range(5), repeat=3)))
def test_all_125_wilt_vectors_against_independent_specification(scores):
    for intent, params in REQUESTS:
        action = reference_action(scores, intent, params)
        assert action == specification(scores, intent, params), (scores, intent, params)
        assert all(type(value) is int and 0 <= value <= 3 for value in action)


@pytest.mark.parametrize("intent,params,expected", [
    ("AUTO", {}, [0, 1, 3]), ("WORST_ONLY", {}, [0, 0, 3]),
    ("ZONE_ONLY", {"zone": 2}, [0, 1, 0]), ("CAP_LEVEL", {"cap": 1}, [0, 1, 1]),
    ("CAP_LEVEL", {"cap": 2}, [0, 1, 2]), ("EXCLUDE_ZONE", {"zone": 3}, [0, 1, 0]),
    ("STOP_ALL", {}, [0, 0, 0]),
])
def test_published_example_024(intent, params, expected):
    assert reference_action([0, 2, 4], intent, params) == expected


@pytest.mark.parametrize("scores,expected", [
    ([4, 4, 1], [3, 3, 0]), ([2, 2, 2], [1, 1, 1]), ([3, 3, 4], [0, 0, 3]),
    ([1, 1, 0], [0, 0, 0]), ([0, 0, 0], [0, 0, 0]),
])
def test_worst_all_ties_and_below_threshold(scores, expected):
    assert reference_action(scores, "WORST_ONLY") == expected


def test_permutation_equivariance_for_all_cases_with_zone_parameters_remapped():
    for scores in itertools.product(range(5), repeat=3):
        for intent, params in REQUESTS:
            original = reference_action(scores, intent, params)
            for permutation in itertools.permutations(range(3)):
                remapped = dict(params)
                if "zone" in remapped:
                    remapped["zone"] = permutation.index(params["zone"] - 1) + 1
                assert reference_action([scores[i] for i in permutation], intent, remapped) == [original[i] for i in permutation]


def test_auto_monotonicity_and_other_zones_unchanged():
    for scores in itertools.product(range(5), repeat=3):
        before = reference_action(scores, "AUTO")
        for zone in range(3):
            if scores[zone] < 4:
                changed = list(scores)
                changed[zone] += 1
                after = reference_action(changed, "AUTO")
                assert after[zone] >= before[zone]
                assert all(after[i] == before[i] for i in range(3) if i != zone)


@pytest.mark.parametrize("scores", [None, "024", {"a": 0}, [], [0, 2], [0, 1, 2, 3],
                                        [0, 2, 5], [-1, 2, 4], [True, 2, 4], [0.0, 2, 4],
                                        ["0", 2, 4], [float("nan"), 2, 4]])
def test_invalid_wilt_is_rejected_even_for_stop(scores):
    with pytest.raises(PolicyError):
        reference_action(scores, "STOP_ALL")


@pytest.mark.parametrize("intent,params", [
    ("auto", {}), ("AUTO+STOP_ALL", {}), (["AUTO", "STOP_ALL"], {}), (None, {}),
    ("AUTO", {"zone": 1}), ("STOP_ALL", {"cap": 1}), ("WORST_ONLY", {"tie": "first"}),
    ("ZONE_ONLY", {}), ("ZONE_ONLY", {"zone": 0}), ("ZONE_ONLY", {"zone": 4}),
    ("ZONE_ONLY", {"zone": True}), ("ZONE_ONLY", {"zone": "2"}), ("ZONE_ONLY", {"zone": 2.0}),
    ("ZONE_ONLY", {"zone": 2, "cap": 1}), ("EXCLUDE_ZONE", {"zone": None}),
    ("CAP_LEVEL", {"cap": 0}), ("CAP_LEVEL", {"cap": 3}), ("CAP_LEVEL", {"cap": True}),
    ("CAP_LEVEL", {"cap": 1, "zone": 2}), ("AUTO", []), ("AUTO", {"soil_moisture": 10}),
])
def test_invalid_or_contradictory_requests_are_rejected(intent, params):
    with pytest.raises(PolicyError):
        reference_action([0, 2, 4], intent, params)


def test_no_mutation_or_rng_dependency_and_complete_request_enumeration():
    scores, params = [0, 2, 4], {"zone": 2}
    before = random.getstate()
    action = reference_action(scores, "ZONE_ONLY", params)
    action[1] = 99
    assert reference_action(scores, "ZONE_ONLY", params) == [0, 1, 0]
    assert scores == [0, 2, 4] and params == {"zone": 2}
    assert random.getstate() == before
    assert {(str(r.intent), tuple(r.params.items())) for r in all_requests()} == {(i, tuple(p.items())) for i, p in REQUESTS}
    with pytest.raises(PolicyError):
        Request(Intent.STOP_ALL, zone=2)


def test_load_canonical_policy():
    policy = load_policy(project_root() / "configs/policy_v0.yaml")
    assert policy.action([0, 2, 4], "AUTO") == [0, 1, 3]
    assert len(policy.config_sha256) == len(policy.semantic_sha256) == 64


@pytest.mark.parametrize("extra", ["actionable_wilt_min: 1\n", "actionable_wilt_min: 2\n"])
def test_duplicate_yaml_keys_are_rejected_even_when_values_match(tmp_path, extra):
    path = tmp_path / "duplicate.yaml"
    path.write_bytes((project_root() / "configs/policy_v0.yaml").read_bytes() + extra.encode())
    with pytest.raises(PolicyError, match="Duplicate YAML key"):
        load_policy(path)


@pytest.mark.parametrize("field,value", [
    ("worst_tie_policy", "first_max"), ("actionable_wilt_min", 1), ("multiple_intents", "prefer_stop"),
    ("wilt_to_level", {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}),
    ("wilt_to_level", {0: 0, True: 0, 2: 1, 3: 2, 4: 3}),
    ("config_version", True), ("caps", [1, 2, 3]), ("extra", "silently ignored"),
])
def test_config_cannot_silently_change_v0_semantics(tmp_path, field, value):
    config = yaml.safe_load((project_root() / "configs/policy_v0.yaml").read_text())
    config[field] = value
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(PolicyError, match="contradicts"):
        load_policy(path)
