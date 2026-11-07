from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import unicodedata

import pytest
import yaml

from agri_vla.language import load_registry, registry_summary
from agri_vla.policy import PolicyError, all_requests, reference_action
from agri_vla.preflight import project_root

LANGUAGE_CONFIG = project_root() / "configs/language_v0.yaml"


def test_registry_coverage_and_actual_heldout_intersections():
    registry = load_registry(LANGUAGE_CONFIG)
    report = registry_summary(registry)
    assert report["families"] == 60 and report["rendered_instructions"] == 220
    assert report["partitions"]["train"] == {"families": 36, "rendered_by_language": {"vi": 66, "en": 66}}
    for partition in ("unseen-val", "unseen-test"):
        assert report["partitions"][partition] == {"families": 12, "rendered_by_language": {"vi": 22, "en": 22}}
    assert all(not v for v in report["family_intersections"].values())
    assert all(not v for language in report["normalized_wording_intersections"].values() for v in language.values())


def test_bilingual_pairs_have_equivalent_metadata_and_no_teacher_slots():
    registry = load_registry(LANGUAGE_CONFIG)
    allowed_keys = set(json.loads((project_root() / "schemas/scene_v0.schema.json").read_text())["properties"])
    for family in registry.families:
        split = "train" if family.partition == "train" else family.partition.removeprefix("unseen-")
        for request in all_requests():
            if request.intent != family.intent:
                continue
            vi, en = registry.render_pair(family.family_id, request.params, split=split)
            assert vi["language"] == "vi" and en["language"] == "en"
            for key in ("intent", "intent_params", "template_family_id", "wording_partition", "policy_version"):
                assert vi[key] == en[key]
            assert set(vi) <= allowed_keys and set(en) <= allowed_keys
            assert "target_action" not in vi and "teacher_annotations" not in vi and "translation_pair_id" not in vi
            assert "{" not in vi["instruction"] and "{" not in en["instruction"]
            assert reference_action([0, 2, 4], vi["intent"], vi["intent_params"]) == reference_action([0, 2, 4], en["intent"], en["intent_params"])


@pytest.mark.parametrize("family,split", [("auto_val_01", "train"), ("auto_test_01", "train"),
                                            ("auto_val_01", "test"), ("auto_test_01", "val")])
def test_heldout_families_cannot_be_rendered_into_wrong_split(family, split):
    with pytest.raises(PolicyError):
        load_registry(LANGUAGE_CONFIG).render_pair(family, split=split)


def test_seen_families_allowed_for_unseen_images_and_seed_is_local():
    registry = load_registry(LANGUAGE_CONFIG)
    before = random.getstate()
    for split in ("train", "val", "test"):
        pair = registry.choose_pair("AUTO", split=split, wording="seen", seed=42)
        assert all(entry["wording_partition"] == "seen" for entry in pair)
        assert pair == registry.choose_pair("AUTO", split=split, wording="seen", seed=42)
    assert random.getstate() == before
    with pytest.raises(PolicyError):
        registry.choose_pair("AUTO", split="train", wording="unseen")
    with pytest.raises(PolicyError):
        registry.choose_pair("AUTO", split="test", wording="unseen", seed=True)
    with pytest.raises(PolicyError):
        registry.pool("unknown", split="test", wording="unseen")


@pytest.mark.parametrize("mutation", ["duplicate_family", "duplicate_en", "duplicate_vi_normalized",
    "teacher_slot", "missing_zone", "missing_english", "worst_missing_tie", "missing_threshold",
    "soil_measurement", "format_expression", "wrong_partition", "too_few", "decomposed_unicode"])
def test_invalid_or_leaking_registry_is_rejected(tmp_path, mutation):
    config = yaml.safe_load(LANGUAGE_CONFIG.read_text())
    auto = config["families"]["AUTO"]
    if mutation == "duplicate_family":
        auto["unseen-val"][0]["id"] = auto["train"][0]["id"]
    elif mutation == "duplicate_en":
        auto["unseen-val"][0]["en"] = auto["train"][0]["en"]
    elif mutation == "duplicate_vi_normalized":
        auto["unseen-test"][0]["vi"] = auto["train"][0]["vi"].upper() + "!!!"
    elif mutation == "teacher_slot":
        auto["train"][0]["vi"] = "Tưới theo ngưỡng và điểm {wilt_scores}."
    elif mutation == "missing_zone":
        config["families"]["ZONE_ONLY"]["train"][0]["vi"] = "Chỉ tưới vùng 2 nếu đạt ngưỡng héo."
    elif mutation == "missing_english":
        auto["train"][0].pop("en")
    elif mutation == "worst_missing_tie":
        config["families"]["WORST_ONLY"]["train"][0]["en"] = "Water the most wilted eligible zone."
    elif mutation == "missing_threshold":
        auto["train"][0]["vi"] = "Tưới mọi vùng đang héo."
    elif mutation == "soil_measurement":
        auto["train"][0]["en"] = "Water the driest soil once the threshold is met."
    elif mutation == "format_expression":
        config["families"]["ZONE_ONLY"]["train"][0]["en"] = "Water zone {zone!r} if eligible."
    elif mutation == "wrong_partition":
        auto["unseen_val"] = auto.pop("unseen-val")
    elif mutation == "too_few":
        auto["train"].pop()
    else:
        auto["train"][0]["vi"] = unicodedata.normalize("NFD", auto["train"][0]["vi"])
    path = tmp_path / "language.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(PolicyError):
        load_registry(path)
