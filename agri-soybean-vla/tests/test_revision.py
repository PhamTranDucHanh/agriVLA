"""Independent v1 override truth table, domain and language isolation fixtures."""
from copy import deepcopy
import itertools
from pathlib import Path

import pytest

from agri_vla.dataset import DatasetError, read_jsonl
from agri_vla.policy import reference_action, all_requests
from agri_vla.preflight import project_root
from agri_vla.revision import (revision_action,load_revision,extension_templates,generate_revision)


@pytest.mark.parametrize("wilt",list(itertools.product(range(5),repeat=3)))
def test_all_wilt_combinations_keep_proxy_and_honor_explicit_override(wilt):
    for request in all_requests():
        assert revision_action(wilt,str(request.intent),request.params) == reference_action(wilt,str(request.intent),request.params)
    for level,expected in ((1,[1,1,1]),(2,[2,2,2]),(3,[3,3,3])):
        assert revision_action(wilt,"WATER_ALL",{"level":level}) == expected
        for permutation in itertools.permutations(range(3)):
            assert revision_action([wilt[i] for i in permutation],"WATER_ALL",{"level":level}) == expected


@pytest.mark.parametrize("params",[{}, {"level":0},{"level":4},{"level":True},{"level":1.},{"cap":2},{"zone":1},{"level":1,"zone":2},None])
def test_force_does_not_guess_missing_level(params):
    with pytest.raises((DatasetError,ValueError)):
        revision_action([0,0,0],"WATER_ALL",params)


def test_explicit_examples_are_not_threshold_gated():
    assert revision_action([0,0,0],"WATER_ALL",{"level":2}) == [2,2,2]
    assert revision_action([0,0,0],"AUTO",{}) == [0,0,0]
    assert revision_action([4,4,4],"STOP_ALL",{}) == [0,0,0]


def test_casual_bilingual_families_and_new_pool_isolation():
    from agri_vla.language import load_registry
    c=load_revision(project_root()/"configs/revision_v1.yaml")
    registry=load_registry(c["_base"]/c["language_config"])
    force=extension_templates(c)
    assert registry.render_pair("stop_train_01",{},split="train")[0]["instruction"] == "Tắt hết."
    seen=set()
    for part,pool in force["templates"].items():
        for f in pool:
            assert f["id"] not in seen
            seen.add(f["id"])
            for level in (1,2,3):
                assert str(level) in f["vi"].format(level=level) and str(level) in f["en"].format(level=level)


def test_revision_generation_preserves_parent_triplets_and_is_reproducible():
    c=load_revision(project_root()/"configs/revision_v1.yaml")
    c=deepcopy(c)
    c["force_cases_per_level"]={s:2 for s in ("train","val","test")}
    # Literal parent fixtures: enough fields for generation, no raw data dependency.
    parents={}
    for split in ("train","val","test"):
        parents[split]=[]
        for i in range(2):
            for lang in ("vi","en"):
                parents[split].append({"sample_id":f"{split}{i}_{lang}","scene_id":f"{split}{i}","template_family_id":"stop_train_01", "intent_params":{},"language":lang,"teacher_annotations":{"wilt_scores":[i,i,i]},"target_action":[0,0,0],"images":{f"zone{z}":f"{split}{i}-{z}.jpg" for z in (1,2,3)},"source_image_ids":[f"{split}{i}-{z}" for z in (1,2,3)],"intent":"STOP_ALL"})
    a=generate_revision(c,parents);b=generate_revision(c,parents)
    assert a==b
    for split,rows in a.items():
        old={r["sample_id"]:r for r in parents[split]}
        for r in rows:
            parent=old[r["revision_metadata"]["parent_sample_id"]]
            assert r["images"]==parent["images"] and r["source_image_ids"]==parent["source_image_ids"]
            if r["intent"]!="WATER_ALL":
                assert r["target_action"]==parent["target_action"]
