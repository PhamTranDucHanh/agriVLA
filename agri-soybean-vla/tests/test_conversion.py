from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
import subprocess
import sys

import pytest
import yaml

from agri_vla.dataset import DatasetError
from agri_vla.lerobot_io import (select_smoke, coverage, load_conversion_config, convert, train_statistics,
                                 features, MODEL_FEATURES, CAMERAS)
from agri_vla.policy import all_requests, reference_action
from agri_vla.preflight import project_root


def synthetic_rows():
    rows = []
    for scores in itertools.product(range(5), repeat=3):
        for request in all_requests():
            pid = str(len(rows) // 2)
            for language in ("vi", "en"):
                rows.append({"sample_id": pid + "_" + language, "translation_pair_id": pid, "split": "train", "language": language,
                             "intent": str(request.intent), "intent_params": request.params,
                             "teacher_annotations": {"wilt_scores": list(scores)},
                             "target_action": reference_action(scores, request.intent, request.params),
                             "generation_metadata": {"kind": "contrastive_base" if scores == (0, 2, 4) else "balanced"}})
    return rows


def test_smoke_selects_complete_pairs_and_every_intent_param_and_zone_action():
    rows = synthetic_rows()
    before = random.getstate()
    selected = select_smoke(rows)
    assert len(selected) == 64 and Counter(r["language"] for r in selected) == {"vi": 32, "en": 32}
    assert len({r["sample_id"] for r in selected}) == 64
    assert {r["intent"] for r in selected} == {str(r.intent) for r in all_requests()}
    assert sum(r["generation_metadata"]["kind"] == "contrastive_base" for r in selected) == 22
    for zone in range(3):
        assert {r["target_action"][zone] for r in selected} == {0, 1, 2, 3}
    assert select_smoke(rows) == selected
    assert select_smoke(rows, seed=43) != selected
    assert before == random.getstate()


@pytest.mark.parametrize("mutation", ["nontrain", "missing_pair", "too_few", "odd_count"])
def test_smoke_refuses_broken_selection(mutation):
    rows = synthetic_rows()
    count = 64
    if mutation == "nontrain":
        rows[0]["split"] = "test"
    elif mutation == "missing_pair":
        rows.pop()
    elif mutation == "too_few":
        rows = rows[:10]
    else:
        count = 63
    with pytest.raises(DatasetError):
        select_smoke(rows, count=count)


@pytest.mark.parametrize("key,value", [("fps", 10), ("fps", True), ("use_videos", True), ("image_size", [512, 512]),
                                     ("image_transform", "stretch"), ("smoke_count", 50), ("seed", -1),
                                     ("lerobot_commit", "main"), ("output_dir", "../outside"), ("extra", True)])
def test_conversion_config_rejects_unsupported_or_ambiguous_settings(tmp_path, key, value):
    config = yaml.safe_load((project_root() / "configs/lerobot_v0.yaml").read_text())
    config[key] = value
    path = tmp_path / "conversion.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_conversion_config(path)


def test_dry_run_works_without_ml_imports_and_writes(tmp_path):
    rows = synthetic_rows()
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    config = yaml.safe_load((project_root() / "configs/lerobot_v0.yaml").read_text())
    config["base_dir"] = "."
    (tmp_path / "conversion.yaml").write_text(yaml.safe_dump(config))
    before = sorted(tmp_path.rglob("*"))
    code = r'''
import importlib.abc, sys, json
from pathlib import Path
from types import SimpleNamespace
attempts=[]
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','lerobot','transformers','peft','numpy'}:
            attempts.append(fullname)
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockML())
from agri_vla import lerobot_io
from agri_vla.cli import main
base=Path(sys.argv[1]);rows=json.loads((base/'rows.json').read_text())
lerobot_io.scene_inputs=lambda cfg: ({'status':'pass'},SimpleNamespace(config={'_base':base,'paths':{'raw_dir':'raw'}}),{'train':rows})
assert main(['convert','--config',str(base/'conversion.yaml'),'--profile','smoke','--dry-run'])==0
assert not attempts,attempts
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("profile,split", [("full", "val"), ("full", "test"), ("smoke", "train")])
def test_normalization_refuses_val_test_and_smoke_stats(tmp_path, profile, split):
    (tmp_path / "export_info.json").write_text(json.dumps({"profile": profile, "source_split": split}))
    with pytest.raises(DatasetError, match="full TRAIN"):
        train_statistics(tmp_path)


def test_feature_contract_contains_no_teacher_or_identifiers():
    schema = features()
    assert set(schema) == MODEL_FEATURES == {*CAMERAS, "observation.state", "action"}
    assert schema["action"]["shape"] == schema["observation.state"]["shape"] == (3,)
    assert all(schema[key]["dtype"] == "image" and schema[key]["shape"] == (3, 256, 256) for key in CAMERAS)
