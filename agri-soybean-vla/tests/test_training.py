"""Data-only preparation contracts and independent evaluator fixtures."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agri_vla.dataset import DatasetError
from agri_vla.evaluation import metrics, quantize_action, model_batch
from agri_vla.training import load_training_config, prepare_training, prompt_text
from agri_vla.preflight import project_root


def test_quantization_half_up_and_explicit_clipping():
    r = quantize_action([-.2, .5, 3.8])
    assert r["continuous_action"] == [-.2, .5, 3.8]
    assert r["clipped_action"] == [0., .5, 3.]
    assert r["quantized_action"] == [0, 1, 3]
    assert r["out_of_range"] == [True, False, True]
    assert quantize_action([1.499,1.5,2.5])["quantized_action"] == [1,2,3]


@pytest.mark.parametrize("values", [[0,1], [0,1,2,3], [0,float("nan"),0], [float("inf"),0,0], [0,True,0], [0,"1",0]])
def test_reject_invalid_prediction(values):
    with pytest.raises(DatasetError):
        quantize_action(values)


def literal_predictions():
    # Expected labels and predictions are literal, independent of the wilt policy.
    return [dict(sample_id="a", language="vi", intent="AUTO", wording_partition="seen", target_action=[0,1,3], **quantize_action([0.,1.,3.])),
            dict(sample_id="b", language="en", intent="STOP_ALL", wording_partition="unseen", target_action=[0,0,0], **quantize_action([.5,-.1,.1]))]


def test_metrics_independent_fixture():
    r = metrics(literal_predictions())
    assert r["overall"]["zone_accuracy"] == 5/6
    assert r["overall"]["exact_action_accuracy"] == .5
    assert r["overall"]["continuous_mae"] == pytest.approx(.7/6)
    assert r["overall"]["out_of_range_components"] == 1
    assert r["overall"]["off_target_components"] == 4
    assert r["overall"]["false_watering_components"] == 1
    assert r["overall"]["stop_exact_accuracy"] == 0
    assert r["by_language"]["vi"]["exact_action_accuracy"] == 1
    assert r["by_wording_partition"]["unseen"]["zone_accuracy"] == 2/3


def test_metric_corruption_and_duplicate_ids_rejected():
    rows = literal_predictions()
    rows[0]["quantized_action"] = [3,3,3]
    with pytest.raises(DatasetError):
        metrics(rows)
    rows = literal_predictions()
    rows[1]["sample_id"] = "a"
    with pytest.raises(DatasetError):
        metrics(rows)


def test_model_input_allowlist_drops_target_and_teacher():
    keys = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3", "observation.state", "task"]
    batch = {k:k for k in keys} | {"action":[3,3,3], "intent":"STOP_ALL", "teacher_annotations":{}, "index":7}
    assert model_batch(batch) == {k:k for k in keys}


def test_newline_matches_upstream_contract():
    assert prompt_text("Không tưới vùng 3.") == "Không tưới vùng 3.\n"
    assert prompt_text("Stop.\n") == "Stop.\n"


@pytest.mark.parametrize("mutate", [
    lambda c:c["base"].update(revision="main"),
    lambda c:c["policy"].update(freeze_vision_encoder=False),
    lambda c:c["policy"].update(chunk_size=50),
    lambda c:c["policy"].update(tokenizer_max_length=48),
    lambda c:c["peft"].update(full_training_modules=["vlm"]),
    lambda c:c["training"].update(dataset_eval_split=.1),
    lambda c:c["policy"].update(gradient_accumulation_steps=4),
    lambda c:c.update(dataset_root="../outside"),
])
def test_invalid_training_config_rejected(tmp_path, mutate):
    import yaml
    path = project_root()/"configs/training_v0.yaml"
    c = yaml.safe_load(path.read_text())
    mutate(c)
    target = tmp_path/"invalid.yaml"
    target.write_text(yaml.safe_dump(c))
    with pytest.raises((DatasetError, ValueError)):
        load_training_config(target)


def test_prepare_dry_run_no_ml_no_writes(tmp_path):
    script = '''
import sys
class BlockML:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','lerobot','transformers','peft','numpy','huggingface_hub'}:
   raise AssertionError('ML import in dry-run: '+fullname)
sys.meta_path.insert(0,BlockML())
from pathlib import Path
from agri_vla.training import load_training_config,prepare_training
c=load_training_config(sys.argv[1]);r=prepare_training(c,dry_run=True,output=Path(sys.argv[2]))
assert r['optimizer_steps']==0 and not Path(sys.argv[2]).exists()
'''
    result = subprocess.run([sys.executable,"-c",script,str(project_root()/"configs/training_v0.yaml"),str(tmp_path/"not_created")],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_prepare_refuses_existing_output_before_import(tmp_path):
    c = load_training_config(project_root()/"configs/training_v0.yaml")
    with pytest.raises(DatasetError,match="exists"):
        prepare_training(c,dry_run=True,output=tmp_path)


def test_checkpoint_contract_addition_preserves_upstream_files(tmp_path):
    from agri_vla.training import attach_checkpoint_contract
    bundle=tmp_path/"bundle";folder=tmp_path/"checkpoint/pretrained_model"
    folder.mkdir(parents=True);(bundle/"tokenizer").mkdir(parents=True)
    for name in ("config.json","adapter_config.json","adapter_model.safetensors","policy_preprocessor.json","policy_postprocessor.json"):
        (folder/name).write_text("upstream")
    for name in ("provenance.json","schema.json","train_stats.json"):
        (bundle/name).write_text("{}")
    (bundle/"tokenizer/tokenizer.json").write_text("{}")
    attach_checkpoint_contract(bundle,folder)
    assert (folder/"adapter_model.safetensors").read_text()=="upstream"
    assert (folder/"agri_train_stats.json").read_text()=="{}"
    assert (folder/"tokenizer/tokenizer.json").read_text()=="{}"
    lock=json.loads((folder/"agri_checkpoint_lock.json").read_text())
    assert "adapter_model.safetensors" in lock["files"]
    with pytest.raises(DatasetError,match="exists"):
        attach_checkpoint_contract(bundle,folder)


def test_stable_amp_runtime_is_exposed_by_dry_run(tmp_path, monkeypatch):
    import agri_vla.training as training

    c = load_training_config(project_root()/"configs/training_v0.yaml")
    monkeypatch.setattr("agri_vla.evaluation.check_bundle", lambda *_: None)
    monkeypatch.setattr(training, "verify_train_export", lambda *_: {"status": "pass"})
    monkeypatch.setattr(training, "runtime", lambda *_, **__: None)
    monkeypatch.setattr(training, "train_config", lambda *_, **__: type("Cfg", (), {"to_dict": lambda self: {}})())
    result = training.launch_training(c, tmp_path/"bundle", output=tmp_path/"new-output")
    assert result["amp_runtime"] == {
        "mixed_precision": "fp16",
        "init_scale": 1024.0,
        "growth_interval": 2000,
        "nonfinite_guard": True,
    }
    assert not (tmp_path/"new-output").exists()


def test_tiny_overfit_selection_is_bilingual_and_intent_balanced(tmp_path, monkeypatch):
    import agri_vla.training as training

    rows = []
    intents = ("AUTO", "CAP_LEVEL", "EXCLUDE_ZONE", "STOP_ALL", "WATER_ALL", "WORST_ONLY", "ZONE_ONLY")
    episode = 0
    for intent_index, intent in enumerate(intents):
        for case_index in range(4):
            case = f"{intent}-{case_index}"
            target = [case_index % 4, intent_index % 4, (case_index + intent_index) % 4]
            for language in ("vi", "en"):
                rows.append({"episode_index": episode, "source_record": {
                    "intent": intent, "case_id": case, "language": language,
                    "target_action": target, "scene_id": f"scene-{case_index}",
                    "sample_id": f"{case}-{language}",
                }})
                episode += 1
    monkeypatch.setattr(training, "read_jsonl", lambda _: rows)
    result = training.tiny_overfit_selection({"_base": tmp_path, "dataset_root": "unused", "seed": 42}, 56)
    assert len(result["episodes"]) == len(set(result["episodes"])) == 56
    assert result["intent_records"] == {intent: 8 for intent in sorted(intents)}
    assert {row["language"] for row in result["rows"]} == {"vi", "en"}
    with pytest.raises(DatasetError, match="even number"):
        training.tiny_overfit_selection({"_base": tmp_path, "dataset_root": "unused", "seed": 42}, 55)


def test_fixed_validation_selection_reuses_complete_translation_pairs(tmp_path, monkeypatch):
    import agri_vla.evaluation as evaluation

    rows = []
    episode = 0
    for intent in ("AUTO", "CAP_LEVEL", "EXCLUDE_ZONE", "STOP_ALL", "WATER_ALL", "WORST_ONLY", "ZONE_ONLY"):
        for wording in ("seen", "unseen"):
            for pair_index in range(2):
                pair_id = f"{intent}-{wording}-{pair_index}"
                for language in ("vi", "en"):
                    rows.append({"episode_index": episode, "source_record": {
                        "translation_pair_id": pair_id, "wording_partition": wording,
                        "language": language, "intent": intent,
                        "target_action": [pair_index, pair_index, pair_index],
                        "sample_id": f"{pair_id}-{language}",
                    }})
                    episode += 1
    monkeypatch.setattr(evaluation, "read_jsonl", lambda _: rows)
    c = {"_base": tmp_path, "dataset_root": "unused", "seed": 42}
    first = evaluation.fixed_validation_selection(c, 56)
    second = evaluation.fixed_validation_selection(c, 56)
    assert first == second and len(first["episodes"]) == 56
    selected = [rows[i]["source_record"] for i in first["episodes"]]
    pairs = {}
    for row in selected:
        pairs.setdefault((row["translation_pair_id"], row["wording_partition"]), set()).add(row["language"])
    assert all(languages == {"vi", "en"} for languages in pairs.values())


def test_resume_checkpoint_requires_complete_state_and_contract(tmp_path):
    from agri_vla.training import resume_checkpoint, file_sha

    bundle = tmp_path/"bundle"
    checkpoint = tmp_path/"run/checkpoints/004000"
    model = checkpoint/"pretrained_model"
    state = checkpoint/"training_state"
    model.mkdir(parents=True)
    state.mkdir()
    (bundle).mkdir()
    provenance = {"dataset": "locked"}
    schema = {"action": {"shape": [3]}}
    for root, name, value in ((bundle,"provenance.json",provenance), (bundle,"schema.json",schema),
                              (model,"agri_provenance.json",provenance), (model,"agri_schema.json",schema)):
        (root/name).write_text(json.dumps(value))
    (model/"adapter_model.safetensors").write_bytes(b"adapter")
    (model/"agri_checkpoint_lock.json").write_text(json.dumps({"files": {
        "adapter_model.safetensors": file_sha(model/"adapter_model.safetensors")
    }}))
    (state/"training_step.json").write_text(json.dumps({"step": 4000, "num_processes": 1, "batch_size": 1}))
    for name in ("optimizer_state.safetensors", "optimizer_param_groups.json", "scheduler_state.json", "rng_state.safetensors"):
        (state/name).write_bytes(b"state")
    resolved, info = resume_checkpoint({}, bundle, checkpoint, final_steps=6000)
    assert resolved == checkpoint.resolve() and info["step"] == 4000
    (state/"rng_state.safetensors").unlink()
    with pytest.raises(DatasetError, match="Incomplete resume state"):
        resume_checkpoint({}, bundle, checkpoint, final_steps=6000)


@pytest.mark.parametrize("instruction",["Tưới cả 3 vườn","tưới cả ba vườn!","Water all 3 gardens"])
def test_raw_inference_refuses_ambiguous_level_before_model_import(tmp_path,instruction):
    from agri_vla.evaluation import infer_scene
    c=load_training_config(project_root()/"configs/training_v0.yaml")
    with pytest.raises(DatasetError,match="Ambiguous"):
        infer_scene(c,tmp_path/"unused",images=[tmp_path/str(i) for i in (1,2,3)],instruction=instruction,output=tmp_path/"prediction.json")
    assert not (tmp_path/"prediction.json").exists()


def test_stats_source_byte_pin_and_copy_semantics_are_both_checked(tmp_path):
    from agri_vla.evaluation import check_bundle
    from agri_vla.training import BASE,BACKBONE,file_sha
    bundle=tmp_path/"bundle";bundle.mkdir()
    data=tmp_path/"full";(data/"train/meta").mkdir(parents=True)
    stats={"action":{"mean":[1.,2.,3.],"std":[1.,1.,1.]}}
    (data/"train_stats.json").write_text(json.dumps(stats,separators=(',',':')))
    (bundle/"train_stats.json").write_text(json.dumps(stats,indent=2))
    (bundle/"schema.json").write_text('{}');(data/"train/meta/info.json").write_text('{"features":{}}')
    p={"base":BASE,"backbone":BACKBONE,"training_config_sha256":"fixture","train_stats_sha256":file_sha(data/"train_stats.json")}
    (bundle/"provenance.json").write_text(json.dumps(p))
    (bundle/"bundle_lock.json").write_text(json.dumps({"files":{"train_stats.json":file_sha(bundle/"train_stats.json")}}))
    c={"_base":tmp_path,"dataset_root":"full","_sha256":"fixture"}
    assert check_bundle(c,bundle)==p
    (data/"train_stats.json").write_text(json.dumps(stats,indent=1))
    with pytest.raises(DatasetError,match="normalization stats"):
        check_bundle(c,bundle)


def test_train_launcher_hash_gate_never_needs_heldout_roots(tmp_path):
    from agri_vla.training import verify_train_export,file_sha
    root=tmp_path/"full/train";root.mkdir(parents=True)
    bundle=tmp_path/"bundle";bundle.mkdir()
    (root/"data.parquet").write_bytes(b"fixture")
    (root/"export_info.json").write_text(json.dumps({"source_split":"train","profile":"full","records":3}))
    (root/"export_lock.json").write_text(json.dumps({"files_sha256":{"data.parquet":file_sha(root/"data.parquet"),"export_info.json":file_sha(root/"export_info.json")}}))
    (bundle/"input_verification.json").write_text(json.dumps({"splits":{"train":{"status":"pass","export_lock_sha256":file_sha(root/"export_lock.json"),"records_checked":3}}}))
    c={"_base":tmp_path,"dataset_root":"full"}
    assert verify_train_export(c,bundle)["records"]==3
    assert not (tmp_path/"full/val").exists() and not (tmp_path/"full/test").exists()
    (root/"data.parquet").write_bytes(b"changed")
    with pytest.raises(DatasetError,match="Train file changed"):
        verify_train_export(c,bundle)
