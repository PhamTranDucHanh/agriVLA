"""Thin pinned-upstream model preparation/launcher. No optimizer in preparation.

Imports at module scope are data-only. Never call conversion.train_runtime here:
that CPU-only helper intentionally masks CUDA, which a GPU launcher must preserve.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .dataset import require, new_output, read_json, read_jsonl
from .policy import load_unique_yaml
from .preflight import project_root, LEROBOT_COMMIT
from .split import digest, relative_file

BASE = {"repo_id": "lerobot/smolvla_base", "revision": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"}
BACKBONE = {"repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467"}
CAMERAS = tuple(f"observation.images.camera{i}" for i in (1, 2, 3))
INPUT_LOCK = "c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_training_config(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    c = load_unique_yaml(raw, context="training")
    require(set(c) == {"version", "base_dir", "conversion_config", "dataset_root", "bundle", "base", "backbone", "seed", "policy", "peft", "training"}, "Unknown/missing training fields")
    require(c["version"] == "agri_soybean_training_v0" and c["base"] == BASE and c["backbone"] == BACKBONE, "Model revisions/version must be pinned")
    require(type(c["seed"]) is int and c["seed"] >= 0, "Invalid seed")
    require(isinstance(c["base_dir"], str) and not Path(c["base_dir"]).is_absolute(), "base_dir must be relative")
    base = (path.parent / c["base_dir"]).resolve()
    for key in ("conversion_config", "dataset_root", "bundle"):
        relative_file(base, c[key])
    p, t = c["policy"], c["training"]
    required_policy = {"chunk_size", "n_action_steps", "resize_imgs_with_padding", "tokenizer_max_length", "freeze_vision_encoder", "train_expert_only", "use_amp", "compile_model", "optimizer_lr", "scheduler_warmup_steps", "scheduler_decay_steps", "scheduler_decay_lr"}
    require(set(p) == required_policy, "Unknown/missing policy options")
    require(p["chunk_size"] == p["n_action_steps"] == 1 and type(p["chunk_size"]) is int and type(p["n_action_steps"]) is int, "Only singleton horizon supported")
    require(p["resize_imgs_with_padding"] == [256, 256], "Expected 256 model resize")
    require(p["freeze_vision_encoder"] is True and p["train_expert_only"] is True and p["compile_model"] is False and p["use_amp"] is True, "Frozen VLM/vision, AMP and no compile required")
    require(type(p["tokenizer_max_length"]) is int and 64 <= p["tokenizer_max_length"] <= 256, "Invalid tokenizer length")
    require(p["optimizer_lr"] == 3e-4 and p["scheduler_warmup_steps"] == 100 and p["scheduler_decay_steps"] == 6000 and p["scheduler_decay_lr"] == 3e-5, "Unexpected v0 LR/schedule")
    require(c["peft"] == {"method_type": "LORA", "r": 8, "lora_alpha": 8, "full_training_modules": []}, "Fresh r=8/alpha=8 adapters only")
    require(t == {"batch_size": 1, "num_workers": 0, "steps": 6000, "save_freq": 500, "log_freq": 10, "eval_steps": 0, "env_eval_freq": 0, "dataset_eval_split": 0.0}, "Unexpected training options / eval split")
    return c | {"_base": base, "_path": path, "_sha256": digest(raw)}


def runtime(c, *, offline=True, device="cpu"):
    base = c["_base"]
    require(Path(sys.prefix).resolve() == (base / ".venv-train").resolve(), "Use module .venv-train")
    vendor = base / ".vendor/lerobot"
    require(subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip() == LEROBOT_COMMIT, "Wrong LeRobot commit")
    require(not subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True).strip(), "Vendor must be clean")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = str(base / ".cache/hf-datasets")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import lerobot
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    require(Path(lerobot.__file__).resolve().is_relative_to(vendor), "Wrong imported LeRobot")
    require(device in ("cpu", "cuda"), "device must be cpu/cuda")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA GPU required; no CPU training fallback")
    return torch


def asset_path(c, spec):
    return c["_base"] / ".cache/models" / spec["repo_id"] / spec["revision"]


def assets(c, *, download=False):
    from huggingface_hub import snapshot_download
    lock_path = c["_base"] / ".cache/models/assets_lock.json"
    if download:
        require(os.environ.get("HF_HUB_OFFLINE") != "1", "Start download in a fresh online process")
        for spec in (BASE, BACKBONE):
            snapshot_download(**spec, local_dir=asset_path(c, spec), allow_patterns=["*.json", "*.safetensors", "*.txt"], ignore_patterns=["onnx/*"], max_workers=2)
    entries = {}
    for spec in (BASE, BACKBONE):
        root = asset_path(c, spec)
        require((root / "model.safetensors").is_file() and (root / "config.json").is_file(), f"Missing pinned model files: {root}; use --download-models")
        entries[spec["repo_id"]] = {"revision": spec["revision"], "files": {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.iterdir()) if p.is_file()}}
    lock = {"version": 1, "models": entries}
    if lock_path.exists():
        require(read_json(lock_path) == lock, "Model asset checksum/revision changed")
    else:
        write_json(lock_path, lock)
    return lock


def validate_inputs(c):
    """Re-run full step-5 verifier in an isolated CPU subprocess, not GPU process."""
    root = c["_base"] / c["dataset_root"]
    report = subprocess.run([sys.executable, "-m", "agri_vla.cli", "verify-lerobot", "--config", str(c["_base"] / c["conversion_config"]), "--root", str(root)], capture_output=True, text=True)
    require(report.returncode == 0, f"Step5 revalidation failed: {report.stderr[-3000:]}")
    result = json.loads(report.stdout)
    require(result["status"] == "pass", "Step5 must pass")
    return result


def prompt_text(text):
    # Mirrors the actual upstream NewLineTaskProcessorStep.
    return text if text.endswith("\n") else text + "\n"


def audit_tokens(c, tokenizer):
    import numpy as np
    groups = defaultdict(list)
    failures = []
    for split in ("train", "val", "test"):
        path = c["_base"] / c["dataset_root"] / split / "sample_metadata.jsonl"
        for item in read_jsonl(path):
            row = item["source_record"]
            text = prompt_text(row["instruction"])
            encoded = tokenizer(text, truncation=False)["input_ids"]
            limited = tokenizer(text, max_length=c["policy"]["tokenizer_max_length"], truncation=True)["input_ids"]
            require(tokenizer.decode(encoded, skip_special_tokens=True) == text, f"Tokenizer Unicode round-trip changed text: {row['sample_id']}")
            groups[f"{split}/{row['language']}"].append(len(encoded))
            if encoded != limited:
                failures.append(row["sample_id"])
    require(not failures, f"Token truncation: {failures[:10]}; raise max length and prepare NEW bundle")
    return {"status": "pass", "records": sum(map(len, groups.values())), "max_length": c["policy"]["tokenizer_max_length"], "truncated": len(failures), "unicode_roundtrip": True,
            "groups": {k: {"count": len(v), "min": min(v), "max": max(v), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in groups.items()}}


def audit_saved_tokenizer_step(c, step):
    checked, cache = 0, set()
    for split in ("train","val","test"):
        for item in read_jsonl(c["_base"]/c["dataset_root"]/split/"sample_metadata.jsonl"):
            text = prompt_text(item["source_record"]["instruction"])
            if text not in cache:
                result = step._tokenize_text([text])
                actual = result["input_ids"][0][result["attention_mask"][0].bool()].tolist()
                expected = step.input_tokenizer(text,truncation=False)["input_ids"]
                require(actual == expected, "Saved tokenizer processor truncates/changes tokens")
                cache.add(text)
            checked += 1
    return {"records":checked,"unique_prompts":len(cache),"saved_processor_token_sequences_exact":True}


def parameter_report(policy):
    rows = [{"name": n, "shape": list(p.shape), "count": p.numel(), "dtype":str(p.dtype)} for n, p in policy.named_parameters() if p.requires_grad]
    require(rows, "No trainable parameters")
    for row in rows:
        n = row["name"]
        require((".lora_A." in n or ".lora_B." in n) and (".lm_expert." in n or re.search(r"\.model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)\.", n)), f"Unexpected trainable parameter: {n}")
        require(".vlm." not in n, f"VLM trainable: {n}")
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(r["count"] for r in rows)
    require(trainable / total < .02, "Unexpected full finetuning")
    return {"total": total, "trainable": trainable, "percent": 100 * trainable / total, "parameters": rows, "vlm_frozen": True, "only_lora": True}


def make_policy_config(c, *, device="cpu"):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs import NormalizationMode
    root = c["_base"] / c["dataset_root"] / "train"
    features = dataset_to_policy_features(read_json(root / "meta/info.json")["features"])
    cfg = SmolVLAConfig.from_pretrained(asset_path(c, BASE))
    require(cfg.max_state_dim == cfg.max_action_dim == 32, "Base padding dimensions changed")
    cfg.input_features = {k: features[k] for k in (*CAMERAS, "observation.state")}
    cfg.output_features = {"action": features["action"]}
    for k, v in c["policy"].items():
        require(hasattr(cfg, k), f"Unsupported policy option: {k}")
        setattr(cfg, k, tuple(v) if k == "resize_imgs_with_padding" else v)
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.MEAN_STD}
    cfg.device, cfg.use_amp = device, device == "cuda"
    cfg.pretrained_path = c["_base"] / c["bundle"] / "base_policy"
    cfg.pretrained_revision = None  # local immutable base mirror, pin is in provenance
    cfg.vlm_model_name = str(asset_path(c, BACKBONE))
    cfg.load_vlm_weights = True
    cfg.use_peft = False
    cfg.push_to_hub = False
    cfg.compile_model = False
    return cfg


def train_config(c, bundle, *, output, steps=None, device="cuda"):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    root = c["_base"] / c["dataset_root"] / "train"
    cfg = SmolVLAConfig.from_pretrained(bundle / "base_policy")
    cfg.device, cfg.use_amp = device, device == "cuda"
    cfg.pretrained_path = bundle / "base_policy"
    cfg.vlm_model_name = str(asset_path(c, BACKBONE))
    cfg.push_to_hub = False
    t = c["training"]
    result = TrainPipelineConfig(dataset=DatasetConfig(repo_id=read_json(root / "export_info.json")["repo_id"], root=str(root), eval_split=0.0, use_imagenet_stats=False, video_backend="pyav"),
             policy=cfg, peft=PeftConfig(**c["peft"]), output_dir=Path(output), job_name="agri_soybean_v0", seed=c["seed"], num_workers=t["num_workers"], batch_size=1,
             steps=t["steps"] if steps is None else steps, save_freq=t["save_freq"], log_freq=t["log_freq"], env_eval_freq=0, eval_steps=0, wandb=WandBConfig(enable=False), persistent_workers=False)
    result.dataset.image_transforms.enable = False
    result.optimizer, result.scheduler = cfg.get_optimizer_preset(), cfg.get_scheduler_preset()
    return result


def load_processors(c, folder, cfg, *, device="cpu", tokenizer_dir=None):
    from lerobot.policies.factory import make_pre_post_processors
    # Rebase saved tokenizer location when loading a copied checkpoint on another host.
    return make_pre_post_processors(policy_cfg=cfg, pretrained_path=folder,
        preprocessor_overrides={"device_processor": {"device": device}, "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir or folder.parent / "tokenizer")}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})


def prepare_training(c, *, download=False, dry_run=False, output=None):
    bundle = Path(output).resolve() if output else c["_base"] / c["bundle"]
    new_output(bundle)
    if dry_run:
        return {"status": "dry_run", "bundle": str(bundle), "base": BASE, "backbone": BACKBONE, "policy": c["policy"], "peft": c["peft"], "training": c["training"], "optimizer_steps": 0, "gpu_fit_verified": False}
    validated = validate_inputs(c)
    torch = runtime(c, offline=not download)
    asset_lock = assets(c, download=download)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    torch.manual_seed(c["seed"])
    tokenizer = AutoTokenizer.from_pretrained(asset_path(c, BACKBONE), local_files_only=True)
    token_report = audit_tokens(c, tokenizer)
    cfg = make_policy_config(c)
    cfg.pretrained_path = bundle / "base_policy"
    policy = SmolVLAPolicy.from_pretrained(asset_path(c, BASE), config=cfg, strict=True, local_files_only=True)
    stats = read_json(c["_base"] / c["dataset_root"] / "train_stats.json")
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    bundle.mkdir(parents=True)
    write_json(bundle / "input_verification.json", validated)
    tokenizer.save_pretrained(bundle / "tokenizer")
    next(s for s in pre.steps if type(s).__name__ == "TokenizerProcessorStep").tokenizer_name = str(bundle / "tokenizer")
    base_policy = bundle / "base_policy"
    cfg.save_pretrained(base_policy)
    (base_policy / "model.safetensors").symlink_to(os.path.relpath(asset_path(c, BASE) / "model.safetensors", base_policy))
    pre.save_pretrained(base_policy)
    post.save_pretrained(base_policy)
    policy = policy.wrap_with_peft(peft_cli_overrides=c["peft"])
    params = parameter_report(policy)
    require(all(not p.detach().any() for name,p in policy.named_parameters() if ".lora_B." in name), "Fresh LoRA B must initialize to zero")
    policy.save_pretrained(bundle / "initial_adapter")
    policy.config.save_pretrained(bundle / "initial_adapter")
    # The native trainer wraps a NEW adapter from this base, never consumes this proof adapter.
    conversion_pin = load_unique_yaml((c["_base"]/c["conversion_config"]).read_bytes(), context="conversion pin")["dataset_lock_sha256"]
    provenance = {"version": 1, "base": BASE, "backbone": BACKBONE, "lerobot_commit": LEROBOT_COMMIT, "training_config_sha256": c["_sha256"], "dataset_lock_sha256": conversion_pin,
                  "train_stats_sha256": file_sha(c["_base"] / c["dataset_root"] / "train_stats.json"), "train_info_sha256": file_sha(c["_base"] / c["dataset_root"] / "train/meta/info.json"),
                  "assets_lock_sha256": digest(json.dumps(asset_lock, sort_keys=True).encode()), "optimizer_steps": 0, "gpu_fit_verified": False}
    write_json(bundle / "provenance.json", provenance)
    write_json(bundle / "schema.json", read_json(c["_base"] / c["dataset_root"] / "train/meta/info.json")["features"])
    write_json(bundle / "train_stats.json", stats)
    write_json(bundle / "token_audit.json", token_report)
    write_json(bundle / "trainable_parameters.json", params)
    resolved = train_config(c, bundle, output=c["_base"] / "outputs/train/v0", device="cuda")
    write_json(bundle / "resolved_train_config.json", resolved.to_dict())
    # Re-load SAVED pipelines, not freshly created in-memory config.
    saved_pre, saved_post = load_processors(c, base_policy, cfg)
    tok = next(s for s in saved_pre.steps if type(s).__name__ == "TokenizerProcessorStep")
    require(tok.max_length == token_report["max_length"], "Saved tokenizer length differs")
    require(audit_tokens(c, tok.input_tokenizer) == token_report, "Saved tokenizer audit differs")
    saved_token_audit = audit_saved_tokenizer_step(c,tok)
    write_json(bundle / "preparation_report.json", {"status": "pass", "strict_base_load": True, "fresh_lora_B_zero":True, "token_audit": token_report, "trainable": params["trainable"], "total_parameters": params["total"], "saved_processor_reload": True, "saved_tokenizer_audit":saved_token_audit, "step5_status": validated["status"], "optimizer_steps": 0, "gpu_fit_verified": False})
    write_json(bundle / "bundle_lock.json", {"files": {p.relative_to(bundle).as_posix(): file_sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}})
    return read_json(bundle / "preparation_report.json")


def launch_training(c, bundle, *, output, steps=None, execute=False):
    """Only explicit --execute on CUDA calls the unmodified upstream training loop."""
    from .evaluation import check_bundle
    check_bundle(c, bundle)
    verify_train_export(c,bundle)
    require(steps is None or type(steps) is int and steps > 0, "steps must be positive")
    new_output(output)
    runtime(c, device="cuda" if execute else "cpu")
    cfg = train_config(c, bundle, output=output, steps=steps)
    result = {"status": "ready" if execute else "dry_run", "resolved_config": cfg.to_dict(), "gpu_fit_verified": False}
    if not execute:
        return result
    # Do not let the parser re-read this wrapper's unrelated command-line flags.
    # validate() itself consults argv, so keep argv clean for the entire upstream call.
    from lerobot.scripts import lerobot_train as upstream
    previous = sys.argv
    original_save = upstream.save_checkpoint
    def save_with_contract(*args, **kwargs):
        result = original_save(*args, **kwargs)
        folder = Path(kwargs.get("checkpoint_dir", args[0] if args else ""))/"pretrained_model"
        attach_checkpoint_contract(bundle, folder)
        return result
    try:
        sys.argv = ["agri-vla-upstream"]
        upstream.save_checkpoint = save_with_contract
        cfg.validate()
        require(cfg.policy.device == "cuda" and cfg.policy.use_amp and cfg.dataset.eval_split == 0., "CUDA/AMP/split configuration drift")
        upstream.train(cfg)
    finally:
        sys.argv = previous
        upstream.save_checkpoint = original_save
    return result


def verify_train_export(c,bundle):
    """Fast byte checks against the all-record round-trip proof; TRAIN root only."""
    root=c["_base"]/c["dataset_root"]/"train"
    proof=read_json(bundle/"input_verification.json")["splits"]["train"]
    require(proof["status"]=="pass" and proof["export_lock_sha256"]==file_sha(root/"export_lock.json"),"Train export differs from prepared round-trip proof")
    lock=read_json(root/"export_lock.json")
    for name,sha in lock["files_sha256"].items():
        require(file_sha(relative_file(root.resolve(),name))==sha,f"Train file changed after preparation: {name}")
    info=read_json(root/"export_info.json")
    require(info["source_split"]=="train" and info["profile"]=="full" and info["records"]==proof["records_checked"],"Trainer requires the exact full TRAIN export")
    return {"status":"pass","records":info["records"],"files_checked":len(lock["files_sha256"])}


def attach_checkpoint_contract(bundle, folder):
    """Add portable inference assets AFTER the upstream checkpoint was saved.

    Does not touch optimizer/scheduler/RNG states or vendor files. The trainer's
    original checkpoint call remains authoritative for resume state.
    """
    import shutil
    for name in ("config.json", "adapter_config.json", "adapter_model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        require((folder/name).is_file(), f"Incomplete upstream checkpoint: {name}")
    for source,target in (("provenance.json","agri_provenance.json"),("schema.json","agri_schema.json"),("train_stats.json","agri_train_stats.json")):
        new_output(folder/target)
        shutil.copyfile(bundle/source,folder/target)
    new_output(folder/"tokenizer")
    shutil.copytree(bundle/"tokenizer",folder/"tokenizer")
    write_json(folder/"agri_checkpoint_lock.json",{"files":{p.relative_to(folder).as_posix():file_sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}})
