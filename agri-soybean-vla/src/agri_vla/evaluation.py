"""Independent action postprocessing and held-out evaluator; teacher never a model input."""
from collections import defaultdict
import math
from pathlib import Path

from .dataset import require, read_json, read_jsonl, new_output
from .training import (BASE, BACKBONE, CAMERAS, runtime, assets, asset_path, parameter_report,
                       load_processors, file_sha, write_json)


def quantize_action(values):
    """Finite float[3] -> clip[0,3], nearest integer, exact .5 rounds UP."""
    require(len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in values), "Prediction must be finite float[3]")
    continuous = [float(v) for v in values]
    clipped = [min(3., max(0., v)) for v in continuous]
    return {"continuous_action": continuous, "clipped_action": clipped,
            "quantized_action": [math.floor(v + .5) for v in clipped],
            "out_of_range": [not 0. <= v <= 3. for v in continuous],
            "quantization": "clip_0_3_then_floor_x_plus_0.5"}


def metrics(predictions):
    require(predictions and len({r["sample_id"] for r in predictions}) == len(predictions), "Empty/duplicate predictions")
    def score(rows):
        total = len(rows)
        matches = [p == t for r in rows for p, t in zip(r["quantized_action"], r["target_action"], strict=True)]
        abs_error = [abs(p-t) for r in rows for p,t in zip(r["continuous_action"], r["target_action"], strict=True)]
        off = [(p, t) for r in rows for p,t in zip(r["quantized_action"], r["target_action"], strict=True) if t == 0]
        stop = [r for r in rows if r.get("intent") == "STOP_ALL"]
        return {"records": total, "zone_accuracy": sum(matches) / (3*total),
                "exact_action_accuracy": sum(r["quantized_action"] == r["target_action"] for r in rows) / total,
                "continuous_mae": sum(abs_error) / (3*total),
                "out_of_range_components": sum(sum(r["out_of_range"]) for r in rows),
                "off_target_components": len(off), "false_watering_components": sum(p != 0 for p,t in off),
                "stop_records": len(stop), "stop_exact_accuracy": sum(r["quantized_action"] == [0,0,0] for r in stop) / len(stop) if stop else None}
    for row in predictions:
        require(len(row["target_action"]) == 3 and all(type(v) is int and 0 <= v <= 3 for v in row["target_action"]), "Invalid target")
        expected = quantize_action(row["continuous_action"])
        require(all(row[k] == expected[k] for k in ("quantized_action", "out_of_range")), "Prediction postprocess mismatch")
    report = {"overall": score(predictions)}
    for key in ("language", "intent", "wording_partition"):
        groups = defaultdict(list)
        for row in predictions:
            groups[str(row.get(key, "unknown"))].append(row)
        report[f"by_{key}"] = {k: score(rows) for k, rows in sorted(groups.items())}
    return report


def check_bundle(c, bundle):
    provenance = read_json(bundle / "provenance.json")
    require(provenance["base"] == BASE and provenance["backbone"] == BACKBONE, "Bundle model revision differs")
    require(provenance["training_config_sha256"] == c["_sha256"], "Training config changed after preparation")
    source_stats = c["_base"] / c["dataset_root"] / "train_stats.json"
    require(provenance["train_stats_sha256"] == file_sha(source_stats) and read_json(bundle/"train_stats.json") == read_json(source_stats), "Wrong train normalization stats")
    require(read_json(bundle / "schema.json") == read_json(c["_base"] / c["dataset_root"] / "train/meta/info.json")["features"], "Saved model schema differs")
    lock = read_json(bundle / "bundle_lock.json")
    for name, sha in lock["files"].items():
        require(file_sha(bundle / name) == sha, f"Prepared bundle changed: {name}")
    return provenance


def load_model(c, bundle, *, checkpoint=None, device="cpu"):
    torch = runtime(c, device=device)
    assets(c)
    check_bundle(c, bundle)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftModel, PeftConfig
    adapter = Path(checkpoint).resolve() if checkpoint else bundle / "initial_adapter"
    if checkpoint:
        contract = read_json(adapter/"agri_provenance.json")
        require(contract == read_json(bundle/"provenance.json"), "Checkpoint base/dataset/stats contract differs")
        require(read_json(adapter/"agri_schema.json") == read_json(bundle/"schema.json"), "Checkpoint schema differs")
        require(file_sha(adapter/"agri_train_stats.json") == file_sha(bundle/"train_stats.json"), "Checkpoint stats differ")
        for name,sha in read_json(adapter/"agri_checkpoint_lock.json")["files"].items():
            require(file_sha(adapter/name) == sha, f"Checkpoint checksum differs: {name}")
    cfg = SmolVLAConfig.from_pretrained(adapter)
    schema = read_json(bundle / "schema.json")
    require(list(cfg.image_features) == list(CAMERAS) and cfg.action_feature.shape == (3,) and cfg.robot_state_feature.shape == (3,), "Wrong checkpoint cameras/state/action")
    require(cfg.max_state_dim == cfg.max_action_dim == 32 and cfg.chunk_size == cfg.n_action_steps == 1, "Wrong padding/horizon")
    require(cfg.resize_imgs_with_padding == (256,256) and cfg.tokenizer_max_length == c["policy"]["tokenizer_max_length"], "Checkpoint resize/token length differs")
    require({k: str(v.value) for k,v in cfg.normalization_mapping.items()} == {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "MEAN_STD"}, "Wrong normalization mapping")
    cfg.device, cfg.use_amp = device, device == "cuda"
    cfg.vlm_model_name = str(asset_path(c, BACKBONE))
    cfg.pretrained_path = bundle / "base_policy"
    cfg.use_peft = False
    policy = SmolVLAPolicy.from_pretrained(asset_path(c, BASE), config=cfg, local_files_only=True, strict=True)
    adapter_config = PeftConfig.from_pretrained(adapter)
    require(adapter_config.r == adapter_config.lora_alpha == 8 and not adapter_config.modules_to_save and adapter_config.bias == "none", "Unsupported adapter config")
    policy = PeftModel.from_pretrained(policy, adapter, is_trainable=True)
    params = parameter_report(policy)
    processor_folder = adapter if checkpoint else bundle / "base_policy"
    pre, post = load_processors(c, processor_folder, cfg, device=device, tokenizer_dir=adapter/"tokenizer" if checkpoint else bundle / "tokenizer")
    tok = next(s for s in pre.steps if type(s).__name__ == "TokenizerProcessorStep")
    require(tok.max_length == cfg.tokenizer_max_length, "Saved processor/config token length differs")
    # Saved stats must be the full-train stats, never whatever held-out dataset we open later.
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    wanted = read_json(bundle / "train_stats.json")["action"]
    for pipe, kind in ((pre, NormalizerProcessorStep), (post, UnnormalizerProcessorStep)):
        normalizer = next(s for s in pipe.steps if isinstance(s, kind))
        saved_stats = normalizer.state_dict()
        require(all(torch.allclose(saved_stats[f"action.{key}"].cpu(), torch.tensor(wanted[key]), atol=1e-7) for key in ("mean", "std")), "Saved processor has non-train action stats")
        require(str(normalizer.norm_map[next(k for k in normalizer.norm_map if str(k.value) == "STATE")].value) == "IDENTITY" if kind is NormalizerProcessorStep else True, "State must use IDENTITY")
    policy.eval()
    return policy, pre, post, params


def data_loader(c, cfg, split, *, count=None):
    from torch.utils.data import DataLoader, Subset
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.factory import resolve_delta_timestamps
    root = c["_base"] / c["dataset_root"] / split
    info = read_json(root / "export_info.json")
    require(info["source_split"] == split and info["profile"] == "full", "Wrong split root")
    dataset = LeRobotDataset(info["repo_id"], root=root, video_backend="pyav")
    dataset = LeRobotDataset(info["repo_id"], root=root, video_backend="pyav", delta_timestamps=resolve_delta_timestamps(cfg, dataset.meta))
    mapping = read_jsonl(root / "sample_metadata.jsonl")
    indices = range(len(dataset)) if count is None else range(min(count, len(dataset)))
    return DataLoader(Subset(dataset, list(indices)), batch_size=1, shuffle=False, num_workers=0), mapping


def model_batch(batch):
    # Exact allowlist: no labels, structured intents, case, IDs or provenance in inference.
    return {k: batch[k] for k in (*CAMERAS, "observation.state", "task")}


def predict_batch(policy, pre, post, batch, *, seed=42):
    import torch
    policy.reset()  # Every singleton is independent, including its translation pair.
    if hasattr(pre, "reset"):
        pre.reset()
    if hasattr(post, "reset"):
        post.reset()
    torch.manual_seed(seed)
    if policy.config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs = pre(model_batch(batch))
    with torch.inference_mode(), torch.autocast(policy.config.device, enabled=True, dtype=torch.bfloat16 if policy.config.device == "cpu" else torch.float16):
        normalized = policy.select_action(inputs)
    action = post(normalized).float().cpu()
    require(action.shape == (1,3) and torch.isfinite(action).all(), "Inverse-normalized action must be finite [1,3]")
    return {"normalized_action": normalized.float().cpu()[0].tolist(), **quantize_action(action[0].tolist()), "queue_reset": True}


def check_model_inputs(c, bundle, *, output):
    new_output(output)
    policy, pre, post, params = load_model(c, bundle)
    import torch
    from torch.utils.data import DataLoader,Subset
    from lerobot.datasets.factory import make_train_eval_datasets
    from .training import train_config
    candidate = train_config(c,bundle,output=c["_base"]/"outputs/train/input-audit-unused",device="cpu")
    dataset,heldout = make_train_eval_datasets(candidate)
    require(heldout is None and candidate.dataset.eval_split == 0., "Upstream split unexpectedly created eval episodes")
    require(Path(dataset.root).resolve() == (c["_base"]/c["dataset_root"]/"train").resolve(), "Trainer opened wrong split root")
    mapping = read_jsonl(c["_base"]/c["dataset_root"]/"train/sample_metadata.jsonl")
    require(len(dataset) == len(mapping), "Trainer did not retain every train singleton")
    loader=DataLoader(Subset(dataset,[0,1]),batch_size=1,num_workers=0)
    results, encoder_shapes = [], []
    vision = policy.get_base_model().model.vlm_with_expert.get_vlm_model().vision_model
    hook = vision.register_forward_pre_hook(lambda module, args, kwargs: encoder_shapes.append(list(kwargs["pixel_values"].shape)), with_kwargs=True)
    for i, batch in enumerate(loader):
        raw_dtypes = {k:str(batch[k].dtype) for k in CAMERAS}
        # Same conversion used by the pinned upstream loop before its preprocessor.
        for camera in CAMERAS:
            if batch[camera].dtype == torch.uint8:
                batch[camera] = batch[camera].float()/255.
        processed = pre(batch)
        require(processed["action"].shape == (1,1,3) and not processed["action_is_pad"].any(), "Bad action batch/mask")
        require(torch.equal(processed["observation.state"], torch.zeros(1,1,3)), "State identity failed")
        images, masks = policy.prepare_images(processed)
        require(len(images) == 3 and all(im.shape == (1,3,256,256) for im in images), "Wrong actual model resize")
        require(policy.prepare_state(processed).shape == (1,32) and policy.prepare_action(processed).shape == (1,1,32), "Padding must match base")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = policy(processed)
        require(torch.isfinite(loss).all(), "Nonfinite CPU forward loss")
        prediction = predict_batch(policy, pre, post, batch, seed=c["seed"])
        results.append({"sample_id": mapping[i]["sample_id"], "task": batch["task"], "raw_factory_image_dtypes":raw_dtypes, "loss_no_grad": float(loss), "batch_shapes": {k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}, "token_shape": list(processed["observation.language.tokens"].shape), **prediction})
    hook.remove()
    report = {"status": "pass", "device": "cpu", "trainable": params, "examples": results, "encoder_input_shapes": encoder_shapes, "upstream_train_factory":{"root":str(dataset.root),"records":len(dataset),"eval_dataset":None,"eval_split":0.0,"uint8_conversion_matches_trainer":True}, "optimizer_steps": 0, "backward_performed": False, "gpu_fit_verified": False}
    write_json(output, report)
    return report


def evaluate(c, bundle, *, split, output, checkpoint=None, count=None, device="cpu"):
    require(split in ("val", "test"), "Evaluator only accepts held-out val/test roots")
    require(count is None or type(count) is int and count > 0, "Invalid limit")
    new_output(output)
    policy, pre, post, _ = load_model(c, bundle, checkpoint=checkpoint, device=device)
    loader, mapping = data_loader(c, policy.config, split, count=count)
    predictions = []
    for batch in loader:
        index = int(batch["index"].item())
        row = mapping[index]["source_record"]
        result = predict_batch(policy, pre, post, batch, seed=c["seed"])
        predictions.append({"sample_id": row["sample_id"], "episode_index": index, "split": split, "language": row["language"], "intent": row["intent"], "wording_partition": row["wording_partition"], "target_action": row["target_action"], **result})
    output.mkdir(parents=True)
    from .dataset import jsonl_bytes
    (output / "predictions.jsonl").write_bytes(jsonl_bytes(predictions))
    report = {"status": "pass", "split": split, "partial": count is not None, "checkpoint": str(checkpoint or bundle / "initial_adapter"), "training_status": "not_verified" if checkpoint else "untrained_initial_adapter",
              "seed_per_sample": c["seed"], "metrics": metrics(predictions), "normalization_source": "full/train", "base": BASE, "backbone": BACKBONE,
              "predictions_sha256": file_sha(output / "predictions.jsonl"), "dataset_export_lock_sha256": file_sha(c["_base"] / c["dataset_root"] / split / "export_lock.json")}
    write_json(output / "metrics.json", report)
    return report


def infer_scene(c,bundle,*,images,instruction,output,checkpoint=None,device="cpu"):
    new_output(output)
    require(len(images)==3 and isinstance(instruction,str) and bool(instruction.strip()), "Provide three ordered RGB images and a nonempty instruction")
    from .language import normalize_text
    ambiguous={normalize_text(s) for s in ("Tưới cả 3 vườn","Tưới cả ba vườn","Tưới cả 3 vùng","Water all 3 gardens","Water all three gardens")}
    require(normalize_text(instruction) not in ambiguous,"Ambiguous WATER_ALL: specify level 1/2/3; never guess the target")
    policy,pre,post,_=load_model(c,bundle,checkpoint=checkpoint,device=device)
    import torch
    from .training import prompt_text
    tok=next(s for s in pre.steps if type(s).__name__=="TokenizerProcessorStep")
    n=len(tok.input_tokenizer(prompt_text(instruction),truncation=False)["input_ids"])
    require(n<=tok.max_length,f"Instruction has {n} tokens, exceeds saved limit {tok.max_length}; refusing truncation")
    from .lerobot_io import transformed_image
    ordered=[Path(p).resolve() for p in images]
    checksums=[file_sha(p) for p in ordered]
    batch={camera:torch.from_numpy(transformed_image(path,sha)).permute(2,0,1).float()[None,None]/255.
           for camera,path,sha in zip(CAMERAS,ordered,checksums,strict=True)}
    batch.update({"observation.state":torch.zeros(1,1,3,dtype=torch.float32),"task":[instruction]})
    report={"status":"pass","instruction":instruction,"token_length":n,"input_images":[str(p) for p in ordered],"input_sha256":checksums,
            "camera_order":list(CAMERAS),"state_semantics":"assumed_all_valves_initially_closed","base":BASE,"backbone":BACKBONE,
            "training_status":"not_verified" if checkpoint else "untrained_initial_adapter","normalization_source":c["dataset_root"]+"/train",
            **predict_batch(policy,pre,post,batch,seed=c["seed"])}
    write_json(output,report)
    return report
