"""Versioned dataset entrypoint reusing the byte-pinned step5 writer/verifier."""
from pathlib import Path

from .dataset import require, new_output, read_json, SPLITS
from .revision import load_revision, validate_revision
from .split import canonical_bytes
from .training import file_sha


def inputs(config):
    c = load_revision(config["_base"]/config["scene_config"])
    validated = validate_revision(c)
    require(validated[0]["dataset_lock_sha256"] == config["dataset_lock_sha256"], "Revision conversion input lock differs")
    require(config["_base"]/config["manifests"] == c["_base"]/c["output_dir"], "Wrong revision manifests")
    return validated


def verify_root(config, root, validated):
    from .lerobot_io import verify_export
    marker = read_json(root/"revision_conversion_lock.json")
    require(marker == {"implementation_sha256":file_sha(Path(__file__)),"dataset_lock_sha256":config["dataset_lock_sha256"],"export_lock_sha256":file_sha(root/"export_lock.json")}, "Revision writer/export marker changed")
    return verify_export(config,root,validated=validated)


def verify_path(config,root):
    from .lerobot_io import verify_normalization, train_statistics
    validated = inputs(config)
    if (root/"export_info.json").is_file():
        return verify_root(config,root,validated)
    reports = {s:verify_root(config,root/s,validated) for s in SPLITS}
    normal = verify_normalization(root)
    require(read_json(root/"train_stats.json") == train_statistics(root/"train"), "Train stats copy differs")
    require(read_json(root/"normalization_contract.json") == normal, "Normalization contract changed")
    return {"status":"pass","splits":reports,"normalization":normal,"dataset_lock_sha256":config["dataset_lock_sha256"]}


def convert(config,*,profile,output_dir=None,dry_run=False,smoke_root=None):
    from .lerobot_io import (select_smoke,selection_summary,write_dataset,verify_export,verify_normalization,train_statistics,train_runtime)
    validated = inputs(config)
    rows = validated[2]
    base = config["_base"]/config["output_dir"]
    output = output_dir or base/profile
    new_output(output)
    require(not output.resolve().is_relative_to(config["_base"]/"data/raw"), "Output cannot be raw")
    chosen = {"train":select_smoke(rows["train"],count=64,seed=config["seed"])} if profile == "smoke" else rows
    if profile == "smoke":
        force = [r for r in chosen["train"] if r["intent"] == "WATER_ALL"]
        require({r["intent_params"]["level"] for r in force} == {1,2,3}, "64 smoke must cover all 7 intents/force levels")
    result = {"status":"pass","profile":profile,"dry_run":dry_run,"dataset_lock_sha256":config["dataset_lock_sha256"],"selection":{s:selection_summary(r) for s,r in chosen.items()}}
    if dry_run:
        return result
    train_runtime()
    if profile == "full":
        gate = verify_root(config,smoke_root or base/"smoke",validated)
        require(gate["records_checked"] == 64 and gate["profile"] == "smoke", "64 smoke required")
        output.mkdir(parents=True)
    reports = {}
    for split,records in chosen.items():
        root = output if profile == "smoke" else output/split
        write_dataset(config,records,root,split=split,profile=profile)
        report = verify_export(config,root,validated=validated)
        (root/"roundtrip_report.json").write_bytes(canonical_bytes(report))
        (root/"revision_conversion_lock.json").write_bytes(canonical_bytes({"implementation_sha256":file_sha(Path(__file__)),"dataset_lock_sha256":config["dataset_lock_sha256"],"export_lock_sha256":file_sha(root/"export_lock.json")}))
        reports[split] = report
    if profile == "full":
        normal = verify_normalization(output)
        (output/"train_stats.json").write_bytes(canonical_bytes(train_statistics(output/"train")))
        (output/"normalization_contract.json").write_bytes(canonical_bytes(normal))
        result["normalization"] = normal
    result["roundtrip"] = {s:{k:r[k] for k in ("status","records_checked","images_pixel_exact_checked","episodes","frames")} for s,r in reports.items()}
    if profile == "full":
        (output/"roundtrip_report.json").write_bytes(canonical_bytes(result))
    return result
