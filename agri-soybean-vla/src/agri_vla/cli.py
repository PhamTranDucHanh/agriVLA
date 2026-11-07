"""Data CLI; individual commands load only the dependencies they need."""

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys
import zipfile

from . import __version__
from .preflight import collect_report, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agri-vla", description="AgriVLA-Soybean-v0: bilingual three-zone dataset tools."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Check environment without downloading data or weights.")
    preflight.add_argument("--mode", choices=("data", "train"), default="data")
    preflight.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable (train mode only).")
    preflight.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    preflight.add_argument("--output", type=Path, help="Save JSON to a new file; existing files are never overwritten.")
    contract = commands.add_parser("contract", help="Print the current input/target/metadata contract; pinned v0 remains available.")
    contract.add_argument("--revision",choices=("v0","v1"),default="v1")
    for name in ("compose-revision", "validate-revision", "browse-revision"):
        revision = commands.add_parser(name, help="Casual VI/EN + explicit WATER_ALL v1; immutable v0 stays intact.")
        revision.add_argument("--config", type=Path, default=project_root()/"configs/revision_v1.yaml")
        revision.add_argument("--dry-run", action="store_true")
        if name == "browse-revision":
            revision.add_argument("--port", type=int, default=8765)
            revision.add_argument("--replace",action="store_true",help="After QA, replace only the same user's AgriVLA browser on this port.")
    handoff = commands.add_parser("package-handoff", help="Package source, locked data and model cache for a NEW GPU machine; excludes env/vendor.")
    handoff.add_argument("--output", type=Path, required=True, help="NEW .tar archive.")
    handoff.add_argument("--dry-run", action="store_true")
    check_transfer = commands.add_parser("verify-handoff", help="Verify extracted files against a transfer manifest.")
    check_transfer.add_argument("--manifest", type=Path, required=True)
    check_transfer.add_argument("--repo-root", type=Path, default=project_root().parent)
    for name, description in (
        ("prepare-training", "Validate exports, prepare pinned base/processors and fresh LoRA; NO training."),
        ("check-model-inputs", "CPU no-grad forward/inference checks with saved model/processors."),
        ("train-upstream", "Thin upstream CUDA launcher; dry-run unless --execute is explicit."),
        ("infer", "Predict held-out records; no teacher labels enter the policy."),
        ("evaluate", "Predict and score held-out val/test records with saved train stats."),
    ):
        cmd = commands.add_parser(name, help=description)
        cmd.add_argument("--config", type=Path, default=project_root() / "configs/training_v0.yaml")
        cmd.add_argument("--profile", choices=("4gb",), default="4gb", help="Candidate profile, NOT GPU-fit certification.")
        cmd.add_argument("--bundle", type=Path, help="Prepared bundle; default from config.")
        if name == "prepare-training":
            cmd.add_argument("--download-models", action="store_true", help="Download only pinned base/backbone/tokenizer assets.")
            cmd.add_argument("--dry-run", action="store_true", help="No ML imports/downloads/writes.")
            cmd.add_argument("--output-dir", type=Path, help="NEW prepared bundle.")
        else:
            cmd.add_argument("--output", type=Path, required=True, help="NEW report/output path; existing output refused.")
        if name == "train-upstream":
            cmd.add_argument("--steps", type=int)
            cmd.add_argument("--execute", action="store_true", help="Actually start upstream optimizer training; requires CUDA.")
            cmd.add_argument("--dry-run", action="store_true", help="Print config only (also the default).")
        if name in {"infer", "evaluate"}:
            cmd.add_argument("--split", choices=("val", "test"), required=name=="evaluate")
            cmd.add_argument("--checkpoint", type=Path, help="Saved pretrained_model adapter directory; default UNTRAINED proof adapter.")
            cmd.add_argument("--limit", type=int, help="Diagnostic subset only; omit for the whole split.")
            cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            if name == "infer":
                cmd.add_argument("--images",type=Path,nargs=3,metavar=("ZONE1","ZONE2","ZONE3"),help="Three original RGB images, in zone order; conflicts with --split.")
                cmd.add_argument("--instruction",help="Free-form VI/EN instruction; requires --images. Over-length input is rejected.")
    for command, help_text in (
        ("fetch-source", "Fetch/import raw files with checksums; never overwrite existing raw."),
        ("inspect-source", "Audit CSV/ZIP, materialize original image bytes and create source reports."),
    ):
        source = commands.add_parser(command, help=help_text)
        source.add_argument("--config", type=Path, required=True)
        source.add_argument("--dry-run", action="store_true", help="Print the plan without network or writes.")
        if command == "fetch-source":
            source.add_argument("--local-csv", type=Path, help="Reuse this checksum-matching local CSV.")
            source.add_argument("--local-zip", type=Path, help="Reuse this checksum-matching local ZIP.")
        else:
            source.add_argument("--output-dir", type=Path, help="New report directory; existing directories are refused.")
    split = commands.add_parser("split", help="Audit duplicate leakage and lock explicit source-group train/val/test splits.")
    split.add_argument("--config", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, help="New output directory; existing directories are refused.")
    split.add_argument("--dry-run", action="store_true", help="Run source/image/duplicate checks without any writes.")
    for name, description in (("convert", "Convert validated scenes to image-only LeRobot singleton episodes (CPU)."),
                              ("verify-lerobot", "Round-trip every exported record and inspect real singleton batches (CPU).")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", type=Path, default=project_root() / "configs/lerobot_wording_v1.yaml")
        command.add_argument("--dry-run", action="store_true", help="Do not write output; convert dry-run is data-only.")
        if name == "convert":
            command.add_argument("--profile", choices=("smoke", "full"), required=True)
            command.add_argument("--output-dir", type=Path, help="NEW output root; existing output is never overwritten.")
            command.add_argument("--smoke-root", type=Path, help="Verified 64-sample smoke root gating full conversion.")
        else:
            command.add_argument("--root", type=Path, required=True)
            command.add_argument("--output", type=Path, help="Write a NEW verification report; default prints only.")
    browse = commands.add_parser("browse", help="Browse ALL samples and original local images on localhost (read-only).")
    browse.add_argument("--config", type=Path, default=project_root() / "configs/dataset_v0.yaml")
    browse.add_argument("--manifests", type=Path, default=project_root() / "data/manifests/v0")
    browse.add_argument("--port", type=int, default=8765)
    browse.add_argument("--dry-run", action="store_true", help="Validate data and browser index without opening a port or writing files.")
    for name, help_text in (("compose", "Generate and QA bilingual synthetic three-zone manifests."),
                            ("validate", "Verify all dataset files, source locks, pairs, policy and leakage."),
                            ("preview", "Create a local HTML/image review without changing model images.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=project_root() / "configs/dataset_v0.yaml")
        command.add_argument("--dry-run", action="store_true", help="Run checks without any writes.")
        if name == "compose":
            command.add_argument("--output-dir", type=Path, help="New manifest directory; existing output refused.")
        else:
            command.add_argument("--manifests", type=Path, required=True)
            if name == "validate":
                command.add_argument("--output", type=Path, help="Save QA to a NEW file; otherwise print only.")
            else:
                command.add_argument("--output-dir", type=Path, help="New preview directory; defaults to manifests/preview.")
                command.add_argument("--count", type=int, default=60, help="Number of records; stratified by split/language/intent/edge cases.")
    for name, help_text in (("explain-policy", "Explain reference levels with paired VI/EN instructions."),
                            ("audit-policy", "Enumerate policy cases and audit bilingual template isolation; no scenes.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--policy-config", type=Path, default=project_root() / "configs/policy_v0.yaml")
        command.add_argument("--language-config", type=Path, default=project_root() / "configs/language_v0.yaml")
        if name == "explain-policy":
            command.add_argument("--wilt", type=int, nargs=3, required=True, metavar=("W1", "W2", "W3"))
            command.add_argument("--intent", help="One of AUTO, WORST_ONLY, ZONE_ONLY, CAP_LEVEL, EXCLUDE_ZONE, STOP_ALL.")
            command.add_argument("--zone", type=int)
            command.add_argument("--cap", type=int)
            command.add_argument("--split", choices=("train", "val", "test"), default="train")
            command.add_argument("--wording", choices=("seen", "unseen"), default="seen")
            command.add_argument("--seed", type=int)
            command.add_argument("--json", action="store_true")
        else:
            command.add_argument("--output-dir", type=Path, help="New audit directory; existing output is refused.")
            command.add_argument("--dry-run", action="store_true", help="Run checks without writing artifacts.")
    args = parser.parse_args(argv)
    if args.command in {"compose-revision", "validate-revision", "browse-revision"}:
        from .dataset import DatasetError, load_compose_config
        from .revision import load_revision, compose_revision, validate_revision
        try:
            c = load_revision(args.config)
            if args.command == "compose-revision":
                result = compose_revision(c,dry_run=args.dry_run)
            elif args.command == "validate-revision":
                result = validate_revision(c)[0]
            else:
                from .browser import serve_dataset
                validated = validate_revision(c)
                result = serve_dataset(load_compose_config(c["_base"]/c["parent_config"]),c["_base"]/c["output_dir"],port=args.port,dry_run=args.dry_run,validated=validated,replace=args.replace)
                if not args.dry_run:
                    return 0
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0
        except (DatasetError, ValueError, OSError, KeyError, TypeError) as error:
            print(f"Revision error: {error}",file=sys.stderr)
            return 2
    if args.command in {"package-handoff", "verify-handoff"}:
        from .dataset import DatasetError, safe_output, load_compose_config
        from .handoff import package_handoff, verify_handoff
        try:
            if args.command == "package-handoff":
                safe_output(load_compose_config(project_root() / "configs/dataset_v0.yaml"), args.output)
                result = package_handoff(project_root(), args.output, dry_run=args.dry_run)
            else:
                result = verify_handoff(args.repo_root, args.manifest)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DatasetError, ValueError, OSError, KeyError) as error:
            print(f"Handoff error: {error}", file=sys.stderr)
            return 2
    if args.command in {"prepare-training", "check-model-inputs", "train-upstream", "infer", "evaluate"}:
        from .dataset import DatasetError, safe_output, load_compose_config
        from .training import load_training_config, prepare_training, launch_training
        try:
            c = load_training_config(args.config)
            scene = load_compose_config(c["_base"] / "configs/dataset_v0.yaml")
            bundle = args.bundle.resolve() if args.bundle else c["_base"] / c["bundle"]
            if args.command == "prepare-training":
                destination = args.output_dir or bundle
                safe_output(scene, destination)
                result = prepare_training(c, download=args.download_models, dry_run=args.dry_run, output=destination)
            else:
                safe_output(scene, args.output)
                if args.command == "train-upstream":
                    if args.execute and args.dry_run:
                        raise ValueError("--execute and --dry-run conflict")
                    result = launch_training(c, bundle, output=args.output, steps=args.steps, execute=args.execute)
                else:
                    from .evaluation import check_model_inputs, evaluate, infer_scene
                    if args.command == "check-model-inputs":
                        result = check_model_inputs(c, bundle, output=args.output)
                    else:
                        if args.command == "infer" and (args.images is not None or args.instruction is not None):
                            if args.split is not None or args.limit is not None or args.images is None or args.instruction is None:
                                raise ValueError("Use either --split/--limit OR --images with --instruction")
                            result = infer_scene(c,bundle,images=args.images,instruction=args.instruction,output=args.output,checkpoint=args.checkpoint,device=args.device)
                        else:
                            if args.split is None:
                                raise ValueError("infer requires --split or --images with --instruction")
                            result = evaluate(c, bundle, split=args.split, output=args.output, checkpoint=args.checkpoint, count=args.limit, device=args.device)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DatasetError, OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
            print(f"Model pipeline error: {error}", file=sys.stderr)
            return 2
    if args.command in {"convert", "verify-lerobot"}:
        from .dataset import DatasetError, safe_output, load_compose_config
        from .lerobot_io import load_conversion_config, convert, verify_path
        from .policy import PolicyError
        from .source import SourceError
        try:
            config = load_conversion_config(args.config)
            from .policy import load_unique_yaml
            source_path = config["_base"]/config["scene_config"]
            source_config = load_unique_yaml(source_path.read_bytes(), context="conversion source") if source_path.is_file() else {}
            revision_mode = source_config.get("version") == "agri_soybean_revision_v1"
            if revision_mode:
                from .revision_io import convert, verify_path
            if args.command == "convert":
                result = convert(config, profile=args.profile, output_dir=args.output_dir, smoke_root=args.smoke_root, dry_run=args.dry_run)
            else:
                if args.output is not None:
                    safe_output(load_compose_config(config["_base"] / ("configs/dataset_v0.yaml" if revision_mode else config["scene_config"])), args.output)
                result = verify_path(config, args.root)
                from .lerobot_checks import audit_factory_batches
                result["upstream_factory_batches"] = audit_factory_batches(args.root)
                if args.output is not None and not args.dry_run:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("x", encoding="utf-8") as stream:
                        json.dump(result, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DatasetError, PolicyError, SourceError, OSError, ValueError, KeyError, TypeError, ImportError) as error:
            print(f"LeRobot IO error: {error}", file=sys.stderr)
            return 2
    if args.command == "browse":
        from .browser import serve_dataset
        from .dataset import DatasetError, load_compose_config
        from .policy import PolicyError
        from .source import SourceError
        try:
            result = serve_dataset(load_compose_config(args.config), args.manifests, port=args.port, dry_run=args.dry_run)
            if args.dry_run:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DatasetError, PolicyError, SourceError, OSError, ValueError, KeyError, TypeError) as error:
            print(f"Browser error: {error}", file=sys.stderr)
            return 2
    if args.command in {"compose", "validate", "preview"}:
        from .dataset import DatasetError, load_compose_config, safe_output
        from .policy import PolicyError
        from .source import SourceError

        try:
            config = load_compose_config(args.config)
            if args.command == "compose":
                from .scenes import compose
                result = compose(config, output_dir=args.output_dir, dry_run=args.dry_run)
            elif args.command == "validate":
                from .validate import validate_dataset
                if args.output is not None:
                    safe_output(config, args.output)
                result, _, _ = validate_dataset(config, args.manifests)
                if args.output is not None and not args.dry_run:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("x", encoding="utf-8") as stream:
                        json.dump(result, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
            else:
                from .preview import preview_dataset
                result = preview_dataset(config, args.manifests, output_dir=args.output_dir, count=args.count, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DatasetError, PolicyError, SourceError, OSError, ValueError, KeyError, TypeError) as error:
            print(f"Dataset error: {error}", file=sys.stderr)
            return 2
    if args.command in {"explain-policy", "audit-policy"}:
        from .language import load_registry
        from .policy import PolicyError, load_policy, validate_request
        from .policy_audit import audit_policy, explain_policy

        try:
            policy = load_policy(args.policy_config)
            registry = load_registry(args.language_config)
            if args.command == "audit-policy":
                result = audit_policy(policy, registry, output_dir=args.output_dir, dry_run=args.dry_run)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                params = {key: getattr(args, key) for key in ("zone", "cap") if getattr(args, key) is not None}
                if args.intent is None and params:
                    raise PolicyError("--zone/--cap require an explicit --intent")
                request = validate_request(args.intent, params) if args.intent is not None else None
                result = explain_policy(policy, registry, args.wilt, request=request, split=args.split, wording=args.wording, seed=args.seed)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"{result['policy_version']} | teacher wilt={args.wilt} | wilt 0–1 OFF, actionable threshold=2")
                    print("Synthetic proxy action levels 0–3; not calibrated water quantities.")
                    for item in result["examples"]:
                        print(f"\n{item['intent']} {json.dumps(item['intent_params'])} -> {item['target_action']}")
                        for instruction in item["instructions"]:
                            print(f"  {instruction['language']}: {instruction['instruction']}")
            return 0
        except (PolicyError, OSError, ValueError) as error:
            print(f"Policy error: {error}", file=sys.stderr)
            return 2
    if args.command == "split":
        from .source import SourceError
        from .split import load_split_config, split_sources

        try:
            result = split_sources(load_split_config(args.config), output_dir=args.output_dir, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "pass" else 1
        except (SourceError, OSError, ValueError) as error:
            print(f"Split error: {error}", file=sys.stderr)
            return 2
    if args.command in {"fetch-source", "inspect-source"}:
        from .source import SourceError, fetch_source, load_config

        try:
            config = load_config(args.config)
            if args.command == "fetch-source":
                result = fetch_source(config, local_csv=args.local_csv, local_zip=args.local_zip, dry_run=args.dry_run)
            else:
                from .ingest import inspect_source

                result = inspect_source(config, output_dir=args.output_dir, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status", "pass") != "fail" else 1
        except (SourceError, OSError, zipfile.BadZipFile) as error:
            print(f"Source error: {error}", file=sys.stderr)
            return 2
    if args.command == "contract":
        result = json.loads(files("agri_vla").joinpath("model_contract.json").read_text(encoding="utf-8"))
        if args.revision == "v1":
            result["schema_version"] = "agri-soybean-scene-v1"
            result["training_target"]["label_source"] = "wilt_proxy_with_operator_override_v1"
            result["training_target"]["label_type"] = ["synthetic_policy_action","synthetic_operator_override_action"]
            result["metadata_only"].append("revision_metadata")
            result["intent_extension"] = {"WATER_ALL":{"required_parameter":"level","allowed_levels":[1,2,3],"target":"[level,level,level] independent of wilt","missing_level":"reject"}}
            result["train_statistics"] = "data/lerobot/wording_v1/full/train_stats.json"
            result.pop("runtime_lerobot_compatibility_verified",None)
            result["runtime_checks"] = {"cpu_report":"reports/model-inputs-step6.json","gpu_fit_verified":False}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    if args.require_cuda and args.mode != "train":
        parser.error("--require-cuda requires --mode train")

    report = collect_report(args.mode, require_cuda=args.require_cuda)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(serialized)
        except OSError as error:
            print(f"Cannot write report: {error}", file=sys.stderr)
            return 2
    if args.json:
        print(serialized, end="")
    else:
        print(f"Preflight {args.mode}: {report['status'].upper()} (environment only)")
        for check in report["checks"]:
            print(f"  {check['status'].upper():4} {check['name']}: {check['detail']}")
        print("Not checked: " + ", ".join(report["not_checked"]))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
