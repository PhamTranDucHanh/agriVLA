"""Policy/template diagnostics, not scene generation or a training dataset."""

import csv
import hashlib
from io import StringIO
import itertools
import json
from pathlib import Path

from .language import TemplateRegistry, registry_summary
from .policy import Intent, Policy, PolicyError, all_requests, validate_request, validate_wilt


def example_requests():
    return tuple(validate_request(intent, {"zone": 2} if intent == Intent.ZONE_ONLY else
                                  {"cap": 1} if intent == Intent.CAP_LEVEL else
                                  {"zone": 3} if intent == Intent.EXCLUDE_ZONE else {}) for intent in Intent)


def explain_policy(policy: Policy, registry: TemplateRegistry, wilt_scores, *, request=None, split="train", wording="seen", seed=None) -> dict:
    scores = validate_wilt(wilt_scores)
    examples = []
    for selected in (request,) if request is not None else example_requests():
        pair = registry.choose_pair(selected.intent, selected.params, split=split, wording=wording, seed=seed)
        examples.append({"intent": str(selected.intent), "intent_params": selected.params,
                         "instructions": pair, "target_action": policy.action(scores, selected.intent, selected.params)})
    return {"artifact_kind": "policy_explanation_not_dataset", "policy_version": policy.version,
            "label_type": "synthetic_policy_action", "wilt_scores_teacher_only": list(scores),
            "actionable_wilt_min": 2, "note": "Wilt 0–1 stay OFF; levels are synthetic proxy targets, not calibrated water quantities.",
            "policy_config_sha256": policy.config_sha256, "language_config_sha256": registry.config_sha256,
            "split": split, "wording": wording, "examples": examples}


def csv_bytes(records: list[dict], fields: list[str]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def audit_policy(policy: Policy, registry: TemplateRegistry, *, output_dir: Path | None = None, dry_run=False) -> dict:
    if output_dir is not None and (output_dir.exists() or output_dir.is_symlink()):
        raise PolicyError(f"Output already exists; choose a NEW policy audit directory: {output_dir}")
    requests = all_requests()
    truth = []
    for scores in itertools.product(range(5), repeat=3):
        for request in requests:
            action = policy.action(scores, request.intent, request.params)
            if len(action) != 3 or any(type(value) is not int or value not in range(4) for value in action):
                raise PolicyError("Reference policy produced invalid target levels")
            truth.append(dict(zip(("wilt1", "wilt2", "wilt3"), scores, strict=True)) |
                         {"intent": str(request.intent), "zone": request.zone, "cap": request.cap} |
                         dict(zip(("action1", "action2", "action3"), action, strict=True)))
    catalog = [{"template_partition": item["template_partition"], "template_family_id": item["template_family_id"],
                "template_id": item["template_id"], "language": item["language"], "intent": item["intent"],
                "intent_params": json.dumps(item["intent_params"], sort_keys=True),
                "wording_partition": item["wording_partition"], "instruction": item["instruction"]} for item in registry.catalog()]
    examples = explain_policy(policy, registry, [0, 2, 4])
    markdown = ["# Policy examples — not a training dataset", "", "Teacher-only wilt scores: [0, 2, 4]. Wilt 0–1 stay OFF; action levels 0–3 are synthetic proxy targets.", ""]
    for item in examples["examples"]:
        markdown += [f"## {item['intent']} {json.dumps(item['intent_params'])}", ""]
        markdown += [f"- {entry['language']}: {entry['instruction']}" for entry in item["instructions"]]
        markdown += [f"- Target: {item['target_action']}", ""]
    artifacts = {
        "policy_truth_table.csv": csv_bytes(truth, ["wilt1", "wilt2", "wilt3", "intent", "zone", "cap", "action1", "action2", "action3"]),
        "template_catalog.csv": csv_bytes(catalog, ["template_partition", "template_family_id", "template_id", "language", "intent", "intent_params", "wording_partition", "instruction"]),
        "policy_examples.md": ("\n".join(markdown) + "\n").encode("utf-8"),
    }
    lock = {"policy_version": policy.version, "registry_version": registry.version, "seed": registry.seed,
            "policy_config_sha256": policy.config_sha256, "policy_semantic_sha256": policy.semantic_sha256,
            "language_config_sha256": registry.config_sha256, "language_semantic_sha256": registry.semantic_sha256,
            "artifacts_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}}
    lock_bytes = (json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    summary = registry_summary(registry)
    summary["instruction_character_lengths"] = {
        language: {"min": min(len(r["instruction"]) for r in catalog if r["language"] == language),
                   "max": max(len(r["instruction"]) for r in catalog if r["language"] == language)} for language in ("vi", "en")}
    summary["actual_dataset_family_coverage"] = "not_checked: scenes have not been generated"
    report = {"status": "pass", "artifact_kind": "policy_audit_not_dataset", "dry_run": dry_run,
              "wilt_combinations": 125, "intent_parameter_cases": len(requests), "policy_evaluations": len(truth),
              "registry": summary, "lock": lock, "policy_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
              "examples": examples["examples"],
              "validation_scope": ["strict v0 config", "request domains", "all finite ordinal target vectors", "template slots/threshold/tie wording guards", "family and normalized wording isolation"],
              "not_proven": ["automatic natural-language equivalence of arbitrary text", "tokenizer lengths/truncation", "model language/vision competence", "physical irrigation efficacy"],
              "not_performed": ["source scene generation", "dataset manifests", "training"]}
    if output_dir is not None and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in artifacts.items():
            with (output_dir / name).open("xb") as stream:
                stream.write(data)
        with (output_dir / "policy_lock.json").open("xb") as stream:
            stream.write(lock_bytes)
        with (output_dir / "policy_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return report
