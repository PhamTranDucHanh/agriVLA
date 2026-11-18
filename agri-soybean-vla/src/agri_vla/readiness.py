"""Recompute held-out sanity checks before spending a full training budget."""
from pathlib import Path

from .dataset import require, read_json, read_jsonl
from .preflight import project_root
from .training import file_sha, load_training_config


def load_policy(path=None):
    path = Path(path) if path else project_root() / 'configs/readiness_balanced_v1.json'
    policy = read_json(path)
    require(policy['version'] == 'balanced_v1' and policy['declared_before_pilot280_validation']
            and policy['never_use_test_for_tuning'] and policy['small_train_fit_is_not_generalization'],
            'Unknown readiness policy')
    require(policy['capacity_train60_each_seed'] == {'exact': .95, 'zone': .98, 'stop': 1.}
            and policy['validation_probe'] == {'records': 112, 'selection_version': 2,
                'exact_margin_over_always_off': .05, 'zone_margin_over_always_off': .02, 'stop': 1.0},
            'Readiness thresholds drifted; create a new policy version explicitly')
    return policy, {'file': str(path.resolve()), 'sha256': file_sha(path), 'version': policy['version']}


def assess_validation(rows, policy):
    from .evaluation import metrics, quantize_action
    measured = metrics(rows)
    baseline_rows = [{k: r[k] for k in ('sample_id', 'language', 'intent', 'wording_partition', 'target_action')}
                     | quantize_action([0., 0., 0.]) for r in rows]
    baseline = metrics(baseline_rows)
    actual, off = measured['overall'], baseline['overall']
    rule = policy['validation_probe']
    checks = {
        'exact_above_off_plus_margin': actual['exact_action_accuracy'] >= off['exact_action_accuracy'] + rule['exact_margin_over_always_off'],
        'zone_above_off_plus_margin': actual['zone_accuracy'] >= off['zone_accuracy'] + rule['zone_margin_over_always_off'],
        'all_stop_correct': actual['stop_exact_accuracy'] == rule['stop'],
    }
    return {'passed': all(checks.values()), 'checks': checks, 'metrics': measured, 'always_off': baseline,
            'scope': 'partial VAL sanity only; does not certify full-model accuracy'}


def check_validation_evidence(path, checkpoint, policy_path=None):
    from .evaluation import fixed_validation_selection
    policy, identity = load_policy(policy_path)
    checkpoint = Path(checkpoint).resolve()
    report = read_json(path)
    require(Path(report['checkpoint']).resolve() == checkpoint
            and report['adapter_sha256'] == file_sha(checkpoint / 'adapter_model.safetensors'),
            'VAL evidence belongs to different weights')
    c = load_training_config(project_root() / 'configs/training_v0.yaml')
    wanted = fixed_validation_selection(c, policy['validation_probe']['records'])
    require(report['val112']['selection'] == wanted, 'VAL selection differs from frozen balanced v2')
    rows = report['val112']['predictions']
    metadata = read_jsonl(c['_base'] / c['dataset_root'] / 'val/sample_metadata.jsonl')
    targets = {metadata[i]['source_record']['sample_id']: metadata[i]['source_record'] for i in wanted['episodes']}
    require(len(rows) == 112 and {r['sample_id'] for r in rows} == targets.keys(), 'Missing/duplicate VAL evidence')
    for row in rows:
        require(row['seed'] == 42 and all(row[k] == targets[row['sample_id']][k]
                for k in ('target_action', 'language', 'intent', 'wording_partition')), 'VAL sample/target/seed mapping differs')
    result = assess_validation(rows, policy)
    require(result['passed'], f"VAL readiness not met: {result['checks']}")
    return {'file': str(Path(path).resolve()), 'sha256': file_sha(path), 'policy': identity,
            'validation_probe_passed': True, **result}
