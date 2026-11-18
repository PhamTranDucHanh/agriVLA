"""Decision guards only. Fixtures are not model quality or GPU evidence."""
import json
from pathlib import Path

import pytest

from agri_vla.dataset import DatasetError
from agri_vla import readiness, evaluation


POLICY = Path(__file__).resolve().parents[1] / 'configs/readiness_balanced_v1.json'


def records():
    intents = ('AUTO', 'WORST_ONLY', 'ZONE_ONLY', 'CAP_LEVEL', 'EXCLUDE_ZONE', 'STOP_ALL', 'WATER_ALL')
    return [{'sample_id': f'{intent}-{i}', 'language': 'vi' if i % 2 else 'en',
             'intent': intent, 'wording_partition': 'seen' if i < 8 else 'unseen',
             'target_action': [0, 0, 0] if intent == 'STOP_ALL' else [1, 2, 3]}
            for intent in intents for i in range(16)]


def predictions(source, *, correct=True):
    return [r | {'seed': 42} | evaluation.quantize_action(r['target_action'] if correct else [1., 1., 1.]) for r in source]


def test_validation_requires_baseline_margin_and_every_stop():
    policy, _ = readiness.load_policy(POLICY)
    source = records()
    good = predictions(source)
    assert readiness.assess_validation(good, policy)['passed']
    off = [r | evaluation.quantize_action([0., 0., 0.]) for r in source]
    assert not readiness.assess_validation(off, policy)['passed']
    # Overall99.1% is insufficient if STOP is wrong.
    stop_index = next(i for i, r in enumerate(source) if r['intent'] == 'STOP_ALL')
    good[stop_index] |= evaluation.quantize_action([1., 0., 0.])
    result = readiness.assess_validation(good, policy)
    assert result['metrics']['overall']['exact_action_accuracy'] > .99
    assert not result['passed'] and not result['checks']['all_stop_correct']


def evidence_fixture(tmp_path, monkeypatch):
    source = records()
    folder = tmp_path / 'data/val'
    folder.mkdir(parents=True)
    (folder / 'sample_metadata.jsonl').write_text('\n'.join(json.dumps({'source_record': r}) for r in source))
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture, not trained model')
    selected = {'version': 2, 'split': 'val', 'records': 112, 'episodes': list(range(112))}
    monkeypatch.setattr(readiness, 'load_training_config', lambda _: {'_base': tmp_path, 'dataset_root': 'data'})
    monkeypatch.setattr(evaluation, 'fixed_validation_selection', lambda *a: selected)
    report = {'checkpoint': str(checkpoint), 'adapter_sha256': readiness.file_sha(checkpoint / 'adapter_model.safetensors'),
              'val112': {'selection': selected, 'predictions': predictions(source), 'metrics': {'fake': 'ignored'}}}
    return checkpoint, report


def test_evidence_recomputes_raw_scores_and_checks_actual_model(tmp_path, monkeypatch):
    checkpoint, report = evidence_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(report))
    assert readiness.check_validation_evidence(path, checkpoint, POLICY)['validation_probe_passed']
    report['val112']['predictions'] = predictions(records(), correct=False)
    # A claimed successful summary must not unlock full training.
    report['val112']['metrics'] = {'overall': {'exact_action_accuracy': 1.}}
    path.write_text(json.dumps(report))
    with pytest.raises(DatasetError, match='readiness not met'):
        readiness.check_validation_evidence(path, checkpoint, POLICY)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(DatasetError, match='different weights'):
        readiness.check_validation_evidence(path, checkpoint, POLICY)


@pytest.mark.parametrize('corrupt', ['duplicate', 'target', 'seed', 'selection'])
def test_evidence_rejects_wrong_protocol(tmp_path, monkeypatch, corrupt):
    checkpoint, report = evidence_fixture(tmp_path, monkeypatch)
    rows = report['val112']['predictions']
    if corrupt == 'duplicate': rows[0] = rows[1]
    elif corrupt == 'target': rows[0]['target_action'] = [3, 3, 3]
    elif corrupt == 'seed': rows[0]['seed'] = 0
    else: report['val112']['selection'] = {'split': 'test'}
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(report))
    with pytest.raises(DatasetError):
        readiness.check_validation_evidence(path, checkpoint, POLICY)


def capacity_fixture(tmp_path, monkeypatch, *, errors=0):
    from agri_vla import candidate_train, small_cases
    source = [{'sample_id': str(i), 'target_action': [0, 0, 0] if i < 6 else [1, 2, 3],
               'intent': 'STOP_ALL' if i < 6 else 'AUTO', 'language': 'vi', 'wording_partition': 'seen'}
              for i in range(60)]
    monkeypatch.setattr(candidate_train, 'load_training_config', lambda _: {})
    monkeypatch.setattr(small_cases, 'selection', lambda *a: {'items': [{'source_record': r} for r in source]})
    checkpoint = tmp_path / 'capacity-model'; checkpoint.mkdir()
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'capacity guard fixture, not a model')
    rows = []
    for seed in (0, 1, 2, 3, 42):
        for i, r in enumerate(source):
            action = [0, 2, 3] if 6 <= i < 6 + errors else r['target_action']
            rows.append(r | {'seed': seed} | evaluation.quantize_action(action))
    report = {'checkpoint': str(checkpoint), 'adapter_sha256': readiness.file_sha(checkpoint / 'adapter_model.safetensors'),
              'stage': 'cases60', 'batch_size': 1, 'seeds': [0, 1, 2, 3, 42], 'records_per_seed': 60,
              'predictions': rows, 'reload_comparison': {'passed': True}}
    path = tmp_path / 'capacity-evidence.json'; path.write_text(json.dumps(report))
    return checkpoint, path


def test_balanced_capacity_is_explicit_and_does_not_change_old_strict_gate(tmp_path, monkeypatch):
    from agri_vla.candidate_train import check_evidence
    checkpoint, path = capacity_fixture(tmp_path, monkeypatch, errors=2)
    with pytest.raises(DatasetError, match='declared gate'):
        check_evidence(path, checkpoint)
    result = check_evidence(path, checkpoint, POLICY)
    assert result['gate'] == {'exact': .95, 'zone': .98, 'stop': 1.}
    assert result['policy']['version'] == 'balanced_v1'


def test_capacity_evidence_pins_actual_weights_and_rejects_duplicate_rows(tmp_path, monkeypatch):
    from agri_vla.candidate_train import check_evidence
    checkpoint, path = capacity_fixture(tmp_path, monkeypatch)
    assert check_evidence(path, checkpoint)['train60_every_seed_passed']
    report = json.loads(path.read_text()); report['predictions'][0] = report['predictions'][1]
    path.write_text(json.dumps(report))
    with pytest.raises(DatasetError, match='Duplicate'):
        check_evidence(path, checkpoint)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'new weights at same path')
    with pytest.raises(DatasetError, match='pin these adapter weights'):
        check_evidence(path, checkpoint)
