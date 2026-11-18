"""Analyze the predeclared K/V experiment; never train or select on TEST."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import math

from .ablation_analysis import scores, summarize, conditional_scene_interval
from .dataset import read_json, read_jsonl, require
from .preflight import project_root
from .training import file_sha, write_json


def decision(candidate, control, interval, criteria):
    a, b = candidate['validation_mean'], control['validation_mean']
    checks = {
        'mean_validation_exact_margin': a['exact'] - b['exact'] >= criteria['mean_validation_exact_margin'],
        'mean_validation_zone_margin': a['zone'] - b['zone'] >= criteria['mean_validation_zone_margin'],
        'positive_noise_seeds': sum(candidate['validation_by_seed'][s]['exact'] > control['validation_by_seed'][s]['exact']
                                    for s in candidate['validation_by_seed']) >= criteria['minimum_positive_noise_seeds'],
        'conditional_scene_interval_positive': interval['lower'] > 0,
        'STOP_not_worse': a['STOP'] >= b['STOP'],
        'false_watering_not_worse': a['false_watering_rate'] <= b['false_watering_rate'],
        'old60_guard': candidate['old60_42']['exact'] >= control['old60_42']['exact'] - criteria['old60_exact_max_drop'],
        'new220_guard': candidate['new220_42']['exact'] >= control['new220_42']['exact'],
    }
    return {'passed': all(checks.values()), 'checks': checks}


def analyze(report_dir, tag):
    root = project_root(); out = Path(report_dir).resolve()
    reports = {arm: read_json(out / f'{arm}-final.json') for arm in ('control', 'candidate')}
    protocol = read_json(out / 'protocol.json')
    for name, sha in protocol['files_sha256'].items():
        require(file_sha(root / name) == sha, f'Frozen file changed: {name}')
    from .small_cases import selection
    from .training import load_training_config
    old = {x['source_record']['sample_id'] for x in selection(load_training_config(root / 'configs/training_v0.yaml'), 'cases60')['items']}
    metadata = {r['sample_id']: r for r in read_jsonl(root / 'data/manifests/wording_v1/val.jsonl')}
    result = {}
    for arm, report in reports.items():
        require(report['seeds'] == [0, 1, 2, 3, 42], 'Five fixed noise seeds required')
        require(report['validation_selection'] == protocol['val112'], 'Frozen VAL panel differs')
        require(len(report['val_predictions']) == 560 and len(report['train']['predictions']) == 280, 'Incomplete evaluation')
        expected = set(protocol['val112']['sample_ids'])
        for seed in report['seeds']:
            rows = [r for r in report['val_predictions'] if r['seed'] == seed]
            require(len(rows) == 112 and {r['sample_id'] for r in rows} == expected, 'Missing/duplicate seed rows')
            for row in rows:
                require(all(row[k] == metadata[row['sample_id']][k] for k in ('target_action', 'intent', 'language', 'wording_partition')), 'VAL labels or grouping changed')
        require(file_sha(Path(report['checkpoint']) / 'adapter_model.safetensors') == report['adapter_sha256'], 'Checkpoint changed')
        result[arm] = summarize(report, old)
        result[arm]['validation_groups_five_noise_mean'] = {
            key:{value:scores([r for r in report['val_predictions'] if r[key]==value])
                 for value in sorted({r[key] for r in report['val_predictions']})}
            for key in ('intent','language','wording_partition')}
        result[arm]['milestone100'] = scores(read_json(out / f'{arm}-val100.json')['val112']['predictions'])
    interval = conditional_scene_interval(reports['candidate']['val_predictions'], reports['control']['val_predictions'], metadata)
    verdict = decision(result['candidate'], result['control'], interval, protocol['recipe']['comparison']['acceptance'])
    orders = {}
    resources = {}; reloads = {}; checkpoints = {}
    for arm in reports:
        rows = []
        phases = []
        for step in (2, 100, 300):
            folder = root / 'reports/small-cases' / f'latekv-{tag}-{arm}-s{step}'
            rows.extend(read_jsonl(folder / 'updates.jsonl'))
            d = read_json(folder / 'result.json')
            phases.append({'final_step': step, 'new_updates': d['optimizer_steps_performed'],
                           'torch_allocated_mib': d['train_peak_allocated'] / 2**20,
                           'torch_reserved_mib': d['train_peak_reserved'] / 2**20,
                           'nvml_peak_mib': max(p['used_mib'] for p in d['gpu_samples']) if d.get('gpu_samples') else None,
                           'update_seconds_sum': sum(d['update_seconds']), 'data_seconds_sum': d['data_seconds_sum'],
                           'wall_seconds': d['wall_seconds_including_setup_screens_saves'],
                           'memory': d.get('process_memory')})
        require([r['step'] for r in rows] == list(range(1, 301)), 'Not exactly 300 cumulative updates')
        orders[arm] = rows
        resources[arm] = {'phases': phases, 'mean_update_seconds': statistics.mean(r['seconds'] for r in rows),
                          'mean_dataload_seconds': sum(p['data_seconds_sum'] for p in phases)/300,
                          'loss_curve_20_updates': [{'last_step':end,'mean_loss':statistics.mean(r['loss'] for r in rows[end-20:end])}
                                                    for end in range(20,301,20)]}
        require(all(math.isfinite(r[k]) for r in rows for k in ('loss','grad_norm','lr')), 'Nonfinite training metric')
        from .small_cases import compare_predictions
        saved = read_json(root / 'reports/small-cases' / f'latekv-{tag}-{arm}-s300/evaluation-000300.json')
        reference = [r for r in saved['screen']['predictions'] if r['seed'] == 42]
        reloaded = [r for r in reports[arm]['train']['predictions'] if r['sample_id'] in old]
        reloads[arm] = compare_predictions(reference, reloaded, atol=1e-3, rtol=1e-3)
        require(reloads[arm]['passed'], 'Fresh-process TRAIN60 inference differs from pre-save screen')
        from safetensors import safe_open
        import numpy as np
        cp = Path(reports[arm]['checkpoint']); audits = {}
        for name in ('adapter_model.safetensors', '../training_state/optimizer_state.safetensors'):
            path = cp / name
            with safe_open(path, framework='np') as state:
                require(all(np.isfinite(state.get_tensor(k)).all() for k in state.keys()), 'Nonfinite saved tensors')
                audits[name] = {'tensors':len(state.keys()),'finite':True}
                if 'optimizer' in name:
                    steps = sorted({float(state.get_tensor(k).item()) for k in state.keys() if k.endswith('/step')})
                    require(steps == [300.0], 'Saved optimizer steps do not all equal300')
                    audits[name]['unique_optimizer_steps'] = steps
        audits['adapter_squared_delta_last_phase'] = sum(saved['weight_squared_delta'].values())
        require(audits['adapter_squared_delta_last_phase'] > 0, 'No adapter change')
        if arm == 'candidate':
            gradients = [g for step in (2,100,300) for g in read_json(root / 'reports/small-cases' / f'latekv-{tag}-{arm}-s{step}/late-vlm-runtime.json')['KV_gradient_checks']]
            require(len(gradients) == 300 and all(math.isfinite(v) and v > 0 for g in gradients for v in g['squared_norms'].values()), 'K/V gradients incomplete')
            audits['KV_nonzero_finite_gradient_updates'] = len(gradients)
            audits['KV_squared_delta_last_phase'] = sum(v for k,v in saved['weight_squared_delta'].items() if '.vlm.' in k)
        checkpoints[arm] = audits
    fairness = {'same_episode_order_all_300_updates': all(a['episodes'] == b['episodes'] for a, b in zip(orders['control'], orders['candidate'])),
                'same_rng_all_300_updates': all(a['rng_before'] == b['rng_before'] for a, b in zip(orders['control'], orders['candidate']))}
    require(all(fairness.values()), 'Arms are not matched')
    from .readiness import assess_validation, load_policy
    readiness = {arm: assess_validation([r for r in rep['val_predictions'] if r['seed'] == 42], load_policy()[0]) for arm, rep in reports.items()}
    final = {'arms': result, 'candidate_minus_control_scene_interval': interval, 'relative_improvement': verdict,
             'fairness': fairness, 'resources': resources, 'validation_readiness': readiness,
             'fresh_process_inference_comparison': reloads, 'saved_tensor_audits': checkpoints,
             'candidate_checkpoint': reports['candidate']['checkpoint'], 'candidate_adapter_sha256': reports['candidate']['adapter_sha256'],
             'full_ready': False, 'TEST_opened': False, 'full_training_performed': False,
             'limits': 'One training seed. Five inference-noise seeds are not five independent trainings. Fixed VALID112 is a development panel, not full VAL720 or TEST accuracy.',
             'report_hashes': {arm: file_sha(out / f'{arm}-final.json') for arm in reports}}
    write_json(out / 'decision.json', final)
    fields = ['arm', 'seed', 'sample_id', 'scene_id', 'intent', 'language', 'wording_partition',
              'instruction', 'zone1_image', 'zone2_image', 'zone3_image', 'target_action', 'continuous_action', 'quantized_action', 'correct']
    with (out / 'validation-examples.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for arm, report in reports.items():
            for row in report['val_predictions']:
                raw = metadata[row['sample_id']]
                output = {k: row[k] for k in ('seed', 'sample_id', 'intent', 'language', 'wording_partition')}
                output.update(arm=arm, scene_id=raw['scene_id'], instruction=raw['instruction'],
                              correct=row['quantized_action'] == row['target_action'])
                output.update({z + '_image': str(root / raw['images'][z]) for z in ('zone1', 'zone2', 'zone3')})
                output.update({k: json.dumps(row[k]) for k in ('target_action', 'continuous_action', 'quantized_action')})
                writer.writerow(output)
    print(json.dumps({'relative_improvement': verdict, 'full_ready': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, required=True); p.add_argument('--tag', required=True)
    a = p.parse_args(); analyze(a.report_dir, a.tag)
