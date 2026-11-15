"""Package a measured sampling-only variant of genuinely trained checkpoints.

Adapter bytes and native optimizer/scheduler/RNG/step/AMP files are copied exactly.
Only native num_steps and the variant's output/provenance paths are reconfigured.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .dataset import require, new_output, read_json
from .preflight import project_root
from .training import file_sha, write_json
from .small_cases import load_experiment, passes, compatible_inference_variant


def run(source_run, checkpoints, config, profile_name, reference, output):
    root = project_root()
    source_run, config, reference, output = (Path(x).resolve() for x in (source_run, config, reference, output))
    require(source_run.is_relative_to(root / 'outputs/small-cases') and output.is_relative_to(root / 'outputs/small-cases'), 'Use isolated small-cases paths')
    require(config.parent == root / 'configs', 'Config must live in project configs')
    spec = load_experiment(config); target = spec['profiles'][profile_name]
    require(checkpoints and len(checkpoints) == len(set(checkpoints)) and all(n > 0 for n in checkpoints), 'Invalid steps')
    new_output(output)
    report_dir = root / 'reports/small-cases' / output.name
    new_output(report_dir); report_dir.mkdir(parents=True)
    source_exp = read_json(source_run / 'checkpoints' / f'{checkpoints[0]:06d}' / 'pretrained_model/small_experiment.json')
    require(compatible_inference_variant(source_exp['profile'], target), 'Variant may change only native sampling num_steps')
    measurement = read_json(reference)
    final_step = max(checkpoints)
    expected_source = source_run / 'checkpoints' / f'{final_step:06d}' / 'pretrained_model'
    require(measurement['status'] == 'complete' and Path(measurement['checkpoint']) == expected_source
            and measurement['stage'] == source_exp['stage'], 'Wrong sampling measurement')
    panel = measurement['results'][str(target.get('num_steps', 10))]
    require(panel['batch_size'] == 1 and panel['seeds'] == spec['certificate_seeds'], 'Wrong measured seed/batch protocol')
    require(passes([m['overall'] for m in panel['by_seed'].values()], spec['gates'][source_exp['stage']]), 'Sampling candidate has not passed every seed')
    exp = deepcopy(source_exp)
    exp.update(version=spec['version'], profile_name=profile_name, profile=target,
               experiment_config_file=config.name, experiment_config_sha256=spec['_sha256'],
               inference={'sampler': 'native Euler', 'num_steps': target.get('num_steps', 10)},
               experiment_kind='sampling-only derivative; no new training', inference_variant_of_run=str(source_run),
               source_training_experiment=source_exp)
    receipts = []
    for step in checkpoints:
        src = source_run / 'checkpoints' / f'{step:06d}'
        model = src / 'pretrained_model'
        require(read_json(model / 'small_experiment.json') == source_exp, 'Source checkpoint experiments differ')
        require(read_json(src / 'training_state/training_step.json')['step'] == step, 'Untrained/wrong source step')
        old_lock = read_json(model / 'agri_checkpoint_lock.json')
        for name, sha in old_lock['files'].items():
            require(file_sha(model / name) == sha, f'Source checksum differs: {name}')
        cp = output / 'checkpoints' / f'{step:06d}'
        shutil.copytree(src, cp)
        native = read_json(cp / 'pretrained_model/config.json'); native['num_steps'] = target.get('num_steps', 10)
        train = read_json(cp / 'pretrained_model/train_config.json')
        train['policy']['num_steps'] = target.get('num_steps', 10); train['output_dir'] = str(output)
        for name, value in [('config.json', native), ('train_config.json', train), ('small_experiment.json', exp)]:
            (cp / 'pretrained_model' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        unchanged = {p.name: file_sha(p) for p in (src / 'training_state').iterdir() if p.is_file()}
        require(all(file_sha(cp / 'training_state' / k) == v for k,v in unchanged.items()), 'Native training state changed')
        weight_sha = file_sha(model / 'adapter_model.safetensors')
        require(file_sha(cp / 'pretrained_model/adapter_model.safetensors') == weight_sha, 'Adapter bytes changed')
        lock = deepcopy(old_lock)
        lock['files'] = {name: file_sha(cp / 'pretrained_model' / name) for name in old_lock['files']}
        (cp / 'pretrained_model/agri_checkpoint_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        item = {'step': step, 'source_checkpoint': str(src), 'checkpoint': str(cp), 'adapter_sha256': weight_sha,
                'adapter_bytes_identical': True, 'native_training_files_sha256': unchanged}
        write_json(cp / 'pretrained_model/agri_inference_variant.json', item)
        receipts.append(item)
    write_json(report_dir / 'reference-evaluation.json', {'step': final_step, 'screen': panel, 'certificate': panel})
    result = {'status': 'packaged_pending_fresh_reload', 'optimizer_steps_performed': 0,
              'experiment_kind': 'sampling-only derivative, native state inherited from real training',
              'source_run': str(source_run), 'profile': target, 'checkpoints': receipts,
              'sampling_measurement': str(reference), 'reference_evaluation': str(report_dir / 'reference-evaluation.json')}
    write_json(report_dir / 'variant_report.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', required=True)
    p.add_argument('--experiment-config', type=Path, required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source_run, a.checkpoints, a.experiment_config, a.profile, a.reference, a.output)


if __name__ == '__main__':
    main()
