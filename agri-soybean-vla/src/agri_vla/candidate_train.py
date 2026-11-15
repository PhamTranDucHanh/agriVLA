"""Prepare or run full WORDING_V1 training with a measured small-case profile.

Default: CPU config/evidence checks only. --execute is explicit; a new full run
loads adapter weights only, never tiny optimizer state. Native full-state resume
is restricted to checkpoints produced by this runner with identical settings.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from .dataset import require, read_json, new_output
from .preflight import project_root
from .training import (load_training_config, train_config, runtime, verify_train_export,
                       write_json, file_sha, attach_checkpoint_contract)
from .small_cases import load_experiment, expert_target_modules, compatible_inference_variant


def check_evidence(path, checkpoint):
    d = read_json(path)
    require(Path(d['checkpoint']).resolve() == checkpoint, 'Evidence belongs to another checkpoint')
    require(d['stage'] == 'cases60' and d['batch_size'] == 1 and d['seeds'] == [0, 1, 2, 3, 42]
            and d['records_per_seed'] == 60 and len(d['predictions']) == 300, 'Need complete TRAIN60/five-seed/B1 evidence')
    from .evaluation import metrics
    from .small_cases import passes
    require(len({(p['sample_id'], p['seed']) for p in d['predictions']}) == 300, 'Duplicate evidence predictions')
    from .small_cases import selection
    c = load_training_config(project_root() / 'configs/training_v0.yaml')
    targets = {x['source_record']['sample_id']: x['source_record']['target_action']
               for x in selection(c, 'cases60')['items']}
    require(all(p['sample_id'] in targets and p['target_action'] == targets[p['sample_id']] for p in d['predictions'])
            and all(sum(p['seed'] == s for p in d['predictions']) == 60 for s in d['seeds']), 'Evidence sample/target/seed mapping differs')
    scores = [metrics([p for p in d['predictions'] if p['seed'] == seed])['overall'] for seed in d['seeds']]
    require(passes(scores, {'exact': 1., 'zone': 1., 'stop': 1.}), 'TRAIN60 has not passed every noise seed')
    comparison = d.get('reload_comparison', d.get('comparison_with_fresh_diagnostic'))
    require(comparison and comparison['passed'], 'Evidence lacks a passing save/fresh-load comparison')
    return {'file': str(path), 'sha256': file_sha(path), 'train60_every_seed_passed': True,
            'scope': 'finite TRAIN panel only; no full-data generalization claim'}


def resolve(config, profile_name, output, *, steps=1000, init_checkpoint=None, resume_from=None):
    root = project_root()
    # HF datasets captures its cache path at import time, before runtime().
    os.environ['HF_DATASETS_CACHE'] = str(root / '.cache/hf-datasets')
    from lerobot.configs.default import PeftConfig
    from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig
    c = load_training_config(root / 'configs/training_v0.yaml')
    spec = load_experiment(config); profile = spec['profiles'][profile_name]
    require(profile.get('time_sampling', 'upstream_beta') == 'upstream_beta', 'Full runner supports native Beta-time loss only')
    require(profile['rank'] == profile['alpha'] == 8 and not profile['full_projections'], 'Deploy-compatible r8 LoRA required')
    require(type(steps) is int and 1 <= steps <= 12000, 'Use 1..12000 cumulative steps; validate between phases')
    require(bool(init_checkpoint) != bool(resume_from), 'Choose weights-only initialization or full-state resume')
    output = Path(output).resolve()
    require(output.is_relative_to(root / 'outputs/train'), 'Full-run output must be a NEW outputs/train directory')
    new_output(output)
    cfg = train_config(c, root / c['bundle'], output=output, steps=steps, device='cuda', episodes=None)
    cfg.use_policy_training_preset = False
    cfg.batch_size = profile.get('batch_size', 1); cfg.num_workers = 0
    cfg.policy.num_steps = profile.get('num_steps', 10)
    cfg.optimizer.lr = profile['lr']; cfg.policy.optimizer_lr = profile['lr']
    cfg.scheduler = ConstantWithWarmupSchedulerConfig(num_warmup_steps=spec['warmup_steps'])
    cfg.peft = PeftConfig(method_type='LORA', r=profile['rank'], lora_alpha=profile['alpha'], full_training_modules=[])
    cfg.peft.target_modules = expert_target_modules(profile)
    cfg.log_freq = 20; cfg.save_checkpoint = True; cfg.save_freq = min(500, steps)
    start = 0
    if resume_from:
        from lerobot.configs.train import TrainPipelineConfig
        resume_from = Path(resume_from).resolve()
        meta = read_json(resume_from / 'pretrained_model/agri_full_candidate.json')
        require(meta['profile'] == profile and meta['config_sha256'] == spec['_sha256'], 'Full resume profile drift')
        step_info = read_json(resume_from / 'training_state/training_step.json'); start = step_info['step']
        require(type(start) is int and 0 < start < steps and step_info['batch_size'] == cfg.batch_size
                and step_info['num_processes'] == 1, 'Invalid resume step/batch/world size')
        cfg = TrainPipelineConfig.from_pretrained(resume_from / 'pretrained_model', local_files_only=True, cli_args=[])
        require(cfg.dataset.episodes is None and cfg.num_workers == 0 and not cfg.dataset.streaming
                and not cfg.dataset.image_transforms.enable and cfg.scheduler.type == 'constant_with_warmup', 'Resume loader differs')
        require(Path(cfg.dataset.root).resolve() == (root / c['dataset_root'] / 'train').resolve()
                and cfg.batch_size == profile.get('batch_size', 1)
                and cfg.policy.num_steps == profile.get('num_steps', 10)
                and cfg.optimizer.lr == profile['lr'], 'Full resume dataset/batch/policy/optimizer drift')
        cfg.output_dir = output; cfg.steps = steps; cfg.resume = True
        model = resume_from / 'pretrained_model'
    else:
        model = Path(init_checkpoint).resolve()
        meta = read_json(model / 'small_experiment.json')
        require(compatible_inference_variant(meta['profile'], profile)
                and meta['profile'].get('num_steps', 10) == profile.get('num_steps', 10), 'Initialization differs from measured profile')
    for name, sha in read_json(model / 'agri_checkpoint_lock.json')['files'].items():
        require(file_sha(model / name) == sha, f'Initialization checksum differs: {name}')
    from .evaluation import check_bundle
    bundle = root / c['bundle']; check_bundle(c, bundle); verify_train_export(c, bundle)
    require(read_json(model / 'agri_provenance.json') == read_json(bundle / 'provenance.json')
            and read_json(model / 'agri_schema.json') == read_json(bundle / 'schema.json')
            and read_json(model / 'agri_train_stats.json') == read_json(bundle / 'train_stats.json'), 'Dataset/schema/full-train stats differ')
    return c, spec, profile, cfg, model, start


def run(args):
    c, spec, profile, cfg, model, start = resolve(args.experiment_config, args.profile, args.output,
                                               steps=args.steps, init_checkpoint=args.init_checkpoint, resume_from=args.resume_from)
    # Dry run checks native config and local contracts without loading model/CUDA.
    runtime(c, device='cuda' if args.execute else 'cpu')
    if args.resume_from:
        meta = read_json(model / 'agri_full_candidate.json')
        evidence = meta['small_case_evidence']
        require(file_sha(Path(evidence['file'])) == evidence['sha256'], 'Original small-case evidence changed')
        evidence = check_evidence(Path(evidence['file']), Path(read_json(evidence['file'])['checkpoint']).resolve())
    else:
        evidence = check_evidence(args.evidence, model) if args.evidence else None
    plan = {'status': 'dry_run', 'scope': 'full TRAIN3600 only; validation is a separate process between phases',
            'profile': profile, 'config_file': Path(args.experiment_config).name, 'config_sha256': spec['_sha256'],
            'initialization': {'native_full_state_resume' if args.resume_from else 'weights_only_optimizer_reset': str(model)},
            'small_case_evidence': evidence, 'resolved_config': cfg.to_dict(), 'start_step': start,
            'full_execution_gate_passed': evidence is not None,
            'final_step': args.steps, 'planned_new_optimizer_updates': args.steps - start,
            'estimated_train_epochs_this_phase': (args.steps - start) * cfg.batch_size / 3600,
            'optimizer_steps_performed': 0, 'runner_sha256': file_sha(Path(__file__))}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True); return plan
    require(evidence is not None, '--execute requires passing TRAIN60 + fresh-reload evidence; inspect small-case reports first')
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs
    from lerobot.scripts import lerobot_train as upstream
    from lerobot.common.train_utils import save_training_state
    from .small_train import load_checkpoint, parameter_audit
    from .small_resume import continued_cycle, compare_tensor_files
    import torch
    report_dir = project_root() / 'reports/full-candidate' / cfg.output_dir.name
    new_output(report_dir); report_dir.mkdir(parents=True)
    shutil.copyfile(__file__, report_dir / 'runner-source.py')
    write_json(report_dir / 'plan.json', plan)
    originals = {n: getattr(upstream, n) for n in ('make_policy', 'make_pre_post_processors', 'make_train_eval_datasets',
                                                  'update_policy', 'save_checkpoint', 'load_training_state', 'cycle')}
    state = {'step': start, 'train_peak_allocated': 0, 'train_peak_reserved': 0, 'update_seconds_sum': 0.}
    initial_weights = {}
    telemetry, stop_poll = [], threading.Event()
    def poll_gpu():
        while not stop_poll.is_set():
            try:
                r = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    used, total = map(int, r.stdout.strip().splitlines()[0].split(','))
                    telemetry.append({'time': time.time(), 'used_mib': used, 'total_mib': total})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop_poll.wait(1)
    thread = threading.Thread(target=poll_gpu, daemon=True)

    def make_policy(*a, **kw):
        policy, state['pre'], state['post'] = load_checkpoint(c, model)
        policy.config.use_peft = True
        write_json(report_dir / 'parameters.json', parameter_audit(policy))
        initial_weights.update({n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad})
        return policy

    def make_data(*a, **kw):
        ds, ev = originals['make_train_eval_datasets'](*a, **kw)
        require(len(ds) == 3600 and ds.episodes is None and ev is None, 'Wrong full TRAIN dataset or implicit split')
        return ds, ev

    def save(*a, **kw):
        weights = {n: p for n, p in kw['policy'].named_parameters() if p.requires_grad}
        require(all(bool(torch.isfinite(p).all()) for p in weights.values()), 'Nonfinite weights at save')
        delta = sum(float((p.detach().cpu() - initial_weights[n]).square().sum()) for n, p in weights.items())
        require(delta > 0, 'Adapter weights did not change')
        state['adapter_weight_squared_delta'] = delta
        result = originals['save_checkpoint'](*a, **kw)
        cp = Path(kw['checkpoint_dir']); folder = cp / 'pretrained_model'
        write_json(cp / 'training_state/amp_scaler_state.json', accelerator.scaler.state_dict())
        write_json(folder / 'agri_full_candidate.json', {k: plan[k] for k in ('profile', 'config_file', 'config_sha256',
                                                                          'small_case_evidence', 'initialization', 'runner_sha256')})
        attach_checkpoint_contract(project_root() / c['bundle'], folder)
        return result

    def restore(*a, **kw):
        step, opt, sched = originals['load_training_state'](*a, **kw)
        require(step == start, 'Native restored step differs')
        receipt = cfg.output_dir / 'restored_before_update'
        save_training_state(receipt, step, opt, sched, num_processes=1, batch_size=cfg.batch_size)
        checks = {}
        for name in ('optimizer_state.safetensors', 'rng_state.safetensors'):
            checks[name] = compare_tensor_files(Path(args.resume_from) / 'training_state' / name,
                                                receipt / 'training_state' / name)
            require(checks[name]['exact'], f'Native restore mismatch: {name}')
        for name in ('scheduler_state.json', 'training_step.json'):
            checks[name] = read_json(Path(args.resume_from) / 'training_state' / name) == read_json(receipt / 'training_state' / name)
            require(checks[name], f'Native restore mismatch: {name}')
        checks['amp_scaler_exact'] = accelerator.scaler.state_dict() == read_json(Path(args.resume_from) / 'training_state/amp_scaler_state.json')
        require(checks['amp_scaler_exact'], 'AMP resume differs')
        write_json(report_dir / 'restored-state.json', checks)
        return step, opt, sched

    def cycle(loader):
        yield from continued_cycle(loader, originals['cycle'], step=start, records=3600, batch_size=cfg.batch_size)

    def update(*a, **kw):
        torch.cuda.reset_peak_memory_stats()
        updated = originals['update_policy'](*a, **kw); tracker = updated[0]
        require(math.isfinite(tracker.loss.val) and math.isfinite(tracker.grad_norm.val)
                and not getattr(a[3], 'step_was_skipped', False), 'Nonfinite/skipped full-train update')
        state['step'] += 1
        state['update_seconds_sum'] += float(tracker.update_s.val)
        if state['step'] == start + 1:
            batch = a[2]
            write_json(report_dir / 'first-batch.json', {
                'shapes': {k: list(v.shape) for k,v in batch.items() if isinstance(v, torch.Tensor)},
                'actual_token_counts': batch['observation.language.attention_mask'].sum(-1).detach().cpu().tolist(),
                'episodes': batch['episode_index'].detach().cpu().tolist()})
        state['train_peak_allocated'] = max(state['train_peak_allocated'], torch.cuda.max_memory_allocated())
        state['train_peak_reserved'] = max(state['train_peak_reserved'], torch.cuda.max_memory_reserved())
        with (report_dir / 'updates.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state['step'], 'loss': tracker.loss.val, 'grad_norm': tracker.grad_norm.val,
                                     'lr': tracker.lr.val, 'seconds': tracker.update_s.val}) + '\n')
        return updated

    previous = sys.argv; failure = None; wall_start = time.perf_counter()
    try:
        sys.argv = ['agri-full-candidate']
        if args.resume_from:
            sys.argv.append(f'--config_path={model / "train_config.json"}')
            upstream.load_training_state = restore; upstream.cycle = cycle
        cfg.validate()
        upstream.make_policy = make_policy
        upstream.make_pre_post_processors = lambda *a, **kw: (state['pre'], state['post'])
        upstream.make_train_eval_datasets = make_data
        upstream.save_checkpoint = save; upstream.update_policy = update
        accelerator = Accelerator(step_scheduler_with_optimizer=False, mixed_precision='fp16',
                                  kwargs_handlers=[GradScalerKwargs(init_scale=1024., growth_interval=2000)])
        require(accelerator.device.type == 'cuda' and accelerator.scaler is not None
                and accelerator.num_processes == 1, 'Single-GPU FP16 AMP required')
        if args.resume_from:
            accelerator.scaler.load_state_dict(read_json(Path(args.resume_from) / 'training_state/amp_scaler_state.json'))
        thread.start()
        upstream.train(cfg, accelerator=accelerator)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        (report_dir / 'failure.txt').write_text(traceback.format_exc())
    finally:
        stop_poll.set()
        if thread.is_alive():
            thread.join(timeout=6)
        sys.argv = previous
        for name, fn in originals.items():
            setattr(upstream, name, fn)
    process_memory = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'VmSwap'):
            process_memory[key + '_bytes'] = int(value.strip().split()[0]) * 1024
    result = {**state, 'status': 'error' if failure else 'completed', 'failure': failure,
              'optimizer_steps_performed': state['step'] - start, 'validation_selection_performed': False,
              'wall_seconds_including_loading_and_saves': time.perf_counter() - wall_start,
              'process_memory': process_memory, 'gpu_samples': telemetry,
              'nvml_peak_mib': max((x['used_mib'] for x in telemetry), default=None),
              'torch_peak_scope': 'update-only; reset before every forward/backward/update'}
    write_json(report_dir / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    require(failure is None, failure)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-config', type=Path, required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--init-checkpoint', type=Path)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
