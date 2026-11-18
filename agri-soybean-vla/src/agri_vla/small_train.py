"""Small, versioned SmolVLA experiments using the pinned upstream training loop.

Existing dataset/training_v0/checkpoints/loaders are never changed. Evaluation is
train-only and preserves RNG. A passing result is a finite-panel certificate,
not an estimate of held-out quality.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import traceback

from .dataset import require, new_output, read_json
from .preflight import project_root
from .small_cases import load_experiment, selection, passes, compare_predictions, compatible_warm_start, expert_target_modules, PROJECTIONS, STAGES
from .training import (load_training_config, runtime, train_config, write_json, file_sha,
                       verify_train_export, attach_checkpoint_contract, CAMERAS)


def rng_fingerprint():
    import hashlib
    import numpy as np
    import torch
    digest = lambda b: hashlib.sha256(b).hexdigest()
    ns = np.random.get_state()
    return {'torch_cpu': digest(torch.get_rng_state().numpy().tobytes()),
            'torch_cuda': digest(torch.cuda.get_rng_state().cpu().numpy().tobytes()),
            'python': digest(repr(random.getstate()).encode()),
            'numpy': digest(ns[1].tobytes() + repr((ns[0], *ns[2:])).encode())}


def parameter_audit(policy):
    result = []
    for name, p in policy.named_parameters():
        if p.requires_grad:
            require('.vlm.' not in name, f'Unexpected VLM training: {name}')
            allowed = '.lora_A.' in name or '.lora_B.' in name or any(
                f'.{module}.modules_to_save.default.' in name for module in PROJECTIONS)
            require(allowed, f'Unexpected trainable weight: {name}')
            require(str(p.dtype) == 'torch.float32', 'Trainable master weights must remain FP32')
            result.append({'name': name, 'parameters': p.numel(), 'dtype': str(p.dtype)})
    require(result and sum(x['parameters'] for x in result) < 10_000_000, 'Unexpected full fine-tuning')
    return {'parameters': result, 'trainable': sum(x['parameters'] for x in result), 'vlm_frozen': True}


def configure_training_time(policy, profile):
    """Endpoint experiment uses native sample_time; stock Beta profiles stay intact."""
    mode = profile.get('time_sampling', 'upstream_beta')
    audit = {'mode': mode}
    if mode == 'upstream_beta':
        return audit
    require(mode == 'endpoint_t1' and policy.config.num_steps == 1, 'Unsupported timestep/inference combination')
    import torch
    core = policy.get_base_model().model
    audit.update(calls=0, samples=0, minimum=None, maximum=None)
    def sample_time(batch_size, device):
        time = torch.ones((batch_size,), dtype=torch.float32, device=device)
        audit['calls'] += 1; audit['samples'] += batch_size
        lo, hi = float(time.min()), float(time.max())
        audit['minimum'] = lo if audit['minimum'] is None else min(audit['minimum'], lo)
        audit['maximum'] = hi if audit['maximum'] is None else max(audit['maximum'], hi)
        return time
    core.sample_time = sample_time
    core._agri_time_sampling_audit = audit
    return audit


def panel_evaluate(policy, pre, post, dataset, selected, seeds, *, batch_size=4):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .evaluation import model_batch, quantize_action, metrics
    records = {x['episode_index']: x['source_record'] for x in selected['items']}
    was_training = policy.training
    np_state, py_state = np.random.get_state(), random.getstate()
    by_seed, predictions = {}, []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            policy.eval()
            for seed in seeds:
                rows = []
                for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
                    for cam in policy.config.image_features:
                        if batch[cam].dtype == torch.uint8:
                            batch[cam] = batch[cam].float() / 255.
                    policy.reset(); pre.reset(); post.reset()
                    inputs = pre(model_batch(batch))
                    bsize = len(batch['task'])
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    # Match B1 upstream sample_noise, using the same noise for every
                    # independent sample and its translation at this evaluation seed.
                    noise = torch.normal(0., 1., size=(1, 1, 32), device='cuda').repeat(bsize, 1, 1)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        normalized = policy.predict_action_chunk(inputs, noise=noise)[:, 0, :]
                        action = post(normalized).float().cpu()
                    require(tuple(action.shape) == (bsize, 3) and bool(torch.isfinite(action).all()), 'Bad sampled action')
                    for i in range(bsize):
                        episode = int(batch['episode_index'][i])
                        r = records[episode]
                        require(batch['task'][i] == r['instruction'], 'Evaluation task/sample mapping differs')
                        rows.append({'sample_id': r['sample_id'], 'language': r['language'], 'intent': r['intent'],
                                     'wording_partition': r['wording_partition'], 'target_action': r['target_action'],
                                     'seed': seed, **quantize_action(action[i].tolist())})
                require(len(rows) == selected['records'], 'Missing panel records')
                by_seed[str(seed)] = metrics(rows)
                predictions.extend(rows)
    finally:
        policy.train(was_training)
        np.random.set_state(np_state); random.setstate(py_state)
    return {'batch_size': batch_size, 'by_seed': by_seed, 'predictions': predictions,
            'records_per_seed': selected['records'], 'seeds': list(seeds)}


class PanelPassed(Exception):
    pass


def run(stage, profile_name, output, *, steps=None, init_checkpoint=None, resume_checkpoint=None, experiment_config=None, evaluate_stage=None):
    run_wall_start = time.perf_counter()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs
    from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig
    from lerobot.configs.default import PeftConfig
    from lerobot.scripts import lerobot_train as upstream
    from .evaluation import check_bundle

    root = project_root()
    c = load_training_config(root / 'configs/training_v0.yaml')
    spec_path = Path(experiment_config).resolve() if experiment_config else root / 'configs/small_cases_v1.yaml'
    require(spec_path.parent == root / 'configs', 'Experimental config must live in project configs/')
    spec = load_experiment(spec_path)
    require(profile_name in spec['profiles'], 'Unknown experimental profile')
    profile = spec['profiles'][profile_name]
    require(not (init_checkpoint and resume_checkpoint), 'Choose weights-only initialization or native full-state resume')
    selected = selection(c, stage)
    augmentation_records = {x['episode_index']: x['source_record'] for x in selected['items']}
    eval_stage=evaluate_stage or stage
    eval_selected=selection(c,eval_stage)
    require(set(eval_selected['episodes'])<=set(selected['episodes']), 'Evaluation panel must nest in training panel')
    bundle = root / c['bundle']
    check_bundle(c, bundle); verify_train_export(c, bundle)
    runtime(c, device='cuda')
    output = Path(output).resolve(); new_output(output)
    require(output.is_relative_to(root / 'outputs/small-cases'), 'Use outputs/small-cases for experimental runs')
    report_dir = root / 'reports/small-cases' / output.name
    new_output(report_dir); report_dir.mkdir(parents=True)
    import shutil
    shutil.copyfile(__file__, report_dir / 'runner-source.py')
    max_steps = spec['max_steps'] if steps is None else steps
    require(type(max_steps) is int and 1 <= max_steps <= spec['max_steps'], 'Diagnostic budget exceeds 1000')
    cfg = train_config(c, bundle, output=output, steps=max_steps, device='cuda', episodes=selected['episodes'])
    cfg.use_policy_training_preset = False
    cfg.batch_size = profile.get('batch_size', 1)
    cfg.policy.num_steps = profile.get('num_steps', 10)
    cfg.optimizer.lr = profile['lr']; cfg.policy.optimizer_lr = profile['lr']
    cfg.scheduler = ConstantWithWarmupSchedulerConfig(num_warmup_steps=spec['warmup_steps'])
    cfg.peft = PeftConfig(method_type='LORA', r=profile['rank'], lora_alpha=profile['alpha'],
                          full_training_modules=PROJECTIONS if profile['full_projections'] else [])
    cfg.peft.target_modules = expert_target_modules(profile)
    cfg.log_freq = 20
    # Milestone save uses native save_checkpoint after evaluation and before an
    # optional early exit. Avoid a second automatic save to the same directory.
    cfg.save_checkpoint = False
    start_step, resume_exp, resume_reference = 0, None, None
    if resume_checkpoint:
        from lerobot.configs.train import TrainPipelineConfig
        resume_checkpoint = Path(resume_checkpoint).resolve()
        resume_exp = read_json(resume_checkpoint / 'pretrained_model/small_experiment.json')
        require(resume_exp['profile'] == profile and resume_exp['stage'] == stage
                and resume_exp['episodes'] == selected['episodes']
                and resume_exp['experiment_config_sha256'] == spec['_sha256'], 'Resume experiment differs')
        step_info = read_json(resume_checkpoint / 'training_state/training_step.json')
        start_step = step_info['step']
        require(type(start_step) is int and 0 < start_step < max_steps, 'Invalid trained resume step/budget')
        cfg = TrainPipelineConfig.from_pretrained(resume_checkpoint / 'pretrained_model', local_files_only=True, cli_args=[])
        require(cfg.batch_size == profile.get('batch_size', 1) == step_info['batch_size']
                and step_info['num_processes'] == 1 and cfg.num_workers == 0
                and not cfg.dataset.streaming and not cfg.dataset.image_transforms.enable
                and cfg.scheduler.type == 'constant_with_warmup', 'Unsupported resume loader/schedule/world size')
        cfg.output_dir = output; cfg.steps = max_steps; cfg.resume = True; cfg.save_checkpoint = False
        source_rows = root / 'reports/small-cases' / resume_checkpoint.parents[1].name / 'updates.jsonl'
        if source_rows.exists():
            resume_reference = next((json.loads(line) for line in source_rows.read_text().splitlines()
                                     if json.loads(line)['step'] == start_step + 1), None)
    experiment = {'version': spec['version'], 'profile_name': profile_name, 'profile': profile,
                  'stage': stage, 'evaluation_stage':eval_stage, 'selection_sha256': selected['source_metadata_sha256'],
                  'episodes': selected['episodes'], 'coverage': selected['coverage'],
                  'source_config_sha256': c['_sha256'], 'experiment_config_sha256': spec['_sha256'],
                  'experiment_config_file': spec_path.name, 'actual_batch_size': cfg.batch_size,
                  'runner_sha256': file_sha(report_dir / 'runner-source.py'),
                  'schedule': {'type': 'constant_with_warmup', 'warmup_steps': spec['warmup_steps']},
                  'inference': {'sampler': 'native Euler', 'num_steps': cfg.policy.num_steps},
                  'training_time_sampling': profile.get('time_sampling', 'upstream_beta'),
                  'peft_target_modules': cfg.peft.target_modules,
                  'gate': spec['gates'][eval_stage], 'screen_seeds': spec['screen_seeds'],
                  'certificate_seeds': spec['certificate_seeds'], 'max_steps': max_steps,
                  'initialization': 'fresh LoRA from pinned base', 'test_or_validation_used': False,
                  'amp': {'init_scale': 1024., 'growth_interval': 2000}, 'seed': c['seed']}
    experiment['memory_measurement'] = 'torch peak reset before each update; eval peaks reported separately; NVML entire run'
    if profile.get('training_augmentation'):
        from . import augmentation
        experiment['augmentation_source_sha256'] = file_sha(Path(augmentation.__file__))
    if resume_checkpoint:
        experiment['initialization'] = {'native_full_state_resume': str(resume_checkpoint),
                                        'start_step': start_step, 'optimizer_reset': False,
                                        'source_training_experiment': resume_exp}
    if init_checkpoint:
        init_checkpoint = Path(init_checkpoint).resolve()
        old = read_json(init_checkpoint / 'small_experiment.json')
        require(compatible_warm_start(old['profile'], profile), 'Warm-start architecture differs')
        require(read_json(init_checkpoint / 'agri_provenance.json') == read_json(bundle / 'provenance.json'), 'Warm-start base contract differs')
        require(read_json(init_checkpoint / 'agri_schema.json') == read_json(bundle / 'schema.json'), 'Warm-start schema differs')
        require(read_json(init_checkpoint / 'agri_train_stats.json') == read_json(bundle / 'train_stats.json'), 'Warm-start statistics differ')
        for name, sha in read_json(init_checkpoint / 'agri_checkpoint_lock.json')['files'].items():
            require(file_sha(init_checkpoint / name) == sha, 'Warm-start checksum differs')
        experiment['initialization'] = {'weights_only_curriculum': str(init_checkpoint),
                                        'adapter_sha256': file_sha(init_checkpoint / 'adapter_model.safetensors'),
                                        'source_profile': old['profile'], 'source_version': old['version'],
                                        'optimizer_reset': True}
    write_json(report_dir / 'experiment.json', experiment)
    write_json(report_dir / 'selection.json', selected)
    write_json(report_dir / 'resolved_config.json', cfg.to_dict())

    original = {name: getattr(upstream, name) for name in ('make_train_eval_datasets', 'make_pre_post_processors',
                                                         'make_policy', 'update_policy', 'save_checkpoint', 'load_training_state', 'cycle')}
    state = {'step': start_step, 'dataset': None, 'pre': None, 'post': None, 'initial': None,
             'gradient_audits': [], 'evaluations': [], 'checkpoints': [], 'train_peak_allocated': 0,
             'train_peak_reserved': 0, 'update_seconds': [], 'data_seconds_sum': 0., 'passed': False,
             'time_sampling_audit': None, 'scope_expansion': None}
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

    def make_data(*args, **kwargs):
        ds, ev = original['make_train_eval_datasets'](*args, **kwargs)
        require(ev is None and len(ds) == selected['records'], 'Wrong training subset')
        from torch.utils.data import Subset
        wanted=set(eval_selected['episodes'])
        state['dataset'] = Subset(ds,[i for i,ep in enumerate(selected['episodes']) if ep in wanted])
        return ds, ev

    def make_processors(*args, **kwargs):
        if resume_checkpoint:
            return state['pre'], state['post']
        pre, post = original['make_pre_post_processors'](*args, **kwargs)
        state['pre'], state['post'] = pre, post
        return pre, post

    def make_policy(*args, **kwargs):
        if resume_checkpoint:
            policy, state['pre'], state['post'] = load_checkpoint(c, resume_checkpoint / 'pretrained_model')
            policy.config.use_peft = True
            return policy
        base = original['make_policy'](*args, **kwargs)
        if init_checkpoint:
            from peft import PeftModel
            if old['profile'].get('expert_targets', 'qv') != profile.get('expert_targets', 'qv'):
                from dataclasses import asdict
                from peft import set_peft_model_state_dict, get_peft_model_state_dict
                from safetensors.torch import load_file
                expanded = base.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
                saved = load_file(init_checkpoint / 'adapter_model.safetensors')
                loaded = set_peft_model_state_dict(expanded, saved, adapter_name='default')
                require(not loaded.unexpected_keys, 'Unexpected adapter keys during scope expansion')
                actual = get_peft_model_state_dict(expanded)
                require(saved.keys() <= actual.keys() and all(torch.equal(actual[k].detach().cpu(), v) for k, v in saved.items()),
                        'Existing adapter weights changed during expansion')
                extra = set(actual) - set(saved)
                require(extra and all('.lm_expert.' in k and ('.lora_A.' in k or '.lora_B.' in k) for k in extra), 'Unexpected new trainable modules')
                new_b = [k for k in extra if '.lora_B.' in k]
                require(new_b and all(bool((actual[k] == 0).all()) for k in new_b), 'New LoRA B must be zero for function-preserving expansion')
                state['scope_expansion'] = {'copied_tensors': len(saved), 'added_tensors': len(extra),
                                            'new_B_zero_tensors': len(new_b), 'existing_weights_exact': True,
                                            'initial_function': 'same frozen base + copied old LoRA + zero new LoRA contributions'}
                write_json(report_dir / 'scope-expansion.json', state['scope_expansion'])
                return expanded
            return PeftModel.from_pretrained(base, init_checkpoint, is_trainable=True)
        return base

    def restore_training_state(*args, **kwargs):
        from lerobot.common.train_utils import save_training_state
        from .small_resume import compare_tensor_files
        step, optimizer, scheduler = original['load_training_state'](*args, **kwargs)
        require(step == start_step, 'Native restored step differs')
        receipt = output / 'restored_before_update'
        save_training_state(receipt, step, optimizer, scheduler, num_processes=1, batch_size=cfg.batch_size)
        checks = {}
        for name in ('optimizer_state.safetensors', 'rng_state.safetensors'):
            checks[name] = compare_tensor_files(resume_checkpoint / 'training_state' / name,
                                                receipt / 'training_state' / name)
            require(checks[name]['exact'], f'Native restore differs: {name}')
        for name in ('scheduler_state.json', 'training_step.json'):
            checks[name] = read_json(resume_checkpoint / 'training_state' / name) == read_json(receipt / 'training_state' / name)
            require(checks[name], f'Native restore differs: {name}')
        checks['amp_scaler_exact'] = accelerator.scaler.state_dict() == read_json(resume_checkpoint / 'training_state/amp_scaler_state.json')
        require(checks['amp_scaler_exact'], 'AMP restore differs')
        write_json(report_dir / 'restored-state.json', checks)
        return step, optimizer, scheduler

    def continued_iterator(loader):
        from .small_resume import continued_cycle
        yield from continued_cycle(loader, original['cycle'], step=start_step,
                                   records=selected['records'], batch_size=cfg.batch_size)

    def save(step, policy, optimizer, scheduler):
        cp = output / 'checkpoints' / f'{step:06d}'
        original['save_checkpoint'](checkpoint_dir=cp, step=step, cfg=cfg, policy=policy,
                                    optimizer=optimizer, scheduler=scheduler, preprocessor=state['pre'],
                                    postprocessor=state['post'], num_processes=1, batch_size=cfg.batch_size)
        # Native API stores optimizer/scheduler/RNG/step, but not GradScaler.
        # Keep its growth tracker too, so a later resume can restore AMP exactly.
        write_json(cp / 'training_state/amp_scaler_state.json', accelerator.scaler.state_dict())
        attach_checkpoint_contract(bundle, cp / 'pretrained_model')
        write_json(cp / 'pretrained_model/small_experiment.json', experiment)
        state['checkpoints'].append(str(cp))
        return cp

    def update(*args, **kwargs):
        policy = args[1] if len(args) > 1 else kwargs['policy']
        optimizer = args[3] if len(args) > 3 else kwargs['optimizer']
        accelerator = kwargs['accelerator']
        unwrapped = accelerator.unwrap_model(policy)
        trainable = {n: p for n, p in unwrapped.named_parameters() if p.requires_grad}
        if state['initial'] is None:
            write_json(report_dir / 'parameters.json', parameter_audit(unwrapped))
            state['time_sampling_audit'] = configure_training_time(unwrapped, profile)
            state['initial'] = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        probe_step = state['step'] + 1
        batch = args[2] if len(args) > 2 else kwargs['batch']
        episodes = batch['episode_index'].detach().cpu().tolist()
        before_rng = rng_fingerprint() if probe_step == start_step + 1 or probe_step in (1, 99, 100, 101, 300, 301, 600, 601, 1000) else None
        opt = getattr(optimizer, 'optimizer', optimizer)
        hook = None
        if probe_step in (1, 100, 300, 600, 1000):
            def before_step(_opt, _args, _kwargs):
                groups = defaultdict(float)
                for name, p in trainable.items():
                    if p.grad is not None:
                        require(bool(torch.isfinite(p.grad).all()), f'Nonfinite gradient {name}')
                        family = next((s for s in PROJECTIONS + ['state_proj', 'lm_expert'] if f'.{s}.' in name), 'other')
                        groups[family] += float(p.grad.detach().float().square().sum())
                state['gradient_audits'].append({'step': probe_step, 'squared_norm_by_family': dict(groups)})
                with (report_dir / 'gradient-audits.jsonl').open('a') as stream:
                    stream.write(json.dumps(state['gradient_audits'][-1]) + '\n')
            hook = opt.register_step_pre_hook(before_step)
        try:
            torch.cuda.reset_peak_memory_stats()
            from .augmentation import mix_override_images
            mixed = mix_override_images(batch, augmentation_records, step=probe_step, profile=profile)
            if mixed:
                with (report_dir / 'augmentation.jsonl').open('a') as stream:
                    stream.write(json.dumps(mixed) + '\n')
            updated = original['update_policy'](*args, **kwargs)
        finally:
            if hook is not None:
                hook.remove()
        tracker = updated[0]
        import math
        require(math.isfinite(tracker.loss.val) and math.isfinite(tracker.grad_norm.val), 'Nonfinite training')
        require(not getattr(optimizer, 'step_was_skipped', False), 'AMP skipped an optimizer update')
        if resume_checkpoint and probe_step == start_step + 1 and resume_reference is not None:
            comparison = {'step': probe_step, 'sample_order_equal': episodes == resume_reference['episodes'],
                          'rng_before_equal': before_rng == resume_reference['rng_before'],
                          'reference_loss': resume_reference['loss'], 'actual_loss': tracker.loss.val,
                          'loss_close': math.isclose(tracker.loss.val, resume_reference['loss'], abs_tol=1e-5, rel_tol=1e-4)}
            write_json(report_dir / 'first-resumed-update.json', comparison)
            require(all(comparison[k] for k in ('sample_order_equal', 'rng_before_equal', 'loss_close')),
                    'First resumed update differs from recorded uninterrupted source')
        state['step'] += 1
        state['train_peak_allocated'] = max(state['train_peak_allocated'], torch.cuda.max_memory_allocated())
        state['train_peak_reserved'] = max(state['train_peak_reserved'], torch.cuda.max_memory_reserved())
        state['update_seconds'].append(float(tracker.update_s.val))
        state['data_seconds_sum'] += float(tracker.dataloading_s.val)
        with (report_dir / 'updates.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state['step'], 'loss': tracker.loss.val, 'grad_norm': tracker.grad_norm.val,
                                     'lr': tracker.lr.val, 'seconds': tracker.update_s.val,
                                     'episodes': episodes, 'rng_before': before_rng}) + '\n')
        if state['step'] in spec['evaluate_at'] or state['step'] == max_steps:
            torch.cuda.reset_peak_memory_stats()
            screen = panel_evaluate(unwrapped, state['pre'], state['post'], state['dataset'], eval_selected,
                                    spec['screen_seeds'], batch_size=4)
            screen_scores = [x['overall'] for x in screen['by_seed'].values()]
            event = {'step': state['step'], 'screen': screen,
                     'screen_peak_allocated': torch.cuda.max_memory_allocated(),
                     'screen_peak_reserved': torch.cuda.max_memory_reserved()}
            print('SMALL_SCREEN', state['step'], {k: v['overall'] for k, v in screen['by_seed'].items()}, flush=True)
            if passes(screen_scores, spec['gates'][eval_stage]):
                torch.cuda.reset_peak_memory_stats()
                certificate = panel_evaluate(unwrapped, state['pre'], state['post'], state['dataset'], eval_selected,
                                             spec['certificate_seeds'], batch_size=1)
                event['certificate'] = certificate
                event['certificate_peak_allocated'] = torch.cuda.max_memory_allocated()
                event['certificate_peak_reserved'] = torch.cuda.max_memory_reserved()
                state['passed'] = passes([x['overall'] for x in certificate['by_seed'].values()], spec['gates'][eval_stage])
                print('SMALL_CERTIFICATE', state['step'], state['passed'], flush=True)
            delta = {n: float((p.detach().cpu() - state['initial'][n]).square().sum()) for n, p in trainable.items()}
            require(all(bool(torch.isfinite(p).all()) for p in trainable.values()), 'Nonfinite weights')
            require(sum(delta.values()) > 0, 'Adapter did not update')
            event['weight_squared_delta'] = delta
            cp = save(state['step'], unwrapped, optimizer, kwargs.get('lr_scheduler'))
            event['checkpoint'] = str(cp)
            write_json(report_dir / f'evaluation-{state["step"]:06d}.json', event)
            state['evaluations'].append({'step': state['step'], 'passed': state['passed'], 'checkpoint': str(cp)})
            if state['passed']:
                raise PanelPassed()
        elif state['step'] == 99:
            # Adjacent real checkpoints99/100 support a one-update resume proof
            # against the uninterrupted run, even if the panel passes at100.
            save(state['step'], unwrapped, optimizer, kwargs.get('lr_scheduler'))
        return updated

    previous_argv = sys.argv
    failure = None
    try:
        sys.argv = ['agri-vla-small-experiment']
        if resume_checkpoint:
            sys.argv.append(f'--config_path={resume_checkpoint / "pretrained_model/train_config.json"}')
            upstream.load_training_state = restore_training_state
            upstream.cycle = continued_iterator
        upstream.make_train_eval_datasets = make_data
        upstream.make_pre_post_processors = make_processors
        upstream.make_policy = make_policy
        upstream.update_policy = update
        cfg.validate()
        thread.start()
        accelerator = Accelerator(step_scheduler_with_optimizer=False, mixed_precision='fp16',
                                  kwargs_handlers=[GradScalerKwargs(init_scale=1024., growth_interval=2000)])
        require(accelerator.scaler is not None and accelerator.device.type == 'cuda', 'CUDA AMP required')
        if resume_checkpoint:
            accelerator.scaler.load_state_dict(read_json(resume_checkpoint / 'training_state/amp_scaler_state.json'))
        upstream.train(cfg, accelerator=accelerator)
    except PanelPassed:
        pass
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        (report_dir / 'failure.txt').write_text(traceback.format_exc())
    finally:
        stop_poll.set()
        if thread.is_alive():
            thread.join(timeout=6)
        for name, value in original.items():
            setattr(upstream, name, value)
        sys.argv = previous_argv
    result = {k: state[k] for k in ('step', 'gradient_audits', 'evaluations', 'checkpoints', 'train_peak_allocated',
                                   'train_peak_reserved', 'update_seconds', 'passed', 'time_sampling_audit', 'scope_expansion')}
    result |= {'status': 'error' if failure else 'passed' if state['passed'] else 'gate_not_met', 'failure': failure,
               'experiment': experiment, 'gpu_samples': telemetry, 'scope': 'TRAIN diagnostic only',
               'report_dir': str(report_dir), 'output': str(output),
               'start_step': start_step, 'optimizer_steps_performed': state['step'] - start_step}
    result['data_seconds_sum'] = state['data_seconds_sum']
    result['wall_seconds_including_setup_screens_saves'] = time.perf_counter() - run_wall_start
    result['process_memory'] = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'VmSwap'):
            result['process_memory'][key + '_bytes'] = int(value.strip().split()[0]) * 1024
    write_json(report_dir / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'failure', 'step', 'passed', 'checkpoints', 'report_dir')}, indent=2), flush=True)
    return result


def load_checkpoint(c, checkpoint):
    """Separate experimental loader; preserves strict original r8 deploy loader."""
    import torch
    from .evaluation import check_bundle, load_model
    from .training import assets, asset_path, BASE, BACKBONE, load_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftModel, PeftConfig
    # Match upstream train() numerics. The original recipe does not request
    # cudnn_deterministic; both the train-time panel and reload use this mode.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = Path(checkpoint).resolve()
    bundle = c['_base'] / c['bundle']
    if not (checkpoint / 'small_experiment.json').exists():
        return load_model(c, bundle, checkpoint=checkpoint, device='cuda')[:3]
    runtime(c, device='cuda'); assets(c); check_bundle(c, bundle)
    exp = read_json(checkpoint / 'small_experiment.json')
    config_name = exp.get('experiment_config_file', 'small_cases_v1.yaml')
    require(config_name in ('small_cases_v1.yaml', 'small_cases_v2.yaml', 'small_cases_v3.yaml', 'small_cases_v4.yaml', 'small_cases_v5.yaml', 'small_cases_v6.yaml', 'small_cases_v7.yaml', 'small_cases_v8.yaml', 'small_cases_v9.yaml', 'small_cases_v10.yaml'), 'Unknown experimental config file')
    spec = load_experiment(project_root() / 'configs' / config_name)
    if exp['profile'].get('training_augmentation'):
        from . import augmentation
        require(exp['augmentation_source_sha256'] == file_sha(Path(augmentation.__file__)), 'Augmentation source drift')
    require(exp['version'] == spec['version'] and exp['profile'] == spec['profiles'][exp['profile_name']]
            and exp['experiment_config_sha256'] == spec['_sha256'], 'Experimental profile/config drift')
    require(read_json(checkpoint / 'agri_provenance.json') == read_json(bundle / 'provenance.json'), 'Base contract differs')
    require(read_json(checkpoint / 'agri_schema.json') == read_json(bundle / 'schema.json'), 'Feature schema differs')
    require(read_json(checkpoint / 'agri_train_stats.json') == read_json(bundle / 'train_stats.json'), 'Train statistics differ')
    for name, sha in read_json(checkpoint / 'agri_checkpoint_lock.json')['files'].items():
        require(file_sha(checkpoint / name) == sha, f'Checkpoint changed: {name}')
    cfg = SmolVLAConfig.from_pretrained(checkpoint)
    require(cfg.num_steps == exp['profile'].get('num_steps', 10), 'Inference sampling steps differ from experiment')
    require(list(cfg.image_features) == list(CAMERAS), 'Wrong image inputs')
    require(cfg.action_feature.shape == (3,) and cfg.robot_state_feature.shape == (3,)
            and cfg.chunk_size == cfg.n_action_steps == 1 and cfg.max_action_dim == cfg.max_state_dim == 32
            and cfg.tokenizer_max_length == 96 and cfg.resize_imgs_with_padding == (256, 256), 'Schema/horizon drift')
    require({k: v.value for k, v in cfg.normalization_mapping.items()} ==
            {'VISUAL': 'IDENTITY', 'STATE': 'IDENTITY', 'ACTION': 'MEAN_STD'}, 'Normalization drift')
    cfg.device = 'cuda'; cfg.use_amp = True; cfg.use_peft = False
    cfg.pretrained_path = bundle / 'base_policy'; cfg.vlm_model_name = str(asset_path(c, BACKBONE))
    base = SmolVLAPolicy.from_pretrained(asset_path(c, BASE), config=cfg, local_files_only=True, strict=True)
    ac = PeftConfig.from_pretrained(checkpoint)
    if expert_target_modules(exp['profile']) is not None:
        require(ac.target_modules == expert_target_modules(exp['profile']), 'Expert target modules differ')
    require(ac.r == exp['profile']['rank'] and ac.lora_alpha == exp['profile']['alpha'], 'Wrong adapter rank')
    require(set(ac.modules_to_save or []) == set(PROJECTIONS if exp['profile']['full_projections'] else []), 'Wrong full modules')
    policy = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    parameter_audit(policy)
    configure_training_time(policy, exp['profile'])
    pre, post = load_processors(c, checkpoint, cfg, device='cuda', tokenizer_dir=checkpoint / 'tokenizer')
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    wanted = read_json(bundle / 'train_stats.json')['action']
    for pipe, kind in ((pre, NormalizerProcessorStep), (post, UnnormalizerProcessorStep)):
        normalizer = next(s for s in pipe.steps if isinstance(s, kind))
        saved = normalizer.state_dict()
        require(all(torch.allclose(saved[f'action.{k}'].cpu(), torch.tensor(wanted[k]), atol=1e-7)
                    for k in ('mean', 'std')), 'Processor statistics differ')
    policy.eval()
    from .inference_sampling import configure_inference
    configure_inference(policy,exp['profile'])
    return policy, pre, post


def evaluate_saved(stage, checkpoint, output, *, reference_evaluation=None):
    from lerobot.datasets.factory import make_train_eval_datasets
    root = project_root()
    c = load_training_config(root / 'configs/training_v0.yaml')
    contract_file = Path(checkpoint) / 'small_experiment.json'
    config_name = read_json(contract_file).get('experiment_config_file', 'small_cases_v1.yaml') if contract_file.exists() else 'small_cases_v1.yaml'
    require(config_name in ('small_cases_v1.yaml', 'small_cases_v2.yaml', 'small_cases_v3.yaml', 'small_cases_v4.yaml', 'small_cases_v5.yaml', 'small_cases_v6.yaml', 'small_cases_v7.yaml', 'small_cases_v8.yaml', 'small_cases_v9.yaml', 'small_cases_v10.yaml'), 'Unknown experimental config file')
    spec = load_experiment(root / 'configs' / config_name)
    selected = selection(c, stage)
    output = Path(output).resolve(); new_output(output)
    policy, pre, post = load_checkpoint(c, checkpoint)
    cfg = train_config(c, root / c['bundle'], output=output.parent / 'unused-no-training',
                       device='cuda', episodes=selected['episodes'])
    ds, ev = make_train_eval_datasets(cfg)
    require(ev is None and len(ds) == selected['records'], 'Wrong evaluation panel')
    result = panel_evaluate(policy, pre, post, ds, selected, spec['certificate_seeds'], batch_size=1)
    if reference_evaluation:
        reference = read_json(Path(reference_evaluation))
        panel = reference.get('certificate', reference['screen'])
        if panel['batch_size'] == 1 and panel['seeds'] == result['seeds']:
            matched = result
        else:
            matched = panel_evaluate(policy, pre, post, ds, selected, panel['seeds'], batch_size=panel['batch_size'])
        result['reload_comparison'] = compare_predictions(panel['predictions'], matched['predictions'])
        result['reload_comparison']['reference_evaluation'] = str(Path(reference_evaluation).resolve())
    gate_passed = passes([x['overall'] for x in result['by_seed'].values()], spec['gates'][stage])
    comparison_ok = result.get('reload_comparison', {}).get('passed', True)
    result |= {'status': 'reload_mismatch' if not comparison_ok else 'passed' if gate_passed else 'gate_not_met',
               'load_and_inference_verified': True, 'stage': stage, 'checkpoint': str(Path(checkpoint).resolve()),
               'adapter_sha256': file_sha(Path(checkpoint) / 'adapter_model.safetensors'),
               'gate_passed': gate_passed, 'num_steps': policy.config.num_steps,
               'scope': 'fresh-process reload, TRAIN diagnostic only', 'optimizer_steps_performed': 0}
    write_json(output, result)
    print(json.dumps({'stage': stage, 'gate_passed': result['gate_passed'],
                      'by_seed': {k: v['overall'] for k, v in result['by_seed'].items()}}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--profile')
    parser.add_argument('--evaluate-stage', choices=tuple(STAGES))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path, help='Native checkpoint directory including training_state; same profile/panel only')
    parser.add_argument('--evaluate-checkpoint', type=Path)
    parser.add_argument('--reference-evaluation', type=Path)
    parser.add_argument('--experiment-config', type=Path)
    args = parser.parse_args()
    if args.evaluate_checkpoint:
        r = evaluate_saved(args.stage, args.evaluate_checkpoint, args.output, reference_evaluation=args.reference_evaluation)
        raise SystemExit(1 if r['status'] == 'reload_mismatch' else 0)
    else:
        require(args.profile is not None, 'Training requires --profile')
        require(args.reference_evaluation is None, 'Reference evaluation is only used for reload checks')
        r = run(args.stage, args.profile, args.output, steps=args.steps, init_checkpoint=args.init_checkpoint,
                resume_checkpoint=args.resume_checkpoint, evaluate_stage=args.evaluate_stage,
                experiment_config=args.experiment_config)
        raise SystemExit(1 if r['status'] == 'error' else 0)


if __name__ == '__main__':
    main()
