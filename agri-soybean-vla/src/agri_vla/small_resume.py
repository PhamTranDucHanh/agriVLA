"""One-update native resume proof against a real uninterrupted control checkpoint.

This is a diagnostic for small_cases checkpoints, not a replacement full-run CLI.
It restores native optimizer/scheduler/RNG/step and the saved AMP scaler, then
compares the next update with the uninterrupted run on the same sample/noise.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from .dataset import require, new_output, read_json
from .preflight import project_root
from .training import load_training_config, write_json, attach_checkpoint_contract
from .small_train import load_checkpoint, parameter_audit, rng_fingerprint


def continued_cycle(loader, original_cycle, *, step, records, batch_size):
    """Continue the prepared loader at its saved epoch without an extra RNG draw."""
    import torch
    from lerobot.datasets.sampler import compute_sampler_state
    sampler_state = compute_sampler_state(step, records, batch_size, 1)
    require(hasattr(loader, 'set_epoch'), 'Prepared loader needs the public set_epoch API')
    loader.set_epoch(sampler_state['epoch'])
    iterator = original_cycle(loader)
    if sampler_state['start_index']:
        with torch.random.fork_rng(devices=[]):
            first = next(iterator)
    else:
        first = next(iterator)
    yield first
    yield from iterator


def compare_tensor_files(a, b):
    import torch
    from safetensors.torch import load_file
    x, y = load_file(a), load_file(b)
    require(x.keys() == y.keys(), f'Tensor names differ: {a.name}')
    maximum, close, exact = 0., True, True
    for key in x:
        require(x[key].shape == y[key].shape and x[key].dtype == y[key].dtype, f'Tensor shape/dtype differs: {key}')
        require(bool(torch.isfinite(x[key]).all()) and bool(torch.isfinite(y[key]).all()), 'Nonfinite saved state')
        maximum = max(maximum, float((x[key].double() - y[key].double()).abs().max()))
        close &= torch.allclose(x[key], y[key], atol=1e-6, rtol=1e-5)
        exact &= torch.equal(x[key], y[key])
    return {'tensors': len(x), 'max_absolute_error': maximum, 'allclose': bool(close), 'exact': bool(exact),
            'atol': 1e-6, 'rtol': 1e-5}


def run(checkpoint, control, output, *, restore_loader_continuation=False):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # register native policy choice
    from lerobot.common.train_utils import save_training_state
    from lerobot.scripts import lerobot_train as upstream
    root = project_root()
    c = load_training_config(root / 'configs/training_v0.yaml')
    checkpoint, control, output = (Path(p).resolve() for p in (checkpoint, control, output))
    require(output.is_relative_to(root / 'outputs/small-cases'), 'Use a new outputs/small-cases directory')
    new_output(output)
    report_dir = root / 'reports/small-cases' / output.name
    new_output(report_dir); report_dir.mkdir(parents=True)
    exp = read_json(checkpoint / 'pretrained_model/small_experiment.json')
    step_info = read_json(checkpoint / 'training_state/training_step.json')
    start = step_info['step']
    require(type(start) is int and start > 0 and read_json(control / 'training_state/training_step.json')['step'] == start + 1,
            'Need adjacent genuinely trained checkpoints')
    require(read_json(control / 'pretrained_model/small_experiment.json') == exp, 'Control experiment differs')
    cfg = TrainPipelineConfig.from_pretrained(checkpoint / 'pretrained_model', local_files_only=True, cli_args=[])
    require(cfg.num_workers == 0 and not cfg.dataset.streaming and not cfg.dataset.image_transforms.enable,
            'Proof requires deterministic image loading, no augmentation or workers')
    require(step_info['num_processes'] == 1 and step_info['batch_size'] == cfg.batch_size, 'Batch/world-size drift')
    require(cfg.dataset.episodes == exp['episodes'], 'Training panel drift')
    from .dataset import read_jsonl
    augmentation_records = {x['episode_index']: x['source_record'] for x in
                            read_jsonl(root / c['dataset_root'] / 'train/sample_metadata.jsonl')}
    require(cfg.scheduler.type == 'constant_with_warmup', 'Only tested constant schedule supported')
    cfg.output_dir = output; cfg.steps = start + 1; cfg.resume = True
    cfg.save_checkpoint = False; cfg.log_freq = 1
    training_run_name = Path(exp['inference_variant_of_run']).name if 'inference_variant_of_run' in exp else checkpoint.parents[1].name
    control_rows = [json.loads(x) for x in (root / 'reports/small-cases' / training_run_name / 'updates.jsonl').read_text().splitlines()]
    reference = next(x for x in control_rows if x['step'] == start + 1)
    require(reference.get('rng_before') is not None and reference.get('episodes'), 'Control lacks sample/RNG evidence')
    originals = {n: getattr(upstream, n) for n in ('make_policy', 'make_pre_post_processors', 'load_training_state', 'update_policy', 'cycle')}
    state = {'optimizer_steps_performed': 0, 'native_restore': {}, 'continuation': {}}

    def make_policy(*args, **kwargs):
        policy, pre, post = load_checkpoint(c, checkpoint / 'pretrained_model')
        policy.config.use_peft = True
        state['policy'], state['pre'], state['post'] = policy, pre, post
        state['vision_inputs'] = []
        def vision_input(_module, _args, inputs):
            x = inputs['pixel_values']
            state['vision_inputs'].append({'shape': list(x.shape), 'dtype': str(x.dtype),
                                           'min': float(x.min()), 'max': float(x.max())})
        policy.get_base_model().model.vlm_with_expert.get_vlm_model().vision_model.register_forward_pre_hook(vision_input, with_kwargs=True)
        write_json(report_dir / 'parameters.json', parameter_audit(policy))
        return policy

    def processors(*args, **kwargs):
        return state['pre'], state['post']

    def load_state(*args, **kwargs):
        step, opt, sched = originals['load_training_state'](*args, **kwargs)
        require(step == start, 'Native step restore differs')
        receipt = output / 'restored_before_update'
        save_training_state(receipt, step, opt, sched, num_processes=1, batch_size=cfg.batch_size)
        checks = state['native_restore']
        for name in ('optimizer_state.safetensors', 'rng_state.safetensors'):
            checks[name] = compare_tensor_files(checkpoint / 'training_state' / name, receipt / 'training_state' / name)
            require(checks[name]['exact'], f'Native restore differs: {name}')
        for name in ('scheduler_state.json', 'training_step.json'):
            checks[name] = read_json(checkpoint / 'training_state' / name) == read_json(receipt / 'training_state' / name)
            require(checks[name], f'Native restore differs: {name}')
        checks['amp_scaler_exact'] = accelerator.scaler.state_dict() == read_json(checkpoint / 'training_state/amp_scaler_state.json')
        require(checks['amp_scaler_exact'], 'AMP state not restored')
        write_json(report_dir / 'restored-state.json', checks)
        return step, opt, sched

    def update(*args, **kwargs):
        batch = args[2] if len(args) > 2 else kwargs['batch']
        observed = {'episodes': batch['episode_index'].detach().cpu().tolist(), 'rng_before': rng_fingerprint()}
        state['batch_shapes'] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        state['real_token_counts'] = batch['observation.language.attention_mask'].sum(-1).detach().cpu().tolist()
        from .augmentation import mix_override_images
        state['augmentation'] = mix_override_images(batch, augmentation_records, step=start + 1, profile=exp['profile'])
        result = originals['update_policy'](*args, **kwargs)
        tracker = result[0]
        optimizer = args[3] if len(args) > 3 else kwargs['optimizer']
        require(math.isfinite(tracker.loss.val) and math.isfinite(tracker.grad_norm.val)
                and not getattr(optimizer, 'step_was_skipped', False), 'Nonfinite/skipped resumed update')
        state['optimizer_steps_performed'] += 1
        require(state['optimizer_steps_performed'] == 1, 'Resume proof exceeded one update')
        observed.update(loss=tracker.loss.val, grad_norm=tracker.grad_norm.val, lr=tracker.lr.val)
        state['continuation'] = {'reference': reference, 'actual': observed,
                                 'sample_order_equal': observed['episodes'] == reference['episodes'],
                                 'rng_before_equal': observed['rng_before'] == reference['rng_before'],
                                 'loss_close': math.isclose(observed['loss'], reference['loss'], abs_tol=1e-5, rel_tol=1e-4)}
        cp = output / 'checkpoints' / f'{start+1:06d}'
        policy = kwargs['accelerator'].unwrap_model(args[1] if len(args) > 1 else kwargs['policy'])
        upstream.save_checkpoint(checkpoint_dir=cp, step=start+1, cfg=cfg, policy=policy, optimizer=optimizer,
                                 scheduler=kwargs.get('lr_scheduler'), preprocessor=state['pre'], postprocessor=state['post'],
                                 num_processes=1, batch_size=cfg.batch_size)
        attach_checkpoint_contract(root / c['bundle'], cp / 'pretrained_model')
        write_json(cp / 'pretrained_model/small_experiment.json', exp)
        write_json(cp / 'training_state/amp_scaler_state.json', accelerator.scaler.state_dict())
        state['resumed_checkpoint'] = str(cp)
        comparisons = {}
        for name in ('pretrained_model/adapter_model.safetensors', 'training_state/optimizer_state.safetensors', 'training_state/rng_state.safetensors'):
            comparisons[name] = compare_tensor_files(control / name, cp / name)
        for name in ('scheduler_state.json', 'training_step.json', 'amp_scaler_state.json'):
            comparisons[name] = read_json(control / 'training_state' / name) == read_json(cp / 'training_state' / name)
        state['control_comparison'] = comparisons
        return result

    def continued_iterator(loader):
        # Accelerate's prepared loader starts at iteration0 and calls set_epoch
        # before reading its sampler, overriding the epoch restored upstream.
        # A mid-epoch resume creates an extra iterator/base_seed draw. No
        # augmentation/workers are allowed here; restore only this first draw.
        yield from continued_cycle(loader, originals['cycle'], step=start,
                                   records=len(cfg.dataset.episodes), batch_size=cfg.batch_size)

    previous_argv = sys.argv
    failure = None
    try:
        sys.argv = ['agri-small-resume', f'--config_path={checkpoint / "pretrained_model/train_config.json"}']
        cfg.validate()
        if restore_loader_continuation:
            upstream.cycle = continued_iterator
        upstream.make_policy = make_policy; upstream.make_pre_post_processors = processors
        upstream.load_training_state = load_state; upstream.update_policy = update
        accelerator = Accelerator(step_scheduler_with_optimizer=False, mixed_precision='fp16',
                                  kwargs_handlers=[GradScalerKwargs(init_scale=1024., growth_interval=2000)])
        accelerator.scaler.load_state_dict(read_json(checkpoint / 'training_state/amp_scaler_state.json'))
        upstream.train(cfg, accelerator=accelerator)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        (report_dir / 'failure.txt').write_text(traceback.format_exc())
    finally:
        sys.argv = previous_argv
        for n, f in originals.items():
            setattr(upstream, n, f)
    comparisons = state.get('control_comparison', {})
    passed = not failure and state['optimizer_steps_performed'] == 1 and all(
        state['continuation'].get(k, False) for k in ('sample_order_equal', 'rng_before_equal', 'loss_close')) and bool(comparisons)
    passed &= all(v.get('exact' if 'rng_state' in k else 'allclose', False) if isinstance(v, dict) else v
                  for k, v in comparisons.items())
    report = {k: v for k, v in state.items() if k not in ('policy', 'pre', 'post')}
    report.update(status='error' if failure else 'passed' if passed else 'mismatch', failure=failure,
                  source_checkpoint=str(checkpoint), control_checkpoint=str(control),
                  restore_loader_continuation=restore_loader_continuation, scope='one real native resumed update; TRAIN only')
    write_json(report_dir / 'resume_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--restore-loader-continuation', action='store_true')
    a = p.parse_args()
    report = run(a.checkpoint, a.control, a.output, restore_loader_continuation=a.restore_loader_continuation)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
