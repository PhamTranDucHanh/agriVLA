"""Real longest-TRAIN-prompt forward/backward memory check, without any update."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .dataset import require, new_output, read_json, read_jsonl
from .preflight import project_root
from .training import load_training_config, train_config, CAMERAS, write_json, file_sha, prompt_text
from .small_train import load_checkpoint, parameter_audit


def run(checkpoint, output):
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.common.train_utils import load_training_state
    root = project_root(); output = Path(output).resolve(); new_output(output)
    c = load_training_config(root / 'configs/training_v0.yaml')
    checkpoint = Path(checkpoint).resolve()
    exp = read_json(checkpoint / 'small_experiment.json')
    batch_size = exp['profile'].get('batch_size', 1)
    policy, pre, post = load_checkpoint(c, checkpoint)
    parameter_audit(policy)
    tok = next(s for s in pre.steps if type(s).__name__ == 'TokenizerProcessorStep')
    metadata = root / c['dataset_root'] / 'train/sample_metadata.jsonl'
    items = read_jsonl(metadata)
    lengths = {r['source_record']['instruction']: len(tok.input_tokenizer(prompt_text(r['source_record']['instruction']), truncation=False)['input_ids']) for r in items}
    longest = max(lengths.values()); require(longest <= 96, 'Longest prompt truncated')
    chosen = [r for r in items if lengths[r['source_record']['instruction']] == longest][:batch_size]
    require(len(chosen) == batch_size, 'Need enough longest-prompt TRAIN records')
    episodes = [r['episode_index'] for r in chosen]
    cfg = train_config(c, root / c['bundle'], output=output.parent / 'unused-no-training', device='cuda', episodes=episodes)
    dataset, ev = make_train_eval_datasets(cfg)
    require(ev is None and len(dataset) == batch_size, 'Stress data differs')
    raw = next(iter(DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)))
    for cam in CAMERAS:
        if raw[cam].dtype == torch.uint8:
            raw[cam] = raw[cam].float() / 255.
    inputs = pre(raw)
    require(inputs['observation.language.tokens'].shape[-1] == 96, 'Tokenizer capacity drift')
    counts = inputs['observation.language.attention_mask'].sum(-1).detach().cpu().tolist()
    require(counts == [longest] * batch_size, 'Actual longest token counts differ')
    images = []
    def vision_input(_module, _args, kwargs):
        x = kwargs['pixel_values']
        images.append({'shape': list(x.shape), 'dtype': str(x.dtype), 'min': float(x.min()), 'max': float(x.max())})
    handle = policy.get_base_model().model.vlm_with_expert.get_vlm_model().vision_model.register_forward_pre_hook(vision_input, with_kwargs=True)
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    saved_cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True, cli_args=[])
    opt, scheduler = make_optimizer_and_scheduler(saved_cfg, policy)
    restored_step, opt, scheduler = load_training_state(checkpoint.parent, opt, scheduler)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_interval=2000)
    policy.train(); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        loss, _ = policy.forward(inputs)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 10.)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(norm)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable.values())
    unchanged = all(torch.equal(p.detach().cpu(), before[n]) for n, p in trainable.items())
    require(finite and unchanged and len(images) == 3, 'Bad stress forward/backward or mutated weights')
    require(all(x['shape'] == [batch_size, 3, 256, 256] for x in images), 'Camera resolution/batch differs')
    nv = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used, total = map(int, nv.stdout.strip().splitlines()[0].split(','))
    report = {'status': 'pass', 'scope': 'longest TRAIN prompt, resource check only', 'checkpoint': str(checkpoint),
              'source_metadata_sha256': file_sha(metadata), 'batch_size': batch_size, 'episodes': episodes,
              'longest_train_tokens': longest, 'actual_token_counts': counts, 'tokenizer_capacity': 96,
              'instructions': raw['task'], 'vision_encoder_inputs': images, 'loss': float(loss), 'grad_norm_before_clip': float(norm),
              'finite_gradients': finite, 'weights_unchanged': unchanged, 'optimizer_steps_performed': 0,
              'native_optimizer_state_loaded': True, 'restored_training_step': restored_step,
              'torch_peak_allocated': torch.cuda.max_memory_allocated(), 'torch_peak_reserved': torch.cuda.max_memory_reserved(),
              'nvml_end_used_mib': used, 'nvml_total_mib': total, 'nvml_is_end_snapshot_not_peak': True,
              'forward_backward_seconds': elapsed}
    write_json(output, report); handle.remove(); opt.zero_grad(set_to_none=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.checkpoint, a.output)


if __name__ == '__main__':
    main()
