"""Frozen TRAIN/VAL diagnostic. This command never opens TEST or updates weights."""
import argparse
import os
from pathlib import Path
import time

from .dataset import require, new_output, read_json
from .preflight import project_root
from .training import load_training_config, train_config, write_json, file_sha


def audit_inputs(output, *, stage='cases280'):
    """Compare real native tensors with raw records, not model predictions."""
    root = project_root()
    os.environ['HF_DATASETS_CACHE'] = str(root / '.cache/hf-datasets')
    import torch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from .evaluation import fixed_validation_selection, data_loader
    from .small_cases import selection
    from .training import CAMERAS
    from .lerobot_io import transformed_image

    output = Path(output).resolve()
    new_output(output)
    torch.set_num_threads(1)
    c = load_training_config(root / 'configs/training_v0.yaml')
    cfg = SmolVLAConfig.from_pretrained(root / c['bundle'] / 'base_policy')
    image_cache = {}
    summaries = {}
    for split, episodes in (('train', selection(c, stage)['episodes']),
                            ('val', fixed_validation_selection(c, 112)['episodes'])):
        loader, mapping = data_loader(c, cfg, split, indices=episodes)
        pixel_max, records = 0., 0
        for batch in loader:
            episode = int(batch['episode_index'].item())
            record = mapping[episode]['source_record']
            require(int(batch['index'].item()) == episode and record['split'] == split, 'Singleton index/split differs')
            require(batch['task'] == [record['instruction']], 'Native task differs from raw instruction')
            require(batch['action'].reshape(-1).tolist() == record['target_action'], 'Native action differs from raw target')
            require(bool((batch['observation.state'] == 0).all()), 'Unexpected state')
            for camera, zone in zip(CAMERAS, ('zone1', 'zone2', 'zone3'), strict=True):
                path = root / record['images'][zone]
                if path not in image_cache:
                    raw = transformed_image(path, file_sha(path))
                    image_cache[path] = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.
                actual = batch[camera].reshape(3, 256, 256)
                pixel_max = max(pixel_max, float((actual - image_cache[path]).abs().max()))
            records += 1
        require(pixel_max == 0. and records == len(episodes), 'Native pixels/count differ from raw input')
        summaries[split] = {'records': records, 'pixel_tensors': records * 3, 'max_pixel_error': pixel_max,
                            'instruction_action_state_mapping_exact': True}
    result = {'status': 'pass', 'scope': 'real TRAIN/VAL CPU input audit; no model/optimizer/TEST',
              'stage': stage, 'splits': summaries, 'train_stats_sha256': file_sha(root / c['dataset_root'] / 'train_stats.json')}
    write_json(output, result)
    print(result, flush=True)
    return result


def run(checkpoint, output, *, stage='cases280'):
    root = project_root()
    os.environ['HF_DATASETS_CACHE'] = str(root / '.cache/hf-datasets')
    from lerobot.datasets.factory import make_train_eval_datasets
    from .small_cases import selection
    from .small_train import load_checkpoint, panel_evaluate
    from .evaluation import fixed_validation_selection, data_loader, predict_batch, metrics, quantize_action

    output = Path(output).resolve()
    new_output(output)
    checkpoint = Path(checkpoint).resolve()
    original_sha = file_sha(checkpoint / 'adapter_model.safetensors')
    c = load_training_config(root / 'configs/training_v0.yaml')
    policy, pre, post = load_checkpoint(c, checkpoint)
    selected = selection(c, stage)
    cfg = train_config(c, root / c['bundle'], output=root / 'outputs/unused-pilot-eval',
                       device='cuda', episodes=selected['episodes'])
    dataset, heldout = make_train_eval_datasets(cfg)
    require(heldout is None and len(dataset) == selected['records'], 'Wrong TRAIN diagnostic selection')
    start = time.perf_counter()
    train = panel_evaluate(policy, pre, post, dataset, selected, [42], batch_size=4)
    train_seconds = time.perf_counter() - start
    print('TRAIN', stage, train['by_seed']['42']['overall'], flush=True)

    validation = fixed_validation_selection(c, 112)
    loader, mapping = data_loader(c, policy.config, 'val', indices=validation['episodes'])
    rows = []
    start = time.perf_counter()
    for batch in loader:
        record = mapping[int(batch['index'].item())]['source_record']
        require(batch['task'] == [record['instruction']], 'VAL instruction/sample mapping differs')
        require(batch['action'].reshape(-1).tolist() == record['target_action'], 'VAL action/sample mapping differs')
        prediction = predict_batch(policy, pre, post, batch, seed=42)
        rows.append({k: record[k] for k in ('sample_id', 'language', 'intent', 'wording_partition', 'target_action')}
                    | {'seed': 42} | prediction)
    val = {'selection': validation, 'metrics': metrics(rows), 'predictions': rows,
           'seconds': time.perf_counter() - start}
    off = [{k: r[k] for k in ('sample_id', 'language', 'intent', 'wording_partition', 'target_action')}
           | quantize_action([0., 0., 0.]) for r in rows]
    require(file_sha(checkpoint / 'adapter_model.safetensors') == original_sha, 'Frozen model changed')
    report = {'scope': 'TRAIN diagnostic and fixed balanced partial VAL112; seed42; no TEST/updates',
              'checkpoint': str(checkpoint), 'adapter_sha256': original_sha,
              'stage': stage, 'train': train, 'train_seconds': train_seconds, 'val112': val,
              'always_off': metrics(off), 'optimizer_steps_performed': 0,
              'normalization_source': 'saved full TRAIN3600',
              'time_sampling': read_json(checkpoint / 'small_experiment.json')['profile'].get('time_sampling', 'upstream_beta')}
    write_json(output, report)
    print('VAL112', val['metrics']['overall'], 'OFF', report['always_off']['overall'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('cases60', 'cases280'), default='cases280')
    parser.add_argument('--audit-inputs', action='store_true')
    args = parser.parse_args()
    if args.audit_inputs:
        audit_inputs(args.output, stage=args.stage)
    else:
        require(args.checkpoint is not None, 'Provide --checkpoint for frozen inference')
        run(args.checkpoint, args.output, stage=args.stage)


if __name__ == '__main__':
    main()
