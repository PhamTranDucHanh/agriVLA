"""Versioned, train-only nested diagnostic selections; no ML imports or data edits."""
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import require, read_jsonl
from .policy import load_unique_yaml
from .training import file_sha

STAGES = {'micro8': 8, 'cases32': 32, 'cases60': 60}
PROJECTIONS = ['action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out']


def load_experiment(path):
    path = Path(path).resolve()
    c = load_unique_yaml(path.read_bytes(), context='small cases')
    require(c['version'] in ('agri_small_cases_v1', 'agri_small_cases_v2', 'agri_small_cases_v3', 'agri_small_cases_v4', 'agri_small_cases_v5', 'agri_small_cases_v6'), 'Unknown experiment version')
    require(c['seed'] == 42 and c['screen_seeds'] == [0, 42]
            and c['certificate_seeds'] == [0, 1, 2, 3, 42], 'Evaluation seeds must be frozen')
    require(c['warmup_steps'] == 20 and c['max_steps'] == 1000
            and c['evaluate_at'] == [100, 300, 600, 1000], 'Unexpected diagnostic budget')
    for profile in c['profiles'].values():
        fields = {'rank', 'alpha', 'lr', 'full_projections'}
        if c['version'] != 'agri_small_cases_v1':
            fields.add('batch_size')
            require(type(profile.get('batch_size')) is int and profile['batch_size'] in (1, 2, 4), 'Unsupported batch size')
        if c['version'] in ('agri_small_cases_v4', 'agri_small_cases_v5', 'agri_small_cases_v6'):
            fields.add('num_steps')
            require(type(profile.get('num_steps')) is int and profile['num_steps'] in (1, 2, 4, 10, 20), 'Unsupported native sampling steps')
        if c['version'] == 'agri_small_cases_v5':
            fields.add('time_sampling')
            require(profile.get('time_sampling') == 'endpoint_t1' and profile['num_steps'] == 1,
                    'Endpoint training requires one-step inference')
        if c['version'] == 'agri_small_cases_v6':
            fields.add('expert_targets')
            require(profile.get('expert_targets') == 'all_linear' and not profile['full_projections'], 'Unsupported expert scope')
        require(set(profile) == fields, 'Invalid profile fields')
        require(type(profile['rank']) is int and profile['rank'] in (8, 16)
                and profile['alpha'] == profile['rank'], 'Unsupported LoRA scaling')
        require(type(profile['lr']) is float and profile['lr'] in (1e-4, 3e-4), 'Unsupported LR')
        require(type(profile['full_projections']) is bool, 'Invalid projection option')
    for stage in STAGES:
        gate = c['gates'][stage]
        require(set(gate) == {'exact', 'zone', 'stop'} and all(0 < v <= 1 for v in gate.values()), 'Invalid gate')
    return c | {'_path': str(path), '_sha256': file_sha(path)}


def compatible_warm_start(source, target):
    """Weights can transfer across inference/optimizer options with the same architecture."""
    same = all(source[k] == target[k] for k in ('rank', 'alpha', 'full_projections'))
    a, b = source.get('expert_targets', 'qv'), target.get('expert_targets', 'qv')
    return same and (a == b or (a, b) == ('qv', 'all_linear'))


def expert_target_modules(profile):
    if profile.get('expert_targets', 'qv') == 'qv':
        return None  # pinned native default
    return (r'(model\.vlm_with_expert\.lm_expert\..*\.(q|k|v|o|gate|up|down)_proj|'
            r'model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))')


def compatible_inference_variant(source, target):
    def training_options(profile):
        p = dict(profile); p.pop('num_steps', None)
        for k, value in {'batch_size': 1, 'expert_targets': 'qv', 'time_sampling': 'upstream_beta'}.items():
            p.setdefault(k, value)
        return p
    return training_options(source) == training_options(target)


def select_items(items, stage):
    require(stage in STAGES, 'Unknown stage')
    pairs = defaultdict(list)
    for item in items:
        r = item['source_record']
        require(r['split'] == 'train', 'Diagnostic can only select TRAIN')
        pairs[r['case_id']].append(item)
    for pair in pairs.values():
        require(len(pair) == 2 and {x['source_record']['language'] for x in pair} == {'vi', 'en'}, 'Incomplete bilingual pair')
        require(len({tuple(x['source_record']['target_action']) for x in pair}) == 1, 'Translation target mismatch')
        require(len({tuple(x['source_record']['source_image_ids']) for x in pair}) == 1, 'Translation image mismatch')
    cases = sorted(pairs, key=lambda k: min(x['episode_index'] for x in pairs[k]))
    row = lambda k: pairs[k][0]['source_record']
    by_scene = defaultdict(list)
    for k in cases:
        by_scene[row(k)['scene_id']].append(k)
    candidates = []
    for scene, keys in by_scene.items():
        levels = {row(k)['intent_params'].get('level') for k in keys if row(k)['intent'] == 'WATER_ALL'}
        if levels == {1, 2, 3} and any(row(k)['intent'] == 'STOP_ALL' for k in keys):
            candidates.append(scene)
    require(candidates, 'Need same-scene STOP and all three WATER_ALL levels')
    scene = sorted(candidates)[0]
    base = [k for k in cases if row(k)['generation_metadata']['kind'] == 'contrastive_base']
    require(len(base) == 11, 'Need all eleven base intent/parameter controls')
    force = [next(k for k in by_scene[scene] if row(k)['intent'] == 'WATER_ALL'
                  and row(k)['intent_params']['level'] == level) for level in (1, 2, 3)]
    stop = next(k for k in by_scene[scene] if row(k)['intent'] == 'STOP_ALL')
    micro = [stop, *force]
    chosen = [*base, *micro]
    for kind, intent in [('tie', 'WORST_ONLY'), ('below_threshold', 'WORST_ONLY'),
                         ('all_off', 'AUTO'), ('all_high', 'AUTO')]:
        chosen.append(next(k for k in cases if row(k)['generation_metadata']['kind'] == kind and row(k)['intent'] == intent))
    permutations = [k for k in cases if row(k)['generation_metadata']['kind'] == 'permutation']
    require(len(permutations) == 11, 'Expected eleven permutation controls')
    chosen.extend(permutations)
    keys = micro if stage == 'micro8' else chosen[:16] if stage == 'cases32' else chosen
    require(len(keys) == len(set(keys)) == STAGES[stage] // 2, 'Selection case accounting mismatch')
    selected = sorted([x for k in keys for x in pairs[k]], key=lambda x: x['episode_index'])
    records = [x['source_record'] for x in selected]
    return {'version': 'agri_small_cases_v1', 'stage': stage, 'split': 'train', 'records': len(selected),
            'episodes': [x['episode_index'] for x in selected], 'items': selected,
            'coverage': {'intents': dict(Counter(r['intent'] for r in records)),
                         'languages': dict(Counter(r['language'] for r in records)),
                         'kinds': dict(Counter(r['generation_metadata']['kind'] for r in records)),
                         'targets': sorted({tuple(r['target_action']) for r in records})}}


def selection(c, stage):
    path = c['_base'] / c['dataset_root'] / 'train/sample_metadata.jsonl'
    return select_items(read_jsonl(path), stage) | {'source_metadata_sha256': file_sha(path)}


def passes(scores, gate):
    return bool(scores) and all(s['exact_action_accuracy'] >= gate['exact']
                               and s['zone_accuracy'] >= gate['zone']
                               and s['stop_exact_accuracy'] is not None
                               and s['stop_exact_accuracy'] >= gate['stop'] for s in scores)


def compare_predictions(reference, actual, *, atol=1e-3, rtol=1e-3):
    """Compare matching explicit-noise predictions without conflating batches/seeds."""
    import math
    index = lambda rows: {(r['sample_id'], r['seed']): r for r in rows}
    wanted, got = index(reference), index(actual)
    require(len(wanted) == len(reference) and len(got) == len(actual), 'Duplicate reload prediction')
    require(wanted and wanted.keys() == got.keys(), 'Reload prediction IDs/seeds differ')
    maximum = 0.
    within, discrete_equal = True, True
    for key, a in wanted.items():
        b = got[key]
        require(a['target_action'] == b['target_action'], 'Reload target mapping differs')
        require(len(a['continuous_action']) == len(b['continuous_action']) == 3, 'Bad reload action shape')
        for x, y in zip(a['continuous_action'], b['continuous_action']):
            require(math.isfinite(x) and math.isfinite(y), 'Nonfinite reload action')
            maximum = max(maximum, abs(x - y))
            within &= abs(x - y) <= atol + rtol * abs(x)
        discrete_equal &= a['quantized_action'] == b['quantized_action']
    return {'passed': bool(within and discrete_equal), 'records': len(wanted),
            'max_absolute_error': maximum, 'atol': atol, 'rtol': rtol,
            'continuous_within_tolerance': bool(within), 'quantized_equal': bool(discrete_equal)}
