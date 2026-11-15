from copy import deepcopy
from pathlib import Path
import random

import pytest

from agri_vla.dataset import DatasetError
from agri_vla.small_cases import select_items, passes, compare_predictions, load_experiment, compatible_warm_start, compatible_inference_variant, expert_target_modules


def fixture_items():
    requests = [('AUTO', {}, [0, 1, 3]), ('WORST_ONLY', {}, [0, 0, 3]),
                *[('ZONE_ONLY', {'zone': z}, [0, 0, 0]) for z in (1, 2, 3)],
                *[('CAP_LEVEL', {'cap': z}, [0, 1, z]) for z in (1, 2)],
                *[('EXCLUDE_ZONE', {'zone': z}, [0, 0, 0]) for z in (1, 2, 3)],
                ('STOP_ALL', {}, [0, 0, 0])]
    cases = [(kind, kind, intent, params, target) for kind in ('contrastive_base', 'permutation')
             for intent, params, target in requests]
    cases += [('all_off', 'all_off', 'STOP_ALL', {}, [0, 0, 0])]
    cases += [('operator_override_all', 'all_off', 'WATER_ALL', {'level': z}, [z] * 3) for z in (1, 2, 3)]
    cases += [('tie', 'tie', 'WORST_ONLY', {}, [3, 3, 0]),
              ('below_threshold', 'below', 'WORST_ONLY', {}, [0, 0, 0]),
              ('all_off', 'all_off', 'AUTO', {}, [0, 0, 0]),
              ('all_high', 'high', 'AUTO', {}, [3, 3, 3])]
    items = []
    for i, (kind, scene, intent, params, target) in enumerate(cases):
        for language in ('vi', 'en'):
            items.append({'episode_index': len(items), 'source_record': {
                'split': 'train', 'case_id': str(i), 'scene_id': scene, 'language': language,
                'intent': intent, 'intent_params': params, 'target_action': target,
                'source_image_ids': [scene + str(j) for j in (1, 2, 3)],
                'generation_metadata': {'kind': kind}}})
    return items


def test_nested_bilingual_coverage_and_permutation_controls():
    items = fixture_items()
    panels = {k: select_items(items, k) for k in ('micro8', 'cases32', 'cases60')}
    a, b, c = [set(panels[k]['episodes']) for k in panels]
    assert a < b < c
    assert [len(x) for x in (a, b, c)] == [8, 32, 60]
    assert panels['micro8']['coverage']['targets'] == [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert panels['cases60']['coverage']['languages'] == {'vi': 30, 'en': 30}
    assert len(panels['cases60']['coverage']['intents']) == 7
    assert panels['cases60']['coverage']['kinds']['permutation'] == 22
    random.Random(7).shuffle(items)
    assert select_items(items, 'cases60')['episodes'] == panels['cases60']['episodes']


def test_selection_rejects_heldout_or_corrupt_pairs():
    items = fixture_items()
    items[0]['source_record']['split'] = 'val'
    with pytest.raises(DatasetError, match='TRAIN'):
        select_items(items, 'micro8')
    items = fixture_items()
    items[0]['source_record']['target_action'] = [3, 3, 3]
    with pytest.raises(DatasetError, match='target mismatch'):
        select_items(items, 'cases60')
    with pytest.raises(DatasetError, match='Incomplete'):
        select_items(fixture_items()[:-1], 'cases60')


def test_gate_requires_every_seed_and_stop_not_only_average():
    gate = {'exact': .95, 'zone': .98, 'stop': 1.}
    good = {'exact_action_accuracy': 1., 'zone_accuracy': 1., 'stop_exact_accuracy': 1.}
    assert passes([good, good], gate)
    assert not passes([], gate)
    for field, value in [('exact_action_accuracy', .9), ('zone_accuracy', .97), ('stop_exact_accuracy', .5)]:
        bad = good | {field: value}
        assert not passes([good, bad], gate)
    assert not passes([good | {'stop_exact_accuracy': None}], gate)


def test_reload_checks_matching_noise_and_discrete_boundary():
    a = {'sample_id': 'a', 'seed': 42, 'target_action': [0, 1, 3],
         'continuous_action': [0., .4999, 3.], 'quantized_action': [0, 0, 3]}
    assert compare_predictions([a], [deepcopy(a)])['passed']
    b = deepcopy(a); b['continuous_action'][1] = .5001; b['quantized_action'][1] = 1
    assert not compare_predictions([a], [b])['passed']
    b = deepcopy(a); b['continuous_action'][0] = .01
    assert not compare_predictions([a], [b])['passed']
    b = a | {'seed': 0}
    with pytest.raises(DatasetError, match='IDs/seeds'):
        compare_predictions([a], [b])


def test_batch_control_version_preserves_v1_and_rejects_unsupported_batch(tmp_path):
    configs = Path(__file__).resolve().parents[1] / 'configs'
    old = load_experiment(configs / 'small_cases_v1.yaml')
    new = load_experiment(configs / 'small_cases_v2.yaml')
    for key in ('screen_seeds', 'certificate_seeds', 'gates', 'warmup_steps', 'max_steps'):
        assert old[key] == new[key]
    assert {p['batch_size'] for p in new['profiles'].values()} == {2, 4}
    bad = tmp_path / 'bad.yaml'
    bad.write_text((configs / 'small_cases_v2.yaml').read_text().replace('batch_size: 2', 'batch_size: 8'))
    with pytest.raises(DatasetError, match='batch size'):
        load_experiment(bad)
    strict = load_experiment(configs / 'small_cases_v3.yaml')
    assert strict['profiles']['r8_b2_lr3e4'] == new['profiles']['r8_b2_lr3e4']
    assert all(g == {'exact': 1., 'zone': 1., 'stop': 1.} for g in strict['gates'].values())


def test_sampling_profile_keeps_architecture_but_rejects_rank_or_module_changes():
    configs = Path(__file__).resolve().parents[1] / 'configs'
    source = load_experiment(configs / 'small_cases_v3.yaml')['profiles']['r8_b4_lr3e4']
    target = load_experiment(configs / 'small_cases_v4.yaml')['profiles']['r8_b4_n2_lr3e4']
    assert compatible_warm_start(source, target)
    assert not compatible_warm_start(source, target | {'rank': 16, 'alpha': 16})
    assert not compatible_warm_start(source, target | {'full_projections': True})


def test_sampling_only_variant_rejects_training_changes():
    source = {'rank': 8, 'alpha': 8, 'lr': .0003, 'full_projections': False, 'batch_size': 4}
    assert compatible_inference_variant(source, source | {'num_steps': 2, 'expert_targets': 'qv', 'time_sampling': 'upstream_beta'})
    for change in ({'lr': .0001}, {'batch_size': 2}, {'expert_targets': 'all_linear'}, {'time_sampling': 'endpoint_t1'}):
        assert not compatible_inference_variant(source, source | {'num_steps': 2} | change)


def test_endpoint_profile_requires_matching_single_step_inference(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'configs/small_cases_v5.yaml'
    c = load_experiment(path)
    assert all(p['time_sampling'] == 'endpoint_t1' and p['num_steps'] == 1 for p in c['profiles'].values())
    bad = tmp_path / 'mismatched.yaml'
    bad.write_text(path.read_text().replace('num_steps: 1', 'num_steps: 2'))
    with pytest.raises(DatasetError, match='Endpoint training'):
        load_experiment(bad)


def test_expert_scope_expands_without_targeting_frozen_vlm_or_dropping_adapters():
    import re
    configs = Path(__file__).resolve().parents[1] / 'configs'
    old = load_experiment(configs / 'small_cases_v4.yaml')['profiles']['r8_b4_n2_lr3e4']
    wide = load_experiment(configs / 'small_cases_v6.yaml')['profiles']['r8_wide_b4_n2_lr3e4']
    assert compatible_warm_start(old, wide)
    assert not compatible_warm_start(wide, old)
    regex = expert_target_modules(wide)
    for name in ['self_attn.k_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        assert re.fullmatch(regex, 'model.vlm_with_expert.lm_expert.layers.0.' + name)
        assert not re.fullmatch(regex, 'model.vlm_with_expert.vlm.text_model.layers.0.' + name)
    assert re.fullmatch(regex, 'model.action_out_proj')
