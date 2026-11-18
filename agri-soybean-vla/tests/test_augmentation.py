"""Policy-valid counterfactual construction; these fixtures do not prove accuracy."""
import copy

import pytest

from agri_vla.augmentation import mix_override_images
from agri_vla.dataset import DatasetError
from agri_vla.training import CAMERAS


def batch_and_records():
    import torch
    records = {
        5: {'split': 'train', 'intent': 'STOP_ALL', 'target_action': [0, 0, 0], 'intent_params': {}},
        9: {'split': 'train', 'intent': 'WATER_ALL', 'target_action': [2, 2, 2], 'intent_params': {'level': 2}},
        11: {'split': 'train', 'intent': 'AUTO', 'target_action': [0, 1, 3], 'intent_params': {}},
        13: {'split': 'train', 'intent': 'EXCLUDE_ZONE', 'target_action': [3, 0, 1], 'intent_params': {'zone': 2}},
    }
    batch = {'episode_index': torch.tensor([5, 9, 11, 13]), 'task': ['stop', 'all2', 'auto', 'exclude2'],
             'observation.state': torch.zeros(4, 1, 3), 'action': torch.arange(12.).reshape(4, 1, 3)}
    for z, camera in enumerate(CAMERAS):
        batch[camera] = torch.arange(4.).reshape(4, 1, 1, 1, 1).expand(4, 1, 3, 256, 256).clone() + z * 10
    return batch, records


def test_mix_only_image_independent_targets_whole_triplets_and_no_rng_draw():
    import torch
    batch, records = batch_and_records()
    before = copy.deepcopy(batch)
    rng = torch.random.get_rng_state().clone()
    report = mix_override_images(batch, records, step=17, profile={'training_augmentation': 'override_image_mix_v1'})
    assert torch.equal(rng, torch.random.get_rng_state())
    assert report['rows_replaced'] == 2 and {x['episode'] for x in report['mappings']} == {5, 9}
    assert set(batch) == set(before)  # No intent/teacher feature is inserted.
    for key in ('episode_index', 'action', 'observation.state'):
        assert torch.equal(batch[key], before[key])
    assert batch['task'] == before['task']
    for camera in CAMERAS:
        assert torch.equal(batch[camera][2:], before[camera][2:])
        for mapping in report['mappings']:
            i = before['episode_index'].tolist().index(mapping['episode'])
            donor = before['episode_index'].tolist().index(mapping['images_from_episode'])
            assert i != donor and torch.equal(batch[camera][i], before[camera][donor])
    again = copy.deepcopy(before)
    assert mix_override_images(again, records, step=17, profile={'training_augmentation': 'override_image_mix_v1'}) == report
    assert all(torch.equal(again[k], batch[k]) for k in CAMERAS)


def test_old_profiles_do_not_mutate_or_require_override_metadata():
    assert mix_override_images({}, {}, step=1, profile={}) is None


@pytest.mark.parametrize('fault', ['val_donor', 'wrong_target', 'wrong_camera'])
def test_mix_rejects_invalid_supervision_or_schema(fault):
    batch, records = batch_and_records()
    if fault == 'val_donor': records[13]['split'] = 'val'
    elif fault == 'wrong_target': records[9]['target_action'] = [1, 2, 3]
    else: batch.pop(CAMERAS[-1])
    with pytest.raises((DatasetError, KeyError)):
        mix_override_images(batch, records, step=1, profile={'training_augmentation': 'override_image_mix_v1'})
