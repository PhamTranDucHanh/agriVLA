"""TRAIN-only counterfactual images for actions provably independent of images."""
import hashlib

from .dataset import require
from .training import CAMERAS


def mix_override_images(batch, records, *, step, profile):
    """Mix whole real TRAIN triplets; keep task/state/action and RNG untouched.

    Structured intent chooses valid supervised augmentation only. It is never
    added to policy inputs, and inference has no such transform or override.
    """
    mode = profile.get('training_augmentation', 'none')
    if mode == 'none':
        return None
    require(mode == 'override_image_mix_v1', 'Unknown training augmentation')
    import torch
    episodes = batch['episode_index'].detach().cpu().tolist()
    require(type(step) is int and step > 0 and len(episodes) == 4, 'Mix v1 requires B4 and cumulative step')
    require(all(ep in records and records[ep]['split'] == 'train' for ep in episodes), 'Only TRAIN donors allowed')
    swaps = []
    for i, episode in enumerate(episodes):
        record = records[episode]
        intent, target = record['intent'], record['target_action']
        if intent not in ('STOP_ALL', 'WATER_ALL'):
            continue
        expected = [0, 0, 0] if intent == 'STOP_ALL' else [record['intent_params']['level']] * 3
        require(target == expected, 'Override target does not satisfy image-independent policy')
        digest = hashlib.sha256(f'override_image_mix_v1:{step}:{episode}'.encode()).digest()
        donor = (i + 1 + int.from_bytes(digest[:4], 'big') % 3) % 4
        swaps.append((i, donor, episode, intent, target))
    changed = {i: False for i, *_ in swaps}
    # Read from the original tensor for every donor, including cyclic swaps.
    for camera in CAMERAS:
        original = batch[camera]
        require(tuple(original.shape) == (4, 1, 3, 256, 256), 'Mix requires native three-camera256 inputs')
        if swaps:
            mixed = original.clone()
            for i, donor, *_ in swaps:
                mixed[i].copy_(original[donor])
                changed[i] |= not torch.equal(original[i], original[donor])
            batch[camera] = mixed
    return {'version': mode, 'step': step, 'rows_replaced': len(swaps),
            'mappings': [{'episode': ep, 'images_from_episode': episodes[donor], 'intent': intent,
                          'target_action': target, 'pixels_changed': changed[i]}
                         for i, donor, ep, intent, target in swaps]}
