"""Native loader regression only; this does not certify trained GPU resume."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('lerobot')

from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from lerobot.datasets.sampler import EpisodeAwareSampler, compute_sampler_state
from lerobot.utils.utils import cycle
from agri_vla.small_resume import continued_cycle


def test_endpoint_loss_matches_one_step_action_regression_and_preserves_rng():
    from types import SimpleNamespace
    from agri_vla.small_train import configure_training_time
    core = SimpleNamespace()
    policy = SimpleNamespace(config=SimpleNamespace(num_steps=1), get_base_model=lambda: SimpleNamespace(model=core))
    audit = configure_training_time(policy, {'time_sampling': 'endpoint_t1'})
    rng = torch.get_rng_state().clone()
    assert torch.equal(core.sample_time(4, 'cpu'), torch.ones(4))
    assert torch.equal(torch.get_rng_state(), rng)
    assert audit['calls'] == 1 and audit['minimum'] == audit['maximum'] == 1.
    action, noise, velocity = (torch.randn(4, 1, 3) for _ in range(3))
    flow_loss = ((noise - action - velocity) ** 2).mean()
    action_loss = ((action - (noise - velocity)) ** 2).mean()
    assert torch.allclose(flow_loss, action_loss)


@pytest.mark.parametrize('step,records,batch_size', [(99, 8, 2), (100, 8, 2), (99, 60, 4), (100, 60, 4),
                                                  (300, 60, 4), (899, 3600, 4), (900, 3600, 4), (1000, 3600, 4)])
def test_prepared_loader_resume_keeps_epoch_offset_and_cpu_rng(step, records, batch_size):
    accelerator = Accelerator(cpu=True)
    dataset = TensorDataset(torch.arange(records))
    def loader(step=None):
        sampler = EpisodeAwareSampler(list(range(records)), list(range(1, records + 1)), shuffle=True, seed=42)
        if step is not None:
            sampler.load_state_dict(compute_sampler_state(step, records, batch_size, 1))
        return accelerator.prepare(DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0))
    torch.manual_seed(42)
    live = cycle(loader())
    for _ in range(step):
        next(live); torch.randn(7)
    rng = torch.get_rng_state().clone()
    expected = []
    for _ in range(20):
        batch = next(live)[0].clone()
        expected.append((batch, torch.get_rng_state().clone(), torch.randn(7)))
    raw = loader(step); torch.set_rng_state(rng)
    raw_batch = next(cycle(raw))[0]
    assert not torch.equal(raw_batch, expected[0][0]) or not torch.equal(torch.get_rng_state(), expected[0][1])
    fixed = loader(step); torch.set_rng_state(rng)
    iterator = continued_cycle(fixed, cycle, step=step, records=records, batch_size=batch_size)
    for batch, wanted_rng, noise in expected:
        assert torch.equal(next(iterator)[0], batch)
        assert torch.equal(torch.get_rng_state(), wanted_rng)
        assert torch.equal(torch.randn(7), noise)


def test_mc2_uses_two_independent_draws_and_same_pair_for_b1_and_b4():
    from types import SimpleNamespace
    from agri_vla.inference_sampling import configure_inference
    from lerobot.policies.common.flow_matching import sample_noise
    calls=[]
    base=SimpleNamespace(config=SimpleNamespace(chunk_size=1,n_action_steps=1,max_action_dim=32),
                         model=SimpleNamespace(sample_noise=sample_noise))
    def native(batch,noise=None,**kwargs):
        calls.append(noise.clone());return noise[:,:,:3]
    base._get_action_chunk=native
    base.predict_action_chunk=lambda batch,noise=None:base._get_action_chunk(batch,noise=noise)
    base.select_action=lambda batch:base._get_action_chunk(batch)[:,0]
    policy=SimpleNamespace(get_base_model=lambda:base)
    configure_inference(policy,{'inference_samples':2})
    torch.manual_seed(42);expected=(sample_noise((1,1,32),'cpu')+sample_noise((1,1,32),'cpu'))[:,:,:3]/2
    torch.manual_seed(42);first=sample_noise((1,1,32),'cpu')
    batch={'observation.state':torch.zeros(4,3)}
    actual=base.predict_action_chunk(batch,noise=first.repeat(4,1,1))
    assert len(calls)==2 and not torch.equal(calls[0],calls[1])
    assert torch.equal(actual,expected.repeat(4,1,1))
    torch.manual_seed(42)
    assert torch.equal(base.predict_action_chunk({'observation.state':torch.zeros(1,3)}),expected)
    torch.manual_seed(42)
    assert torch.equal(base.select_action({'observation.state':torch.zeros(1,3)}),expected[:,0])


def test_mc2_real_upstream_select_action_and_chunk_share_hook():
    # Real upstream entrypoints, synthetic decoder only. No model training claimed.
    from collections import deque
    from types import SimpleNamespace
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.common.flow_matching import sample_noise
    from agri_vla.inference_sampling import configure_inference
    base=SmolVLAPolicy.__new__(SmolVLAPolicy);torch.nn.Module.__init__(base)
    base.config=SimpleNamespace(chunk_size=1,n_action_steps=1,max_action_dim=32,adapt_to_pi_aloha=False,rtc_config=None)
    base.model=SimpleNamespace(sample_noise=sample_noise)
    base._queues={'action':deque(maxlen=1)}
    base._get_action_chunk=lambda batch,noise=None,**kwargs:noise[:,:,:3]
    policy=SimpleNamespace(get_base_model=lambda:base)
    configure_inference(policy,{'inference_samples':2})
    inputs={'observation.state':torch.zeros(1,3)}
    torch.manual_seed(42);chunk=base.predict_action_chunk(inputs)
    torch.manual_seed(42);action=base.select_action(inputs)
    assert torch.equal(chunk[:,0],action)
