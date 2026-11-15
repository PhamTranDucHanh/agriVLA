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
