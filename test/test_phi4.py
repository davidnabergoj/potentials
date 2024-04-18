import pytest
import torch
from potentials.synthetic.phi4 import Phi4


@pytest.mark.parametrize('length', [2, 4, 8, 16, 32])
@pytest.mark.parametrize('temperature', [0.1, 0.8, 1.0, 1.2, 1.4, 1.6])
def test_exhaustive(length: int, temperature: float):
    potential = Phi4(length=length, temperature=temperature)

    torch.manual_seed(0)
    batch_size = 10
    x = torch.randn(size=(batch_size, *potential.event_shape))
    out = potential(x)

    assert torch.all(torch.isfinite(out))
