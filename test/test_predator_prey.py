import torch

from potentials.real.predator_prey import PredatorPrey


def test_basic():
    torch.manual_seed(0)
    potential = PredatorPrey()
    x0 = torch.randn(size=(1, 8))
    u = potential(x0)
    assert u.shape == (1,)
    assert ~torch.isnan(u)
    assert ~torch.isinf(u)
