import pytest
import torch

from potentials.real.biochemical_oxygen_demand import BiochemicalOxygenDemand


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (10,), (1, 3, 5, 7)])
def test_basic(batch_shape):
    torch.manual_seed(0)
    potential = BiochemicalOxygenDemand()
    x0 = torch.randn(size=(*batch_shape, 2))

    u = potential(x0)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))
