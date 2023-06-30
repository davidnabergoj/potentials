import pytest
import torch

from potentials.synthetic.funnel import Funnel


@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_shape(batch_shape, n_dim):
    torch.manual_seed(0)
    potential = Funnel(n_dim=n_dim)
    x = torch.randn(size=(*batch_shape, potential.n_dim))
    u = potential(x)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))
