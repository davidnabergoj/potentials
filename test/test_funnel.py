import pytest
import torch

from potentials.synthetic.funnel import Funnel, BimodalFunnel


@pytest.mark.parametrize('potential_class', [Funnel, BimodalFunnel])
@pytest.mark.parametrize('n_dim', [2, 10, 100])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_evaluation(batch_shape, n_dim, potential_class):
    torch.manual_seed(0)
    potential = potential_class(n_dim)
    x = torch.randn(size=(*batch_shape, *potential.event_shape))
    u = potential(x)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))


@pytest.mark.parametrize('potential_class', [Funnel, BimodalFunnel])
@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_sampling(batch_shape, n_dim, potential_class):
    torch.manual_seed(0)
    potential = potential_class(n_dim)
    x = potential.sample(batch_shape)
    assert x.shape == (*batch_shape, *potential.event_shape)
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))


@pytest.mark.parametrize('potential_class', [Funnel, BimodalFunnel])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_fixed_n_dim_evaluation(batch_shape, potential_class):
    torch.manual_seed(0)
    potential = potential_class()
    x = torch.randn(size=(*batch_shape, *potential.event_shape))
    u = potential(x)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))


@pytest.mark.parametrize('potential_class', [Funnel, BimodalFunnel])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_fixed_n_dim_sampling(batch_shape, potential_class):
    torch.manual_seed(0)
    potential = potential_class()
    x = potential.sample(batch_shape)
    assert x.shape == (*batch_shape, *potential.event_shape)
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))
