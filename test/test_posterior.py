import pytest
import torch

from potentials.real.german_credit import GermanCredit, SparseGermanCredit
from potentials.real.radon import (
    RadonVaryingIntercepts,
    RadonVaryingInterceptsAndSlopes,
    RadonVaryingSlopes
)
from potentials.real.eight_schools import EightSchools
from potentials.real.item_response import SyntheticItemResponseTheory
from potentials.real.stochastic_volatility import StochasticVolatilityModel

_all = [
    GermanCredit,
    SparseGermanCredit,
    RadonVaryingIntercepts,
    RadonVaryingInterceptsAndSlopes,
    RadonVaryingSlopes,
    EightSchools,
    SyntheticItemResponseTheory,
    StochasticVolatilityModel
]

_reduced_dataset = [
    GermanCredit,
    SparseGermanCredit,
    RadonVaryingIntercepts,
    RadonVaryingInterceptsAndSlopes,
    RadonVaryingSlopes,
]


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_compute(batch_shape, _class):
    torch.manual_seed(0)

    u = _class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    ret = u(x)

    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_lppd(batch_shape, _class):
    torch.manual_seed(0)

    u = _class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    lppd = u.normalized_log_posterior_predictive_density(x)
    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_reduced_dataset_compute(batch_shape, _class):
    torch.manual_seed(0)

    u = _class(n_data=50)
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    ret = u(x)

    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_reduced_dataset_lppd(batch_shape, _class):
    torch.manual_seed(0)
    
    u = _class(n_data=50)
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    lppd = u.normalized_log_posterior_predictive_density(x)

    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()
