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

_optional_hyperprior = [
    GermanCredit,
    SparseGermanCredit,
    SyntheticItemResponseTheory
]


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_compute(batch_shape, _class):
    torch.manual_seed(0)

    u = _class()
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ret = u(x)

    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all)
def test_lppd(batch_shape, _class):
    torch.manual_seed(0)

    u = _class()
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    lppd = u.normalized_log_posterior_predictive_density(x)
    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('n_draws', [1, 2, 10])
@pytest.mark.parametrize('_class', _all)
def test_posterior_predictive_draws(batch_shape, _class, n_draws):
    torch.manual_seed(0)

    u = _class()
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ppd = u.posterior_predictive_draws(x, n_draws)
    assert isinstance(ppd, torch.Tensor)
    assert torch.isfinite(ppd).all()
    assert ppd.shape[:-1] == (n_draws, *batch_shape)


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _reduced_dataset)
@pytest.mark.parametrize('n_data', [0, 50])
def test_reduced_dataset_compute(batch_shape, _class, n_data):
    torch.manual_seed(0)

    u = _class(n_data=n_data)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ret = u(x)

    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _reduced_dataset)
@pytest.mark.parametrize('n_data', [0, 50])
def test_reduced_dataset_lppd(batch_shape, _class, n_data):
    torch.manual_seed(0)

    u = _class(n_data=n_data)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    lppd = u.normalized_log_posterior_predictive_density(x)

    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('n_draws', [1, 2, 10])
@pytest.mark.parametrize('_class', _reduced_dataset)
@pytest.mark.parametrize('n_data', [0, 50])
def test_posterior_predictive_draws_reduced_dataset(batch_shape, _class, n_draws, n_data):
    torch.manual_seed(0)

    u = _class(n_data=n_data)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ppd = u.posterior_predictive_draws(x, n_draws)
    assert isinstance(ppd, torch.Tensor)
    assert torch.isfinite(ppd).all()
    assert ppd.shape[:-1] == (n_draws, *batch_shape)

@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _optional_hyperprior)
def test_hyperprior_compute(batch_shape, _class):
    torch.manual_seed(0)

    u = _class(use_hyperprior=True)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ret = u(x)

    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _optional_hyperprior)
def test_hyperprior_lppd(batch_shape, _class):
    torch.manual_seed(0)

    u = _class(use_hyperprior=True)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    lppd = u.normalized_log_posterior_predictive_density(x)

    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('n_draws', [1, 2, 10])
@pytest.mark.parametrize('_class', _optional_hyperprior)
def test_posterior_predictive_draws_hyperprior(batch_shape, _class, n_draws):
    torch.manual_seed(0)

    u = _class(use_hyperprior=True)
    x = torch.rand(size=(*batch_shape, *u.event_shape)) * 2 - 1

    ppd = u.posterior_predictive_draws(x, n_draws)
    assert isinstance(ppd, torch.Tensor)
    assert torch.isfinite(ppd).all()
    assert ppd.shape[:-1] == (n_draws, *batch_shape)
