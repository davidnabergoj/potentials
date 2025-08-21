import pytest
import torch

from potentials.real.german_credit import GermanCredit, SparseGermanCredit


@pytest.mark.local_only
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('gc_class', [GermanCredit, SparseGermanCredit])
def test_compute(batch_shape, gc_class):
    torch.manual_seed(0)
    u = gc_class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))
    ret = u(x)
    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.local_only
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('gc_class', [GermanCredit, SparseGermanCredit])
def test_lppd(batch_shape, gc_class):
    torch.manual_seed(0)
    u = gc_class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    lppd = u.normalized_log_posterior_predictive_density(x)
    assert isinstance(lppd, torch.Tensor)
    assert torch.isfinite(lppd)
    assert lppd.shape == ()
