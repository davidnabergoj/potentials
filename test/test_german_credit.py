import pytest
import torch

from potentials.real.german_credit import GermanCredit, SparseGermanCredit


@pytest.mark.skip(reason="Avoid re-downloading the German credit dataset")
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
def test_regular(batch_shape):
    torch.manual_seed(0)
    u = GermanCredit()
    x = torch.randn(size=(*batch_shape, 26))
    ret = u(x)
    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))


@pytest.mark.skip(reason="Avoid re-downloading the German credit dataset")
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
def test_sparse(batch_shape):
    torch.manual_seed(0)
    u = SparseGermanCredit()
    x = torch.randn(size=(*batch_shape, 51))
    ret = u(x)
    assert ret.shape == batch_shape
    assert torch.all(~torch.isnan(ret))
    assert torch.all(~torch.isinf(ret))
