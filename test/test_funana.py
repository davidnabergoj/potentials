import pytest
import torch

from potentials.synthetic.funana import Funana


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (3,), (1, 3, 5)])
def test_compute(batch_shape):
    torch.manual_seed(0)
    u = Funana()
    x = torch.randn(size=(*batch_shape, *u.event_shape))
    out = u(x)
    assert out.shape == batch_shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('sample_shape', [(1,), (2,), (3,), (1, 3, 5)])
def test_sample(sample_shape):
    torch.manual_seed(0)
    u = Funana()
    out = u.sample(sample_shape)
    assert out.shape == (*sample_shape, *u.event_shape)
    assert torch.isfinite(out).all()


def test_moments_easy():
    # Simplified case so we do not need a huge number of samples
    torch.manual_seed(0)
    u = Funana(event_shape=(4,),
               sigma_0=1.2,
               sigma_1=1.2,
               alpha=0.5,
               beta=0.5)
    x_sampled = u.sample((1000000,))

    assert torch.allclose(torch.mean(x_sampled, dim=0), u.mean, atol=1e-2)
    assert torch.allclose(torch.var(x_sampled, dim=0), u.variance, atol=1e-2)
