import pytest
import torch

from potentials.synthetic.student import StudentT0, StudentT1


@pytest.mark.parametrize('target_class', [StudentT0, StudentT1])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (3,), (1, 3, 5)])
def test_compute(target_class, batch_shape):
    torch.manual_seed(0)
    u = target_class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))
    out = u(x)
    assert out.shape == batch_shape
    assert torch.isfinite(out).all()

@pytest.mark.parametrize('target_class', [StudentT0, StudentT1])
@pytest.mark.parametrize('sample_shape', [(1,), (2,), (3,), (1, 3, 5)])
def test_sample(target_class, sample_shape):
    torch.manual_seed(0)
    u = target_class()
    out = u.sample(sample_shape)
    assert out.shape == (*sample_shape, *u.event_shape)
    assert torch.isfinite(out).all()

# Do not check for T0, as its empirical mean has infinite variance
@pytest.mark.skip  # problematic case, as empirical mean has infinite/huge variance and variance can be infinite...
@pytest.mark.parametrize('target_class', [StudentT1])
def test_moments(target_class):
    # Simplified case so we do not need a huge number of samples
    torch.manual_seed(0)
    u = target_class(event_shape=(2,))
    x_sampled = u.sample((10000000,))

    assert torch.allclose(torch.mean(x_sampled, dim=0), u.mean, atol=1e-2)
    assert torch.allclose(torch.var(x_sampled, dim=0), u.variance, atol=1e-2)
