import pytest
import torch
import math

from potentials.synthetic.gaussian.diagonal import DiagonalGaussian, gaussian_potential_v2


def test_basic():
    potential = DiagonalGaussian(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 2.0]))
    x = torch.tensor([[1.0, 1.0]])
    u = potential(x)
    true_log_prob = math.log(1 / (4 * math.pi)) - 5 / 8
    true_u = -true_log_prob
    assert torch.isclose(u, torch.tensor(true_u))


@pytest.mark.parametrize('batch_shape', [(2,), (5,), (10,), (1, 3, 5)])
@pytest.mark.parametrize('event_shape', [(2,), (5,), (10,), (1, 3, 5)])
def test_gaussian_potential_v2(batch_shape, event_shape):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))
    mu = torch.randn(size=event_shape)
    sigma = torch.randn(size=event_shape) ** 2

    u = gaussian_potential_v2(x, mu, sigma)
    assert u.shape == batch_shape
    assert torch.isfinite(u).all()
