import math

import pytest
import torch

from potentials.synthetic.gaussian.diagonal import DiagonalGaussian
from potentials.synthetic.gaussian.full_rank import FullRankGaussian1
from potentials.synthetic.gaussian.mixture import (
    GaussianMixture0,
    GaussianMixture1,
    GaussianMixture2,
    GaussianMixture3,
    GaussianMixture4,
    GaussianMixture5,
    GaussianMixture6,
    GaussianMixture7,
    GaussianMixture8,
    GaussianMixture9,
)

_n_dim_classes = [
    GaussianMixture0,
    GaussianMixture1,
    GaussianMixture2,
    GaussianMixture3,
    GaussianMixture4,
    GaussianMixture5,
    GaussianMixture6,
    GaussianMixture7,
    GaussianMixture8,
]


def _construct(_class):
    if _class is GaussianMixture9:
        return _class()
    return _class(n_dim=5)


def _component_log_prob(potential, x):
    if isinstance(potential, DiagonalGaussian):
        return torch.distributions.Normal(potential.mu, potential.sigma).log_prob(x).sum(dim=-1)
    return potential.dist.log_prob(x)


@pytest.mark.parametrize('_class', _n_dim_classes + [GaussianMixture9])
def test_weights_valid(_class):
    u = _construct(_class)
    assert torch.all(u.weights >= 0)
    assert torch.isclose(u.weights.sum(), torch.tensor(1.0))
    assert len(u.weights) == u.n_components


@pytest.mark.parametrize('batch_shape', [(1,), (17,), (2, 3)])
@pytest.mark.parametrize('_class', _n_dim_classes + [GaussianMixture9])
def test_compute_matches_reference(_class, batch_shape):
    torch.manual_seed(0)
    u = _construct(_class)
    x = u.sample(batch_shape)
    ret = u(x)
    assert ret.shape == batch_shape
    assert torch.all(torch.isfinite(ret))

    component_log_probs = torch.stack([_component_log_prob(p, x) for p in u.potentials], dim=-1)
    expected = -torch.logsumexp(torch.log(u.weights) + component_log_probs, dim=-1)
    assert torch.allclose(ret, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize('_class', _n_dim_classes + [GaussianMixture9])
def test_moments_match_samples(_class):
    torch.manual_seed(0)
    u = _construct(_class)
    n = 50_000
    x = u.sample((n,))
    assert x.shape == (n, *u.event_shape)

    mean = u.mean
    variance = u.variance
    assert torch.all(variance > 0)
    # Mean error is within 5 standard errors
    assert torch.all(torch.abs(x.mean(dim=0) - mean) < 5 * torch.sqrt(variance / n))
    assert torch.allclose(x.var(dim=0), variance, rtol=0.1)


@pytest.mark.parametrize('_class', [GaussianMixture1, GaussianMixture2, GaussianMixture3])
def test_components_are_distinct(_class):
    u = _construct(_class)
    assert not torch.allclose(u.potentials[0].mean, u.potentials[1].mean)


def test_gaussian_mixture8_variance():
    u = GaussianMixture8(n_dim=3)
    # Law of total variance: E[Var] + Var[E] = 1 + 0.8 * 0.2 * (10 - (-10)) ** 2
    assert torch.allclose(u.variance, torch.full((3,), 65.0))
    assert torch.allclose(u.mean, torch.full((3,), -6.0))


@pytest.mark.parametrize('_class', [GaussianMixture5, GaussianMixture7])
def test_seeded_weights_reproducible(_class):
    assert torch.equal(_class(n_dim=3, seed=0).weights, _class(n_dim=3, seed=0).weights)
    assert not torch.equal(_class(n_dim=3, seed=0).weights, _class(n_dim=3, seed=1).weights)


def test_full_rank_gaussian_moments():
    p = FullRankGaussian1(n_dim=4)
    assert torch.equal(p.mean, p.loc)
    assert torch.allclose(p.variance, torch.diagonal(p.cov), atol=1e-5)
    assert math.isclose(p.normalization_constant, 1.0)
