import torch

import pytest
from potentials.synthetic.rosenbrock import Rosenbrock
from potentials.synthetic.double_well import DoubleWell
from potentials.synthetic.student import StudentT0, StudentT1
from potentials.synthetic.unpreconditionable import UnpreconditionablePotential1
from potentials.synthetic.phi4 import Phi4
from potentials.synthetic.gaussian_mixture import GaussianMixture2D
from potentials.synthetic.multimodal import (
    DoubleGaussian0,
    DoubleGaussian1,
    TenRandomlyPositionedGaussians,
    SimpleTripleGaussian1D,
)
from potentials.synthetic.banana import Banana
from potentials.synthetic.funana import Funana
from potentials.synthetic.funnel import Funnel

_all_classes = [
    Rosenbrock,
    DoubleWell,
    StudentT0,
    StudentT1,
    UnpreconditionablePotential1,
    Phi4,
    GaussianMixture2D,
    DoubleGaussian0,
    DoubleGaussian1,
    TenRandomlyPositionedGaussians,
    SimpleTripleGaussian1D,
    Banana,
    Funana,
    Funnel,
]

_classes_with_known_moments = [
    Banana,
    Funana,
    Funnel
]

_classes_with_known_sample = [
    Banana,
    Funana,
    Funnel
]


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _all_classes)
def test_compute(_class, batch_shape):
    torch.manual_seed(0)

    u = _class()
    x = torch.randn(size=(*batch_shape, *u.event_shape))
    ret = u(x)
    assert ret.shape == batch_shape
    assert torch.all(torch.isfinite(ret))


@pytest.mark.parametrize('sample_shape', [(1,), (2,), (17,), (2, 3, 7, 13)])
@pytest.mark.parametrize('_class', _classes_with_known_sample)
def test_sample(_class, sample_shape):
    torch.manual_seed(0)

    u = _class()
    x = u.sample(sample_shape)
    assert x.shape == (*sample_shape, *u.event_shape)
    assert torch.all(torch.isfinite(x))


@pytest.mark.parametrize('_class', _classes_with_known_moments)
def test_valid_moments(_class):
    torch.manual_seed(0)

    u = _class()

    assert u.mean.shape == u.event_shape
    assert u.second_moment.shape == u.event_shape
    assert u.variance.shape == u.event_shape

    assert torch.all(torch.isfinite(u.mean))
    assert torch.all(torch.isfinite(u.second_moment))
    assert torch.all(torch.isfinite(u.variance))
