import pytest

import torch

from potentials.real import (
    BiochemicalOxygenDemand,
    EightSchools,
    GermanCredit,
    SparseGermanCredit,
    SyntheticItemResponseTheory,
    RadonVaryingSlopes,
    RadonVaryingIntercepts,
    RadonVaryingInterceptsAndSlopes,
    StochasticVolatilityModel
)

from potentials.synthetic.rosenbrock import Rosenbrock
from potentials.synthetic.phi4 import Phi4


@pytest.mark.parametrize('potential_class', [
    BiochemicalOxygenDemand,
    EightSchools,
    GermanCredit,
    SparseGermanCredit,
    SyntheticItemResponseTheory,
    RadonVaryingSlopes,
    RadonVaryingIntercepts,
    RadonVaryingInterceptsAndSlopes,
    StochasticVolatilityModel,
])
def test_basic(potential_class):
    potential = potential_class()
    assert torch.all(torch.isfinite(potential.mean))
    assert torch.all(torch.isfinite(potential.second_moment))


def test_rosenbrock():
    potential = Rosenbrock(event_shape=(100,), scale=10.0)
    assert torch.all(torch.isfinite(potential.mean))
    assert torch.all(torch.isfinite(potential.second_moment))


@pytest.mark.parametrize('length', [8, 16, 32, 64, 128, 256])
def test_phi4(length):
    potential = Phi4(length=length)
    assert torch.all(torch.isfinite(potential.mean))
    assert torch.all(torch.isfinite(potential.second_moment))
