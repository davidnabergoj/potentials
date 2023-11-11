import pytest

from potentials.synthetic.gaussian.full_rank import (
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5
)


@pytest.mark.parametrize("n_dim", [2, 4, 10, 50, 100])
@pytest.mark.parametrize("u_class", [
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5
])
def test_constructor(n_dim, u_class):
    u_class(n_dim)
