import torch

from potentials.synthetic.mixture import Mixture
from potentials.synthetic.gaussian.full_rank import (
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5
)


class MultimodalFullRankGaussian0(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian0(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian1(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian1(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian2(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian2(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian3(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian3(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian4(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian4(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian5(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [FullRankGaussian5(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
