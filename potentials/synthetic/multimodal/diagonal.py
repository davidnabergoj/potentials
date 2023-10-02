import torch

from potentials.synthetic.mixture import Mixture
from potentials.synthetic.gaussian.diagonal import (
    DiagonalGaussian0,
    DiagonalGaussian1,
    DiagonalGaussian2,
    DiagonalGaussian3,
    DiagonalGaussian4,
    DiagonalGaussian5
)


class MultimodalDiagonalGaussian0(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian0(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalDiagonalGaussian1(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian1(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalDiagonalGaussian2(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian2(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalDiagonalGaussian3(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian3(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalDiagonalGaussian4(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian4(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalDiagonalGaussian5(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [DiagonalGaussian5(n_dim=n_dim) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
