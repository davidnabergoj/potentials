import torch
import numpy as np
from potentials.synthetic.mixture import Mixture
from potentials.synthetic.gaussian.full_rank import (
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5
)
from potentials.synthetic.multimodal.util import generate_mode_locations


class MultimodalFullRankGaussian0(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian0(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian1(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian1(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian2(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian2(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian3(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian3(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian4(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian4(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalFullRankGaussian5(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [FullRankGaussian5(n_dim=n_dim) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
