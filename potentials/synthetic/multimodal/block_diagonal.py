import torch

from potentials.synthetic.mixture import Mixture
from potentials.synthetic.gaussian.block_diagonal import (
    BlockDiagonal0,
    BlockDiagonal1,
    BlockDiagonal2,
    BlockDiagonal3,
    BlockDiagonal4,
    BlockDiagonal5
)
from potentials.synthetic.multimodal.util import generate_mode_locations


class MultimodalBlockDiagonal0(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal0(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalBlockDiagonal1(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal1(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalBlockDiagonal2(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal2(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalBlockDiagonal3(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal3(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalBlockDiagonal4(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal4(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)


class MultimodalBlockDiagonal5(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0, **kwargs):
        potentials = [BlockDiagonal5(n_dim=n_dim, **kwargs) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.dist.loc.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
