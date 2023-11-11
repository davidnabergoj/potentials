import torch

from potentials.synthetic.gaussian.diagonal import DiagonalGaussian
from potentials.synthetic.mixture import Mixture


class RandomlyPositionedGaussians(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode has mean ~ N(0, sigma) in all dimensions.
    Each mode has variance 1 in all dimensions.
    """

    def __init__(self, n_dim: int, n_components: int, weights, seed: int = 0, scale: float = 10.0):
        torch.random.fork_rng()
        torch.manual_seed(seed)
        locations = torch.randn(size=(n_components, n_dim)) * scale
        potentials = [DiagonalGaussian(loc, torch.ones(n_dim)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class TenRandomlyPositionedGaussians(RandomlyPositionedGaussians):
    def __init__(self, n_dim: int):
        super().__init__(n_dim, n_components=10, weights=torch.ones(10) / 10)


class GaussianChain0(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode has mean 0 across all but the first dimension.
    Each mode's mean in the first dimension is at a distance of 8 from the previous mode's mean.
    In other words: mu[i+1] = mu[i] + 8.
    The first mean is at the origin in all dimensions, i.e. mu[0] = 0.
    Each mode has variance 1 across all dimensions.
    """

    def __init__(self, n_dim: int, weights):
        n_components = len(weights)
        scales = torch.ones(n_components)
        tmp = torch.zeros(n_components)
        for i in range(1, n_components):
            tmp[i] = 4 * scales[i - 1] + 4 * scales[i]
        first_dim_offsets = torch.cumsum(tmp, dim=0)
        locations = torch.zeros(size=(n_components, n_dim))

        # Change first dim locations of all but the first component
        locations[1:, 0] = first_dim_offsets[1:]
        potentials = [DiagonalGaussian(loc, torch.ones(n_dim)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class DoubleGaussian0(GaussianChain0):
    def __init__(self, n_dim: int, w: float = 0.5):
        if w < 0 or w > 1:
            raise ValueError(f"Mode weight must be between 0 and 1, but got {w}")
        super().__init__(n_dim, torch.tensor([w, 1 - w]))


class TripleGaussian0(GaussianChain0):
    def __init__(self, n_dim: int, w0: float = 1 / 3, w1: float = 1 / 3):
        super().__init__(n_dim, torch.tensor([w0, w1, 1 - w0 - w1]))


class GaussianChain1(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode's mean in each dimension is at a distance of 8 from the previous mode's mean.
    In other words: mu[i+1] = mu[i] + 8.
    The first mean is at the origin in all dimensions, i.e. mu[0] = 0.
    Each mode has variance 1 across all dimensions.
    """

    def __init__(self, n_dim: int, weights):
        n_components = len(weights)
        if any(w < 0 for w in weights):
            raise ValueError(f"Expected weights to be non-negative, but found {min(weights)} in weights.")
        if sum(weights) != 1:
            raise ValueError(f"Expected weights to sum to 1, but got {sum(weights)}")
        scales = torch.ones(n_components)
        tmp = torch.zeros(n_components)
        for i in range(1, n_components):
            tmp[i] = 4 * scales[i - 1] + 4 * scales[i]
        first_dim_offsets = torch.cumsum(tmp, dim=0)
        all_dims_offsets = first_dim_offsets[:, None].repeat(1, n_dim)
        locations = torch.zeros(size=(n_components, n_dim))

        # Change all dim locations of all but the first component
        locations[1:] += all_dims_offsets[1:]
        potentials = [DiagonalGaussian(loc, torch.ones(n_dim)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class DoubleGaussian1(GaussianChain1):
    def __init__(self, n_dim: int, w: float = 0.5):
        if w < 0 or w > 1:
            raise ValueError(f"Mode weight must be between 0 and 1, but got {w}")
        super().__init__(n_dim, torch.tensor([w, 1 - w]))


class TripleGaussian1(GaussianChain1):
    def __init__(self, n_dim: int, w0: float = 1 / 3, w1: float = 1 / 3):
        super().__init__(n_dim, torch.tensor([w0, w1, 1 - w0 - w1]))
