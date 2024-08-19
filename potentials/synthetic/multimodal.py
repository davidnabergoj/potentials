from typing import List, Tuple, Union

import torch

from potentials.base import Potential
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian
from potentials.synthetic.mixture import Mixture


class RandomlyPositionedGaussians(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode has mean ~ N(0, sigma) in all dimensions.
    Each mode has variance 1 in all dimensions.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int], n_components: int, weights, seed: int = 0,
                 scale: float = 10.0):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        locations = torch.randn(size=(n_components, *event_shape)) * scale
        potentials = [DiagonalGaussian(loc, torch.ones(size=event_shape)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class TenRandomlyPositionedGaussians(RandomlyPositionedGaussians):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (10,)):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        super().__init__(event_shape, n_components=10, weights=torch.ones(10) / 10)


class GaussianChain(Mixture):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], int],
                 means: list[float],
                 scales: list[float],
                 weights: list[float]):
        assert len(torch.as_tensor(means).shape) == 1
        assert len(torch.as_tensor(scales).shape) == 1
        assert len(torch.as_tensor(weights).shape) == 1
        assert len(means) == len(scales)
        assert len(means) == len(weights)

        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        assert n_dim >= 1

        n_components = len(means)

        potentials = []
        for i in range(n_components):
            potentials.append(DiagonalGaussian(
                mu=torch.full(size=event_shape, fill_value=means[i]),
                sigma=torch.full(size=event_shape, fill_value=scales[i]),
            ))
        super().__init__(potentials, torch.as_tensor(weights))


class GaussianChain0(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode has mean 0 across all but the first dimension.
    Each mode's mean in the first dimension is at a distance of 8 from the previous mode's mean.
    In other words: mu[i+1] = mu[i] + 8.
    The first mean is at the origin in all dimensions, i.e. mu[0] = 0.
    Each mode has variance 1 across all dimensions.
    """

    def __init__(self, event_shape: Union[int, Tuple[int, ...]], weights):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        n_components = len(weights)
        if any(w < 0 for w in weights):
            raise ValueError(f"Expected weights to be non-negative, but found {min(weights)} in weights.")
        if sum(weights) != 1:
            raise ValueError(f"Expected weights to sum to 1, but got {sum(weights)}")
        locations = 8 * torch.arange(n_components)[[None] * len(event_shape)].T.repeat(1, *event_shape)
        mask = torch.ones(size=event_shape).bool()
        mask[[0] * len(event_shape)] = False
        locations[:, mask] = 0
        potentials = [DiagonalGaussian(loc, torch.ones(size=event_shape)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class DoubleGaussian0(GaussianChain0):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), w: float = 0.5):
        if w < 0 or w > 1:
            raise ValueError(f"Mode weight must be between 0 and 1, but got {w}")
        super().__init__(event_shape, torch.tensor([w, 1 - w]))


class TripleGaussian0(GaussianChain0):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), w0: float = 1 / 3, w1: float = 1 / 3):
        super().__init__(event_shape, torch.tensor([w0, w1, 1 - w0 - w1]))


class GaussianChain1(Mixture):
    """
    Multimodal distribution with unit Gaussian modes.

    Each mode's mean in each dimension is at a distance of 8 from the previous mode's mean.
    In other words: mu[i+1] = mu[i] + 8.
    The first mean is at the origin in all dimensions, i.e. mu[0] = 0.
    Each mode has variance 1 across all dimensions.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int], weights):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        n_components = len(weights)
        if any(w < 0 for w in weights):
            raise ValueError(f"Expected weights to be non-negative, but found {min(weights)} in weights.")
        if sum(weights) != 1:
            raise ValueError(f"Expected weights to sum to 1, but got {sum(weights)}")
        locations = 8 * torch.arange(n_components)[[None] * len(event_shape)].T.repeat(1, *event_shape)
        potentials = [DiagonalGaussian(loc, torch.ones(size=event_shape)) for loc in locations]
        super().__init__(potentials, torch.as_tensor(weights))


class DoubleGaussian1(GaussianChain1):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), w: float = 0.5):
        if w < 0 or w > 1:
            raise ValueError(f"Mode weight must be between 0 and 1, but got {w}")
        super().__init__(event_shape, torch.tensor([w, 1 - w]))


class TripleGaussian1(GaussianChain1):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), w0: float = 1 / 3, w1: float = 1 / 3):
        super().__init__(event_shape, torch.tensor([w0, w1, 1 - w0 - w1]))


class TripleGaussian2(GaussianChain):
    def __init__(self, event_shape: Union[Tuple[int, ...], int]):
        means = [0., 8., 16.]
        scales = [0.2, 0.4, 1.6]
        weights = [4 / 7, 2 / 7, 1 / 7]
        super().__init__(event_shape, means, scales, weights)


class TripleGaussian3(GaussianChain):
    def __init__(self, event_shape: Union[Tuple[int, ...], int]):
        means = [-3., 0., 3.]
        scales = [0.1, 0.1, 0.1]
        weights = [1 / 3, 1 / 3, 1 / 3]
        super().__init__(event_shape, means, scales, weights)


class SimpleTripleGaussian1D(GaussianChain):
    def __init__(self):
        event_shape = (1,)
        means = [-5.0, 0.0, 5.0]
        scales = [0.7, 0.7, 0.7]
        weights = [1 / 3, 1 / 3, 1 / 3]
        super().__init__(event_shape, means, scales, weights)
