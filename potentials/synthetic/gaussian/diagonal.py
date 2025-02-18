import math
from typing import Union, Tuple
import numpy as np
import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape, unsqueeze_to_batch, sum_except_batch


def gaussian_potential(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    return 0.5 * ((x - mu.to(x)) / sigma.to(x)) ** 2 + 0.5 * math.log(2 * math.pi) + torch.log(sigma.to(x))


class DiagonalGaussian(Potential):
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        event_shape = mu.shape
        super().__init__(event_shape=event_shape)
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        mu = unsqueeze_to_batch(self.mu, batch_shape)
        sigma = unsqueeze_to_batch(self.sigma, batch_shape)
        return sum_except_batch(gaussian_potential(x, mu, sigma), batch_shape)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        mu = unsqueeze_to_batch(self.mu, batch_shape)
        sigma = unsqueeze_to_batch(self.sigma, batch_shape)
        return torch.randn(*batch_shape, *self.event_shape).to(self.mu) * sigma + mu

    @property
    def variance(self):
        return self.sigma ** 2

    @property
    def mean(self):
        return self.mu

    @property
    def normalization_constant(self) -> float:
        return 1.0


class DiagonalGaussian0(DiagonalGaussian):
    """
    Eigenvalues are reciprocals of Gamma distribution samples.
    """

    def __init__(self,
                 event_shape: Union[Tuple[int, ...], int] = (100,),
                 gamma_shape: float = 0.5,
                 seed: int = 0):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        mu = torch.zeros(event_shape)
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(1 / np.sort(rng.gamma(shape=gamma_shape, scale=1.0, size=event_shape)))
        super().__init__(mu, eigenvalues)


class DiagonalGaussian1(DiagonalGaussian):
    """
    Eigenvalues are linearly spaced between 1 and 10.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,)):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        mu = torch.zeros(event_shape)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        eigenvalues = torch.linspace(1, 10, n_dim).view_as(mu)
        super().__init__(mu, eigenvalues)


class DiagonalGaussian2(DiagonalGaussian):
    """
    Log of the eigenvalues sampled from standard normal.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), seed: int = 0):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        mu = torch.zeros(event_shape)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(np.exp(rng.randn(n_dim))).view_as(mu)
        super().__init__(mu, eigenvalues)


class DiagonalGaussian3(DiagonalGaussian):
    """
    First eigenvalue is 1000, remainder are 1.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,)):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        mu = torch.zeros(event_shape)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        eigenvalues = torch.ones(n_dim).view_as(mu)
        eigenvalues[0] = 1000
        super().__init__(mu, eigenvalues)


class DiagonalGaussian4(DiagonalGaussian):
    """
    First eigenvalue is 1000, second eigenvalue is 1/1000, remainder are 1.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,)):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        assert n_dim >= 2
        mu = torch.zeros(event_shape)
        eigenvalues = torch.ones(n_dim)
        eigenvalues[0] = 1000
        eigenvalues[1] = 1 / 1000
        eigenvalues = eigenvalues.view_as(mu)
        super().__init__(mu, eigenvalues)


class DiagonalGaussian5(DiagonalGaussian):
    """
    Eigenvalues linearly space between 1/1000 and 1000.
    """

    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,)):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        n_dim = int(torch.prod(torch.as_tensor(event_shape)))
        assert n_dim >= 2
        mu = torch.zeros(event_shape)
        eigenvalues = torch.linspace(1 / 1000, 1000, n_dim).view_as(mu)
        super().__init__(mu, eigenvalues)
