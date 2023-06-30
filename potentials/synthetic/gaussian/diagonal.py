import math
from typing import Union, Tuple

import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape, unsqueeze_to_batch


def gaussian_potential(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    return 0.5 * ((x - mu) / sigma) ** 2 + 0.5 * torch.log(2 * math.pi * sigma)


class DiagonalGaussian(Potential):
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        event_shape = mu.shape
        super().__init__(event_shape=event_shape)
        self.mu = mu
        self.sigma = sigma

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        sum_dims = list(range(len(batch_shape), len(batch_shape) + len(self.event_shape)))
        mu = unsqueeze_to_batch(self.mu, batch_shape)
        sigma = unsqueeze_to_batch(self.sigma, batch_shape)
        return torch.sum(gaussian_potential(x, mu, sigma), dim=sum_dims)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        mu = unsqueeze_to_batch(self.mu, batch_shape)
        sigma = unsqueeze_to_batch(self.sigma, batch_shape)
        return torch.randn(*batch_shape, *self.event_shape) * sigma + mu
