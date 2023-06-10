import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape


class DiagonalGaussian(Potential):
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        event_shape = mu.shape
        super().__init__(event_shape=event_shape)
        self.mu = mu
        self.sigma = sigma

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        sum_dims = list(range(len(batch_shape), len(batch_shape) + len(self.event_shape)))
        mu = self.mu.view(*([1] * len(batch_shape)), self.event_shape)
        sigma = self.sigma.view(*([1] * len(batch_shape)), self.event_shape)
        return torch.sum(((x - mu) / sigma) ** 2, dim=sum_dims)
