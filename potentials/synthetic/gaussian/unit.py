import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape


class StandardGaussian(Potential):
    def __init__(self, event_shape):
        super().__init__(event_shape=event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        sum_dims = list(range(len(batch_shape), len(batch_shape) + len(self.event_shape)))
        return torch.sum(x ** 2, dim=sum_dims)
