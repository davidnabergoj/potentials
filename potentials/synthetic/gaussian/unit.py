from typing import Union, Tuple

import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape


class StandardGaussian(Potential):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,)):
        super().__init__(event_shape=event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        sum_dims = list(range(len(batch_shape), len(batch_shape) + len(self.event_shape)))
        return torch.sum(0.5 * (x ** 2), dim=sum_dims)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        return torch.randn(*batch_shape, *self.event_shape)

    @property
    def mean(self):
        return torch.zeros(size=self.event_shape)

    @property
    def variance(self):
        return torch.ones(size=self.event_shape)
