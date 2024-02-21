import torch

from potentials.base import PotentialSimple
from potentials.utils import sum_except_batch, get_batch_shape


class DoubleWell(PotentialSimple):
    def __init__(self, n_dim: int, distance: float = 4.0):
        # Contains 2^{n_dim} modes
        super().__init__(n_dim)
        assert distance > 0
        self.distance = distance

    @property
    def mean(self) -> torch.Tensor:
        return torch.zeros(self.n_dim)

    def compute(self, x: torch.Tensor):
        y = (x ** 2 - self.distance) ** 2
        return sum_except_batch(y, batch_shape=get_batch_shape(y, self.event_shape))
