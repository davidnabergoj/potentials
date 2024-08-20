import torch

from potentials.base import Potential


class StochasticVolatilityModel(Potential):
    def __init__(self):
        super().__init__(event_shape=(3003,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        pass
