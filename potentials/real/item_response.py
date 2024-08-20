import torch

from potentials.base import Potential


class SyntheticItemResponseTheory(Potential):
    def __init__(self):
        super().__init__(event_shape=(501,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        pass
