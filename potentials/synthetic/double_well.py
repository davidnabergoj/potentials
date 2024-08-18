from typing import Union, Tuple

import torch

from potentials.base import PotentialSimple
from potentials.utils import sum_except_batch, get_batch_shape


class DoubleWell(PotentialSimple):
    def __init__(self, event_shape: Union[int, Tuple[int, ...]], distance: float = 4.0):
        # Contains 2^{n_dim} modes
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        super().__init__(event_shape)
        assert distance > 0
        self.distance = distance

    @property
    def mean(self) -> torch.Tensor:
        return torch.zeros(size=self.event_shape)

    @property
    def variance(self) -> torch.Tensor:
        if self.distance == 4:
            marginal_var = 3.9341046423344384502056756017641461798253050095405973357175507668
            # Integrate[exp(-(x^2-4)^2) * x^2,{x,-∞,∞}] / Integrate[exp(-(y^2-4)^2),{y,-∞,∞}]
        elif self.distance == 2:
            marginal_var = 1.8353417214904593599217904198784863635172262491493630166971601894
        else:
            raise ValueError
        return torch.full(size=self.event_shape, fill_value=marginal_var)

    def compute(self, x: torch.Tensor):
        y = (x ** 2 - self.distance) ** 2
        return sum_except_batch(y, batch_shape=get_batch_shape(y, self.event_shape))
