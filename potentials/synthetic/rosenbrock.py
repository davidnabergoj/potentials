from typing import Union, Tuple

import torch

from potentials.base import Potential


class Rosenbrock(Potential):
    def __init__(self, n_dim: int = 10, scale: float = 10.0):
        super().__init__(event_shape=(n_dim,))
        self.scale = scale

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        first_term_mask = torch.arange(self.n_dim) % 2 == 0
        second_term_mask = ~first_term_mask
        return torch.sum(
            torch.add(
                self.scale * (x[:, first_term_mask] ** 2 - x[:, second_term_mask]) ** 2,
                (x[:, first_term_mask] - 1) ** 2
            ),
            dim=1
        )

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        raise NotImplementedError
