from typing import Union, Tuple

import torch

from potentials.base import Potential


class Rosenbrock(Potential):
    def __init__(self, event_shape: Union[int, Tuple[int, ...]] = (10,), scale: float = 10.0):
        super().__init__(event_shape=event_shape)
        self.scale = scale

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = torch.flatten(x, start_dim=-len(self.event_shape), end_dim=-1)
        first_term_mask = torch.arange(self.n_dim) % 2 == 0
        second_term_mask = ~first_term_mask
        return torch.sum(
            torch.add(
                self.scale * (x_flat[..., first_term_mask] ** 2 - x_flat[..., second_term_mask]) ** 2,
                (x_flat[..., first_term_mask] - 1) ** 2
            ),
            dim=1
        )

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        raise NotImplementedError
