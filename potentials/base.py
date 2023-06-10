from typing import Union, Tuple

import torch


class Potential:
    def __init__(self, event_shape: Union[torch.Size, Tuple[int]]):
        self.event_shape = event_shape

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        raise NotImplementedError
