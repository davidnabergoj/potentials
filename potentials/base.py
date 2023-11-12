from typing import Union, Tuple

import torch
import torch.nn as nn


class Potential(nn.Module):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__()
        self.event_shape = event_shape
        self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))

    @property
    def variance(self):
        """
        :return: marginal variances for each dimension.
        """
        try:
            x = self.sample((10000,))
            return torch.var(x, dim=0)
        except NotImplementedError as e:
            raise e

    @property
    def mean(self):
        """
        :return: mean for each dimension.
        """
        try:
            x = self.sample((10000,))
            return torch.mean(x, dim=0)
        except NotImplementedError as e:
            raise e

    @property
    def second_moment(self):
        return self.variance + self.mean ** 2

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_grad(self, x: torch.Tensor):
        x_clone = torch.clone(x)
        x_clone.requires_grad_(True)
        return torch.autograd.grad(self.compute(x_clone).sum(), x_clone)[0].detach()

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        raise NotImplementedError


class PotentialSimple(Potential):
    """
    Potential with a length one event shape.
    """

    def __init__(self, n_dim: int):
        self.n_dim = n_dim
        event_shape = (n_dim,)
        super().__init__(event_shape=event_shape)
