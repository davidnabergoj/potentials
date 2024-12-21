from typing import Union, Tuple

import torch
import torch.nn as nn


class Potential(nn.Module):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...], int]):
        super().__init__()
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        self.event_shape = event_shape
        self.n_dim = int(torch.prod(torch.as_tensor(event_shape)))

    @property
    def normalization_constant(self) -> float:
        """
        Normalization constant value for an overarching statistical distribution.

        If the potential U(x) defines a statistical distribution p(x) via p(x) = exp(U(x)) / z, then z > 0 is the
         normalization constant.
        """
        raise NotImplementedError

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

    def __init__(self, event_shape: Union[Tuple[int, ...], int]):
        assert len(event_shape) == 1
        n_dim = event_shape[0]
        self.n_dim = n_dim
        event_shape = (n_dim,)
        super().__init__(event_shape=event_shape)


class StructuredPotential(Potential):
    def __init__(self, event_shape: Union[torch.Size, Tuple[int, ...]]):
        super().__init__(event_shape)

    @property
    def edge_list(self):
        raise NotImplementedError


class PosteriorPotential(Potential):
    """
    Potential U(x) = P(x) + L(x) consisting of two components:
    * a "prior potential" P (defined by the negative log density of a prior distribution)
    * a "likelihood potential" L (defined by the negative log density of a likelihood function)
    """

    def __init__(self,
                 prior_potential: Potential,
                 likelihood_potential: Potential):
        assert prior_potential.event_shape == likelihood_potential.event_shape
        super().__init__(event_shape=prior_potential.event_shape)
        self.prior_potential = prior_potential
        self.likelihood_potential = likelihood_potential

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior_potential(x) + self.likelihood_potential(x)
