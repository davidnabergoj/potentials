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
        """
        Compute the negative log probability density of samples x under this model.
        """
        raise NotImplementedError

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score of samples x under this model.
        The score is the gradient of the negative log probability density under inputs x with respect to inputs x.
        """
        v = x.clone()  # Clone the inputs
        v.requires_grad_(True)  # Ensure v requires grad
        with torch.enable_grad():
            neg_log_prob = self.compute(v)
            score = -torch.autograd.grad(
                neg_log_prob.sum(),
                v,
                create_graph=True
            )[0].detach()
        return score

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


class SplitPosteriorPotential(Potential):
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


class ConcatenatedPotential(Potential):
    def __init__(self,
                 *potentials: Potential):
        n_dim = 0
        for p in potentials:
            assert len(p.event_shape) == 1
            n_dim += p.event_shape[0]
        event_shape = (n_dim,)
        self.potentials = potentials
        super().__init__(event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        dim_start = 0
        dim_end = None
        u = torch.zeros(size=x.shape[:-1]).to(x)
        for potential in self.potentials:
            if dim_end is None:
                dim_end = potential.event_shape[0]
            else:
                dim_start = dim_end
                dim_end = dim_start + potential.event_shape[0]
            u += potential.compute(x[..., dim_start:dim_end])
        return u

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        return torch.concat([
            p.sample(batch_shape) for p in self.potentials
        ], dim=-1)
