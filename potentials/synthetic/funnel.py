import math

import torch
from typing import Union, Tuple

from potentials.synthetic.mixture import Mixture
from potentials.base import StructuredPotential, Potential
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian, gaussian_potential
from potentials.utils import get_batch_shape, sum_except_batch


class ShiftedFunnelBase(StructuredPotential):
    """
    Implements the distribution p(x1, x2, ... xd) = p(x1) p(x2, ..., xd | x1), where
    p(x1) is a given 1d distribution and p(xi|x1) = N(.; mu, sigma=exp((x1 - mu) / 2)) with mu = E[p(x1)].
    """

    def __init__(self,
                 event_shape: Union[int, Tuple[int, ...]],
                 base_potential_1d: Potential):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        assert len(event_shape) == 1
        n_dim = event_shape[0]
        assert n_dim >= 2
        super().__init__(event_shape=(n_dim,))
        assert base_potential_1d.event_shape == (1,)
        self.base_potential_1d = base_potential_1d
        self.funnel_loc = self.base_potential_1d.mean  # shape = (1,)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        xi = x[..., 1:]  # (*batch_shape, n_dim - 1)
        funnel_scale = torch.exp((x[..., 0] - self.funnel_loc) / 2)[..., None].repeat(
            tuple([1] * len(batch_shape) + [self.n_dim - 1]))  # (*batch_shape, n_dim - 1)
        u_xi = sum_except_batch(gaussian_potential(
            xi, self.funnel_loc, funnel_scale), batch_shape)
        u_x1 = self.base_potential_1d(x[..., 0][..., None])
        return u_x1 + u_xi

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        x = torch.zeros(*batch_shape, self.n_dim)
        x[..., 0] = self.base_potential_1d.sample(batch_shape)[..., 0]
        x[..., 1:self.n_dim] = torch.randn(
            *batch_shape, self.n_dim - 1) * torch.exp(x[..., 0][..., None] / 2) + self.funnel_loc
        return x

    @property
    def mean(self):
        return torch.concat([self.base_potential_1d.mean for _ in range(self.n_dim)], dim=0)

    @property
    def variance(self):
        # x0 has variance V
        # We write the derivation for x1, x2, ...
        # > Law of total variance says: Var[Y] = E[Var[Y|X]] + Var[E[X|Y]]
        # > Second term is 0, so this becomes : Var[Y] = E[Var[Y|X]]
        # > We know Var[Y|X] = exp(X)
        # > E[exp(X)] is related to the Moment generating function E[exp(tX)] at t = 1.
        # > If X ~ N(mu, sigma), then the MGF E[exp(tX)] equals exp(mu*t + sigma**2 * t**2 / 2)
        # > In our case X = x0 and e.g. Y = x1
        # > We have mu = 0, sigma = sqrt(V)
        # > Therefore at t = 1, the MGF equals exp(V/2)
        var_x0 = self.base_potential_1d.variance
        var_rest = torch.full(size=(self.n_dim - 1,),
                              fill_value=float(math.exp(float(var_x0 / 2))))
        return torch.concat([var_x0, var_rest], dim=0)

    @property
    def edge_list(self):
        return [(0, i) for i in range(1, self.n_dim)]


class FunnelBase(StructuredPotential):
    """
    Implements the distribution p(x1, x2, ... xd) = p(x1) p(x2, ..., xd | x1), where
    p(x1) is a given 1d distribution and p(xi|x1) = N(.; 0, sigma=exp(x1/2)).
    """

    def __init__(self, event_shape: Union[int, Tuple[int, ...]], base_potential_1d: Potential):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        assert len(event_shape) == 1
        n_dim = event_shape[0]
        assert n_dim >= 2
        self.n_dim = n_dim
        super().__init__(event_shape=(n_dim,))
        assert base_potential_1d.event_shape == (1,)
        self.base_potential_1d = base_potential_1d

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        u_x1 = self.base_potential_1d(
            x[..., 0][..., None])  # Take the first dim

        xi = x[..., 1:]  # (*batch_shape, n_dim - 1)
        mu = torch.zeros_like(xi)  # (*batch_shape, n_dim - 1)

        # (*batch_shape, n_dim - 1)
        sigma = torch.exp(x[..., 0] / 2)[..., None].repeat(tuple([1]
                                                                 * len(batch_shape) + [self.n_dim - 1]))

        u_xi = sum_except_batch(gaussian_potential(xi, mu, sigma), batch_shape)
        return u_x1 + u_xi

    def _conditional_sample(self, x0: torch.Tensor) -> torch.Tensor:
        """Sample dimensions 1 and onwards conditional on samples from dimension 0.

        :param torch.Tensor x0: tensor with shape `(*batch_shape,)`
        :return: tensor with shape `(*batch_shape, n_dim - 1)`.
        """
        u = torch.randn(*x0.shape, self.n_dim - 1)
        return u * torch.exp(x0[..., None] / 2)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        x = torch.zeros(*batch_shape, self.n_dim)
        x[..., 0] = self.base_potential_1d.sample(batch_shape)[..., 0]
        x[..., 1:self.n_dim] = self._conditional_sample(x[..., 0])
        return x

    @property
    def mean(self):
        return torch.concat([self.base_potential_1d.mean, torch.zeros(size=(self.n_dim - 1,))], dim=0)

    @property
    def variance(self):
        # x0 has variance V
        # We write the derivation for x1, x2, ...
        # > Law of total variance says: Var[Y] = E[Var[Y|X]] + Var[E[X|Y]]
        # > Second term is 0, so this becomes : Var[Y] = E[Var[Y|X]]
        # > We know Var[Y|X] = exp(X)
        # > E[exp(X)] is related to the Moment generating function E[exp(tX)] at t = 1.
        # > If X ~ N(mu, sigma), then the MGF E[exp(tX)] equals exp(mu*t + sigma**2 * t**2 / 2)
        # > In our case X = x0 and e.g. Y = x1
        # > We have mu = 0, sigma = sqrt(V)
        # > Therefore at t = 1, the MGF equals exp(V/2)
        var_x0 = self.base_potential_1d.variance
        var_rest = torch.full(size=(self.n_dim - 1,),
                              fill_value=float(math.exp(float(var_x0 / 2))))
        return torch.concat([var_x0, var_rest], dim=0)

    @property
    def edge_list(self):
        return [(0, i) for i in range(1, self.n_dim)]


class Funnel(FunnelBase):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), scale: float = 3.0):
        # p(x1) = N(.; 0, sigma=3)
        # p(xi|x1) = N(.; 0, sigma=exp(x1/2))
        super().__init__(
            event_shape=event_shape,
            base_potential_1d=DiagonalGaussian(
                mu=torch.Tensor([0.0]),
                sigma=torch.Tensor([scale])
            )
        )


class BimodalFunnel(FunnelBase):
    def __init__(self,
                 event_shape: Union[Tuple[int, ...], int] = (100,),
                 scale: float = 1.0,
                 weight: float = 0.5):
        # p(x1) = Mixture[N(.; -3, sigma=scale), N(.; 3, sigma=scale), (weight, 1 - weight)]
        # p(xi|x1) = N(.; 0, sigma=exp(x1/2))

        super().__init__(
            event_shape=event_shape,
            base_potential_1d=Mixture(
                potentials=[
                    DiagonalGaussian(
                        mu=torch.Tensor([-3.0]),
                        sigma=torch.Tensor([scale])
                    ),
                    DiagonalGaussian(
                        mu=torch.Tensor([3.0]),
                        sigma=torch.Tensor([scale])
                    )
                ],
                weights=torch.tensor([weight, 1 - weight])
            )
        )


class ShiftedFunnel(ShiftedFunnelBase):
    def __init__(self, event_shape: Union[Tuple[int, ...], int] = (100,), mu: float = 0.0, scale: float = 3.0):
        super().__init__(
            event_shape=event_shape,
            base_potential_1d=DiagonalGaussian(
                mu=torch.tensor([mu]),
                sigma=torch.tensor([scale])
            )
        )


class GaussianExponentialPosterior(Potential):
    """
    Potential for the posterior distribution p(theta, z | y) = p(y | z, theta) * p(z | theta) * p(theta).
    The parameter shapes are (n_dim - 1,) for z and (1,) for theta.

    We have:
    - p(theta) = N(theta; 0, 3^2)
    - p(z_i | theta) = N(z_i; 0, exp(theta))
    - p(y_i | z_i, theta) = N(y_i; z_i, sigma^2)

    The likelihood variance, sigma^2, is provided by the user.

    Reference: https://arxiv.org/pdf/2205.14240 (DLMC paper).
    """

    def __init__(self,
                 n_observations: int = 99,
                 hyperprior_scale: float = 3.0,
                 likelihood_scale: float = 1.0):
        """
        :param hyperprior_scale: scale of the hyperprior p(theta) = N(theta; 0, hyperprior_scale^2)
        :param likelihood_scale: scale of the likelihood p(y_i | z_i, theta) = N(y_i; z_i, likelihood_scale^2)
        """
        event_shape = (n_observations + 1,)
        super().__init__(event_shape)
        self.hyperprior_scale = hyperprior_scale
        self.likelihood_scale = likelihood_scale
        # Dimension 0 is theta, the remaining dimensions are observations.
        self.prior = Funnel(event_shape=(self.n_dim,),
                            scale=self.hyperprior_scale)

        torch.random.fork_rng()
        z = torch.randn(size=(n_observations,))
        self.y = torch.randn(size=(n_observations,)) * likelihood_scale + z

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        negative_log_prior = self.prior.compute(x)
        negative_log_likelihood = sum_except_batch(
            gaussian_potential(
                self.y,  # TODO test for shape
                x[..., 1:],
                torch.tensor(self.likelihood_scale)
            ),
            x.shape[:-1]
        )
        return negative_log_prior + negative_log_likelihood
