import math

import torch
from typing import Union, Tuple

from potentials.base import PotentialSimple
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian, gaussian_potential
from potentials.utils import get_batch_shape, sum_except_batch


class Funnel(PotentialSimple):
    def __init__(self, n_dim: int = 100, scale: float = 3.0):
        # p(x1) = N(.; 0, sigma=3)
        # p(xi|x1) = N(.; 0, sigma=exp(x1/2))
        super().__init__(n_dim=n_dim)
        self.n_dim = n_dim
        self.base_potential = DiagonalGaussian(mu=torch.Tensor([0.0]), sigma=torch.Tensor([scale]))
        self.scale = scale

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # p(x) = p(x1) * prod_{i=2}^100 p(xi|x1)
        # x.shape = (*batch_shape, n_dim)
        batch_shape = get_batch_shape(x, self.event_shape)
        u_x1 = self.base_potential(x[..., 0][..., None])  # Take the first dim

        xi = x[..., 1:]  # (*batch_shape, n_dim - 1)
        mu = torch.zeros_like(xi)  # (*batch_shape, n_dim - 1)

        # (*batch_shape, n_dim - 1)
        sigma = torch.exp(x[..., 0] / 2)[..., None].repeat(tuple([1] * len(batch_shape) + [self.n_dim - 1]))

        u_xi = sum_except_batch(gaussian_potential(xi, mu, sigma), batch_shape)
        return u_x1 + u_xi

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        x = torch.zeros(*batch_shape, self.n_dim)
        x[..., 0] = torch.randn(*batch_shape) * self.scale
        x[..., 1:self.n_dim] = torch.randn(*batch_shape, self.n_dim - 1) * torch.exp(x[..., 0][..., None] / 2)
        return x

    @property
    def mean(self):
        return torch.zeros(size=(self.n_dim,))

    @property
    def variance(self):
        # x0 has variance 3^2
        # We write the derivation for x1, x2, ...
        # > Law of total variance says: Var[Y] = E[Var[Y|X]] + Var[E[X|Y]]
        # > Second term is 0, so this becomes : Var[Y] = E[Var[Y|X]]
        # > We know Var[Y|X] = exp(X)
        # > E[exp(X)] is related to the Moment generating function E[exp(tX)] at t = 1.
        # > If X ~ N(mu, sigma), then the MGF E[exp(tX)] equals exp(mu*t + sigma**2 * t**2 / 2)
        # > In our case X = x0 and e.g. Y = x1
        # > We have mu = 0, sigma = 3
        # > Therefore at t = 1, the MGF equals exp(3**2/2) = exp(9/2) = exp(4.5) \approx 90.0
        return torch.tensor([self.scale ** 2] + [math.exp(self.scale ** 2 / 2)] * (self.n_dim - 1))
