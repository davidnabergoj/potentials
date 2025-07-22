import math
from typing import Union, Tuple

import torch

from potentials.base import StructuredPotential
from potentials.synthetic.gaussian.diagonal import gaussian_potential


class Funana(StructuredPotential):
    def __init__(self,
                 event_shape=(100,),
                 mu_0: float = 0.0,
                 sigma_0: float = 3.0,
                 mu_1: float = 0.0,
                 sigma_1: float = 10.0,
                 alpha: float = 0.03,
                 beta: float = -3.0):
        """
        x1 ~ N(mu_0, sigma_0^2)
        x2 ~ N(mu_1, sigma_1^2)
        xi | x1 ~ N(alpha * x1^2 + beta, exp(x1 / 2)^2)

        References:
        - Abhinav Agrawal, Justin Domke. Disentangling impact of capacity, objective, batchsize, estimators, and step-size on flow VI. https://arxiv.org/abs/2412.08824
        """
        assert len(event_shape) == 1
        super().__init__(event_shape)
        assert self.n_dim >= 2

        self.mu_0 = torch.tensor(mu_0)
        self.mu_1 = torch.tensor(mu_1)
        self.sigma_0 = torch.tensor(sigma_0)
        self.sigma_1 = torch.tensor(sigma_1)
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)

    @property
    def variance(self):
        var_x1_squared = 2 * self.sigma_0 ** 4 + 4 * self.mu_0 ** 2 * self.sigma_0 ** 2
        return torch.tensor(
            [
                self.sigma_0 ** 2,
                self.sigma_1 ** 2,
            ] + [
                math.exp(self.sigma_0 ** 2 / 2) + self.alpha ** 2 * var_x1_squared
            ] * (self.n_dim - 2)
        )

    @property
    def mean(self):
        return torch.tensor(
            [
                self.mu_0,
                self.mu_1,
            ] + [
                self.alpha * (self.sigma_0 ** 2 + self.mu_0 ** 2) + self.beta
            ] * (self.n_dim - 2)
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        u_0 = gaussian_potential(x[..., 0], self.mu_0, self.sigma_0)
        u_1 = gaussian_potential(x[..., 1], self.mu_1, self.sigma_1)
        if self.n_dim > 2:
            mu_rest = self.alpha * x[..., 0] ** 2 + self.beta
            sigma_rest = torch.exp(x[..., 0] / 2)
            u_rest = gaussian_potential(x[..., 2:], mu_rest[..., None], sigma_rest[..., None])
            return u_0 + u_1 + u_rest.sum(dim=-1)
        else:
            return u_0 + u_1

    def sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        x_0 = torch.randn(size=sample_shape) * self.sigma_0 + self.mu_0
        x_1 = torch.randn(size=sample_shape) * self.sigma_1 + self.mu_1
        if self.n_dim > 2:
            mu_rest = self.alpha * x_0 ** 2 + self.beta
            sigma_rest = torch.exp(x_1 / 2)
            x_rest = torch.randn(size=(*sample_shape, self.n_dim - 2)) * sigma_rest[..., None] + mu_rest[..., None]
            return torch.concat([
                x_0[..., None],
                x_1[..., None],
                x_rest
            ], dim=-1)
        else:
            return torch.concat([
                x_0[..., None],
                x_1[..., None]
            ], dim=-1)
