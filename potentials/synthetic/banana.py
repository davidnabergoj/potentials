from typing import Union, Tuple

import torch

from potentials.base import StructuredPotential
from potentials.synthetic.gaussian.diagonal import gaussian_potential, gaussian_potential_v2


class Banana(StructuredPotential):
    def __init__(self,
                 event_shape=(100,),
                 mu=0.0,
                 sigma=10.0,
                 alpha=0.03,
                 beta=-3.0,
                 gamma=1.0):
        """
        x1 ~ N(mu, sigma^2)
        x2 | x1 ~ N(alpha * x1^2 + beta, gamma^2)
        xi ~ N(0, 1)

        References:
        - Heikki Haario, Eero Saksman, and Johanna Tamminen. Adaptive proposal distribution for random walk metropolis algorithm. Computational statistics, 1999.
        - Abhinav Agrawal, Justin Domke. Disentangling impact of capacity, objective, batchsize, estimators, and step-size on flow VI. https://arxiv.org/abs/2412.08824
        """
        assert len(event_shape) == 1
        super().__init__(event_shape)
        assert self.n_dim >= 2

        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)

    @property
    def variance(self):
        var_x1_squared = 2 * self.sigma ** 4 + 4 * self.mu ** 2 * self.sigma ** 2
        return torch.tensor(
            [
                self.sigma ** 2,
                self.gamma ** 2 + self.alpha ** 2 * var_x1_squared,
            ] + [1.0] * (self.n_dim - 2)
        )

    @property
    def mean(self):
        return torch.tensor(
            [
                self.mu,
                self.alpha * (self.sigma ** 2 + self.mu ** 2) + self.beta,
            ] + [0.0] * (self.n_dim - 2)
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        u_0 = gaussian_potential(x[..., 0], self.mu, self.sigma)
        u_1 = gaussian_potential(x[..., 1], self.alpha * x[..., 0] ** 2 + self.beta, self.gamma)
        if self.n_dim > 2:
            u_rest = gaussian_potential(x[..., 2:], torch.tensor(0.0), torch.tensor(1.0))
            return u_0 + u_1 + u_rest.sum(dim=-1)
        else:
            return u_0 + u_1

    def sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        x_0 = torch.randn(size=sample_shape) * self.sigma + self.mu
        x_1 = torch.randn(size=sample_shape) * self.gamma + (self.alpha * x_0 ** 2 + self.beta)
        if self.n_dim > 2:
            x_rest = torch.randn(size=(*sample_shape, self.n_dim - 2))
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
