from typing import Tuple, Union

import numpy as np
import torch

from potentials.base import Potential, PotentialSimple


class FullRankGaussian(Potential):
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        assert mu.shape == cov.shape[:-1]
        event_shape = mu.shape
        super().__init__(event_shape)
        self.dist = torch.distributions.MultivariateNormal(
            loc=mu.float(),
            covariance_matrix=cov.float()
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return -self.dist.log_prob(x)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]) -> torch.Tensor:
        return self.dist.sample(batch_shape)


class IllConditionedGaussian(FullRankGaussian):
    """
    Covariance has eigenvalues sampled from a Gamma, then we rotate it with a random orthogonal matrix.

    https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/targets/ill_conditioned_gaussian.py
    """

    def __init__(self, n_dim: int = 100, gamma_shape: float = 0.5, seed: int = 0):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed & (2 ** 32 - 1))
        eigvals = 1 / np.sort(rng.gamma(shape=gamma_shape, scale=1.0, size=n_dim))
        q, r = torch.linalg.qr(torch.tensor(rng.randn(n_dim, n_dim)))
        q *= torch.sign(torch.diag(r))
        cov = (q * eigvals) @ q.T
        self.n_dim = n_dim
        super().__init__(mu, cov)
