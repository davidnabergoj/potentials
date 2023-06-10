import torch

from potentials.base import Potential


class FullRankGaussian(Potential):
    def __init__(self, mu, cov: torch.Tensor):
        assert mu.shape == cov.shape[:-1]
        event_shape = mu.shape
        super().__init__(event_shape)
        self.dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return -self.dist.log_prob(x)
