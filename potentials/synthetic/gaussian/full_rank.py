from typing import Tuple, Union

import numpy as np
import torch

from potentials.base import Potential


def generate_rotation_matrix(n_dim: int, seed):
    rng = np.random.RandomState(seed=seed)
    noise = rng.randn(n_dim, n_dim)
    q, r = np.linalg.qr(noise)
    # q *= np.sign(np.diag(r))

    # Multiply with a random unitary diagonal matrix to get a uniform sample from the Stiefel manifold.
    # https://stackoverflow.com/a/38430739
    q *= np.diag(np.exp(np.pi * 2 * np.random.rand(n_dim)))
    return torch.as_tensor(q)


class FullRankGaussian(Potential):
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        assert mu.shape == cov.shape[:-1], f"{mu.shape = }, {cov.shape = }"
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


class DecomposedFullRankGaussian(FullRankGaussian):
    """
    Full rank Gaussian distribution with its covariance matrix decomposed as q @ d @ q.T
    where q is orthonormal and d is diagonal with eigenvalues.
    """

    def __init__(self, mu: torch.Tensor, eigenvalues: torch.Tensor, seed: int = 0):
        assert len(eigenvalues.shape) == 1
        q = generate_rotation_matrix(n_dim=len(eigenvalues), seed=seed)
        self.q = q
        self.eigenvalues = eigenvalues
        self.cov = q @ torch.diag(eigenvalues).to(q) @ q.T
        # print(f'{mu.shape = } {q.shape = } {eigenvalues.shape = } {self.cov.shape = }')
        super().__init__(mu, self.cov)


class FullRankGaussian0(DecomposedFullRankGaussian):
    """
    Eigenvalues are reciprocals of Gamma distribution samples.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, gamma_shape: float = 0.5, seed: int = 0):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(1 / np.sort(rng.gamma(shape=gamma_shape, scale=1.0, size=n_dim)))
        super().__init__(mu, eigenvalues)


class FullRankGaussian1(DecomposedFullRankGaussian):
    """
    Eigenvalues are linearly spaced between 1 and 10.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100):
        mu = torch.zeros(n_dim)
        eigenvalues = torch.linspace(1, 10, n_dim)
        super().__init__(mu, eigenvalues)


class FullRankGaussian2(DecomposedFullRankGaussian):
    """
    Log of the eigenvalues sampled from standard normal.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, seed: int = 0):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(np.exp(rng.randn(n_dim)))
        super().__init__(mu, eigenvalues)


class FullRankGaussian3(DecomposedFullRankGaussian):
    """
    First eigenvalue is 1000, remainder are 1.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100):
        mu = torch.zeros(n_dim)
        eigenvalues = torch.ones(n_dim)
        eigenvalues[0] = 1000
        super().__init__(mu, eigenvalues)


class FullRankGaussian4(DecomposedFullRankGaussian):
    """
    First eigenvalue is 1000, second eigenvalue is 1/1000, remainder are 1.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100):
        assert n_dim >= 2
        mu = torch.zeros(n_dim)
        eigenvalues = torch.ones(n_dim)
        eigenvalues[0] = 1000
        eigenvalues[1] = 1 / 1000
        super().__init__(mu, eigenvalues)


class FullRankGaussian5(DecomposedFullRankGaussian):
    """
    Eigenvalues linearly space between 1/1000 and 1000.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100):
        assert n_dim >= 2
        mu = torch.zeros(n_dim)
        eigenvalues = torch.linspace(1 / 1000, 1000, n_dim)
        super().__init__(mu, eigenvalues)
