from typing import Tuple, Union
import numpy as np
import torch

from potentials.base import Potential


def generate_rotation_matrix(n_dim: int, seed):
    # Generates a random rotation matrix (not uniform over SO(3))
    torch.random.fork_rng()
    torch.manual_seed(seed)

    # Apparently numpy.linalg.qr is more stable than torch.linalg.qr? Perhaps torch is not calling the best method in
    # LAPACK?
    q, r = np.linalg.qr(torch.randn(size=(n_dim, n_dim)))

    q = torch.as_tensor(q)
    r = torch.as_tensor(r)
    # This line gives q determinant 1. Otherwise, it will have +1 if n_dim odd and -1 if n_dim even.
    # q = q @ torch.diag(torch.sign(torch.diag(r)))
    q *= torch.sign(torch.diag(r))
    return q


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

    def __init__(self, n_dim: int = 100, gamma_shape: float = 0.5, seed: int = 10):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed & (2 ** 32 - 1))
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
    Eigenvalues linearly spaced between 1/100 and 100.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100):
        assert n_dim >= 2
        mu = torch.zeros(n_dim)
        eigenvalues = torch.linspace(1 / 100, 100, n_dim)
        super().__init__(mu, eigenvalues)
