from typing import Tuple, Union
import torch

from potentials.base import Potential
from potentials.utils import sample_from_gamma, generate_rotation_matrix


class FullRankGaussian(Potential):
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor = None, cholesky_lower: torch.Tensor = None):
        if cov is None and cholesky_lower is None:
            raise ValueError("At least one of covariance and the cholesky factor must be provided")
        if cov is not None:
            assert mu.shape == cov.shape[:-1], f"{mu.shape = }, {cov.shape = }"
        elif cholesky_lower is not None:
            assert mu.shape == cholesky_lower.shape[:-1], f"{mu.shape = }, {cholesky_lower.shape = }"
        event_shape = mu.shape
        super().__init__(event_shape)

        self.register_buffer('loc', mu)
        if cholesky_lower is not None:
            self.tril_parametrization = True
            self.register_buffer('scale_tril', cholesky_lower)
        else:
            self.tril_parametrization = False
            self.register_buffer('covariance_matrix', cov)

    @property
    def dist(self):
        if self.tril_parametrization:
            return torch.distributions.MultivariateNormal(loc=self.loc, scale_tril=self.scale_tril)
        else:
            return torch.distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

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
        rotation = generate_rotation_matrix(n_dim=len(eigenvalues), seed=seed).to(mu)
        _, r = torch.linalg.qr(torch.diag(torch.sqrt(eigenvalues)) @ rotation.T)
        r *= torch.sign(torch.diag(r))[:, None]  # For uniqueness: negate row of R with negative diagonal element

        self.rotation = rotation
        self.eigenvalues = eigenvalues
        self.cov = (rotation * eigenvalues.to(rotation)) @ rotation.T
        super().__init__(mu, cholesky_lower=r.T)


class FullRankGaussian0(DecomposedFullRankGaussian):
    """
    Eigenvalues are reciprocals of Gamma distribution samples.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, gamma_shape: float = 0.5, seed: int = 10):
        mu = torch.zeros(n_dim)
        tmp = sample_from_gamma((n_dim,), gamma_shape, 1.0, seed=seed)
        eigenvalues = (1 / torch.sort(tmp)[0]).to(mu)
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
        torch.random.fork_rng()
        torch.manual_seed(seed)
        eigenvalues = torch.exp(torch.randn(n_dim))
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
