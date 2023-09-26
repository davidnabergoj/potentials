from potentials.synthetic.gaussian import FullRankGaussian
import torch

from potentials.synthetic.gaussian.full_rank import generate_rotation_matrix
import numpy as np


class BlockDiagonal(FullRankGaussian):
    def __init__(self, mu: torch.Tensor, eigenvalues: torch.Tensor, n_blocks: int):
        n_dim = len(mu)
        block_sizes = compute_block_sizes(n_dim, n_blocks)
        q_seeds = [i * 1000 for i in range(n_blocks)]
        q_blocks = [generate_rotation_matrix(b, s) for b, s in zip(block_sizes, q_seeds)]
        q = torch.block_diag(*q_blocks)
        cov = q @ eigenvalues @ q.T
        super().__init__(mu, cov)


def compute_block_sizes(n_dim, n_blocks):
    assert n_dim >= n_blocks
    block_sizes = [n_dim // n_blocks] * n_blocks

    diff = n_dim - sum(block_sizes)
    if diff > 0:
        block_sizes[0] += diff
    return block_sizes


class BlockDiagonal0(BlockDiagonal):
    """
    Eigenvalues are reciprocals of Gamma distribution samples.
    """

    def __init__(self, n_dim: int, gamma_shape: float = 0.5, seed: int = 0, **kwargs):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(1 / np.sort(rng.gamma(shape=gamma_shape, scale=1.0, size=n_dim)))
        super().__init__(mu, eigenvalues, **kwargs)


class BlockDiagonal1(BlockDiagonal):
    """
    Eigenvalues are linearly spaced between 1 and 10.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, n_blocks: int = None):
        mu = torch.zeros(n_dim)
        eigenvalues = torch.linspace(1, 10, n_dim)
        super().__init__(mu, eigenvalues, n_blocks)


class BlockDiagonal2(BlockDiagonal):
    """
    Log of the eigenvalues sampled from standard normal.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, n_blocks: int = None, seed: int = 0):
        mu = torch.zeros(n_dim)
        rng = np.random.RandomState(seed=seed)
        eigenvalues = torch.as_tensor(np.exp(rng.randn(n_dim)))
        super().__init__(mu, eigenvalues, n_blocks)


class BlockDiagonal3(BlockDiagonal):
    """
    First eigenvalue is 1000, remainder are 1.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, n_blocks: int = None):
        mu = torch.zeros(n_dim)
        eigenvalues = torch.ones(n_dim)
        eigenvalues[0] = 1000
        super().__init__(mu, eigenvalues, n_blocks)


class BlockDiagonal4(BlockDiagonal):
    """
    First eigenvalue is 1000, second eigenvalue is 1/1000, remainder are 1.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, n_blocks: int = None):
        assert n_dim >= 2
        mu = torch.zeros(n_dim)
        eigenvalues = torch.ones(n_dim)
        eigenvalues[0] = 1000
        eigenvalues[1] = 1 / 1000
        super().__init__(mu, eigenvalues, n_blocks)


class BlockDiagonal5(BlockDiagonal):
    """
    Eigenvalues linearly space between 1/1000 and 1000.
    Rotation matrix sampled uniformly from Stiefel manifold.
    """

    def __init__(self, n_dim: int = 100, n_blocks: int = None):
        assert n_dim >= 2
        mu = torch.zeros(n_dim)
        eigenvalues = torch.linspace(1 / 1000, 1000, n_dim)
        super().__init__(mu, eigenvalues, n_blocks)
