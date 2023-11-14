import math

import torch
from potentials.synthetic.gaussian.full_rank import FullRankGaussian

from potentials.utils import sample_from_gamma, generate_rotation_matrix, generate_cholesky_factor


class BlockDiagonal(FullRankGaussian):
    def __init__(self, mu: torch.Tensor, eigenvalues: torch.Tensor, n_blocks: int = None):
        n_dim = len(mu)

        # Set default number of blocks
        if n_blocks is None:
            n_blocks = max(int(math.log(n_dim)), 1)
            if n_blocks > n_dim:
                n_blocks = n_dim

        block_sizes = compute_block_sizes(n_dim, n_blocks)
        small_cholesky_factors = []
        block_start = 0
        for i, block_size in enumerate(block_sizes):
            block_end = block_start + block_size
            block_cholesky = generate_cholesky_factor(
                eigenvalues[block_start:block_end],
                seed=(i * 1000) & (2 ** 32 - 1)
            )
            small_cholesky_factors.append(block_cholesky)
            block_start = block_end
        cholesky_factor = torch.block_diag(*small_cholesky_factors)
        self.cov = cholesky_factor @ cholesky_factor.T
        super().__init__(mu, cholesky_lower=cholesky_factor)


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

    def __init__(self, n_dim: int = 100, gamma_shape: float = 0.5, seed: int = 0, **kwargs):
        mu = torch.zeros(n_dim)
        tmp = sample_from_gamma((n_dim,), gamma_shape, 1.0, seed=seed)
        eigenvalues = (1 / torch.sort(tmp)[0]).to(mu)
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
        torch.random.fork_rng()
        torch.manual_seed(seed)
        eigenvalues = torch.exp(torch.randn(size=(n_dim,)))
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
        eigenvalues = torch.linspace(1 / 100, 100, n_dim)
        super().__init__(mu, eigenvalues, n_blocks)
