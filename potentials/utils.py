import torch
import numpy as np


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]


def unsqueeze_to_batch(x: torch.Tensor, batch_shape: torch.Size):
    return x[(None,) * len(batch_shape)]


def sum_except_batch(x: torch.Tensor, batch_shape: torch.Size):
    sum_dims = tuple(range(len(batch_shape), len(x.shape)))
    return torch.sum(x, dim=sum_dims)


def sample_from_gamma(sample_shape, gamma_shape: float = 0.5, gamma_scale: float = 1.0, seed: int = 0):
    torch.random.fork_rng()
    torch.manual_seed(seed)
    r = 1 / gamma_scale
    return torch.distributions.Gamma(concentration=gamma_shape, rate=r).sample(sample_shape)


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


def generate_cholesky_factor(eigenvalues: torch.Tensor, seed: int = 0):
    """
    Generate a random Choleksy factor L s.t. distribution of rotations of LL.T is uniform over the Stiefel manifold.

    How it works: we have a diagonal matrix of eigenvalues D. We can also sample a random orthogonal matrix U.
    This gives rise to an eigendecomposition S = U @ D @ U.T where S is a covariance matrix.
    Now perform QR decomposition Q @ R = qr(sqrt(D) @ U.T).
    Note that (sqrt(D) @ U.T).T @ (sqrt(D) @ U.T) = U @ D @ U.T = S.
    Plug in Q @ R to obtain (Q @ R).T @ (Q @ R) = S.
    Cancel the orthogonal matrices: S = (Q @ R).T @ (Q @ R) = (R.T @ Q.T) @ (Q @ R) = R.T @ Q.T @ Q @ R = R.T @ I @ R =
        R.T @ R.
    Therefore L = R.T is the lower Cholesky factor of covariance S.

    Source: https://scicomp.stackexchange.com/a/34648.
    """
    rotation = generate_rotation_matrix(n_dim=len(eigenvalues), seed=seed).to(eigenvalues)
    _, r = torch.linalg.qr(torch.diag(torch.sqrt(eigenvalues)) @ rotation.T)
    r *= torch.sign(torch.diag(r))[:, None]  # For uniqueness: negate row of R with negative diagonal element
    return r.T
