import torch
import numpy as np
from potentials.base import Potential


class DistributionFromPotential(torch.distributions.Distribution):
    def __init__(self, potential: Potential):
        super().__init__(event_shape=potential.event_shape, validate_args=False)
        self.potential = potential

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -self.potential(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.potential.sample(sample_shape)


def as_distribution(potential: Potential) -> DistributionFromPotential:
    return DistributionFromPotential(potential)


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]


def unsqueeze_to_batch(x: torch.Tensor, batch_shape: torch.Size):
    return x[(None,) * len(batch_shape)]


def sum_except_batch(x: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
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


def plot_2d(potential_2d,
            ax,
            xmin: float = -5.0,
            xmax: float = 6.0,
            ymin: float = -5.0,
            ymax: float = 30.0,
            resolution: int = 500,
            n_levels: int = 7,
            min_level: float = None,
            max_level: float = None):
    """
    Make a contour plot of a 2D potential.
    """
    xs = torch.linspace(xmin, xmax, resolution)
    ys = torch.linspace(ymin, ymax, resolution)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    xx_flat, yy_flat = xx.ravel(), yy.ravel()
    zz_flat = -potential_2d(torch.concat([xx_flat[:, None], yy_flat[:, None]], dim=1))
    zz_flat = zz_flat.exp()
    zz = zz_flat.view_as(xx)

    levels = n_levels
    if min_level is not None and max_level is not None:
        levels = np.geomspace(min_level, max_level, n_levels),

    ax.contour(
        xx.numpy(),
        yy.numpy(),
        zz.numpy(),
        levels=levels,
        linewidths=0.5
    )
    ax.contourf(
        xx.numpy(),
        yy.numpy(),
        zz.numpy(),
        levels=levels,
        alpha=0.1
    )
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")


def reduce_two_key_dataset(key1_index: torch.Tensor,
                           key2_index: torch.Tensor,
                           data: torch.Tensor,
                           new_n_unique_key1_ids: int,
                           reindex_key2: bool = False):
    """
    Processes a dataset with entries of the form (key1, key2, data).
    The dataset should have `n` unique key1 values, `m` unique key2 values, and `k` data points.
    The function reduces the number of unique key1 values to `new_unique_key1_counts`, whilst
     also removing the associated data values.

    :param torch.Tensor key1_values: values for key1 objects, indexed from 0 (inclusive) to n (exclusive).
    :param torch.Tensor key2_values: values for key2 objects, indexed from 0 (inclusive) to m (exclusive).
    :param bool reindex_key2: if True, reindexes key2, associated with reduced key1 entries.
    :return: (tuple with 6 elements) new key1 index, new key2 index, new data, number of new key1 ids, 
     number of new key2 ids, number of new data entries.
    """
    n_key1_ids = len(torch.unique(key1_index))

    if new_n_unique_key1_ids == n_key1_ids:
        n_key2_ids = len(torch.unique(key2_index))
        return (
            key1_index,
            key2_index,
            data,
            n_key1_ids,
            n_key2_ids,
            len(data)
        )
    elif new_n_unique_key1_ids > n_key1_ids:
        raise ValueError(
            "Cannot have more IDs than the original dataset"
        )

    _count = [0] * n_key1_ids
    _kept = []
    for i in range(len(key1_index)):
        if _count[key1_index[i]] < new_n_unique_key1_ids:
            _count[key1_index[i]] += 1
            _kept.append(i)
    _kept = torch.tensor(_kept, dtype=torch.long)

    # Modify data
    new_data = data[_kept]
    new_n_data = len(new_data)

    # Modify key1 index
    new_key1_index = key1_index[_kept]
    
    # Modify key2 index
    reduced_key2_index = key2_index[_kept]

    if not reindex_key2:
        new_n_key2_ids = len(torch.unique(reduced_key2_index))
        return (
            new_key1_index,
            reduced_key2_index,
            new_data,
            n_key1_ids,
            new_n_key2_ids,
            new_n_data
        )

    reduced_key2_ids = torch.unique(reduced_key2_index)
    new_key2_ids = torch.arange(len(reduced_key2_ids))

    replacement = dict(zip(reduced_key2_ids.tolist(), new_key2_ids.tolist()))
    for i in range(len(reduced_key2_index)):
        reduced_key2_index[i] = replacement[int(reduced_key2_index[i])]
    
    new_key2_index = reduced_key2_index
    new_n_key2_ids = len(new_key2_ids)

    return (
        new_key1_index,
        new_key2_index,
        new_data,
        n_key1_ids,
        new_n_key2_ids,
        new_n_data
    )