import torch

from potentials.synthetic.gaussian.full_rank import generate_rotation_matrix
import pytest


@pytest.mark.parametrize('n_dim', [2, 4, 10, 100])
def test_basic(n_dim):
    q = generate_rotation_matrix(n_dim, seed=0)
    s = q.T @ q
    assert torch.allclose(s, torch.eye(n_dim), atol=1e-6)

    tmp = torch.clone(q)
    tmp[range(n_dim), range(n_dim)] = 0
    assert not torch.allclose(tmp, torch.tensor(0.0))
