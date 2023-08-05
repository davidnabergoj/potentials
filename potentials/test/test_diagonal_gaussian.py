import torch
import math

from potentials.synthetic.gaussian import DiagonalGaussian


def test_basic():
    potential = DiagonalGaussian(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 2.0]))
    x = torch.tensor([[1.0, 1.0]])
    u = potential(x)
    true_log_prob = math.log(1 / (4 * math.pi)) - 5 / 8
    true_u = -true_log_prob
    assert torch.isclose(u, torch.tensor(true_u))
