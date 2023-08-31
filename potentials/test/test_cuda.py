import torch

from potentials import DoubleGaussian


def test_double_gaussian():
    n_dim = 10
    pot = DoubleGaussian(n_dim).cuda()

    x = torch.randn(n_dim)
    pot(x)

    x = pot.sample((11, 3))
    pot(x)
