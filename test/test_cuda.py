import torch

from potentials.synthetic.gaussian_mixture import DoubleGaussian


def test_double_gaussian():
    n_dim = 10
    u = DoubleGaussian(n_dim).cuda()

    x = torch.randn(n_dim)
    ret = u(x)
    assert torch.all(torch.isfinite(ret))

    x = u.sample((11, 3))
    ret = u(x)
    assert torch.all(torch.isfinite(ret))
    assert torch.all(torch.isfinite(x))
