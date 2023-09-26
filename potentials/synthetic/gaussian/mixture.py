import torch

from potentials.synthetic.gaussian.diagonal import DiagonalGaussian1
from potentials.synthetic.gaussian.full_rank import FullRankGaussian1
from potentials.synthetic.mixture import Mixture


class GaussianMixture0(Mixture):
    def __init__(self, n_dim: int = 100):
        p0 = DiagonalGaussian1(n_dim=n_dim)
        p0.mu = torch.full_like(p0.mu, fill_value=-100.0)

        p1 = DiagonalGaussian1(n_dim=n_dim)
        p1.mu = torch.full_like(p1.mu, fill_value=-100.0)
        super().__init__([p0, p1], weights=torch.tensor([0.5, 0.5]))


class GaussianMixture1(Mixture):
    """
    Components: two diagonal Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = DiagonalGaussian1(n_dim=n_dim)
        p0.mu = torch.full_like(p0.mu, fill_value=-100.0)

        p1 = DiagonalGaussian1(n_dim=n_dim)
        p1.mu = torch.full_like(p1.mu, fill_value=-100.0)

        w0 = max(1 / n_dim, 0.8)
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture2(Mixture):
    """
    Components: two correlated Gaussians with linearly spaced scales 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = FullRankGaussian1(n_dim=n_dim)
        p0.mu = torch.full_like(p0.mu, fill_value=-100.0)

        p1 = FullRankGaussian1(n_dim=n_dim)
        p1.mu = torch.full_like(p1.mu, fill_value=-100.0)

        w0 = 0.5
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture3(Mixture):
    """
    Components: two correlated Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = FullRankGaussian1(n_dim=n_dim)
        p0.mu = torch.full_like(p0.mu, fill_value=-100.0)

        p1 = FullRankGaussian1(n_dim=n_dim)
        p1.mu = torch.full_like(p1.mu, fill_value=-100.0)

        w0 = max(1 / n_dim, 0.8)
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture4(Mixture):
    """
    Components: ten diagonal Gaussians with linearly spaced scales 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [DiagonalGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        super().__init__(ps, weights=1 / torch.ones(10))


class GaussianMixture5(Mixture):
    """
    Components: ten diagonal Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.

    TODO investigate weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [DiagonalGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        super().__init__(ps, weights=torch.softmax(torch.randn(10) * 2, dim=0))


class GaussianMixture6(Mixture):
    """
    Components: ten correlated Gaussians with linearly spaced scales 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [FullRankGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        super().__init__(ps, weights=1 / torch.ones(10))


class GaussianMixture7(Mixture):
    """
    Components: ten correlated Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [FullRankGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        super().__init__(ps, weights=torch.softmax(torch.randn(10) * 2, dim=0))
