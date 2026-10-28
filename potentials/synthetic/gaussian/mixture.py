import torch

from potentials.synthetic.gaussian.diagonal import DiagonalGaussian1, DiagonalGaussian
from potentials.synthetic.gaussian.full_rank import FullRankGaussian1
from potentials.synthetic.mixture import Mixture


class GaussianMixture0(Mixture):
    def __init__(self, n_dim: int = 100, loc0: float = -100.0, loc1: float = 100.0, prob0: float = 0.5):
        p0 = DiagonalGaussian1(event_shape=(n_dim,))
        p0.mu = torch.full_like(p0.mu, fill_value=loc0)

        p1 = DiagonalGaussian1(event_shape=(n_dim,))
        p1.mu = torch.full_like(p1.mu, fill_value=loc1)
        super().__init__([p0, p1], weights=torch.tensor([prob0, 1 - prob0]))


class GaussianMixture1(Mixture):
    """
    Components: two diagonal Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = DiagonalGaussian1(event_shape=(n_dim,))
        p0.mu = torch.full_like(p0.mu, fill_value=-100.0)

        p1 = DiagonalGaussian1(event_shape=(n_dim,))
        p1.mu = torch.full_like(p1.mu, fill_value=100.0)

        w0 = 0.8
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture2(Mixture):
    """
    Components: two correlated Gaussians with linearly spaced eigenvalues 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = FullRankGaussian1(n_dim=n_dim)
        p0.loc = torch.full_like(p0.loc, fill_value=-100.0)

        p1 = FullRankGaussian1(n_dim=n_dim)
        p1.loc = torch.full_like(p1.loc, fill_value=100.0)

        w0 = 0.5
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture3(Mixture):
    """
    Components: two correlated Gaussians with linearly spaced eigenvalues 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100):
        p0 = FullRankGaussian1(n_dim=n_dim)
        p0.loc = torch.full_like(p0.loc, fill_value=-100.0)

        p1 = FullRankGaussian1(n_dim=n_dim)
        p1.loc = torch.full_like(p1.loc, fill_value=100.0)

        w0 = 0.8
        super().__init__([p0, p1], weights=torch.tensor([w0, 1 - w0]))


class GaussianMixture4(Mixture):
    """
    Components: ten diagonal Gaussians with linearly spaced scales 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [DiagonalGaussian1(event_shape=(n_dim,)) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        super().__init__(ps, weights=torch.ones(10) / 10)


class GaussianMixture5(Mixture):
    """
    Components: ten diagonal Gaussians with linearly spaced scales 1 to 10.
    Skewed weight.

    TODO investigate weight.
    """

    def __init__(self, n_dim: int = 100, seed: int = 0):
        ps = [DiagonalGaussian1(event_shape=(n_dim,)) for _ in range(10)]
        for i in range(10):
            ps[i].mu = torch.full_like(ps[i].mu, fill_value=200.0 * i)
        generator = torch.Generator().manual_seed(seed)
        super().__init__(ps, weights=torch.softmax(torch.randn(10, generator=generator) * 2, dim=0))


class GaussianMixture6(Mixture):
    """
    Components: ten correlated Gaussians with linearly spaced eigenvalues 1 to 10.
    Equal weight.
    """

    def __init__(self, n_dim: int = 100):
        ps = [FullRankGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].loc = torch.full_like(ps[i].loc, fill_value=200.0 * i)
        super().__init__(ps, weights=torch.ones(10) / 10)


class GaussianMixture7(Mixture):
    """
    Components: ten correlated Gaussians with linearly spaced eigenvalues 1 to 10.
    Skewed weight.
    """

    def __init__(self, n_dim: int = 100, seed: int = 0):
        ps = [FullRankGaussian1(n_dim=n_dim) for _ in range(10)]
        for i in range(10):
            ps[i].loc = torch.full_like(ps[i].loc, fill_value=200.0 * i)
        generator = torch.Generator().manual_seed(seed)
        super().__init__(ps, weights=torch.softmax(torch.randn(10, generator=generator) * 2, dim=0))


class GaussianMixture8(Mixture):
    def __init__(self, n_dim: int = 100):
        self.loc0 = torch.full(size=(n_dim,), fill_value=-10.0)
        self.scale0 = torch.full(size=(n_dim,), fill_value=1.0)
        self.loc1 = torch.full(size=(n_dim,), fill_value=10.0)
        self.scale1 = torch.full(size=(n_dim,), fill_value=1.0)
        potentials = [
            DiagonalGaussian(mu=self.loc0, sigma=self.scale0),
            DiagonalGaussian(mu=self.loc1, sigma=self.scale1)
        ]
        weights = torch.tensor([0.8, 0.2])
        super().__init__(potentials, weights)


class GaussianMixture9(Mixture):
    def __init__(self):
        u = 10.0
        potentials = [
            DiagonalGaussian(mu=torch.tensor([-u, u]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([0.0, u]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([u, u]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([-u, 0.0]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([0.0, 0.0]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([u, 0.0]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([-u, -u]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([0.0, -u]), sigma=torch.ones(2)),
            DiagonalGaussian(mu=torch.tensor([u, -u]), sigma=torch.ones(2)),
        ]
        weights = torch.tensor([0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.3, 0.01, 0.19])
        super().__init__(potentials, weights)
