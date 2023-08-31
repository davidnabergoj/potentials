import torch
from potentials.synthetic.gaussian import DiagonalGaussian
from potentials.synthetic.mixture import Mixture


class GaussianMixture2D(Mixture):
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


class DoubleGaussian(Mixture):
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

    @property
    def mean(self):
        return self.weights[0] * self.loc0 + (1 - self.weights[0]) * self.loc1

    @property
    def variance(self):
        alpha = self.weights[0]
        var_alpha = alpha * (1 - alpha)
        var_alpha_x = torch.sub(
            (var_alpha + alpha ** 2) * (self.potentials[0].variance + self.potentials[0].mean ** 2),
            alpha ** 2 * self.potentials[0].mean ** 2
        )
        var_alpha_y = torch.sub(
            (var_alpha + alpha ** 2) * (self.potentials[1].variance + self.potentials[1].mean ** 2),
            alpha ** 2 * self.potentials[1].mean ** 2
        )
        cov = -alpha * self.potentials[0].mean * (1 - alpha) * self.potentials[1].mean
        return var_alpha_x + var_alpha_y + 2 * cov
