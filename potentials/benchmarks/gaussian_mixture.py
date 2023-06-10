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
