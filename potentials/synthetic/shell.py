from typing import Union, Tuple

import torch

from potentials.base import Potential
from potentials.synthetic.mixture import Mixture


class GammaShell(Potential):
    """
    Potential that defines a 2D shell.

    The distribution of points on the shall is given by independently sampling a radius and an angle.
    The radius r is sampled from a Gamma distribution with mean r_mean and variance r_var.
    The angle theta is sampled uniformly from U(0, 2 * pi)

    Sampling from the potentials yields points (x, y) in the Cartesian coordinate system according to
        x = r * cos(theta), y = r * sin(theta)
    The Jacobian determinant of the transformation is equal to r.
    """

    def __init__(self, r_mean: float, r_var: float):
        # Get the shape and rate parameters for the gamma distribution
        self.r_shape = r_mean ** 2 / r_var
        self.r_rate = r_mean / r_var
        self.r_dist = torch.distributions.Gamma(concentration=self.r_shape, rate=self.r_rate)
        self.theta_dist = torch.distributions.Uniform(low=0.0, high=2 * torch.pi)
        super().__init__(event_shape=(2,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # Transform to spherical coordinates
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        theta = torch.arccos(x[..., 1] / r)

        # Compute log determinant
        log_det = -torch.log(r)

        # Compute base densities
        log_prob_r = self.r_dist.log_prob(r)
        log_prob_theta = self.theta_dist.log_prob(theta)

        log_prob = log_prob_r + log_prob_theta + log_det

        return -log_prob

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
        # Sample r and theta
        r = self.r_dist.sample(batch_shape)
        theta = self.theta_dist.sample(batch_shape)

        # Transform spherical to Cartesian coordinates
        return torch.concat([
            (r * torch.cos(theta))[:, None],
            (r * torch.sin(theta))[:, None]
        ], dim=-1)


class DoubleGammaShell(Mixture):
    def __init__(self):
        super().__init__(
            potentials=[
                GammaShell(r_mean=5.0, r_var=1.0),
                GammaShell(r_mean=15.0, r_var=1.0),
            ],
            weights=torch.tensor([0.2, 0.8])
        )
