import torch
import torch.distributions as td
from PIL.ImageOps import scale

from potentials.base import Potential
from potentials.transformations import bound_parameter
from potentials.utils import sum_except_batch


class StochasticVolatilityModel(Potential):
    """

    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf
    """

    def __init__(self):
        super().__init__(event_shape=(3003,))
        self.measurements: torch.Tensor = ...  # (3000,)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # (z, unconstrained_sigma, unconstrained_mu, unconstrained_phi)
        batch_shape = x.shape[:-1]

        z = x[..., :3000]
        unconstrained_sigma = x[..., 3001]
        unconstrained_mu = x[..., 3002]
        unconstrained_phi = x[..., 3003]

        phi_transformed, log_det_phi_transformed = bound_parameter(unconstrained_phi, batch_shape, low=0.0, high=1.0)
        sigma, log_det_sigma = bound_parameter(unconstrained_sigma, batch_shape, low=0.0, high=torch.inf)
        mu, log_det_mu = bound_parameter(unconstrained_mu, batch_shape, low=0.0, high=torch.inf)
        log_det = log_det_phi_transformed + log_det_sigma + log_det_mu

        log_prob_z = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(z), batch_shape)
        log_prob_sigma = td.HalfCauchy(scale=2.0).log_prob(sigma)
        log_prob_mu = td.Exponential(rate=1.0).log_prob(mu)
        log_prob_phi_transformed = td.Beta(concentration0=20.0, concentration1=1.5).log_prob(phi_transformed)
        log_prior = log_prob_z + log_prob_sigma + log_prob_mu + log_prob_phi_transformed

        phi = phi_transformed * 2 - 1

        h = torch.zeros(size=(batch_shape, 3000), device=x.device, dtype=x.dtype)
        h[..., 0] = mu + sigma * z[..., 0] - torch.sqrt(1 - phi ** 2)
        for i in range(1, 3000):
            h[..., i] = mu + sigma * z[..., i] + phi * (h[..., i - 1] - mu)

        y_scale = torch.exp(h / 2)
        y_loc = torch.zeros_like(y_scale)
        log_likelihood = td.Independent(
            td.Normal(loc=y_loc, scale=y_scale),
            reinterpreted_batch_ndims=1
        ).log_prob(self.measurements)

        log_prob = log_likelihood + log_prior + log_det
        return -log_prob
