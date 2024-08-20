import torch
import torch.distributions as td
from potentials.base import Potential
from potentials.transformations import bound_parameter
from potentials.utils import sum_except_batch


class EightSchools(Potential):
    """

    Reference: https://www.tensorflow.org/probability/examples/Eight_Schools
    """

    def __init__(self):
        super().__init__(event_shape=(10,))
        self.measurements = ...
        self.scales = ...  # (8,)
        self.index = ...  # (n_measurements,) with values between 0 and 7 inclusive

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # (mu, log_tau, theta_prime)
        batch_shape = x.shape[:-1]
        mu = x[..., 0]
        log_tau = x[..., 1]
        theta_prime = x[..., 2:]

        tau, log_det_tau = bound_parameter(log_tau, batch_shape, low=0.0, high=torch.inf)
        log_det = log_det_tau

        theta = mu[..., None] + tau[..., None] * theta_prime

        log_prob_mu = td.Normal(loc=0.0, scale=10.0).log_prob(mu)
        log_prob_tau = td.LogNormal(loc=5.0, scale=1.0).log_prob(tau)
        log_prob_theta_prime = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(theta_prime), batch_shape)
        log_prior = log_prob_mu + log_prob_tau + log_prob_theta_prime

        log_likelihood = td.Independent(
            td.Normal(loc=theta[self.index], scale=self.sigma[self.index]),
            reinterpreted_batch_ndims=11
        ).log_prob(self.measurements)

        log_prob = log_likelihood + log_prior + log_det
        return - log_prob
