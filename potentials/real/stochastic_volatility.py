import pathlib
from pathlib import Path
import csv

import torch
import torch.distributions as td

from potentials.base import Potential
from potentials.transformations import bound_parameter
from potentials.utils import sum_except_batch


class StochasticVolatilityModel(Potential):
    """

    Data retrieved: August 21, 2024
    Data url: https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1277424000&period2=1593043200&interval=1d&events=history
    Reference: https://github.com/tensorflow/probability/blob/a4852982c5f40a24b20be58b0e32daa52de2464e/spinoffs/inference_gym/inference_gym/internal/datasets/sp500_closing_prices.py
    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf
    """

    def __init__(self):
        data_path = Path(__file__).parent / 'data' / '^GSPC.csv'
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            closing_prices = torch.tensor([float(row[4]) for row in reader], dtype=torch.float)

        self.measurements: torch.Tensor = closing_prices
        self.n_measurements = len(self.measurements)
        super().__init__(event_shape=(self.n_measurements + 3,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # (z, unconstrained_sigma, unconstrained_mu, unconstrained_phi)
        batch_shape = x.shape[:-1]

        z = x[..., :self.n_measurements]
        unconstrained_sigma = x[..., self.n_measurements]
        unconstrained_mu = x[..., self.n_measurements + 1]
        unconstrained_phi = x[..., self.n_measurements + 2]

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

        h = torch.zeros(size=(*batch_shape, self.n_measurements), device=x.device, dtype=x.dtype)
        h[..., 0] = mu + sigma * z[..., 0] - torch.sqrt(1 - phi ** 2)
        for i in range(1, self.n_measurements):
            h[..., i] = mu + sigma * z[..., i] + phi * (h[..., i - 1] - mu)

        y_scale = torch.exp(h / 2)
        y_loc = torch.zeros_like(y_scale)
        log_likelihood = td.Independent(
            td.Normal(loc=y_loc, scale=y_scale),
            reinterpreted_batch_ndims=1
        ).log_prob(self.measurements)

        log_prob = log_likelihood + log_prior + log_det
        return -log_prob

    @property
    def mean(self):
        return torch.load(
            pathlib.Path(__file__).absolute().parent.parent / 'true_moments' / 'stochastic_volatility_moments.pt'
        )[0]

    @property
    def second_moment(self):
        return torch.load(
            pathlib.Path(__file__).absolute().parent.parent / 'true_moments' / 'stochastic_volatility_moments.pt'
        )[1]


if __name__ == '__main__':
    u = StochasticVolatilityModel()
    print(u.mean.shape)
    print(u.second_moment.shape)
    print(u.mean.isfinite().all())
    print(u.second_moment.isfinite().all())

    torch.manual_seed(0)
    print(u(1.5 + torch.randn(size=(5, *u.event_shape)) / 10))

    torch.manual_seed(0)
    print(u(torch.randn(size=(2, 3, *u.event_shape))))
