import pathlib
from pathlib import Path
import csv
from typing import Dict

import torch
import torch.distributions as td

from potentials.real.posterior.base import Posterior1D
from potentials.real.posterior.parameter import ParameterDAG, ParameterVector, ParameterScalar


class StochasticVolatilityModel(Posterior1D):
    """
    Stochastic volatility model.

    Data retrieved: August 21, 2024
    Data url: https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1277424000&period2=1593043200&interval=1d&events=history
    Reference: https://github.com/tensorflow/probability/blob/a4852982c5f40a24b20be58b0e32daa52de2464e/spinoffs/inference_gym/inference_gym/internal/datasets/sp500_closing_prices.py
    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf
    """

    def __init__(self, n_measurements: int = 2517):
        """
        StochasticVolatilityModel constructor.

        :param int n_measurements: maximum number of measurements to use. A smaller number results in a simpler model 
         whose log probability density is computed faster.
        """
        data_path = Path(__file__).parent / 'data' / '^GSPC.csv'
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            closing_prices = torch.tensor(
                [float(row[4]) for row in reader], dtype=torch.float)

        self.measurements: torch.Tensor = closing_prices[:n_measurements]
        self.n_measurements = len(self.measurements)
        self._modified = self.n_measurements != 2517

        def compute_h(mu, sigma, phi, z):
            batch_shape = z.shape[:-1]
            h = torch.zeros(
                size=(*batch_shape, self.n_measurements),
                device=z.device,
                dtype=z.dtype
            )
            h[..., 0] = (
                mu
                + (
                    sigma * z[..., [0]]
                    / torch.sqrt(1 - phi ** 2)
                )
            )[..., 0]
            for i in range(1, self.n_measurements):
                h[..., i] = (
                    mu
                    + (
                        sigma
                        * z[..., [i]] + phi
                        * (
                            h[..., [i - 1]] - mu
                        )
                    )
                )[..., 0]
            return h

        super().__init__(
            posterior_parameters=ParameterDAG(
                {
                    'z': ParameterVector(
                        self.n_measurements,
                        prior=td.Normal(0.0, 1.0)
                    ),
                    'sigma': ParameterScalar(
                        bound='positive',
                        prior=td.HalfCauchy(scale=2.0)
                    ),
                    'mu': ParameterScalar(
                        bound='positive',
                        prior=td.Exponential(rate=1.0)
                    ),
                    'phi_prime': ParameterScalar(
                        bound=(0, 1),
                        prior=td.Beta(
                            concentration0=20.0,
                            concentration1=1.5
                        )
                    ),
                },
                additional_parameters={
                    'phi': (lambda phi_prime, **kwargs: phi_prime * 2 - 1),
                    'h': (
                        lambda mu, sigma, phi, z, **kwargs:
                        compute_h(mu, sigma, phi, z)
                    )
                }
            )
        )

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        y_scale = torch.exp(extracted['h'] / 2)
        y_loc = torch.zeros_like(y_scale)
        return td.Independent(
            td.Normal(
                loc=y_loc,
                scale=y_scale
            ),
            1
        )

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.measurements)

    @property
    def mean(self):
        if self._modified:
            raise ValueError(
                "Reference mean unavailable for modified dataset")
        return torch.load(
            pathlib.Path(__file__).absolute().parent.parent /
            'true_moments' / 'stochastic_volatility_moments.pt',
            weights_only=True
        )[0]

    @property
    def second_moment(self):
        if self._modified:
            raise ValueError(
                "Reference second moment unavailable for modified dataset")
        return torch.load(
            pathlib.Path(__file__).absolute().parent.parent /
            'true_moments' / 'stochastic_volatility_moments.pt',
            weights_only=True
        )[1]

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


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
