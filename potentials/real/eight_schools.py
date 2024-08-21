import json
import urllib.request
from pathlib import Path

import torch
import torch.distributions as td
from potentials.base import Potential
from potentials.transformations import bound_parameter
from potentials.utils import sum_except_batch


class EightSchools(Potential):
    """

    Reference: https://raw.githubusercontent.com/stan-dev/example-models/master/misc/eight_schools/eight_schools.data.json
    Reference: https://www.tensorflow.org/probability/examples/Eight_Schools
    """

    def __init__(self):
        super().__init__(event_shape=(10,))

        download_url = "https://raw.githubusercontent.com/stan-dev/example-models/master/misc/eight_schools/eight_schools.data.json"
        data_dir = Path(__file__).parent / 'downloaded'
        data_file = data_dir / "eight_schools.data.json"
        if not data_file.exists():
            print(f'Downloading {download_url}')
            urllib.request.urlretrieve(download_url, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)

        self.measurements = torch.tensor(data['y'], dtype=torch.float)
        self.scales = torch.tensor(data['sigma'], dtype=torch.float)  # (8,)

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
            td.Normal(loc=theta, scale=self.scales),
            reinterpreted_batch_ndims=1
        ).log_prob(self.measurements)

        log_prob = log_likelihood + log_prior + log_det
        return - log_prob

    @property
    def mean(self):
        return torch.tensor([
            -9.7772e-03, 4.4717e+00, -1.1302e-02, -1.0023e-01, 3.2246e-02,
            1.6387e-01, 6.1710e-02, -4.1238e-04, 2.4907e-02, 1.5544e-01
        ])

    @property
    def second_moment(self):
        return torch.tensor([
            1.1949e+00, 4.2651e+01, 3.8306e-01, 4.1782e-01, 6.7065e-01,
            4.7906e-01, 5.7163e-01, 5.3950e-01, 6.4698e-01, 7.1619e-01
        ])


if __name__ == '__main__':
    u = EightSchools()
    print(u.mean.shape)
    print(u.second_moment.shape)

    torch.manual_seed(0)
    print(u(torch.randn(size=(5, *u.event_shape))))

    torch.manual_seed(0)
    print(u(torch.randn(size=(2, 3, *u.event_shape))))
