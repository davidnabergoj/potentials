import json
from typing import Dict
import urllib.request
from pathlib import Path

import torch
import torch.distributions as td
from potentials.real.posterior_base import Posterior1D
from potentials.real.posterior_util import Parameter1D, ParameterSet1D
from potentials.transformations import bound_parameter
from potentials.utils import sum_except_batch


class EightSchools(Posterior1D):
    """

    Reference: https://raw.githubusercontent.com/stan-dev/example-models/master/misc/eight_schools/eight_schools.data.json
    Reference: https://www.tensorflow.org/probability/examples/Eight_Schools
    """

    def __init__(self):
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

        super().__init__(
            event_shape=(10,),
            posterior_parameters=ParameterSet1D({
                'mu': Parameter1D(1),
                'tau': Parameter1D(1, 'positive'),
                'theta_prime': Parameter1D(8),
            })
        )

    def extract_parameters(self,
                           unconstrained: torch.Tensor,
                           return_log_probs: bool = True) -> Dict[str, torch.Tensor]:
        # (mu_a, log_sigma_a, log_sigma_y, a, b)
        out, log_det = self.posterior_parameters.constrain(
            unconstrained
        )

        # Compute prior probabilities
        if return_log_probs:
            log_prob_mu = td.Normal(
                loc=0.0,
                scale=10.0
            ).log_prob(out['mu'])[..., 0]
            log_prob_tau = td.LogNormal(
                loc=5.0,
                scale=1.0
            ).log_prob(out['tau'])[..., 0]
            log_prob_theta_prime = td.Normal(
                loc=0.0,
                scale=1.0
            ).log_prob(
                out['theta_prime']
            ).sum(dim=-1)

            out['log_prior'] = (
                log_prob_mu
                + log_prob_tau
                + log_prob_theta_prime
                + log_det
            )

        return out

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        theta = (
            extracted['mu']
            + (
                extracted['tau']
                * extracted['theta_prime']
            )
        )

        return td.Independent(
            td.Normal(loc=theta, scale=self.scales),
            reinterpreted_batch_ndims=1
        )

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.measurements)

    @property
    def mean(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'eight_schools_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[0]
        else:
            raise ValueError("Moment file not found")

    @property
    def second_moment(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'eight_schools_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2

    def _compute_likelihood_parameters(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        mu = x[..., 0]
        log_tau = x[..., 1]
        theta_prime = x[..., 2:]

        tau, _ = bound_parameter(
            log_tau,
            batch_shape,
            low=0.0,
            high=torch.inf
        )

        theta = mu[..., None] + tau[..., None] * theta_prime
        return theta

    def posterior_predictive_draws(self, posterior_draws: torch.Tensor, n_draws: int = 100) -> torch.Tensor:
        theta = self._compute_likelihood_parameters(posterior_draws)
        dist = td.Independent(
            td.Normal(loc=theta, scale=self.scales),
            reinterpreted_batch_ndims=1
        )
        return dist.sample((n_draws,))

    def normalized_log_posterior_predictive_density(self, posterior_draws: torch.Tensor):
        theta = self._compute_likelihood_parameters(posterior_draws)
        dist = td.Independent(
            td.Normal(loc=theta, scale=self.scales),
            reinterpreted_batch_ndims=1
        )
        log_likelihood = dist.log_prob(self.measurements)
        return log_likelihood.exp().mean(dim=-1).log().mean()  # Take mean instead of sum


if __name__ == '__main__':
    u = EightSchools()
    print(u.mean.shape)
    print(u.second_moment.shape)

    torch.manual_seed(0)
    print(u(torch.randn(size=(5, *u.event_shape))))

    torch.manual_seed(0)
    print(u(torch.randn(size=(2, 3, *u.event_shape))))
