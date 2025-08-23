import json
from pathlib import Path
from typing import Dict

import torch
import torch.distributions as td
from potentials.real.posterior_base import Posterior1D
from potentials.real.posterior_util import ParameterSet1D, ParameterScalar, ParameterVector
from potentials.utils import reduce_two_key_dataset, sum_except_batch
import urllib.request


class SyntheticItemResponseTheory(Posterior1D):
    """

    Reference: https://github.com/stan-dev/example-models/blob/master/misc/irt/irt.data.json
    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf.
    """

    def __init__(self,
                 n_data: int = None,
                 use_hyperprior: bool = False):
        download_url = "https://raw.githubusercontent.com/stan-dev/example-models/master/misc/irt/irt.data.json"
        data_dir = Path(__file__).parent / 'downloaded'
        data_file = data_dir / "irt.data.json"
        if not data_file.exists():
            print(f'Downloading {download_url}')
            urllib.request.urlretrieve(download_url, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)

        self.n_responses = data['N']  # 30105, N
        self.n_students = data['K']  # 100, K
        self.n_questions = data['J']  # 400, J

        self.responses: torch.Tensor = torch.tensor(
            data['y'], dtype=torch.float)
        self.student_index: torch.Tensor = torch.tensor(
            data['kk'], dtype=torch.long) - 1
        self.question_index: torch.Tensor = torch.tensor(
            data['jj'], dtype=torch.long) - 1

        # Handle smaller data
        if n_data is not None:
            if n_data > self.n_responses:
                raise ValueError(
                    "New dataset cannot have more entries than the original")
            self.responses = self.responses[:n_data]
            self.student_index = self.student_index[:n_data]
            self.question_index = self.question_index[:n_data]
            self.n_responses = n_data

        self._modified = (n_data is not None) and (not use_hyperprior)
        self.use_hyperprior = use_hyperprior

        if use_hyperprior:
            super().__init__(
                event_shape=(504,),
                posterior_parameters=ParameterSet1D({
                    'beta': ParameterVector(400),
                    'alpha': ParameterVector(100),
                    'delta': ParameterScalar(),
                    'beta_prior_scale': ParameterScalar('positive'),
                    'alpha_prior_scale': ParameterScalar('positive'),
                    'delta_prior_scale': ParameterScalar('positive'),
                })
            )
        else:
            super().__init__(
                event_shape=(501,),
                posterior_parameters=ParameterSet1D({
                    'beta': ParameterVector(400),
                    'alpha': ParameterVector(100),
                    'delta': ParameterScalar(),
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
            if self.use_hyperprior:
                # Top-level (unconstrained variance) priors
                log_prob_beta_prior_scale = td.Cauchy(
                    loc=0.0,
                    scale=5.0
                ).log_prob(out['beta_prior_scale'])[..., 0]
                log_prob_alpha_prior_scale = td.Cauchy(
                    loc=0.0,
                    scale=5.0
                ).log_prob(out['alpha_prior_scale'])[..., 0]
                log_prob_delta_prior_scale = td.Cauchy(
                    loc=0.0,
                    scale=5.0
                ).log_prob(out['delta_prior_scale'])[..., 0]

                # Bottom-level priors
                log_prob_beta = td.Independent(
                    td.Normal(
                        loc=0.0,
                        scale=out['beta_prior_scale']
                    ),
                    reinterpreted_batch_ndims=1
                ).log_prob(out['beta'])
                log_prob_alpha = td.Independent(
                    td.Normal(
                        loc=0.0,
                        scale=out['alpha_prior_scale']
                    ),
                    reinterpreted_batch_ndims=1
                ).log_prob(out['alpha'])
                log_prob_delta = td.Independent(
                    td.Normal(
                        loc=0.0,
                        scale=out['delta_prior_scale']
                    ),
                    reinterpreted_batch_ndims=1
                ).log_prob(out['delta'])

                out['log_prior'] = (
                    log_prob_beta_prior_scale
                    + log_prob_alpha_prior_scale
                    + log_prob_delta_prior_scale
                    + log_prob_beta
                    + log_prob_alpha
                    + log_prob_delta
                    + log_det
                )
            else:
                log_prob_beta = td.Normal(
                    loc=0.0,
                    scale=1.0
                ).log_prob(out['beta']).sum(dim=-1)
                log_prob_alpha = td.Normal(
                    loc=0.0,
                    scale=1.0
                ).log_prob(out['alpha']).sum(dim=-1)
                log_prob_delta = td.Normal(
                    loc=3 / 4,
                    scale=1.0
                ).log_prob(out['delta'])[..., 0]

                out['log_prior'] = (
                    log_prob_beta
                    + log_prob_alpha
                    + log_prob_delta
                    + log_det
                )

        return out

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):

        logits = (
            extracted['alpha'][..., self.student_index]
            - extracted['beta'][..., self.question_index]
            + extracted['delta']
        )
        return td.Independent(td.Bernoulli(logits=logits), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.responses)

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'synthetic_item_response_theory_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[0]
        else:
            raise ValueError("Moment file not found")

    @property
    def second_moment(self):
        if self._modified:
            raise ValueError(
                "Reference second moment unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'synthetic_item_response_theory_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


if __name__ == '__main__':
    u = SyntheticItemResponseTheory(n_data=1500)
    print(f'{u.responses.shape=}, {u.n_responses=}')
    print(f'{u.student_index.shape=}, {u.n_students=}')
    print(f'{u.question_index.shape=}, {u.n_questions=}')
    print(f'{u.event_shape=}')
    print()
    print(f'{len(torch.unique(u.student_index))=}')
    print(f'{len(torch.unique(u.question_index))=}')

    torch.manual_seed(0)
    out = u(torch.randn(size=(5, *u.event_shape)))
    print(out)

    torch.manual_seed(0)
    out = u(torch.randn(size=(2, 3, *u.event_shape)))
    print(out)
