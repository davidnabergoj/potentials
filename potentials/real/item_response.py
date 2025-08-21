import json
from pathlib import Path

import torch
import torch.distributions as td
from potentials.real.posterior_base import Posterior1D
from potentials.utils import reduce_two_key_dataset, sum_except_batch
import urllib.request


class SyntheticItemResponseTheory(Posterior1D):
    """

    Reference: https://github.com/stan-dev/example-models/blob/master/misc/irt/irt.data.json
    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf.
    """

    def __init__(self,
                 n_data: int = None):
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

        self._modified = False

        # Handle smaller data
        if n_data is not None:
            if n_data > self.n_responses:
                raise ValueError("New dataset cannot have more entries than the original")
            self.responses = self.responses[:n_data]
            self.student_index = self.student_index[:n_data]
            self.question_index = self.question_index[:n_data]
            self.n_responses = n_data

        # Keep original event shape even if we remove some data entries
        super().__init__(event_shape=(501,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]

        beta = x[..., 0:400]
        alpha = x[..., 400:500]
        delta = x[..., 500]

        log_prob_delta = td.Normal(loc=3 / 4, scale=1.0).log_prob(delta)
        log_prob_alpha = sum_except_batch(
            td.Normal(loc=0.0, scale=1.0).log_prob(alpha), batch_shape)
        log_prob_beta = sum_except_batch(
            td.Normal(loc=0.0, scale=1.0).log_prob(beta), batch_shape)
        log_prior = log_prob_alpha + log_prob_beta + log_prob_delta

        probs = torch.sigmoid(
            alpha[..., self.student_index] - beta[..., self.question_index] + delta[..., None])
        log_likelihood = sum_except_batch(
            torch.distributions.Bernoulli(
                probs=probs).log_prob(self.responses),
            batch_shape
        )
        log_prob = log_likelihood + log_prior

        return -log_prob

    def normalized_log_posterior_predictive_density(self, posterior_draws: torch.Tensor) -> torch.Tensor:
        beta = posterior_draws[..., 0:400]
        alpha = posterior_draws[..., 400:500]
        delta = posterior_draws[..., 500]

        probs = torch.sigmoid(
            alpha[..., self.student_index] - beta[..., self.question_index] + delta[..., None])
        log_likelihood = torch.distributions.Bernoulli(
            probs=probs).log_prob(self.responses)
        return log_likelihood.exp().mean(dim=1).log().mean()  # Take mean instead of sum

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
            raise ValueError("Reference second moment unavailable for modified dataset")
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
    print(f'{u.event_shape = }')
    print()
    print(f'{len(torch.unique(u.student_index)) = }')
    print(f'{len(torch.unique(u.question_index)) = }')

    torch.manual_seed(0)
    out = u(torch.randn(size=(5, *u.event_shape)))
    print(out)

    torch.manual_seed(0)
    out = u(torch.randn(size=(2, 3, *u.event_shape)))
    print(out)
