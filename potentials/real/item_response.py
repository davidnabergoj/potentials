import json
from pathlib import Path

import torch
import torch.distributions as td
from potentials.base import Potential
from potentials.utils import sum_except_batch
import urllib.request


class SyntheticItemResponseTheory(Potential):
    """

    Reference: https://github.com/stan-dev/example-models/blob/master/misc/irt/irt.data.json
    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf.
    """

    def __init__(self):
        super().__init__(event_shape=(501,))

        download_url = "https://raw.githubusercontent.com/stan-dev/example-models/master/misc/irt/irt.data.json"
        data_dir = Path(__file__).parent / 'downloaded'
        data_file = data_dir / "irt.data.json"
        if not data_file.exists():
            print(f'Downloading {download_url}')
            urllib.request.urlretrieve(download_url, data_file)
        with open(data_file, "r") as f:
            data = json.load(f)

        self.n_students = data['K']  # 100, K
        self.n_questions = data['J']  # 400, J
        self.n_responses = data['N']  # 30105, N
        self.responses: torch.Tensor = torch.tensor(data['y'], dtype=torch.float)
        self.student_index: torch.Tensor = torch.tensor(data['kk'], dtype=torch.long) - 1
        self.response_index: torch.Tensor = torch.tensor(data['jj'], dtype=torch.long) - 1

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]

        beta = x[..., 0:400]
        alpha = x[..., 400:500]
        delta = x[..., 500]

        log_prob_delta = td.Normal(loc=3 / 4, scale=1.0).log_prob(delta)
        log_prob_alpha = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(alpha), batch_shape)
        log_prob_beta = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(beta), batch_shape)
        log_prior = log_prob_alpha + log_prob_beta + log_prob_delta

        probs = torch.sigmoid(alpha[..., self.student_index] - beta[..., self.response_index] + delta[..., None])
        log_likelihood = sum_except_batch(
            torch.distributions.Bernoulli(probs=probs).log_prob(self.responses),
            batch_shape
        )
        log_prob = log_likelihood + log_prior

        return -log_prob

    @property
    def mean(self):
        path = Path(__file__).parent.parent / 'true_moments' / f'synthetic_item_response_theory_moments.pt'
        if path.exists():
            return torch.load(path)[0]
        return super().mean

    @property
    def second_moment(self):
        path = Path(__file__).parent.parent / 'true_moments' / f'synthetic_item_response_theory_moments.pt'
        if path.exists():
            return torch.load(path)[1]
        return super().second_moment

if __name__ == '__main__':
    u = SyntheticItemResponseTheory()
    print(u.mean.shape)
    print(u.second_moment.shape)
    print(u.variance.shape)

    torch.manual_seed(0)
    out = u(torch.randn(size=(5, *u.event_shape)))
    print(out)

    torch.manual_seed(0)
    out = u(torch.randn(size=(2, 3, *u.event_shape)))
    print(out)
