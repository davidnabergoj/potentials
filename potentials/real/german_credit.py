import numpy as np

from potentials.base import Potential
import torch
from pathlib import Path
import urllib.request
import zipfile


def load_german_credit():
    download_url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    data_dir = Path(__file__).parent / 'downloaded'
    data_file = data_dir / "statlog+german+credit+data.zip"
    if not data_file.exists():
        print(f'Downloading {download_url}')
        urllib.request.urlretrieve(download_url, data_file)
    with zipfile.ZipFile(data_file, "r") as f:
        data = f.read("german.data-numeric")
        data = data.decode("utf-8")
        data = data.split()
        data = list(map(int, data))
        data = np.array(data, dtype=float)
        data = np.reshape(data, (1000, 25))

        x = data[..., :-1]
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        x = np.c_[x, np.ones(1000)]  # Add a constant factor
        y = data[..., -1] - 1  # from {1, 2} to {0, 1}

        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

    return x, y


class GermanCredit(Potential):
    def __init__(self):
        self.features, self.labels = load_german_credit()
        super().__init__((25,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 26
        batch_shape = x.shape[:-1]
        unnormalized_tau = x[..., 0]
        tau = torch.exp(unnormalized_tau)
        beta = x[..., 1:]

        # Compute the log prior
        log_prior = torch.add(
            torch.distributions.Gamma(0.5, 0.5).log_prob(tau),
            torch.distributions.Normal(0.0, 1.0).log_prob(beta).sum(dim=-1),
        )

        # Compute the log likelihood
        probs = torch.sigmoid(
            torch.einsum(
                'nf,...f->...nf',
                self.features,
                tau.view(*batch_shape, 1) * beta
            ).sum(dim=-1)  # shape = (*batch_shape, features)
        )
        log_likelihood = torch.distributions.Bernoulli(probs=probs).log_prob(self.labels).sum(dim=-1)

        log_probability = log_likelihood + log_prior

        return -log_probability


class SparseGermanCredit(Potential):
    def __init__(self):
        self.features, self.labels = load_german_credit()
        super().__init__((51,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 51
        batch_shape = x.shape[:-1]
        unnormalized_tau = x[..., 0]
        tau = torch.exp(unnormalized_tau)
        beta = x[..., 1:26]
        unconstrained_lambda = x[..., 26:]
        lmbd = torch.exp(unconstrained_lambda)

        # Compute the log prior
        log_prior = torch.add(
            torch.distributions.Gamma(0.5, 0.5).log_prob(tau),
            torch.add(
                torch.distributions.Gamma(0.5, 0.5).log_prob(lmbd).sum(dim=-1),
                torch.distributions.Normal(0.0, 1.0).log_prob(beta).sum(dim=-1)
            )
        )

        # Compute the log likelihood
        probs = torch.sigmoid(
            torch.einsum(
                'nf,...f->...nf',
                self.features,
                tau.view(*batch_shape, 1) * beta * lmbd
            ).sum(dim=-1)  # shape = (*batch_shape, features)
        )
        log_likelihood = torch.distributions.Bernoulli(probs=probs).log_prob(self.labels).sum(dim=-1)

        log_probability = log_likelihood + log_prior

        return -log_probability
