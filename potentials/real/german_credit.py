import numpy as np

from potentials.base import Potential
import torch
from pathlib import Path
import urllib.request
import zipfile


class GermanCredit(Potential):
    def __init__(self):
        event_shape = (25,)
        download_url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
        data_file = Path("downloaded/statlog+german+credit+data.zip")
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
            data = torch.as_tensor(data)

            x = data[..., :-1]
            x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
            y = data[..., -1] - 1  # from {1, 2} to {0, 1}

            self.features = x
            self.labels = y
        super().__init__(event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 25
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

