import numpy as np

from potentials.base import Posterior
import torch
from pathlib import Path
import urllib.request
import zipfile
import torch.distributions as td

from potentials.transformations import bound_parameter, bound_positive


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


class GermanCredit(Posterior):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    """

    def __init__(self, n_data: int = None):
        self.features, self.labels = load_german_credit()

        if n_data is not None:
            if not 0 < n_data <= len(self.labels):
                raise ValueError(
                    "Number of used observations must be between zero and the number of total observations"
                )
            self.features = self.features[:n_data]
            self.labels = self.labels[:n_data]
        
        super().__init__((26,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 26
        batch_shape = x.shape[:-1]

        beta = x[..., 1:]
        unnormalized_tau = x[..., 0]
        tau, log_det_tau = bound_parameter(
            unnormalized_tau,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        log_det = log_det_tau

        # Compute the log prior
        log_prior = torch.add(
            td.Gamma(0.5, 0.5).log_prob(tau),
            td.Normal(0.0, 1.0).log_prob(beta).sum(dim=-1),
        )

        # Compute the log likelihood
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            tau.view(*batch_shape, 1) * beta
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        log_likelihood = td.Bernoulli(
            logits=logits
        ).log_prob(self.labels).sum(dim=-1)
        log_probability = log_likelihood + log_prior + log_det
        return -(log_probability + log_det_tau)

    @property
    def mean(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[0]
        return torch.tensor([
            -5.5876e+01, -1.1380e-01, 1.0360e-01, -1.6707e-01, -4.4986e-02,
            -2.6688e-02, 1.2024e-01, 8.7453e-03, -9.3806e-02, 1.4022e-02,
            1.6742e-01, -4.0864e-02, 4.4368e-02, -1.9163e-02, -2.6597e-04,
            -1.5850e-01, 2.3449e-02, -1.2344e-01, 7.1330e-02, 1.5693e-01,
            1.3266e-01, 1.2540e-01, 3.0680e-02, -1.0913e-01, 3.7934e-03,
            -3.2970e-01
        ])

    @property
    def second_moment(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        return torch.tensor([
            4.2846e+03, 8.5638e-01, 7.9636e-01, 7.2609e-01, 9.1535e-01, 7.0697e-01,
            9.6090e-01, 1.0219e+00, 8.6421e-01, 7.3238e-01, 8.2984e-01, 7.9772e-01,
            7.4951e-01, 6.5836e-01, 8.0805e-01, 1.1717e+00, 6.8116e-01, 8.3310e-01,
            9.4624e-01, 6.5050e-01, 7.3698e-01, 8.1886e-01, 6.6457e-01, 7.0312e-01,
            8.2957e-01, 1.0161e+00
        ])

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2

    def _compute_likelihood_parameters(self, x: torch.Tensor):
        assert x.shape[-1] == 26
        batch_shape = x.shape[:-1]
        beta = x[..., 1:]
        unnormalized_tau = x[..., 0]
        tau, _ = bound_parameter(
            unnormalized_tau,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            tau.view(*batch_shape, 1) * beta
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        return logits

    def posterior_predictive_draws(self, posterior_draws: torch.Tensor, n_draws: int = 100) -> torch.Tensor:
        logits = self._compute_likelihood_parameters(posterior_draws)
        dist = td.Bernoulli(logits=logits)
        return dist.sample((n_draws,))

    def normalized_log_posterior_predictive_density(self, posterior_draws: torch.Tensor):
        logits = self._compute_likelihood_parameters(posterior_draws)
        log_likelihood = td.Bernoulli(logits=logits).log_prob(self.labels)
        return log_likelihood.exp().mean(dim=-1).log().mean()  # Take mean instead of sum


class SparseGermanCredit(Posterior):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    lambda[i] ~ Gamma(0.5, 0.5)
    """

    def __init__(self, n_data: int = None):
        self.features, self.labels = load_german_credit()

        if n_data is not None:
            if not 0 < n_data <= len(self.labels):
                raise ValueError(
                    "Number of used observations must be between zero and the number of total observations"
                )
            self.features = self.features[:n_data]
            self.labels = self.labels[:n_data]

        super().__init__((51,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 51
        batch_shape = x.shape[:-1]

        beta = x[..., 1:26]
        unnormalized_tau = x[..., 0]
        unconstrained_lambda = x[..., 26:]

        tau, log_det_tau = bound_parameter(
            unnormalized_tau,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        lmbd, log_det_lmbd = bound_parameter(
            unconstrained_lambda,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        log_det = log_det_tau + log_det_lmbd

        # Compute the log prior
        log_prior = torch.add(
            td.Gamma(0.5, 0.5).log_prob(tau),
            torch.add(
                td.Gamma(0.5, 0.5).log_prob(lmbd).sum(dim=-1),
                td.Normal(0.0, 1.0).log_prob(beta).sum(dim=-1)
            )
        )

        # Compute the log likelihood
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            tau.view(*batch_shape, 1) * beta * lmbd
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        log_likelihood = td.Bernoulli(
            logits=logits
        ).log_prob(self.labels).sum(dim=-1)
        log_probability = log_likelihood + log_prior + log_det

        return -log_probability

    @property
    def mean(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'sparse_german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[0]
        return torch.tensor([
            -2.7315, -0.9769, 0.6704, -0.7520, 0.4620, -0.6109, -0.1834, -0.4439,
            -0.1633, 0.4071, -0.4058, -0.4201, -0.0725, 0.0516, 0.0628, -0.4283,
            0.5482, -0.4306, 0.2902, 0.3007, 0.2914, -0.3503, -0.2319, -0.1052,
            0.1482, -1.2035, 1.0472, 0.3176, 0.3929, 0.3200, 0.1527, -0.2702,
            -0.2113, -0.2797, -0.0269, -0.1024, -0.2499, -0.4676, -0.1727, -0.2265,
            0.2951, 0.1347, 0.2588, -0.1072, 0.2663, -0.0539, -0.1067, 0.2330,
            -0.2595, -0.0169, 1.3060
        ])

    @property
    def second_moment(self):
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'sparse_german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        return torch.tensor([
            8.4133, 1.7535, 0.9989, 1.2258, 1.0043, 0.9493, 0.8957, 0.8257, 0.7243,
            0.7776, 0.7038, 1.1839, 0.5382, 0.5155, 0.8522, 0.7376, 0.9852, 0.6100,
            1.0263, 0.6785, 0.8097, 0.7912, 0.5397, 0.6826, 0.7234, 2.3983, 6.5461,
            2.9223, 3.1089, 1.9430, 3.5129, 2.6259, 1.2902, 1.4363, 2.2793, 1.5573,
            1.4063, 2.0248, 1.5091, 3.7975, 2.4856, 1.2449, 3.0840, 1.8412, 3.0192,
            1.6737, 3.2266, 3.2037, 1.2882, 1.6431, 5.2194
        ])

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2

    def _compute_likelihood_parameters(self, x: torch.Tensor):
        assert x.shape[-1] == 51
        batch_shape = x.shape[:-1]

        beta = x[..., 1:26]
        unnormalized_tau = x[..., 0]
        unconstrained_lambda = x[..., 26:]

        tau, _ = bound_parameter(
            unnormalized_tau,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        lmbd, _ = bound_parameter(
            unconstrained_lambda,
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            tau.view(*batch_shape, 1) * beta * lmbd
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        return logits

    def posterior_predictive_draws(self, posterior_draws: torch.Tensor, n_draws: int = 100) -> torch.Tensor:
        logits = self._compute_likelihood_parameters(posterior_draws)
        dist = td.Bernoulli(logits=logits)
        return dist.sample((n_draws,))

    def normalized_log_posterior_predictive_density(self, posterior_draws: torch.Tensor):
        logits = self._compute_likelihood_parameters(posterior_draws)
        log_likelihood = td.Bernoulli(logits=logits).log_prob(self.labels)
        return log_likelihood.exp().mean(dim=1).log().mean()  # Take mean instead of sum


class SparseGermanCreditMissingData(Posterior):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    lambda[i] ~ Gamma(0.5, 0.5)

    For each missing entry in columns 1, 3, 9 of the first 100 rows:
    x_ij ~ Normal(mu_ij, 1),  mu_ij ~ Normal(0, 1)

    Total parameters: 1 + 25 + 25 + 300 = 351
    """

    def __init__(self, n_missing_rows=100):
        self.n_missing_rows = n_missing_rows
        self.full_features, self.labels = load_german_credit()

        self.features = self.full_features.clone()
        self.features[:self.n_missing_rows, [1, 3, 9]] = torch.nan

        super().__init__((51 + self.n_missing_rows * 3,))  # tau + beta + lambda + 300 mus

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.event_shape[0]
        batch_shape = x.shape[:-1]

        unnormalized_tau = x[..., 0]
        beta = x[..., 1:26]
        unconstrained_lambda = x[..., 26:51]
        mu_c1 = x[..., 51:51+self.n_missing_rows]
        mu_c3 = x[..., 51+self.n_missing_rows:51+self.n_missing_rows*2]
        mu_c9 = x[..., 51+self.n_missing_rows*2:51+self.n_missing_rows*3]

        # Transform positive-only variables
        tau, log_det_tau = bound_positive(unnormalized_tau, batch_shape)
        lmbd, log_det_lmbd = bound_positive(unconstrained_lambda, batch_shape)

        log_det = log_det_tau + log_det_lmbd  # No transform on mu_* (real)

        # Priors
        log_prior = (
            td.Gamma(0.5, 0.5).log_prob(tau)
            + td.Gamma(0.5, 0.5).log_prob(lmbd).sum(dim=-1)
            + td.Normal(0.0, 1.0).log_prob(beta).sum(dim=-1)
            + td.Normal(0.0, 1.0).log_prob(mu_c1).sum(dim=-1)
            + td.Normal(0.0, 1.0).log_prob(mu_c3).sum(dim=-1)
            + td.Normal(0.0, 1.0).log_prob(mu_c9).sum(dim=-1)
        )

        # Impute missing entries
        imputed_features = self.features.unsqueeze(0).expand(
            x.shape[0], -1, -1).clone().to(x)  # (n_chains, N, D)
        imputed_features[..., :self.n_missing_rows, 1] = mu_c1
        imputed_features[..., :self.n_missing_rows, 3] = mu_c3
        imputed_features[..., :self.n_missing_rows, 9] = mu_c9

        # Bernoulli likelihood for observed labels
        bernoulli_logits = torch.einsum(
            '...nd,...d->...n',
            imputed_features,  # (..., N, D)
            tau.view(*batch_shape, 1) * beta * lmbd  # (..., D)
        )

        log_likelihood_bernoulli = td.Bernoulli(
            logits=bernoulli_logits).log_prob(self.labels).sum(dim=-1)

        # Normal likelihoods for imputed values
        def observed_column(col_idx, mu):
            obs = imputed_features[..., :self.n_missing_rows, col_idx]
            return td.Independent(
                td.Normal(loc=mu, scale=1.0),
                reinterpreted_batch_ndims=1
            ).log_prob(obs).sum(dim=-1)

        log_likelihood_c1 = observed_column(1, mu_c1)
        log_likelihood_c3 = observed_column(3, mu_c3)
        log_likelihood_c9 = observed_column(9, mu_c9)

        log_likelihood = (
            log_likelihood_bernoulli
            + log_likelihood_c1
            + log_likelihood_c3
            + log_likelihood_c9
        )

        log_posterior = log_likelihood + log_prior + log_det
        return -log_posterior

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2
