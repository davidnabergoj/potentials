from typing import Dict
import numpy as np

from potentials.real.posterior.base import Posterior1D
import torch
from pathlib import Path
import urllib.request
import zipfile
import torch.distributions as td

from potentials.real.posterior.parameter import ParameterDAG, ParameterVector, ParameterScalar
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


class GermanCredit(Posterior1D):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    """

    def __init__(self,
                 n_data: int = None,
                 use_hyperprior: bool = False):
        self.features, self.labels = load_german_credit()

        if n_data is not None:
            if not 0 < n_data <= len(self.labels):
                raise ValueError(
                    "Number of used observations must be between zero and the number of total observations"
                )
            self.features = self.features[:n_data]
            self.labels = self.labels[:n_data]

        self._modified = (n_data is not None) and not (use_hyperprior)
        self.use_hyperprior = use_hyperprior

        if use_hyperprior:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'tau': ParameterScalar(
                            bound='positive',
                            prior=td.Gamma(
                                concentration=0.5,
                                rate=0.5
                            )
                        ),
                        'beta': ParameterVector(
                            25,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': torch.zeros(25),
                                'scale': (lambda beta_prior_scale, **kwargs: beta_prior_scale)
                            }
                        ),
                        'beta_prior_scale': ParameterScalar(
                            bound='positive',
                            prior=td.Cauchy(
                                loc=0.0,
                                scale=5.0
                            )
                        )
                    },
                    [
                        ('beta_prior_scale', 'beta')
                    ]
                )
            )
        else:
            super().__init__(
                posterior_parameters=ParameterDAG({
                    'tau': ParameterScalar(
                        bound='positive',
                        prior=td.Gamma(
                            concentration=0.5,
                            rate=0.5
                        )
                    ),
                    'beta': ParameterVector(
                        25,
                        prior=td.Normal(0.0, 1.0)
                    ),
                })
            )

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        batch_shape = extracted['beta'].shape[:-1]
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            extracted['tau'].view(*batch_shape, 1) * extracted['beta']
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        return td.Independent(td.Bernoulli(logits=logits), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.labels)

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'german_credit_moments.pt'
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
            f'german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


class SparseGermanCredit(Posterior1D):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    lambda[i] ~ Gamma(0.5, 0.5)
    """

    def __init__(self,
                 n_data: int = None,
                 use_hyperprior: bool = False):
        self.features, self.labels = load_german_credit()

        if n_data is not None:
            if not 0 < n_data <= len(self.labels):
                raise ValueError(
                    "Number of used observations must be between zero and the number of total observations"
                )
            self.features = self.features[:n_data]
            self.labels = self.labels[:n_data]
        self._modified = (n_data is not None) and (not use_hyperprior)

        self.use_hyperprior = use_hyperprior

        if self.use_hyperprior:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'tau': ParameterScalar(
                            bound='positive',
                            prior=td.Gamma(
                                concentration=0.5,
                                rate=0.5
                            )
                        ),
                        'beta': ParameterVector(
                            25,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': torch.zeros(25),
                                'scale': (lambda beta_prior_scale, **kwargs: beta_prior_scale)
                            }
                        ),
                        'lambda': ParameterVector(
                            25,
                            bound='positive',
                            prior=td.Gamma(0.5, 0.5)
                        ),
                        'beta_prior_scale': ParameterScalar(
                            bound='positive',
                            prior=td.Cauchy(
                                loc=0.0,
                                scale=5.0
                            )
                        )
                    },
                    [
                        ('beta_prior_scale', 'beta')
                    ]
                )
            )
        else:
            super().__init__(
                posterior_parameters=ParameterDAG({
                    'tau': ParameterScalar(
                        bound='positive',
                        prior=td.Gamma(
                            concentration=0.5,
                            rate=0.5
                        )
                    ),
                    'beta': ParameterVector(
                        25,
                        prior=td.Normal(
                            loc=0.0,
                            scale=1.0
                        )
                    ),
                    'lambda': ParameterVector(
                        25,
                        bound='positive',
                        prior=td.Gamma(0.5, 0.5)
                    ),
                })
            )

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        batch_shape = extracted['beta'].shape[:-1]
        logits = torch.einsum(
            'nf,...f->...nf',
            self.features,
            (
                extracted['tau'].view(*batch_shape, 1)
                * extracted['beta']
                * extracted['lambda']
            )
        ).sum(dim=-1)  # shape = (*batch_shape, features)
        return td.Independent(td.Bernoulli(logits=logits), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.labels)

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'sparse_german_credit_moments.pt'
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
            f'sparse_german_credit_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


class SparseGermanCreditMissingData(Posterior1D):
    """
    tau ~ Gamma(0.5, 0.5)
    beta[i] ~ N(0, 1)
    lambda[i] ~ Gamma(0.5, 0.5)

    For each missing entry in columns 1, 3, 9 of the first 100 rows:
    x_ij ~ Normal(mu_ij, 1),  mu_ij ~ Normal(0, 1)

    Total parameters: 1 + 25 + 25 + 300 = 351
    """

    def __init__(self, n_missing_rows=100):
        raise NotImplementedError
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
