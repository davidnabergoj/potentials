from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve
import zipfile
import json

from potentials.real.posterior.base import Posterior1D
import torch
import torch.distributions as td

from potentials.real.posterior.parameter import ParameterDAG, ParameterScalar, ParameterVector
from potentials.utils import LogNormalMixture


# https://www.tensorflow.org/probability/examples/Multilevel_Modeling_Primer

def load_radon(n_counties: int, n_data: int = None):
    # Using Minnesota data
    data_path = Path(__file__).parent / "downloaded/radon_mn.json.zip"
    if not data_path.exists():
        download_url = "https://github.com/stan-dev/posteriordb/raw/master/posterior_database/data/data/radon_mn.json.zip"
        print(f"Downloading Radon MN dataset from {download_url}")
        urlretrieve(download_url, data_path)

    with zipfile.ZipFile(data_path, 'r') as archive:
        file_list = archive.namelist()
        for file_name in file_list:
            if file_name.endswith('.json'):
                with archive.open(file_name) as json_file:
                    json_data = json_file.read().decode('utf-8')
                    data = json.loads(json_data)
                    floor = data["floor_measure"]
                    log_radon = data["log_radon"]
                    log_uppm = data["log_uppm"]
                    county_idx = data["county_idx"]

                    floor = torch.as_tensor(floor)
                    log_radon = torch.as_tensor(log_radon)
                    log_uranium = torch.as_tensor(log_uppm)
                    county_idx = torch.as_tensor(county_idx)

                    mask = county_idx <= n_counties

                    floor = floor[mask]
                    log_radon = log_radon[mask]
                    log_uranium = log_uranium[mask]
                    county_idx = county_idx[mask]

                    # Handle smaller data
                    if n_data is not None:
                        if n_data > len(floor):
                            raise ValueError(
                                "New dataset cannot have more entries than the original"
                            )
                        floor = floor[:n_data]
                        log_radon = log_radon[:n_data]
                        log_uranium = log_uranium[:n_data]
                        county_idx = county_idx[:n_data]

                    return floor, log_radon, log_uranium, county_idx


class RadonVaryingSlopes(Posterior1D):
    def __init__(self, 
                 n_data: int = None,
                 use_multimodal_hyperprior: bool = False):
        n_counties = 85
        (
            self.floor,
            self.log_radon,
            self.log_uranium,
            self.county_idx
        ) = load_radon(
            n_counties=n_counties,
            n_data=n_data
        )
        self.n_counties = n_counties
        
        self._modified = (n_data is not None) and (not use_multimodal_hyperprior)
        self.use_multimodal_hyperprior = use_multimodal_hyperprior

        if self.use_multimodal_hyperprior:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_a': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'a': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_a',
                                'scale': 'sigma_a'
                            }
                        ),
                        'b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                    },
                    [
                        ('mu_a', 'a'),
                        ('sigma_a', 'a'),
                    ]
                )
            )
        else:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_a': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'a': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_a',
                                'scale': 'sigma_a'
                            }
                        ),
                        'b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                    },
                    [
                        ('mu_a', 'a'),
                        ('sigma_a', 'a'),
                    ]
                )
            )

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        batch_shape = extracted['a'].shape[:-1]
        mu = extracted['b'].repeat(
            *([1] * len(batch_shape)),
            len(self.county_idx)
        ) + extracted['a'][..., self.county_idx - 1] * self.floor[None]
        sigma = extracted['sigma_y'].repeat(
            *([1] * len(batch_shape)),
            len(self.county_idx)
        )
        return td.Independent(td.Normal(mu, sigma), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.log_radon)

    @property
    def edge_list(self):
        # (mu_a, log_sigma_a, log_sigma_y, a, b)
        return [(0, i) for i in range(3, 3 + self.n_counties)] + [(1, i) for i in range(3, 3 + self.n_counties)]

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / \
            'true_moments' / f'radon_slopes_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[0]
        else:
            raise ValueError("Moment file not found")

    @property
    def second_moment(self):
        if self._modified:
            raise ValueError(
                "Reference second moment unavailable for modified dataset")
        path = Path(__file__).parent.parent / \
            'true_moments' / f'radon_slopes_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


class RadonVaryingIntercepts(Posterior1D):
    def __init__(self, 
                 n_data: int = None,
                 use_multimodal_hyperprior: bool = False):
        n_counties = 85
        (
            self.floor,
            self.log_radon,
            self.log_uranium,
            self.county_idx
        ) = load_radon(
            n_counties=n_counties,
            n_data=n_data
        )
        self.n_counties = n_counties
        
        self._modified = (n_data is not None) and (not use_multimodal_hyperprior)
        self.use_multimodal_hyperprior = use_multimodal_hyperprior

        if self.use_multimodal_hyperprior:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_b': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'b': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_b',
                                'scale': 'sigma_b'
                            }
                        ),
                    },
                    [
                        ('mu_b', 'b'),
                        ('sigma_b', 'b'),
                    ]
                ))
        else:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_b': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'b': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_b',
                                'scale': 'sigma_b'
                            }
                        ),
                    },
                    [
                        ('mu_b', 'b'),
                        ('sigma_b', 'b'),
                    ]
                ))

    def likelihood_object(self, extracted):
        batch_shape = extracted['b'].shape[:-1]
        mu = extracted['a'].expand(
            *batch_shape,
            len(self.county_idx)
        ) * self.floor[None] + extracted['b'][..., self.county_idx - 1]
        sigma = extracted['sigma_y'].expand(
            *batch_shape,
            len(self.county_idx)
        )
        return td.Independent(td.Normal(mu, sigma), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.log_radon)

    @property
    def edge_list(self):
        # (mu_b, log_sigma_b, log_sigma_y, a, b)
        return [(0, i) for i in range(4, 4 + self.n_counties)] + [(1, i) for i in range(4, 4 + self.n_counties)]

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'radon_intercepts_moments.pt'
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
            f'radon_intercepts_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


class RadonVaryingInterceptsAndSlopes(Posterior1D):
    def __init__(self, 
                 n_data: int = None,
                 use_multimodal_hyperprior: bool = False):
        n_counties = 85
        (
            self.floor,
            self.log_radon,
            self.log_uranium,
            self.county_idx
        ) = load_radon(
            n_counties=n_counties,
            n_data=n_data
        )
        self.n_counties = n_counties

        self._modified = (n_data is not None) and (not use_multimodal_hyperprior)
        self.use_multimodal_hyperprior = use_multimodal_hyperprior
        
        if self.use_multimodal_hyperprior:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_a': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'mu_b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_b': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=LogNormalMixture(
                                loc0=0.0,
                                scale0=1.0,
                                loc1=2,
                                scale1=1/10,
                                weight0=0.5,
                                weight1=0.5
                            )
                        ),
                        'a': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_a',
                                'scale': 'sigma_a'
                            }
                        ),
                        'b': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_b',
                                'scale': 'sigma_b'
                            }
                        ),
                    },
                    [
                        ('mu_a', 'a'),
                        ('sigma_a', 'a'),
                        ('mu_b', 'b'),
                        ('sigma_b', 'b'),
                    ]
                )
            )
        else:
            super().__init__(
                posterior_parameters=ParameterDAG(
                    {
                        'mu_a': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_a': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'mu_b': ParameterScalar(
                            prior=td.Normal(
                                loc=0,
                                scale=1e5
                            )
                        ),
                        'sigma_b': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'sigma_y': ParameterScalar(
                            bound='positive',
                            prior=td.LogNormal(
                                loc=0,
                                scale=1.0
                            )
                        ),
                        'a': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_a',
                                'scale': 'sigma_a'
                            }
                        ),
                        'b': ParameterVector(
                            self.n_counties,
                            prior=td.Normal,
                            prior_kwargs={
                                'loc': 'mu_b',
                                'scale': 'sigma_b'
                            }
                        ),
                    },
                    [
                        ('mu_a', 'a'),
                        ('sigma_a', 'a'),
                        ('mu_b', 'b'),
                        ('sigma_b', 'b'),
                    ]
                )
            )

    def likelihood_object(self, extracted):
        batch_shape = extracted['a'].shape[:-1]
        mu = (
            extracted['a'][..., self.county_idx - 1] * self.floor[None]
            + extracted['b'][..., self.county_idx - 1]
        )

        sigma = extracted['sigma_y'].repeat(
            *([1] * len(batch_shape)),
            len(self.county_idx)
        )
        return td.Independent(td.Normal(mu, sigma), 1)

    def compute_likelihood(self, extracted):
        return self.likelihood_object(extracted).log_prob(self.log_radon)

    @property
    def edge_list(self):
        # (mu_a, log_sigma_a, mu_b, log_sigma_b, log_sigma_y, a, b)
        return (
            [(0, i) for i in range(5, 5 + self.n_counties)]
            + [(1, i) for i in range(5, 5 + self.n_counties)]
            + [(2, i)
                for i in range(5 + self.n_counties, 5 + 2 * self.n_counties)]
            + [(3, i)
                for i in range(5 + self.n_counties, 5 + 2 * self.n_counties)]
        )

    @property
    def mean(self):
        if self._modified:
            raise ValueError("Reference mean unavailable for modified dataset")
        path = Path(__file__).parent.parent / 'true_moments' / \
            f'radon_intercepts_slopes_moments.pt'
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
            f'radon_intercepts_slopes_moments.pt'
        if path.exists():
            return torch.load(path, weights_only=True)[1]
        else:
            raise ValueError("Moment file not found")

    @property
    def variance(self):
        return self.second_moment - self.mean ** 2


if __name__ == '__main__':
    _n_data = 250
    for target in [
        RadonVaryingSlopes(n_data=_n_data),
        RadonVaryingIntercepts(n_data=_n_data),
        RadonVaryingInterceptsAndSlopes(n_data=_n_data)
    ]:
        print(f'{len(torch.unique(target.county_idx))=}')

        # print(target.mean.shape)
        # print(target.second_moment.shape)
        # print(target.mean.isfinite().all())
        # print(target.second_moment.isfinite().all())

        torch.manual_seed(0)
        x = torch.randn(size=(5, *target.event_shape))
        print(target(x))
