from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve
import zipfile
import json

from potentials.base import Posterior
import torch
import torch.distributions as td

from potentials.transformations import bound_parameter


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


class RadonVaryingSlopes(Posterior):
    def __init__(self, n_data: int = None):
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
        self._modified = n_data is not None
        super().__init__(event_shape=(4 + n_counties,))

    def extract_parameters(self,
                           unconstrained: torch.Tensor,
                           return_log_probs: bool = True) -> Dict[str, torch.Tensor]:
        # (mu_a, log_sigma_a, log_sigma_y, a, b)
        batch_shape = unconstrained.shape[:-1]

        out = dict()

        out['mu_a'] = unconstrained[..., 0]
        out['log_sigma_a'] = unconstrained[..., 1]
        out['log_sigma_y'] = unconstrained[..., 2]
        out['a'] = unconstrained[..., 3:3 + self.n_counties]
        out['b'] = unconstrained[..., 3 + self.n_counties]

        # Transform log scales to scales
        out['sigma_a'], out['log_det_sigma_a'] = bound_parameter(
            out['log_sigma_a'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['sigma_y'], out['log_det_sigma_y'] = bound_parameter(
            out['log_sigma_y'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['log_det'] = out['log_det_sigma_a'] + out['log_det_sigma_y']

        # Compute prior probabilities
        if return_log_probs:
            out['log_prob_mu_a'] = td.Normal(
                loc=0, scale=1e5).log_prob(out['mu_a'])
            out['log_prob_b'] = td.Normal(loc=0, scale=1e5).log_prob(out['b'])
            out['log_prob_sigma_a'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_a'])
            out['log_prob_sigma_y'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_y'])
            out['log_prob_a'] = td.Independent(
                td.Normal(
                    out['mu_a'][..., None],
                    out['sigma_a'][..., None]),
                1
            ).log_prob(out['a'])

            out['log_prior'] = (
                out['log_prob_mu_a']
                + out['log_prob_b']
                + out['log_prob_sigma_a']
                + out['log_prob_sigma_y']
                + out['log_prob_a']
                + out['log_det']
            )

        return out

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]):
        batch_shape = extracted['a'].shape[:-1]
        mu = extracted['b'][..., None].repeat(
            *([1] * len(batch_shape)),
            len(self.county_idx)
        ) + extracted['a'][..., self.county_idx - 1] * self.floor[None]
        sigma = extracted['sigma_y'][..., None].repeat(
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


class RadonVaryingIntercepts(Posterior):
    def __init__(self, n_data: int = None):
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
        self._modified = n_data is not None
        super().__init__(event_shape=(4 + n_counties,))

    def extract_parameters(self,
                           unconstrained: torch.Tensor,
                           return_log_probs: bool = True) -> Dict[str, torch.Tensor]:
        # (mu_b, log_sigma_b, log_sigma_y, a, b)
        batch_shape = unconstrained.shape[:-1]

        out = dict()

        out['mu_b'] = unconstrained[..., 0]
        out['log_sigma_b'] = unconstrained[..., 1]
        out['log_sigma_y'] = unconstrained[..., 2]
        out['a'] = unconstrained[..., 3]
        out['b'] = unconstrained[..., 4:4 + self.n_counties]

        # Transform log scales to scales
        out['sigma_b'], out['log_det_sigma_b'] = bound_parameter(
            out['log_sigma_b'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['sigma_y'], out['log_det_sigma_y'] = bound_parameter(
            out['log_sigma_y'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['log_det'] = out['log_det_sigma_b'] + out['log_det_sigma_y']

        # Compute log prior
        if return_log_probs:
            out['log_prob_mu_b'] = td.Normal(
                loc=0, scale=1e5).log_prob(out['mu_b'])
            out['log_prob_a'] = td.Normal(loc=0, scale=1e5).log_prob(out['a'])
            out['log_prob_sigma_b'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_b'])
            out['log_prob_sigma_y'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_y'])

            out['log_prior'] = (
                out['log_prob_mu_b']
                + out['log_prob_sigma_b']
                + out['log_prob_sigma_y']
                + out['log_prob_a']
                + out['log_prob_sigma_y']
                + out['log_det']
            )

        return out

    def likelihood_object(self, extracted):
        batch_shape = extracted['b'].shape[:-1]
        mu = extracted['a'][..., None].expand(
            *batch_shape,
            len(self.county_idx)
        ) * self.floor[None] + extracted['b'][..., self.county_idx - 1]
        sigma = extracted['sigma_y'][..., None].expand(
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


class RadonVaryingInterceptsAndSlopes(Posterior):
    def __init__(self, n_data: int = None):
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
        self._modified = n_data is not None
        super().__init__(event_shape=(5 + 2 * n_counties,))

    def extract_parameters(self, unconstrained, return_log_probs=True):
        # (mu_a, log_sigma_a, mu_b, log_sigma_b, log_sigma_y, a, b)
        batch_shape = unconstrained.shape[:-1]

        out = dict()

        out['mu_a'] = unconstrained[..., 0]
        out['log_sigma_a'] = unconstrained[..., 1]
        out['mu_b'] = unconstrained[..., 2]
        out['log_sigma_b'] = unconstrained[..., 3]
        out['log_sigma_y'] = unconstrained[..., 4]
        out['a'] = unconstrained[..., 5:5 + self.n_counties]
        out['b'] = unconstrained[..., 5 + self.n_counties:5 + 2 * self.n_counties]

        # Transform log scales to scales
        out['sigma_a'], out['log_det_sigma_a'] = bound_parameter(
            out['log_sigma_a'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['sigma_b'], out['log_det_sigma_b'] = bound_parameter(
            out['log_sigma_b'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['sigma_y'], out['log_det_sigma_y'] = bound_parameter(
            out['log_sigma_y'],
            batch_shape,
            low=0.0,
            high=torch.inf
        )
        out['log_det'] = (
            out['log_det_sigma_a']
            + out['log_det_sigma_b']
            + out['log_det_sigma_y']
        )

        # Compute log prior
        if return_log_probs:
            out['log_prob_mu_a'] = td.Normal(
                loc=0, scale=1e5).log_prob(out['mu_a'])
            out['log_prob_sigma_a'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_a'])
            out['log_prob_mu_b'] = td.Normal(
                loc=0, scale=1e5).log_prob(out['mu_b'])
            out['log_prob_sigma_b'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_b'])
            out['log_prob_sigma_y'] = td.HalfCauchy(
                scale=5).log_prob(out['sigma_y'])
            out['log_prob_a'] = td.Independent(
                td.Normal(
                    out['mu_a'][..., None],
                    out['sigma_a'][..., None]
                ),
                1
            ).log_prob(out['a'])
            out['log_prob_b'] = td.Independent(
                td.Normal(
                    out['mu_b'][..., None],
                    out['sigma_b'][..., None]
                ),
                1
            ).log_prob(out['b'])

            out['log_prior'] = (
                out['log_prob_mu_a']
                + out['log_prob_sigma_a']
                + out['log_prob_mu_b']
                + out['log_prob_sigma_b']
                + out['log_prob_sigma_y']
                + out['log_prob_a']
                + out['log_prob_b']
                + out['log_det']
            )

        return out

    def likelihood_object(self, extracted):
        batch_shape = extracted['a'].shape[:-1]
        mu = (
            extracted['a'][..., self.county_idx - 1] * self.floor[None]
            + extracted['b'][..., self.county_idx - 1]
        )

        sigma = extracted['sigma_y'][..., None].repeat(
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
