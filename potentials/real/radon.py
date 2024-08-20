from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import json

from potentials.base import StructuredPotential
import torch
import torch.distributions as td

from potentials.transformations import bound_parameter


# https://www.tensorflow.org/probability/examples/Multilevel_Modeling_Primer

def load_radon(n_counties: int):
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
                    return floor[mask], log_radon[mask], log_uranium[mask], county_idx[mask]


class RadonVaryingSlopes(StructuredPotential):
    def __init__(self, n_counties: int = 85):
        super().__init__(event_shape=(4 + n_counties,))
        self.floor, self.log_radon, self.log_uranium, self.county_idx = load_radon(n_counties=n_counties)
        self.n_counties = n_counties

    def compute(self, model_params):
        # Extract parameters
        # (mu_a, log_sigma_a, log_sigma_y, a, b)
        batch_shape = model_params.shape[:-1]

        mu_a = model_params[..., 0]
        log_sigma_a = model_params[..., 1]
        log_sigma_y = model_params[..., 2]
        a = model_params[..., 3:3 + self.n_counties]
        b = model_params[..., 3 + self.n_counties]

        # Transform log scales to scales
        sigma_a, log_det_sigma_a = bound_parameter(log_sigma_a, batch_shape, low=0.0, high=torch.inf)
        sigma_y, log_det_sigma_y = bound_parameter(log_sigma_y, batch_shape, low=0.0, high=torch.inf)
        log_det = log_det_sigma_a + log_det_sigma_y

        # Compute probabilities
        log_prob_mu_a = td.Normal(loc=0, scale=1e5).log_prob(mu_a)
        log_prob_b = td.Normal(loc=0, scale=1e5).log_prob(b)
        log_prob_sigma_a = td.HalfCauchy(scale=5).log_prob(sigma_a)
        log_prob_sigma_y = td.HalfCauchy(scale=5).log_prob(sigma_y)
        log_prob_a = td.Independent(td.Normal(mu_a[:, None], sigma_a[:, None]), 1).log_prob(a)
        log_prob_y = td.Independent(
            td.Normal(
                a[:, self.county_idx - 1] * self.floor[None] + b[:, None].repeat(1, len(self.county_idx)),
                sigma_y[:, None].repeat(1, len(self.county_idx))
            ),
            1
        ).log_prob(self.log_radon)

        log_prob = (
                log_prob_mu_a
                + log_prob_sigma_a
                + log_prob_sigma_y
                + log_prob_a
                + log_prob_b
                + log_prob_y
        )

        return -(log_prob + log_det)

    @property
    def edge_list(self):
        # (mu_a, log_sigma_a, log_sigma_y, a, b)
        return [(0, i) for i in range(3, 3 + self.n_counties)] + [(1, i) for i in range(3, 3 + self.n_counties)]


class RadonVaryingIntercepts(StructuredPotential):
    def __init__(self, n_counties: int = 85):
        super().__init__(event_shape=(4 + n_counties,))
        self.floor, self.log_radon, self.log_uranium, self.county_idx = load_radon(n_counties=n_counties)
        self.n_counties = n_counties

    def compute(self, model_params):
        # Extract parameters
        # (mu_b, log_sigma_b, log_sigma_y, a, b)
        batch_shape = model_params.shape[:-1]

        mu_b = model_params[..., 0]
        log_sigma_b = model_params[..., 1]
        log_sigma_y = model_params[..., 2]
        a = model_params[..., 3]
        b = model_params[..., 4:4 + self.n_counties]

        # Transform log scales to scales
        sigma_b, log_det_sigma_b = bound_parameter(log_sigma_b, batch_shape, low=0.0, high=torch.inf)
        sigma_y, log_det_sigma_y = bound_parameter(log_sigma_y, batch_shape, low=0.0, high=torch.inf)
        log_det = log_det_sigma_b + log_det_sigma_y

        # Compute probabilities
        log_prob_mu_b = td.Normal(loc=0, scale=1e5).log_prob(mu_b)
        log_prob_a = td.Normal(loc=0, scale=1e5).log_prob(a)
        log_prob_sigma_b = td.HalfCauchy(scale=5).log_prob(sigma_b)
        log_prob_sigma_y = td.HalfCauchy(scale=5).log_prob(sigma_y)
        log_prob_b = td.Independent(td.Normal(mu_b[:, None], sigma_b[:, None]), 1).log_prob(b)
        log_prob_y = td.Independent(
            td.Normal(
                a[:, None].repeat(1, len(self.county_idx)) * self.floor[None] + b[:, self.county_idx - 1],
                sigma_y[:, None].repeat(1, len(self.county_idx))
            ),
            1
        ).log_prob(self.log_radon)

        log_prob = (
                log_prob_mu_b
                + log_prob_sigma_b
                + log_prob_sigma_y
                + log_prob_a
                + log_prob_b
                + log_prob_y
        )

        return -(log_prob + log_det)

    @property
    def edge_list(self):
        # (mu_b, log_sigma_b, log_sigma_y, a, b)
        return [(0, i) for i in range(4, 4 + self.n_counties)] + [(1, i) for i in range(4, 4 + self.n_counties)]


class RadonVaryingInterceptsAndSlopes(StructuredPotential):
    def __init__(self, n_counties: int = 85):
        super().__init__(event_shape=(5 + 2 * n_counties,))
        self.floor, self.log_radon, self.log_uranium, self.county_idx = load_radon(n_counties=n_counties)
        self.n_counties = n_counties

    def compute(self, model_params):
        # Extract parameters
        # (mu_a, log_sigma_a, mu_b, log_sigma_b, log_sigma_y, a, b)
        batch_shape = model_params.shape[:-1]

        mu_a = model_params[..., 0]
        log_sigma_a = model_params[..., 1]
        mu_b = model_params[..., 2]
        log_sigma_b = model_params[..., 3]
        log_sigma_y = model_params[..., 4]
        a = model_params[..., 5:5 + self.n_counties]
        b = model_params[..., 5 + self.n_counties:5 + 2 * self.n_counties]

        # Transform log scales to scales
        sigma_a, log_det_sigma_a = bound_parameter(log_sigma_a, batch_shape, low=0.0, high=torch.inf)
        sigma_b, log_det_sigma_b = bound_parameter(log_sigma_b, batch_shape, low=0.0, high=torch.inf)
        sigma_y, log_det_sigma_y = bound_parameter(log_sigma_y, batch_shape, low=0.0, high=torch.inf)
        log_det = log_det_sigma_a + log_det_sigma_b + log_det_sigma_y

        # Compute probabilities
        log_prob_mu_a = td.Normal(loc=0, scale=1e5).log_prob(mu_a)
        log_prob_sigma_a = td.HalfCauchy(scale=5).log_prob(sigma_a)
        log_prob_mu_b = td.Normal(loc=0, scale=1e5).log_prob(mu_b)
        log_prob_sigma_b = td.HalfCauchy(scale=5).log_prob(sigma_b)
        log_prob_sigma_y = td.HalfCauchy(scale=5).log_prob(sigma_y)
        log_prob_a = td.Independent(td.Normal(mu_a[:, None], sigma_a[:, None]), 1).log_prob(a)
        log_prob_b = td.Independent(td.Normal(mu_b[:, None], sigma_b[:, None]), 1).log_prob(b)
        log_prob_y = td.Independent(
            td.Normal(
                a[:, self.county_idx - 1] * self.floor[None] + b[:, self.county_idx - 1],
                sigma_y[:, None].repeat(1, len(self.county_idx))
            ),
            1
        ).log_prob(self.log_radon)

        log_prob = (
                log_prob_mu_a
                + log_prob_sigma_a
                + log_prob_mu_b
                + log_prob_sigma_b
                + log_prob_sigma_y
                + log_prob_a
                + log_prob_b
                + log_prob_y
        )

        return -(log_prob + log_det)

    @property
    def edge_list(self):
        # (mu_a, log_sigma_a, mu_b, log_sigma_b, log_sigma_y, a, b)
        return (
                [(0, i) for i in range(5, 5 + self.n_counties)]
                + [(1, i) for i in range(5, 5 + self.n_counties)]
                + [(2, i) for i in range(5 + self.n_counties, 5 + 2 * self.n_counties)]
                + [(3, i) for i in range(5 + self.n_counties, 5 + 2 * self.n_counties)]
        )


if __name__ == '__main__':
    for target in [RadonVaryingSlopes(), RadonVaryingIntercepts(), RadonVaryingInterceptsAndSlopes()]:
        torch.manual_seed(0)
        x = torch.randn(size=(5, *target.event_shape))
        print(target(x))
