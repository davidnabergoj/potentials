from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import json

from potentials.base import StructuredPotential
import torch
import torch.distributions as td


class RadonVaryingInterceptsAndSlopes(StructuredPotential):
    # https://www.tensorflow.org/probability/examples/Multilevel_Modeling_Primer
    # Using Minnesota data
    # TODO add dataset
    def load_data(self):
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

                        return (
                            torch.as_tensor(floor),
                            torch.as_tensor(log_radon),
                            torch.as_tensor(log_uppm),
                            torch.as_tensor(county_idx)
                        )

        return None, None

    def __init__(self):
        # number of parameters: 2 + 85 + 1 + 2 + 85 = 175
        super().__init__(event_shape=(175,))
        self.floor, self.log_radon, self.log_uranium, self.county_idx = self.load_data()

    def compute(self, model_params):
        # Extract parameters
        # (mu_a, log_sigma_a, mu_b, log_sigma_b, log_sigma_y, a, b)

        mu_a = model_params[..., 0]
        log_sigma_a = model_params[..., 1]
        mu_b = model_params[..., 2]
        log_sigma_b = model_params[..., 3]
        log_sigma_y = model_params[..., 4]
        a = model_params[..., 5:90]
        b = model_params[..., 90:175]

        # Transform log scales to scales
        sigma_a = torch.exp(log_sigma_a)
        sigma_b = torch.exp(log_sigma_b)
        sigma_y = torch.exp(log_sigma_y)

        log_det_sigma_a = log_sigma_a
        log_det_sigma_b = log_sigma_b
        log_det_sigma_y = log_sigma_y
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
                [(0, i) for i in range(5, 90)]
                + [(1, i) for i in range(5, 90)]
                + [(2, i) for i in range(90, 175)]
                + [(3, i) for i in range(90, 175)]
        )


if __name__ == '__main__':
    target = RadonVaryingInterceptsAndSlopes()
    torch.manual_seed(0)
    x = torch.randn(size=(5, *target.event_shape))
    print(target(x))
