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

    @property
    def mean(self):
        return torch.tensor([
            0.0241, 0.7076, 0.5255, 0.0853, 0.1900, 0.0990, 0.2909, 0.0690,
            0.2946, -0.0117, -0.0879, -0.0107, -0.2639, -0.0284, 0.0206, -0.0412,
            0.1952, -0.0448, -0.0867, -0.1524, -0.2745, -0.0814, 0.2950, 0.0287,
            0.1068, -0.1256, -0.0628, 0.0886, 0.2704, -0.0358, -0.0150, -0.0225,
            0.0435, -0.0284, -0.2268, -0.1070, -0.3231, 0.1632, 0.0391, 0.0600,
            0.1905, 0.3223, -0.0020, 0.1073, -0.1669, -0.3409, -0.0706, 0.0709,
            0.0658, -0.0670, 0.1638, 0.1009, 0.0442, -0.1922, -0.3143, 0.1048,
            0.2213, -0.2651, 0.0169, -0.0608, 0.1142, -0.0728, -0.0719, -0.2311,
            -0.0325, 0.1282, 0.1208, 0.1404, -0.0537, 0.0811, 0.1148, 0.2251,
            0.0057, -0.1732, -0.0661, 0.0239, 0.1456, 0.1303, -0.2662, -0.1424,
            -0.2948, 0.0659, 0.1833, 0.1976, 0.1554, -0.1198, 0.0414, -0.1851,
            0.5584
        ])

    @property
    def second_moment(self):
        return torch.tensor([
            0.8900, 1.7987, 1.1321, 0.9064, 1.0029, 1.4238, 0.9808, 1.4334, 1.1048,
            0.7175, 1.2014, 0.6894, 0.9244, 1.1787, 0.7037, 1.0067, 0.8190, 1.0213,
            1.0011, 1.7850, 0.9562, 0.6848, 1.0255, 0.8433, 1.0209, 0.7751, 0.9022,
            0.9249, 0.7369, 1.0995, 0.7690, 1.3001, 0.6727, 1.0682, 0.8357, 0.5824,
            1.0103, 1.2522, 1.0539, 1.0626, 0.8970, 1.4370, 0.9332, 1.0862, 0.9926,
            1.2699, 0.7470, 0.8950, 1.0115, 1.6899, 1.1140, 0.9274, 1.1596, 1.3328,
            1.0497, 0.8318, 1.2238, 0.8213, 0.8584, 1.1010, 0.9325, 1.1017, 1.2297,
            1.0001, 0.7763, 0.8777, 1.1133, 0.9667, 0.9552, 0.7659, 1.0716, 1.2488,
            1.1172, 1.2292, 0.9069, 1.0884, 1.0382, 0.7648, 1.1863, 1.0100, 1.0720,
            1.1422, 1.1453, 0.9269, 1.0586, 0.7017, 1.2390, 0.8300, 0.8291
        ])


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

    @property
    def mean(self):
        return torch.tensor([
            -3.8755e-03, 2.2379e+00, 6.6476e-01, 1.3949e-01, 1.9831e-01,
            1.7487e-01, 7.1439e-02, -4.3277e-02, 2.2047e-01, 7.3308e-02,
            -3.3929e-02, -7.7369e-02, -1.7558e-01, -1.4463e-01, 1.0245e-01,
            -6.9326e-02, 1.0833e-01, -1.5384e-01, 8.6675e-02, 5.0870e-02,
            -1.3332e-01, -1.6163e-01, 4.1535e-01, 8.1856e-02, -6.4773e-02,
            -1.8058e-01, 9.4885e-02, -4.9711e-02, 2.1792e-01, 2.2982e-01,
            -1.4856e-01, -1.0015e-01, 1.0592e-01, 1.4629e-01, -1.5128e-01,
            1.2062e-02, -2.7484e-01, -4.2599e-02, 1.0802e-02, -7.7924e-02,
            1.2089e-02, 6.2559e-02, -1.5213e-01, 2.5904e-01, -1.9984e-01,
            -1.3630e-01, -1.3199e-02, 7.6267e-02, -7.1080e-02, 1.3201e-01,
            1.7007e-01, -1.8484e-01, -2.2891e-02, -4.3943e-02, -1.2225e-01,
            -8.2760e-03, 2.5852e-01, -8.7325e-02, 7.9065e-02, 1.1337e-01,
            3.5845e-03, -1.0196e-01, 6.5992e-02, -9.8213e-02, -1.7608e-03,
            8.9749e-02, 4.2397e-03, 2.6120e-01, -2.6241e-02, 1.0157e-01,
            2.1592e-01, 3.2129e-01, 7.4226e-02, 4.5357e-04, -2.3393e-02,
            -1.6066e-01, 9.7864e-02, 2.3591e-01, -1.9216e-01, -1.2475e-01,
            -1.6243e-01, 1.1445e-01, 1.1057e-01, 2.9299e-01, 9.7036e-02,
            -1.5331e-01, 1.1169e-01, -4.2060e-02, -1.2196e-01
        ])

    @property
    def second_moment(self):
        return torch.tensor([
            0.8878, 10.7000, 0.6184, 0.6928, 1.0159, 0.9530, 0.9565, 1.2634,
            0.9190, 0.9691, 1.0648, 1.0765, 0.9307, 1.0652, 0.7014, 1.0861,
            0.9580, 1.0709, 0.7358, 1.1494, 1.1590, 0.9787, 0.8486, 0.9577,
            1.1650, 0.7982, 1.1949, 1.1934, 0.8100, 0.8673, 1.0045, 1.1744,
            1.0175, 1.0033, 0.9650, 0.7555, 0.9775, 1.2543, 0.8205, 1.4275,
            0.9372, 1.2655, 0.6934, 1.0302, 1.0883, 1.3375, 0.8157, 0.9765,
            1.1317, 1.5777, 1.0597, 1.0684, 1.0700, 1.5006, 0.8443, 0.7389,
            0.9071, 1.0398, 0.9941, 1.5833, 0.9852, 1.1762, 1.0029, 0.7149,
            0.5930, 0.9057, 1.1354, 1.1316, 0.9475, 1.3735, 1.2303, 1.0025,
            1.2237, 0.7682, 0.8679, 1.0706, 0.8325, 0.9812, 1.3435, 1.1093,
            0.9223, 0.8921, 0.9373, 0.7430, 0.9624, 0.8783, 1.2822, 0.6224,
            0.7281
        ])


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
