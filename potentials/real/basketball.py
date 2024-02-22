import torch

from potentials.base import StructuredPotential


def load_basketball(file_path: str = 'data/basketball.json'):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    labels = torch.as_tensor(data['made'], dtype=torch.float)  # 0.0 or 1.0
    k = int(data['k'][0])
    player_ids = torch.as_tensor(data['playerId'], dtype=torch.long)  # Player ids from 1 to n inclusive
    n_features = int(data['n1'][0])
    features = torch.as_tensor(data['other_data'], dtype=torch.float)  # (n_players, n_features); float
    return k, labels, player_ids, n_features, features


class BasketballV1(StructuredPotential):
    def __init__(self, file_path: str):
        self.n_players, self.labels, self.player_ids, _, _ = load_basketball(file_path)
        self.n_shots = len(self.labels)
        # Note: n_shots >= n_players
        event_shape = (2 + self.n_players,)  # (mu, ss, theta)
        super().__init__(event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        mu = x[..., 0]
        log_ss = x[..., 1]
        theta = x[..., 2:2 + self.n_players]

        ss = torch.exp(log_ss)
        log_abs_det_jac = ss

        # Compute the log prior
        log_prior = (
                torch.distributions.Normal(0.0, 2.0).log_prob(mu)
                + torch.distributions.HalfCauchy(2.0).log_prob(ss)
        )
        for i in range(self.n_players):
            log_prior += torch.distributions.MultivariateNormal(mu, torch.diag(ss) ** 2).log_prob(theta[..., i])

        # Compute the log likelihood
        logits = theta[..., self.player_ids - 1]
        log_likelihood = torch.zeros(size=(len(logits),))
        for i in range(len(logits)):
            log_likelihood[i] = torch.distributions.Bernoulli(logits=logits[i]).log_prob(self.labels).sum(dim=-1)

        log_probability = log_likelihood + log_prior + log_abs_det_jac

        return -log_probability

    @property
    def edge_list(self):
        # mu affects all thetas
        # ss affects all thetas
        return [(0, i) for i in range(2, self.n_dim)] + [(1, i) for i in range(2, self.n_dim)]

    @property
    def mean(self):
        # According to a run of NUTS
        return torch.tensor(
            [1.2495984986, -0.6222534480156974, 1.07395023188, 0.815288626856, 0.896156004974, 1.163478279,
             0.7277601938504, 1.2590343546, 1.07019342308, 1.3565279222000002, 1.5638459609999997, 0.980770136142,
             0.8161523384399999, 1.4608191734, 1.9215176996000003, 1.06636837552, 1.5635881904, 1.13633742194,
             0.734928615152, 0.48968892875978, 1.9256296924000003, 1.7965307566, 1.544489386, 0.8095175535597801,
             0.8147206785333999, 1.5618918414, 1.5649859376, 2.0617403236, 1.6734561999999997])

    @property
    def variance(self):
        # According to a run of NUTS
        return torch.tensor(
            [0.017898767082384876, 0.05476022288914617, 0.09037405955591145, 0.09170848655085606, 0.08558416110161172,
             0.0947213843306633, 0.08488982424833821, 0.09838963558478661, 0.09540567042110418, 0.10322535911256828,
             0.12185991419762021, 0.08973796902651417, 0.08215556953843313, 0.10915338499589372, 0.1542455352796065,
             0.09163809024806249, 0.11161997153341922, 0.09600152277885887, 0.08872794255014992, 0.08693560714833444,
             0.15517564288725966, 0.13203249754532762, 0.11446036732479997, 0.08419583532063162, 0.08493421224845302,
             0.1140198373158458, 0.11317454552801293, 0.17383445257840535, 0.12117303689903819])


class BasketballV2(StructuredPotential):
    def __init__(self, file_path: str):
        self.n_players, self.labels, self.player_ids, self.n_features, self.features = load_basketball(file_path)
        self.n_shots = len(self.labels)
        # Note: n_shots >= n_players
        event_shape = (1 + 1 + self.n_players + 1 + self.n_features + 1,)  # (mu, ss, theta, beta0, beta, ss_beta)
        super().__init__(event_shape)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        mu = x[..., 0]
        log_ss = x[..., 1]
        theta = x[..., 2:2 + self.n_players]
        beta0 = x[..., 2 + self.n_players]
        beta = x[..., 2 + self.n_players + 1:-1]
        log_ss_beta = x[..., -1]

        ss = torch.exp(log_ss)
        ss_beta = torch.exp(log_ss_beta)
        log_abs_det_jac = ss + ss_beta

        # Compute the log prior
        log_prior = (
                torch.distributions.Normal(0.0, 2.0).log_prob(mu)
                + torch.distributions.HalfCauchy(2.0).log_prob(ss)
        )
        for i in range(self.n_features):
            log_prior[i] += torch.distributions.MultivariateNormal(
                torch.zeros(len(ss_beta)), torch.diag(ss_beta) ** 2
            ).log_prob(beta[..., i])
        for i in range(self.n_players):
            log_prior += torch.distributions.MultivariateNormal(mu, torch.diag(ss) ** 2).log_prob(theta[..., i])

        # Compute the log likelihood
        logits = (
                theta[..., self.player_ids - 1]
                + beta0[:, None]
                + torch.einsum('pf,sf->ps', beta, self.features[self.player_ids - 1])
        )
        log_likelihood = torch.zeros(size=(len(logits),))
        for i in range(len(logits)):
            log_likelihood[i] = torch.distributions.Bernoulli(logits=logits[i]).log_prob(self.labels).sum(dim=-1)

        log_probability = log_likelihood + log_prior + log_abs_det_jac

        return -log_probability

    @property
    def edge_list(self):
        # mu affects all thetas
        mu_interactions = [(0, i) for i in range(2, 2 + self.n_players)]

        # ss affects all thetas
        ss_interactions = [(1, i) for i in range(2, 2 + self.n_players)]

        # ss_beta affects all betas except beta0
        ss_beta_interactions = [
            (self.n_dim - 1, i) for i in range(3 + self.n_players, 3 + self.n_players + self.n_features)
        ]

        return mu_interactions + ss_interactions + ss_beta_interactions

    @property
    def mean(self):
        # According to a run of NUTS
        return torch.tensor(
            [0.1393299933926, -0.7337739432537588, -0.12067851640940003, -0.2950483882322, -0.11550470518751994,
             0.07499592918199997, -0.16903622981460004, 0.1189912721824, -0.1029545727954, 0.2239809003719999,
             0.4661536288470261, 0.038700914063000004, -0.265375035601, 0.310718456856, 0.6974284852671999,
             -0.15377605275400003, 0.49008326988999995, 0.08278378681800004, -0.2716441031136, -0.47935773885340005,
             0.759278233207, 0.5271422151545999, 0.36298547710659995, -0.05761762099340003, -0.23508896556600004,
             0.39071256242659996, 0.3811185460928, 0.6905074425982001, 0.44917476922000005, 1.11808411713744,
             -0.07045513315673199, 0.06637397732214, -0.14427018662162, 0.12008643251812, 0.0418195141622666,
             -0.045410063154366, -2.148288833355823])

    @property
    def variance(self):
        return torch.tensor(
            [4.349019392690146, 0.07718174329868129, 4.463947930893879, 4.513246952127813, 4.444038006703312,
             4.471614489351748, 4.474231854790743, 4.462539211600237, 4.490932830807347, 4.4762245250985435,
             4.454366193087127, 4.498935107606987, 4.45998832200406, 4.478760557126554, 4.4640282471559525,
             4.452133711237266, 4.448576993178833, 4.42130878936694, 4.48200373314804, 4.465179661927096,
             4.4916216257362676, 4.458242489610124, 4.43277521140374, 4.491835069345854, 4.45109253993953,
             4.460694204593696, 4.457897354830403, 4.527546534246524, 4.444249779391728, 4.369082873465718,
             0.013255120362671513, 0.01594062700222545, 0.02126243290071545, 0.01899001135449954, 0.012512901568641977,
             0.012206381245539539, 0.8814056278733383])


if __name__ == '__main__':
    # print(load_basketball())
    torch.manual_seed(0)
    target = BasketballV2('data/basketball.json')
    target(torch.randn(size=(10, *target.event_shape)))
    print(target.edge_list)
