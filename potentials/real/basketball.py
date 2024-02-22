import torch

from potentials.base import Potential


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


class BasketballV1(Potential):
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


class BasketballV2(Potential):
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


if __name__ == '__main__':
    # print(load_basketball())
    torch.manual_seed(0)
    target = BasketballV2('data/basketball.json')
    target(torch.randn(size=(10, *target.event_shape)))
