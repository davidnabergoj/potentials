import torch

from potentials.base import Potential


def load_basketball():
    import json
    with open('data/basketball.json', 'r') as f:
        data = json.load(f)
    labels = torch.as_tensor(data['made'], dtype=torch.float64)
    k = int(data['k'][0])
    player_ids = torch.as_tensor(data['playerId'], dtype=torch.long)
    return k, labels, player_ids


class BasketballBaseline(Potential):
    def __init__(self):
        self.n_players, self.labels, self.player_ids = load_basketball()
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


if __name__ == '__main__':
    # print(load_basketball())
    torch.manual_seed(0)
    target = BasketballBaseline()
    target(torch.randn(size=(10, *target.event_shape)))
