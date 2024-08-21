import math

from potentials.base import Potential
import torch

from potentials.utils import unsqueeze_to_batch


class BiochemicalOxygenDemand(Potential):
    def __init__(self):
        event_shape = (2,)
        super().__init__(event_shape)

        self.var_b = 2e-4

        self.theta0_true = 1.0
        self.theta1_true = 0.1
        self.n_observations = 20
        self.t = torch.linspace(1.0, 5.0, steps=self.n_observations)
        self.e = torch.randn_like(self.t) * math.sqrt(self.var_b)
        self.targets = self.theta0_true * (1 - torch.exp(-self.theta1_true * self.t)) + self.e

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        theta0 = x[..., [0]]
        theta1 = x[..., [1]]
        batch_shape = x.shape[:-1]

        t_reshaped = unsqueeze_to_batch(self.t, batch_shape)
        predictions = theta0 * (1 - torch.exp(-theta1 * t_reshaped))
        targets_reshaped = unsqueeze_to_batch(self.targets, batch_shape)

        errors = (predictions - targets_reshaped) ** 2
        assert errors.shape == (*batch_shape, self.n_observations)
        return 2 * math.pi * self.var_b + 0.5 * torch.sum(errors, dim=-1)

    @property
    def mean(self):
        return torch.tensor([2.1695e-01, 9.1293e+01])

    @property
    def second_moment(self):
        return torch.tensor([6.3547e-01, 1.9868e+04])


if __name__ == '__main__':
    u = BiochemicalOxygenDemand()
    print(u.mean.shape)
    print(u.second_moment.shape)
