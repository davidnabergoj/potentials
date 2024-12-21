from typing import List, Tuple, Union
import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape, unsqueeze_to_batch


def validate_weights(weights: torch.Tensor):
    if torch.any(torch.as_tensor(weights < 0.0)):
        raise ValueError(f"Expected weights to be non-negative, but found {weights.min()} in weights.")
    if not torch.isclose(weights.sum(), torch.tensor(1.0)):
        raise ValueError(f"Expected weights to sum close to 1, but got {weights.sum()}")


class Mixture(Potential):
    def __init__(self, potentials: List[Potential], weights: torch.Tensor):
        validate_weights(weights)
        if len(potentials) == 0:
            raise ValueError("Expected at least 1 Potential object, but got 0")
        if len(potentials) != len(weights):
            raise ValueError(
                f"Expected {len(potentials) = } to be equal to {len(weights) = }, but got different values"
            )
        if len(set([p.event_shape for p in potentials])) != 1:
            raise ValueError("Expected all potentials to have the same event shape, but got different values")
        super().__init__(potentials[0].event_shape)
        self.potentials = potentials
        self.weights = weights
        self.n_components = len(self.potentials)
        self.categorical = torch.distributions.Categorical(probs=self.weights)

    @property
    def mean(self):
        means = torch.stack([p.mean for p in self.potentials])
        return torch.sum(self.weights[:, None] * means, dim=0)

    @property
    def variance(self):
        variances = torch.stack([p.variance for p in self.potentials])
        means = torch.stack([p.mean for p in self.potentials])
        return torch.sum(self.weights[:, None] * (variances + means ** 2), dim=0) - self.mean ** 2

    @property
    def log_normalization_constants(self):
        try:
            return torch.log(torch.tensor([p.normalization_constant for p in self.potentials]))
        except NotImplementedError:
            raise ValueError("All potentials must have a specified normalization constant")

    @property
    def log_weights(self):
        return torch.log(self.weights)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        potentials = torch.zeros(*batch_shape, self.n_components).to(x)
        for i in range(self.n_components):
            potentials[..., i] = self.potentials[i](x.to(x))
        log_weights = unsqueeze_to_batch(self.log_weights, batch_shape)
        return -torch.logsumexp(log_weights.to(x) - self.log_normalization_constants.to(x) - potentials, dim=-1)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int, ...]]):
        categories = self.categorical.sample(batch_shape)
        outputs = torch.zeros(*batch_shape, *self.event_shape)
        for i, potential in enumerate(self.potentials):
            outputs[categories == i] = potential.sample(batch_shape)[categories == i]
        return outputs
