from typing import List, Tuple, Union
import torch

from potentials.base import Potential
from potentials.utils import get_batch_shape, unsqueeze_to_batch


class Mixture(Potential):
    def __init__(self, potentials: List[Potential], weights: torch.Tensor):
        assert len(potentials) > 0
        assert len(potentials) == len(weights)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        assert torch.all(weights > 0.0)
        super().__init__(potentials[0].event_shape)
        self.potentials = potentials
        self.weights = weights
        self.log_weights = torch.log(self.weights)
        self.n_components = len(self.potentials)
        self.categorical = torch.distributions.Categorical(probs=self.weights)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        potentials = torch.zeros(*batch_shape, self.n_components)
        for i in range(self.n_components):
            potentials[..., i] = self.potentials[i](x)
        log_weights = unsqueeze_to_batch(self.log_weights, batch_shape)
        return -torch.logsumexp(log_weights - potentials, dim=-1)

    def sample(self, batch_shape: Union[torch.Size, Tuple[int]]):
        categories = self.categorical.sample(batch_shape)
        outputs = torch.zeros(*batch_shape, *self.event_shape)
        for i, potential in enumerate(self.potentials):
            outputs[categories == i] = potential.sample(batch_shape)[categories == i]
        return outputs
