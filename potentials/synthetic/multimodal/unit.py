import torch

from potentials.synthetic.mixture import Mixture
from potentials.synthetic.gaussian.unit import StandardGaussian


class MultimodalStandardGaussian(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10):
        potentials = [StandardGaussian(event_shape=(n_dim,)) for _ in range(n_components)]
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
