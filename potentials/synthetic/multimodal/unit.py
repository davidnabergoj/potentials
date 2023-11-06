import torch

from potentials.synthetic.mixture import Mixture
from potentials.synthetic.multimodal.util import generate_mode_locations
from potentials.synthetic.gaussian.diagonal import DiagonalGaussian


class MultimodalStandardGaussian(Mixture):
    """
    Uniform weights.

    """

    def __init__(self, n_dim, n_components: int = 10, seed: int = 0):
        potentials = [DiagonalGaussian(mu=torch.zeros(n_dim), sigma=torch.ones(n_dim)) for _ in range(n_components)]
        with torch.no_grad():
            modes = generate_mode_locations(n_components, n_dim, seed=seed)
            for u, m in zip(potentials, modes):
                u.mu.data = m
        weights = torch.ones(size=(n_components,)) / n_components
        super().__init__(potentials, weights)
