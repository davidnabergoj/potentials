# energy-based models
from typing import Union, Tuple

import torch

from potentials.base import Potential


class LatentSpaceGANPotential(Potential):
    """
    Reference: Samsonov et al. "Local-Global MCMC kernels: the best of both worlds" (2024).
     Arxiv: https://arxiv.org/abs/2111.02702.
    """
    def __init__(self, latent_shape: Union[Tuple[int, ...], torch.Size]):
        super().__init__(event_shape=latent_shape)


class MNIST(LatentSpaceGANPotential):
    def __init__(self, latent_dim: int = 2):
        super().__init__(latent_shape=(latent_dim,))


class CIFAR10(LatentSpaceGANPotential):
    def __init__(self, latent_dim: int = 128):
        super().__init__(latent_shape=(latent_dim,))
