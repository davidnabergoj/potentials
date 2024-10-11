import torch

from potentials.base import Potential


class GaussianProcess(Potential):
    """
    TODO: implement.
    """

    def __init__(self, mu: callable, cov: torch.Tensor):
        pass


class BrownianBridge(GaussianProcess):
    """
    Brownian bridge potential.

    Reference: Brofos et al. "Adaptation of the Independent Metropolis-Hastings Sampler with Normalizing Flow Proposals"
     (2022). url: https://proceedings.mlr.press/v151/brofos22a.html.
    """
    def __init__(self, n_points: int = 50, eps: float = 1e-2):
        t = torch.linspace(eps, 1 - eps, n_points)
        s = torch.clone(t)
        super().__init__(
            mu=torch.sin(torch.pi * t),
            cov=torch.minimum(t[:, None], s[None]) - s[None] * t[:, None]
        )
