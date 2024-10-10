import torch

from potentials.base import PotentialSimple


class UnpreconditionablePotential1(PotentialSimple):
    """
    Synthetic target distribution for which any non-orthogonal linear preconditioner will cause the condition number to
     increase.

    Reference: Hird and Livingstone: "Quantifying the effectiveness of linear preconditioning in Markov chain Monte
     Carlo" (2023); arxiv: https://arxiv.org/abs/2312.04898, Equation 7.
    """
    def __init__(self, minimum: float, maximum: float):
        if minimum > maximum:
            raise ValueError("minimum is greater than maximum")
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(event_shape=(2,))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        t0 = (self.minimum - self.maximum) / 2 * (torch.cos(x[..., 0]) + torch.cos(x[..., 1]))
        t1 = (self.maximum + self.minimum) / 2 * (x[..., 0] ** 2 / 2 + x[..., 1] ** 2 / 2)
        return t0 + t1
