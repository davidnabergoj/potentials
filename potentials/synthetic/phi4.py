import torch

from potentials.base import Potential


class Phi4(Potential):
    """
    Two-dimensional phi^4 model.
    Events are square lattices with shape (L, L) where L is the side length.
    We can use a lattice to compute the magnetization M of the model (M is a scalar quantity).

    The potential depends on a temperature parameter theta.
    By varying theta, we encounter a phase transition from unimodal to bimodal distribution of M at theta = 1.2.
    Authors of "Optimizing Markov Chain Monte Carlo Convergence with Normalizing Flows and Gibbs Sampling"
     provide an example for L = 16.
    When theta = 1, the distribution of M is unimodal.
    When theta = 1.2, the distribution of M has two overlapping modes.
    When theta = 1.6, the distribution of M as two non-overlapping modes.
    """

    def __init__(self, length: int = 16, temperature: float = 1.2):
        self.theta = temperature
        super().__init__(event_shape=(length, length))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the phi^4 potential as defined in Equation 4 of the accompanying paper.

        :param x: lattices with shape (batch_size, L, L) or (*batch_shape, L, L).
        :return: potential for each lattice with shape (batch_size,) or batch_shape.
        """
        lattice_pow_2 = x ** 2
        lattice_pow_4 = lattice_pow_2 ** 2

        lattice_row_product = x
        lattice_row_product[..., :-1, :] *= x[..., 1:, :]

        lattice_col_product = x
        lattice_col_product[..., :, :-1] *= x[..., :, 1:]

        combined_lattice = (
                (2 - self.theta / 2) * lattice_pow_2
                + 1 / 4 * lattice_pow_4
                - lattice_row_product
                - lattice_col_product
        )

        return torch.sum(combined_lattice, dim=(-2, -1))
