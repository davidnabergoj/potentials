import torch

from potentials.base import Potential


class Phi4(Potential):
    """
    Two-dimensional phi^4 model.
    Events are square lattices with shape (L, L) where L is the side length.
    We can use a lattice to compute the magnetization M of the model.
    M is the average of lattice elements, i.e. 1 / L^2 * sum_i { sum_j ( x[i, j] ) }

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
        self.length = length
        super().__init__(event_shape=(length, length))

    @staticmethod
    def compute_magnetization(x: torch.Tensor):
        return torch.mean(x, dim=(-2, -1))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the phi^4 potential as defined in Equation 4 of the accompanying paper.

        :param x: lattices with shape (batch_size, L, L) or (*batch_shape, L, L).
        :return: potential for each lattice with shape (batch_size,) or batch_shape.
        """
        lattice_pow_2 = x ** 2
        lattice_pow_4 = lattice_pow_2 ** 2

        lattice_row_product = torch.clone(x)
        lattice_row_product[..., :-1, :] *= x[..., 1:, :]

        lattice_col_product = torch.clone(x)
        lattice_col_product[..., :, :-1] *= x[..., :, 1:]

        combined_lattice = (
                (2 - self.theta / 2) * lattice_pow_2
                + 1 / 4 * lattice_pow_4
                - lattice_row_product
                - lattice_col_product
        )

        return torch.sum(combined_lattice, dim=(-2, -1))

    @property
    def mean(self):
        if self.length == 8:
            return torch.tensor([
                [0.0451, 0.0809, 0.1125, 0.1455, 0.1865, 0.2465, 0.3406, 0.5312],
                [0.0780, 0.1432, 0.2022, 0.2567, 0.3174, 0.3975, 0.5131, 0.6978],
                [0.1115, 0.2002, 0.2733, 0.3409, 0.4086, 0.4889, 0.5993, 0.7571],
                [0.1476, 0.2575, 0.3384, 0.4100, 0.4773, 0.5516, 0.6466, 0.7836],
                [0.1870, 0.3181, 0.4084, 0.4758, 0.5389, 0.5999, 0.6808, 0.7970],
                [0.2429, 0.3980, 0.4913, 0.5533, 0.6002, 0.6496, 0.7108, 0.8137],
                [0.3410, 0.5133, 0.6021, 0.6483, 0.6793, 0.7110, 0.7572, 0.8357],
                [0.5332, 0.6979, 0.7579, 0.7845, 0.7983, 0.8131, 0.8373, 0.8851]
            ])
        return super().mean

    @property
    def second_moment(self):
        if self.length == 8:
            return torch.tensor([
                [0.3442, 0.3947, 0.4044, 0.4099, 0.4169, 0.4315, 0.4611, 0.5639],
                [0.3920, 0.4758, 0.5050, 0.5221, 0.5378, 0.5671, 0.6229, 0.7594],
                [0.4054, 0.5048, 0.5444, 0.5734, 0.5972, 0.6331, 0.6992, 0.8307],
                [0.4102, 0.5205, 0.5706, 0.6054, 0.6352, 0.6746, 0.7397, 0.8635],
                [0.4168, 0.5391, 0.5971, 0.6343, 0.6708, 0.7073, 0.7713, 0.8801],
                [0.4287, 0.5653, 0.6331, 0.6766, 0.7088, 0.7459, 0.7979, 0.9008],
                [0.4603, 0.6225, 0.7009, 0.7411, 0.7702, 0.7979, 0.8460, 0.9301],
                [0.5652, 0.7586, 0.8323, 0.8653, 0.8829, 0.9005, 0.9311, 0.9905]
            ])
        return super().second_moment


if __name__ == '__main__':
    u = Phi4(length=8)
    print(u.mean.shape)
    print(u.second_moment.shape)
