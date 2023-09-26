from potentials.base import Potential
import torch

from potentials.forward_model import ForwardModel, ODE


def lorenz_system_3d(state, params):
    # Unpack state
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]

    # Unpack parameters
    sigma = params[..., 0]
    rho = params[..., 1]
    beta = params[..., 2]

    # Return the derivative of state with respect to time
    return torch.concat([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ], dim=0)  # TODO maybe fix concatenation dim


class Lorenz(Potential):
    def __init__(self):
        super().__init__(event_shape=(3,))
        initial_state = ...  # TODO set initial state
        self.observations = ...  # TODO set noisy observations
        target_time = ...  # TODO set target time

        self.forward_model = ODE(initial_state, lorenz_system_3d, target_time)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(self.forward_model(x) - self.observations), dim=-1)
