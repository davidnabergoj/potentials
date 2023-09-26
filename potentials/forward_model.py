import torch
from torchdiffeq import odeint
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(self, initial_state: torch.Tensor):
        super().__init__()
        self.initial_state = initial_state

    def forward(self, parameters: torch.Tensor):
        """
        Simulate forward model with given parameters and fixed initial state.
        :param parameters:
        :return:
        """
        raise NotImplementedError


class ODE(ForwardModel):
    def __init__(self, initial_state: torch.Tensor,
                 ode: callable,
                 target_time: torch.Tensor):
        super().__init__(initial_state)
        self.ode = ode
        self.target_time = target_time

    def forward(self, parameters: torch.Tensor):
        return odeint(self.ode, parameters, t=self.target_time)
