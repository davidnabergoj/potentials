import math

import numpy as np
import torch
import scipy

from potentials.base import Potential


class PredatorPrey(Potential):
    """
    Note: this model is gradient-free.
    """

    @staticmethod
    def diff_eq(y, t, r, k, s, a, u, v):
        p, q = y
        return [
            r * p * (1 - p / k) - s * p * q / (a + p),
            u * p * q / (a + p) - v * q
        ]

    def __init__(self):
        event_shape = (8,)
        super().__init__(event_shape)

        self.n_observations = 5

        # Ground truth parameters
        self.p_star_0 = 50
        self.q_star_0 = 5
        self.r_star = 0.6
        self.k_star = 100
        self.s_star = 1.2
        self.a_star = 25
        self.u_star = 0.5
        self.v_star = 0.3

        # Generate ground truth data
        self.t = np.linspace(0, 50, self.n_observations)
        self.solution = torch.tensor(
            scipy.integrate.odeint(
                self.diff_eq,
                [self.p_star_0, self.q_star_0],
                self.t,
                args=(self.r_star, self.k_star, self.s_star, self.a_star, self.u_star, self.v_star)
            )
        )  # first column: P, second column: Q
        self.noise = torch.randn_like(self.solution) * math.sqrt(10.0)
        self.targets = self.solution + self.noise

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) != 1:
            raise ValueError("Batch size > 1 is not supported")
        assert x.shape == (1, 8)

        p0 = float(x[..., 0])
        q0 = float(x[..., 1])
        r = float(x[..., 2])
        k = float(x[..., 3])
        s = float(x[..., 4])
        a = float(x[..., 5])
        u = float(x[..., 6])
        v = float(x[..., 7])

        # Now compute the values of P, Q at times self.t
        solution = torch.tensor(scipy.integrate.odeint(self.diff_eq, [p0, q0], self.t, args=(r, k, s, a, u, v)))
        # first column: P, second column: Q

        # I assume the likelihood is as in the Biochemical Oxygen Demand model, i.e. Gaussian

        # Compute the log likelihood
        log_likelihood = torch.sum(torch.square(solution - self.targets))

        # TODO add the cyclic prior
        log_prior = 0.0

        log_probability = log_likelihood + log_prior
        return -log_probability.view(1,)
