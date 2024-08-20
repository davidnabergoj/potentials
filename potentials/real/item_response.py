import torch
import torch.distributions as td
from potentials.base import Potential
from potentials.utils import sum_except_batch


class SyntheticItemResponseTheory(Potential):
    """

    Reference: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf.
    """

    def __init__(self):
        super().__init__(event_shape=(501,))
        self.n_students = 100  # J
        self.n_questions = 400  # K
        self.n_responses = 30105
        self.responses: torch.Tensor = ...  # TODO fetch responses tensor
        self.student_index: torch.Tensor = ...  # TODO fetch responses index, this has shape `(self.n_responses,)`
        self.response_index: torch.Tensor = ...  # TODO fetch responses index, this has shape `(self.n_responses,)`

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]

        beta = x[..., 0:400]
        alpha = x[..., 400:500]
        delta = x[..., 500]

        log_prob_delta = td.Normal(loc=3 / 4, scale=1.0).log_prob(delta)
        log_prob_alpha = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(alpha), batch_shape)
        log_prob_beta = sum_except_batch(td.Normal(loc=0.0, scale=1.0).log_prob(beta), batch_shape)
        log_prior = log_prob_alpha + log_prob_beta + log_prob_delta

        probs = torch.sigmoid(alpha[..., self.student_index] - beta[..., self.response_index] + delta[..., None])
        log_likelihood = sum_except_batch(
            torch.distributions.Bernoulli(probs=probs).log_prob(self.responses),
            batch_shape
        )
        log_prob = log_likelihood + log_prior

        return -log_prob
