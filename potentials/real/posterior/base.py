from potentials.base import Potential
from potentials.real.posterior.parameter import ParameterDAG


import torch


import math
from typing import Dict, List, Tuple


class Posterior1D(Potential):
    def __init__(self, posterior_parameters: ParameterDAG):
        event_size = sum([
            p.size 
            for p in posterior_parameters.parameters.values()
        ])
        event_shape = (event_size,)
        super().__init__(event_shape)
        self.posterior_parameters = posterior_parameters

    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        """
        Return a directed acyclic graph (DAG) of the prior as an edgelist.
        Each element of the edgelist is a tuple of the form `(source, destination)`,
         where `source` and `destination` are integer indices of model parameters and
         the parameter at `destination` depends on the parameter at `source` in the prior DAG.
        """
        raise NotImplementedError

    def likelihood_object(self,
                          extracted: Dict[str, torch.Tensor]) -> torch.distributions.Distribution:
        """
        Create the likelihood object.

        :param torch.Tensor extracted: extracted parameters (same as `self.extract_parameters` outputs).
        :return: `torch.distributions.Distribution` object.
        """
        raise NotImplementedError

    def compute_likelihood(self, extracted: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the log likelihood.

        :param torch.Tensor extracted: extracted parameters (same as `self.extract_parameters` outputs).
        :return: torch.Tensor with shape `batch_shape`.
        """
        raise NotImplementedError

    def compute(self, unconstrained: torch.Tensor):
        """
        Compute the negative log probability density of this object.

        :param torch.Tensor unconstrained: tensor of unconstrained values with shape `(*batch_shape, *event_shape)`.
        :return: torch.Tensor with shape `batch_shape`.
        """
        extracted = self.extract_parameters(unconstrained)
        log_likelihood = self.compute_likelihood(extracted)
        log_prob = extracted['log_prior'] + log_likelihood
        # log_prior includes the log jacobian determinant of parameter transformations
        return -log_prob

    def extract_parameters(self,
                           unconstrained: torch.Tensor,
                           return_log_probs: bool = True) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary of model parameters and the associated log prior density values from unconstrained inputs.

        :param torch.Tensor unconstrained: tensor of unconstrained values with shape `(*batch_shape, *event_shape)`.
        :param bool return_log_probs: if False, do not return the log prior density values.
        :return: dictionary where keys correspond to constrained inputs or their likelihoods and values are the 
         associated tensors.
        """
        out, log_det = self.posterior_parameters.constrain(
            unconstrained
        )

        # Compute prior probabilities
        if return_log_probs:
            log_prior_without_log_det = self.posterior_parameters.log_prior_without_log_det(out)
            out['log_prior'] = log_prior_without_log_det + log_det
        return out

    def posterior_predictive_draws(self,
                                   unconstrained_posterior_draws: torch.Tensor,
                                   n_draws: int = 100) -> torch.Tensor:
        """
        Sample posterior predictive draws given parameters.

        :param unconstrained_posterior_draws: tensor of parameters with shape `(*batch_shape, *event_shape)`.
        :param n_draws: number of samples to draw for each parameter vector.
        :return: tensor of posterior predictive draws with shape `(n_draws, *batch_shape, ...)`.
        """
        extracted = self.extract_parameters(unconstrained_posterior_draws)
        dist = self.likelihood_object(extracted)
        return dist.sample((n_draws,))

    def normalized_log_posterior_predictive_density(self,
                                                    unconstrained_posterior_draws: torch.Tensor):
        extracted = self.extract_parameters(unconstrained_posterior_draws)

        # log likelihood per posterior draw; event dims are already summed by Independent
        # shape == batch over posterior draws
        log_lik = self.compute_likelihood(extracted)

        # Reduce over ALL batch dims stably:
        if log_lik.ndim == 0:
            return log_lik  # already scalar

        reduce_dims = tuple(range(log_lik.ndim))
        # number of Monte Carlo samples = product of sizes across reduce_dims
        n = int(torch.prod(torch.as_tensor(log_lik.shape)))

        return torch.logsumexp(log_lik, dim=reduce_dims) - math.log(n)
