from typing import Dict, Tuple, Union
import torch

from potentials.transformations import bound_parameter


class Parameter1D:
    def __init__(self,
                 n_dim: int,
                 bound: Union[str, Tuple[float, float]] = None):
        self.n_dim = n_dim

        if bound == 'positive':
            lower_bound = 0.0
            upper_bound = torch.inf
        elif bound == 'negative':
            lower_bound = -torch.inf
            upper_bound = 0.0
        elif bound is None or bound == 'none':
            lower_bound = -torch.inf
            upper_bound = torch.inf
        else:
            lower_bound, upper_bound = bound

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def constrain(self, unconstrained: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the constrained parameter and log Jacobian determinant of the transformation.
        """
        batch_shape = unconstrained.shape[:-1]
        return bound_parameter(
            unconstrained,
            batch_shape=batch_shape,
            low=self.lower_bound,
            high=self.upper_bound
        )

    def constrain_with_prior(self, unconstrained: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the constrained parameter and log prior.
        The log prior includes the log Jacobian determinant of the constraining transformation.
        """
        raise NotImplementedError


class ParameterScalar(Parameter1D):
    def __init__(self,
                 bound: bool = None):
        super().__init__(1, bound)


class ParameterVector(Parameter1D):
    def __init__(self,
                 n_dim: int,
                 bound: bool = None):
        if n_dim <= 1:
            raise ValueError(
                "Can only use ParameterVector with 2 or more elements"
            )
        super().__init__(n_dim, bound)


class ParameterSet1D:
    def __init__(self,
                 parameters: Dict[str, Parameter1D]):
        self.parameters = parameters
        self.n_dim = sum([p.n_dim for p in self.parameters.values()])

    def constrain(self, unconstrained: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Return the constrained parameters and log Jacobian determinant of the transformation.
        """
        constrained_parameters = dict()
        log_det = 0.0

        idx_start = 0
        idx_end = None
        for key, p in self.parameters.items():
            idx_end = idx_start + p.n_dim
            _in = unconstrained[..., idx_start:idx_end]
            ret = p.constrain(_in)
            constrained_parameters[key] = ret[0]
            log_det += ret[1]

        return constrained_parameters, log_det
