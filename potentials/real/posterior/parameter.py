from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Type, Union
import torch
import torch.distributions as td

from potentials.transformations import bound_parameter


class Parameter1D:
    """
    Class for posterior parameters with one event dimension (with one or more elements).
    """

    def __init__(self,
                 size: int,
                 prior: Union[td.Distribution, Type[td.Distribution]],
                 prior_kwargs: Dict[str, Union[Callable, torch.Tensor, str]] = None,
                 bound: Union[str, Tuple[float, float]] = None):
        """
        Initialize a 1D parameter with a given prior distribution.

        :param int size: Number of parameter elements.
        :param prior: Distribution object or class, describing this object's prior.
        :type prior: Union[torch.distributions.Distribution, Type[torch.distributions.Distribution]]
        :param bound: Boundary constraints for the parameter. Options are:

              - ``"positive"`` → parameter constrained to :math:`[0, +∞)`
              - ``"negative"`` → parameter constrained to :math:`(-∞, 0]`
              - ``"none"`` or ``None`` → unconstrained (:math:`(-∞, +∞)`)
              - Tuple of floats ``(lower, upper)`` specifying custom bounds.
        :type bound: Union[str, Tuple[float, float]], optional

        :example:

            >>> import torch.distributions as td
            >>> a = Parameter1D(
            ...     20,
            ...     td.Normal,
            ...     {'loc': 0.0, 'scale': (lambda sigma_a: sigma_a)}
            ... )
            >>> b = Parameter1D(
            ...     20,
            ...     td.Normal,
            ...     {'loc': (lambda a: a),
            ...      'scale': (lambda sigma_b, delta: sigma_b + delta)}
            ... )
        """
        self.size = size

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

        self.prior = prior

        self.parents: Dict[str, Parameter1D] = dict()
        self.prior_kwargs = prior_kwargs or None

    def constrain_with_log_det(self, 
                               unconstrained: torch.Tensor, 
                               eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the constrained parameter and log Jacobian determinant of the transformation.
        """
        batch_shape = unconstrained.shape[:-1]
        output = bound_parameter(
            unconstrained,
            batch_shape=batch_shape,
            low=self.lower_bound + eps,
            high=self.upper_bound - eps
        )
        return output

    def log_prior_without_log_det(self, constrained: torch.Tensor, **kwargs):
        """
        Compute the log prior for this parameter object.

        :param torch.Tensor constrained: constrained values for this parameter.
        :param kwargs: keyword arguments (usually parameters) for the prior distribution
         if the prior does not have fixed parameters.
        """
        if isinstance(self.prior, td.Distribution):
            dist = self.prior
        else:
            # Handle prior_kwargs
            prior_kwargs = {}
            for key, value in self.prior_kwargs.items():
                if isinstance(value, Callable):
                    prior_kwargs[key] = value(**kwargs)
                elif isinstance(value, str):
                    prior_kwargs[key] = kwargs[value]
                else:
                    prior_kwargs[key] = value
            dist = self.prior(**prior_kwargs)
            if isinstance(self, ParameterVector):
                dist = td.Independent(dist, reinterpreted_batch_ndims=1)
                return dist.log_prob(constrained)
        return dist.log_prob(constrained).sum(dim=-1)


class ParameterScalar(Parameter1D):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class ParameterVector(Parameter1D):
    def __init__(self,
                 size: int,
                 **kwargs):
        if size <= 1:
            raise ValueError(
                "Can only use ParameterVector with 2 or more elements"
            )
        super().__init__(size, **kwargs)


class ParameterDAG:
    def __init__(self,
                 parameters: Dict[str, Parameter1D],
                 edge_list: List[Tuple[str, str]] = None,
                 additional_parameters: Dict[str, Callable] = None):
        if edge_list is None:
            edge_list = []
        elif len(edge_list) != len(set(edge_list)):
            raise ValueError("Duplicate element in `edge_list`")

        self.parameters = parameters
        self.edge_list = edge_list
        self.n_dim = sum([p.size for p in self.parameters.values()])

        for parent, child in edge_list:
            self.parameters[child].parents[parent] = self.parameters[parent]

        self.additional_parameters = (additional_parameters or {})

    def constrain(self, unconstrained: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Return the constrained parameters and log Jacobian determinant of the transformation.
        """
        constrained_parameters = dict()
        log_det = 0.0

        idx_start = 0
        idx_end = None
        for key, p in self.parameters.items():
            idx_end = idx_start + p.size
            _in = unconstrained[..., idx_start:idx_end]
            ret = p.constrain_with_log_det(_in)
            constrained_parameters[key] = ret[0]
            log_det += ret[1]
            idx_start = idx_end

        self._add_parameters(constrained_parameters)
        return constrained_parameters, log_det

    def log_prior_without_log_det(self, constrained_parameters: torch.Tensor):
        # Compute log prior once all parameters are constrained
        log_prior_without_log_det = 0.0
        for child_key, child in self.parameters.items():
            if len(child.parents) == 0:
                log_prior_without_log_det += child.log_prior_without_log_det(
                    constrained_parameters[child_key]
                )
            else:
                log_prior_without_log_det += child.log_prior_without_log_det(
                    constrained_parameters[child_key],
                    **{
                        parent_key: constrained_parameters[parent_key]
                        for parent_key in child.parents
                    }
                )

        log_prior_without_log_det
        return log_prior_without_log_det

    def constrain_with_log_prior(self, unconstrained: torch.Tensor):
        # Constrain parameters first
        constrained_parameters, log_det = self.constrain(unconstrained)

        # Compute log prior once all parameters are constrained
        log_prior_without_log_det = 0.0
        for child_key, child in self.parameters.items():
            if len(child.parents) == 0:
                log_prior_without_log_det += child.log_prior_without_log_det(
                    constrained_parameters[child_key]
                )
            else:
                log_prior_without_log_det += child.log_prior_without_log_det(
                    constrained_parameters[child_key],
                    **{
                        parent_key: constrained_parameters[parent_key]
                        for parent_key in child.parents
                    }
                )

        log_prior = log_det + log_prior_without_log_det
        return constrained_parameters, log_prior

    def _add_parameters(self, constrained) -> dict:
        if len(self.additional_parameters) > 0:
            for key in self.additional_parameters:
                constrained[key] = self.additional_parameters[key](
                    **constrained
                )
