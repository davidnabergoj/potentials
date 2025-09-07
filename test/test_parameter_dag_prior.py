import pytest
import torch
import torch.distributions as td

from potentials.real.german_credit import SparseGermanCredit
from potentials.real.posterior.parameter import ParameterVector
from potentials.transformations import bound_parameter


def test_sparse_german_credit_value():
    torch.manual_seed(0)

    batch_shape = (200,)

    u = SparseGermanCredit(n_data=0, use_hyperprior=False)
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    neg_log_prob_sgc = u(x)
    assert neg_log_prob_sgc.shape == batch_shape

    tau, log_det_tau = bound_parameter(
        x[..., 0][..., None],
        batch_shape,
        low=1e-8,
        high=torch.inf
    )
    assert log_det_tau.shape == batch_shape

    beta = x[..., 1:26]

    _lambda, log_det_lambda = bound_parameter(
        x[..., 26:],
        batch_shape,
        low=1e-8,
        high=torch.inf
    )

    assert log_det_lambda.shape == batch_shape

    _tmp_tau = td.Gamma(0.5, 0.5).log_prob(tau)[..., 0] + log_det_tau
    assert _tmp_tau.shape == batch_shape

    _tmp_beta = td.Normal(0, 1).log_prob(beta).sum(dim=-1)
    assert _tmp_beta.shape == batch_shape

    _tmp_lambda = td.Gamma(0.5, 0.5).log_prob(
        _lambda
    ).sum(dim=-1) + log_det_lambda
    assert _tmp_lambda.shape == batch_shape

    log_prob_manual = (
        _tmp_tau
        + _tmp_beta
        + _tmp_lambda
    )
    neg_log_prob_manual = -log_prob_manual.to(neg_log_prob_sgc.dtype)

    assert torch.all(torch.isfinite(neg_log_prob_sgc))
    assert torch.all(torch.isfinite(neg_log_prob_manual))
    assert torch.allclose(neg_log_prob_sgc, neg_log_prob_manual, atol=1e-8)


def test_sparse_german_credit_constrain():
    torch.manual_seed(0)

    batch_shape = (10000,)

    u = SparseGermanCredit(n_data=0, use_hyperprior=False)
    x = torch.randn(size=(*batch_shape, *u.event_shape))

    constrained, _ = u.posterior_parameters.constrain(x)

    tau = bound_parameter(
        x[..., 0],
        batch_shape,
        low=1e-8,
        high=torch.inf
    )[0][..., None]

    beta = x[..., 1:26]

    _test_beta, _ = bound_parameter(
        x[..., 1:26],
        batch_shape,
        low=-torch.inf,
        high=torch.inf
    )
    assert torch.allclose(beta, _test_beta)

    _lambda, _ = bound_parameter(
        x[..., 26:],
        batch_shape,
        low=1e-8,
        high=torch.inf
    )

    assert constrained['tau'].shape == (*batch_shape, 1)
    assert constrained['beta'].shape == (*batch_shape, 25)
    assert constrained['lambda'].shape == (*batch_shape, 25)

    assert tau.shape == (*batch_shape, 1)
    assert beta.shape == (*batch_shape, 25)
    assert _lambda.shape == (*batch_shape, 25)

    assert torch.allclose(tau, constrained['tau'])
    assert torch.allclose(beta, constrained['beta'])
    assert torch.allclose(_lambda, constrained['lambda'])


def test_sparse_german_credit_constrain_beta():
    torch.manual_seed(0)

    batch_shape = (10000,)
    x = torch.randn(size=(*batch_shape, 25))

    vec = ParameterVector(
        25,
        prior=td.Normal(
            loc=0.0,
            scale=1.0
        )
    )
    vector_constrained, log_det = vec.constrain_with_log_det(x)

    assert torch.allclose(vector_constrained, x)
    assert log_det == 0.0


def test_sparse_german_credit_constrain_with_log_prior_beta():
    torch.manual_seed(0)

    batch_shape = (10000,)
    x = torch.randn(size=(*batch_shape, 25))

    vec = ParameterVector(
        25,
        prior=td.Normal(
            loc=0.0,
            scale=1.0
        )
    )
    vector_constrained, log_det = vec.constrain_with_log_det(x)
    log_prior = vec.log_prior_without_log_det(vector_constrained) + log_det

    assert torch.allclose(vector_constrained, x)
    assert log_det == 0.0

    log_prior_manual = td.Normal(0.0, 1.0).log_prob(x).sum(dim=-1)
    assert torch.allclose(log_prior, log_prior_manual)
