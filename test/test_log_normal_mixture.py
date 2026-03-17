import math
import pytest
import torch
import torch.distributions as dist
from torch.distributions import constraints
from potentials.utils import LogNormalMixture

SEED = 42
N_SAMPLES = 200_000
RTOL_MC = 0.05   # 5 % relative tolerance for Monte-Carlo checks


def scalar_dist(**kwargs):
    """Return a LogNormalMixture with default scalar parameters."""
    defaults = dict(loc0=0.0, scale0=0.5, loc1=2.0, scale1=0.3,
                    weight0=0.6, weight1=0.4)
    defaults.update(kwargs)
    return LogNormalMixture(**defaults)


def lognormal_log_prob_ref(x, loc, scale):
    """Reference scalar log-prob for a single LogNormal."""
    lx = math.log(x)
    return -0.5 * ((lx - loc) / scale) ** 2 - math.log(scale) - 0.5 * math.log(2 * math.pi) - lx


# 1. construction & shapes

class TestConstruction:
    def test_scalar_construction(self):
        d = scalar_dist()
        assert d.batch_shape == torch.Size([])
        assert d.event_shape == torch.Size([])

    def test_batched_construction(self):
        locs = torch.zeros(3)
        d = LogNormalMixture(loc0=locs, scale0=1.0, loc1=1.0, scale1=0.5,
                             weight0=0.5, weight1=0.5)
        assert d.batch_shape == torch.Size([3])

    def test_weights_normalized(self):
        d = LogNormalMixture(loc0=0., scale0=1., loc1=1., scale1=1.,
                             weight0=3., weight1=1.)
        assert torch.isclose(d._w0, torch.tensor(0.75), atol=1e-5)
        assert torch.isclose(d._w1, torch.tensor(0.25), atol=1e-5)
        assert torch.isclose(d._w0 + d._w1, torch.tensor(1.0), atol=1e-6)

    def test_equal_weights(self):
        d = LogNormalMixture(loc0=0., scale0=1., loc1=1., scale1=1.,
                             weight0=1., weight1=1.)
        assert torch.isclose(d._w0, torch.tensor(0.5), atol=1e-6)
        assert torch.isclose(d._w1, torch.tensor(0.5), atol=1e-6)


# 2. support

class TestSupport:
    def test_support_is_positive(self):
        assert scalar_dist().support == constraints.positive

    def test_log_prob_positive_values(self):
        d = scalar_dist()
        for x in [0.01, 0.5, 1.0, 10.0, 1000.0]:
            lp = d.log_prob(torch.tensor(x))
            assert torch.isfinite(lp), f"log_prob not finite at x={x}"

    def test_log_prob_zero_is_neg_inf(self):
        d = LogNormalMixture(loc0=0., scale0=0.5, loc1=2., scale1=0.3,
                             weight0=0.6, weight1=0.4, validate_args=False)
        lp = d.log_prob(torch.tensor(0.0))
        assert not torch.isfinite(lp)

    def test_validate_args_rejects_negative(self):
        d = LogNormalMixture(loc0=0., scale0=1., loc1=1., scale1=1.,
                             weight0=0.5, weight1=0.5, validate_args=True)
        with pytest.raises(ValueError):
            d.log_prob(torch.tensor(-1.0))


# 3. log_prob correctness──

class TestLogProb:
    def test_degenerate_weight0_matches_lognormal(self):
        d = LogNormalMixture(loc0=1.0, scale0=0.5, loc1=5.0, scale1=1.0,
                             weight0=1e6, weight1=1.0)
        ref = dist.LogNormal(loc=1.0, scale=0.5)
        xs = torch.tensor([0.5, 1.0, 2.0, 5.0])
        assert torch.allclose(d.log_prob(xs), ref.log_prob(xs), atol=1e-3)

    def test_degenerate_weight1_matches_lognormal(self):
        d = LogNormalMixture(loc0=5.0, scale0=1.0, loc1=1.0, scale1=0.5,
                             weight0=1.0, weight1=1e6)
        ref = dist.LogNormal(loc=1.0, scale=0.5)
        xs = torch.tensor([0.5, 1.0, 2.0, 5.0])
        assert torch.allclose(d.log_prob(xs), ref.log_prob(xs), atol=1e-3)

    def test_manual_formula(self):
        """Compare against hand-computed mixture log-prob."""
        loc0, scale0 = 0.0, 0.5
        loc1, scale1 = 2.0, 0.3
        w0, w1 = 0.6, 0.4

        d = LogNormalMixture(loc0=loc0, scale0=scale0, loc1=loc1, scale1=scale1,
                             weight0=w0, weight1=w1)
        x = 1.5
        p0 = math.exp(lognormal_log_prob_ref(x, loc0, scale0))
        p1 = math.exp(lognormal_log_prob_ref(x, loc1, scale1))
        expected = math.log(w0 * p0 + w1 * p1)
        got = d.log_prob(torch.tensor(x)).item()
        assert abs(got - expected) < 1e-5

    def test_log_prob_batch(self):
        d = scalar_dist()
        xs = torch.tensor([0.1, 1.0, 5.0, 20.0])
        lps = d.log_prob(xs)
        assert lps.shape == (4,)
        assert torch.all(torch.isfinite(lps))

    def test_log_prob_is_normalized(self):
        """Numerical integration of exp(log_prob) over positive reals is approximately 1."""
        d = scalar_dist()
        # Integrate in log-space: int {p(x) dx} = int {p(e^u) e^u du}
        u = torch.linspace(-6, 8, 10_000)
        du = u[1] - u[0]
        x = u.exp()
        log_px = d.log_prob(x)
        integral = (log_px + u).exp().sum() * du
        assert abs(integral.item() - 1.0) < 0.01

    def test_symmetry_of_component_swap(self):
        d1 = LogNormalMixture(loc0=0., scale0=0.5, loc1=2., scale1=0.3,
                              weight0=0.5, weight1=0.5)
        d2 = LogNormalMixture(loc0=2., scale0=0.3, loc1=0., scale1=0.5,
                              weight0=0.5, weight1=0.5)
        xs = torch.tensor([0.5, 1.0, 3.0, 8.0])
        assert torch.allclose(d1.log_prob(xs), d2.log_prob(xs), atol=1e-5)


# 4. sampling

class TestSampling:
    def test_rsample_shape_scalar(self):
        d = scalar_dist()
        s = d.rsample((100,))
        assert s.shape == (100,)

    def test_rsample_shape_batched(self):
        d = LogNormalMixture(loc0=torch.zeros(3), scale0=1., loc1=1., scale1=0.5,
                             weight0=0.5, weight1=0.5)
        s = d.rsample((10,))
        assert s.shape == (10, 3)

    def test_rsample_positive(self):
        d = scalar_dist()
        s = d.rsample((1000,))
        assert torch.all(s > 0)

    def test_rsample_finite(self):
        d = scalar_dist()
        s = d.rsample((1000,))
        assert torch.all(torch.isfinite(s))

    def test_has_rsample_flag(self):
        assert LogNormalMixture.has_rsample is True

    def test_empirical_mean(self):
        torch.manual_seed(SEED)
        d = scalar_dist()
        s = d.rsample((N_SAMPLES,))
        assert abs(s.mean().item() - d.mean.item()) / d.mean.item() < RTOL_MC

    def test_empirical_variance(self):
        torch.manual_seed(SEED)
        d = scalar_dist()
        s = d.rsample((N_SAMPLES,))
        assert abs(s.var().item() - d.variance.item()) / \
            d.variance.item() < RTOL_MC

    def test_empirical_component_mixing(self):
        torch.manual_seed(SEED)
        # Two well-separated components
        d = LogNormalMixture(loc0=0., scale0=0.1, loc1=4., scale1=0.1,
                             weight0=0.7, weight1=0.3)
        s = d.rsample((N_SAMPLES,))
        frac_low = (s < 5).float().mean().item()
        assert abs(frac_low - 0.7) < 0.02


# 5. moments

class TestMoments:
    def test_mean_positive(self):
        assert scalar_dist().mean.item() > 0

    def test_variance_positive(self):
        assert scalar_dist().variance.item() > 0

    def test_mean_degenerate_matches_lognormal(self):
        d = LogNormalMixture(loc0=1.0, scale0=0.5, loc1=5.0, scale1=1.0,
                             weight0=1e6, weight1=1.0)
        ref = dist.LogNormal(loc=torch.tensor(1.0), scale=torch.tensor(0.5))
        assert torch.isclose(d.mean, ref.mean, rtol=1e-3)

    def test_variance_degenerate_matches_lognormal(self):
        d = LogNormalMixture(loc0=1.0, scale0=0.5, loc1=5.0, scale1=1.0,
                             weight0=1e10, weight1=1.0)
        ref = dist.LogNormal(loc=torch.tensor(1.0), scale=torch.tensor(0.5))
        assert torch.isclose(d.variance, ref.variance, rtol=0.01)

    def test_mean_manual_formula(self):
        loc0, scale0, loc1, scale1 = 0.0, 0.5, 2.0, 0.3
        w0, w1 = 0.6, 0.4
        d = LogNormalMixture(loc0=loc0, scale0=scale0, loc1=loc1, scale1=scale1,
                             weight0=w0, weight1=w1)
        expected = (w0 * math.exp(loc0 + 0.5 * scale0 ** 2)
                    + w1 * math.exp(loc1 + 0.5 * scale1 ** 2))
        assert abs(d.mean.item() - expected) < 1e-5

    def test_variance_manual_formula(self):
        loc0, scale0, loc1, scale1 = 0.0, 0.5, 2.0, 0.3
        w0, w1 = 0.6, 0.4
        d = LogNormalMixture(loc0=loc0, scale0=scale0, loc1=loc1, scale1=scale1,
                             weight0=w0, weight1=w1)
        ex2 = (w0 * math.exp(2 * loc0 + 2 * scale0 ** 2)
               + w1 * math.exp(2 * loc1 + 2 * scale1 ** 2))
        mean = (w0 * math.exp(loc0 + 0.5 * scale0 ** 2)
                + w1 * math.exp(loc1 + 0.5 * scale1 ** 2))
        expected_var = ex2 - mean ** 2
        assert abs(d.variance.item() - expected_var) < 1e-4

    def test_batched_mean_shape(self):
        locs = torch.tensor([0.0, 1.0, 2.0])
        d = LogNormalMixture(loc0=locs, scale0=0.5, loc1=3.0, scale1=0.3,
                             weight0=0.5, weight1=0.5)
        assert d.mean.shape == (3,)
        assert torch.all(d.mean > 0)


# 6. gradients

class TestGradients:
    def test_log_prob_grad_wrt_x(self):
        d = scalar_dist()
        x = torch.tensor(2.0, requires_grad=True)
        lp = d.log_prob(x)
        lp.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad)

    def test_log_prob_grad_wrt_params(self):
        loc0 = torch.tensor(0.0, requires_grad=True)
        scale0 = torch.tensor(0.5, requires_grad=True)
        d = LogNormalMixture(loc0=loc0, scale0=scale0, loc1=2.0, scale1=0.3,
                             weight0=0.6, weight1=0.4)
        lp = d.log_prob(torch.tensor(1.5))
        lp.backward()
        assert loc0.grad is not None and torch.isfinite(loc0.grad)
        assert scale0.grad is not None and torch.isfinite(scale0.grad)

    def test_rsample_grad_flows(self):
        loc0 = torch.tensor(0.0, requires_grad=True)
        scale0 = torch.tensor(0.5, requires_grad=True)
        d = LogNormalMixture(loc0=loc0, scale0=scale0, loc1=2.0, scale1=0.3,
                             weight0=0.6, weight1=0.4)
        torch.manual_seed(SEED)
        s = d.rsample((50,))
        loss = s.mean()
        loss.backward()
        assert loc0.grad is not None
        assert scale0.grad is not None


# 7. edge cases

class TestEdgeCases:
    def test_very_small_x(self):
        d = scalar_dist()
        lp = d.log_prob(torch.tensor(1e-10))
        assert torch.isfinite(lp)

    def test_very_large_x(self):
        d = scalar_dist()
        lp = d.log_prob(torch.tensor(1e6))
        assert torch.isfinite(lp)

    def test_x_equals_one(self):
        """log(1)=0, so the Jacobian term vanishes; should still be finite."""
        d = scalar_dist()
        lp = d.log_prob(torch.tensor(1.0))
        assert torch.isfinite(lp)

    def test_large_scale_separation(self):
        """Widely separated components should not cause NaN."""
        d = LogNormalMixture(loc0=-10., scale0=0.1, loc1=10., scale1=0.1,
                             weight0=0.5, weight1=0.5)
        xs = torch.tensor([1e-4, 1.0, 1e4])
        assert torch.all(torch.isfinite(d.log_prob(xs)))

    def test_rsample_default_shape(self):
        d = scalar_dist()
        s = d.rsample()
        assert s.shape == torch.Size([])

    def test_construction_with_tensors(self):
        d = LogNormalMixture(
            loc0=torch.tensor(0.0), scale0=torch.tensor(0.5),
            loc1=torch.tensor(2.0), scale1=torch.tensor(0.3),
            weight0=torch.tensor(0.6), weight1=torch.tensor(0.4),
        )
        lp = d.log_prob(torch.tensor(1.0))
        assert torch.isfinite(lp)
