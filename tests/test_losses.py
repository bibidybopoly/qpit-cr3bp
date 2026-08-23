"""Tests for src/losses.py.  Runs standalone.

Three are load-bearing.  `test_reduces_to_gaussian_with_no_censoring` and
`test_reduces_to_mse_up_to_constant` are what make M1 a genuine control for the
2x2.  `test_censored_gradient_matches_mse_far_above_threshold` checks the
Mills-ratio limit the module docstring uses to predict M2b ~ M1 -- if it fails,
that prediction was reasoning from a false premise.
"""

import sys, os, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import FLOOR_KM
from losses import (gaussian_nll, CensoredGaussianNLL,
                    check_censoring_consistency)

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def _phi(t):
    return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def _Phi(t):
    return 0.5 * math.erfc(-t / math.sqrt(2.0))


def _mills(t):
    """lambda(t) = phi(t)/Phi(t), in float, as the reference value."""
    return _phi(t) / _Phi(t)


def _grad_mu(mu_val, z_L, sigma, censored=True):
    mu = torch.tensor([mu_val], requires_grad=True)
    loss = CensoredGaussianNLL(z_L=z_L, reduction='sum')
    loss(mu, torch.tensor([z_L]), torch.tensor(math.log(sigma)),
         torch.tensor([censored])).backward()
    return float(mu.grad)


# =========================================================================
# NESTING -- M1 IS A SPECIAL CASE OF M2
# =========================================================================

def test_reduces_to_gaussian_with_no_censoring():
    g = torch.Generator().manual_seed(0)
    mu, z = torch.randn(500, generator=g), torch.randn(500, generator=g)
    ls = torch.tensor(-0.3)
    got = CensoredGaussianNLL(z_L=-4.0)(mu, z, ls, torch.zeros(500, dtype=torch.bool))
    assert torch.allclose(got, gaussian_nll(mu, z, ls), rtol=1e-12)


def test_reduces_to_mse_up_to_constant():
    """With sigma fixed the Gaussian NLL is MSE/(2 sigma^2) plus a constant, so
    it has the same argmin in mu.  That is the sense in which M1 is nested."""
    g = torch.Generator().manual_seed(1)
    mu, z = torch.randn(400, generator=g), torch.randn(400, generator=g)
    mse = torch.mean((z - mu) ** 2)
    for s in (0.5, 1.0, 2.0):
        nll = gaussian_nll(mu, z, torch.tensor(math.log(s)))
        # rtol at float32 accumulation over 400 rows, not at float64.  The two
        # sides sum in different orders, so the gap sits at eps ~ 1.2e-7; a
        # 1e-12 tolerance was testing the dtype, not the identity.
        assert torch.allclose(nll, mse / (2 * s ** 2) + math.log(s) + LOG_SQRT_2PI,
                              rtol=1e-5)


def test_sigma_is_not_a_nuisance_under_censoring():
    """Under MSE sigma cannot change the argmin in mu; under censoring it can,
    because it enters the Phi term.  This is why sigma must be estimated."""
    mu, z = torch.full((64,), 0.5), torch.full((64,), -2.0)
    cens = torch.ones(64, dtype=torch.bool)
    loss = CensoredGaussianNLL(z_L=-2.0)
    assert not torch.allclose(loss(mu, z, torch.tensor(0.0), cens),
                              loss(mu, z, torch.tensor(1.0), cens))


# =========================================================================
# THE MILLS-RATIO LIMITS -- THE ARGUMENT FOR THE LOSS
# =========================================================================

def test_censored_gradient_matches_mse_far_above_threshold():
    """mu k sigmas ABOVE z_L: gradient -> (mu - z_L)/sigma^2, the MSE gradient.
    lambda(t) ~ -t - 1/t, so the relative gap closes as 1/k^2.  This is the
    limit the M2b ~ M1 prediction rests on."""
    sigma, z_L = 0.7, -1.5
    for k, tol in ((10.0, 2e-2), (20.0, 5e-3), (40.0, 2e-3)):
        g = _grad_mu(z_L + k * sigma, z_L, sigma)
        mse_grad = (k * sigma) / sigma ** 2
        assert abs(g - mse_grad) / mse_grad < tol


def test_censored_gradient_vanishes_far_below_threshold():
    """mu k sigmas BELOW z_L: no penalty for going further down.  This is the
    behaviour MSE lacks and the reason the point mass at the floor goes away."""
    sigma, z_L = 0.7, -1.5
    for k in (8.0, 15.0, 30.0):
        g = _grad_mu(z_L - k * sigma, z_L, sigma)
        assert abs(g) <= _mills(k) / sigma * 1.01 + 1e-12
        assert abs(g) < 1e-6


def test_censored_gradient_is_finite_at_the_threshold():
    """t = 0 is the branch boundary; lambda(0) = phi(0)/Phi(0) = 0.7979."""
    z_L = -1.5
    mu = torch.tensor([z_L], requires_grad=True)
    loss = CensoredGaussianNLL(z_L=z_L, reduction='sum')
    out = loss(mu, torch.tensor([z_L]), torch.tensor(0.0),
               torch.ones(1, dtype=torch.bool))
    out.backward()
    # float32: the exact value is 0.69314718056, the network dtype gives
    # 0.69314718246.  1e-9 is below the representable gap and was my error.
    assert abs(float(out) - (-math.log(0.5))) < 1e-6
    assert abs(float(mu.grad) - _mills(0.0)) < 1e-6


def test_analytic_gradient_matches_autograd_across_the_branch():
    """d(NLL)/d(mu) = +lambda(t)/sigma for a censored row, either side of and
    on the threshold."""
    sigma, z_L = 0.9, -1.0
    for k in (-6.0, -1.0, 0.0, 1.0, 6.0):
        g = _grad_mu(z_L + k * sigma, z_L, sigma)
        assert abs(g - _mills(-k) / sigma) < 1e-5 * max(1.0, abs(g))


# =========================================================================
# TAIL STABILITY
# =========================================================================

def test_log_ndtr_survives_where_log_cdf_does_not():
    """log(Phi(t)) underflows to -inf here; log_ndtr returns the asymptote
    -t^2/2 - log(-t) - 0.5 log(2 pi)."""
    for t in (-8.0, -20.0, -40.0, -100.0):
        got = float(torch.special.log_ndtr(torch.tensor(t, dtype=torch.float64)))
        expect = -0.5 * t * t - math.log(-t) - LOG_SQRT_2PI
        assert math.isfinite(got)
        assert abs(got - expect) < 2e-3 * abs(expect)


def test_no_nan_poisoning_across_branches():
    """torch.where's backward multiplies the unused branch by the mask, so a
    nan there would propagate.  Neither branch produces one; asserted with both
    branch types live and extreme values in each."""
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(256, generator=g).mul(20.0).requires_grad_(True)
    z = torch.randn(256, generator=g)
    cens = torch.arange(256) % 2 == 0
    z = torch.where(cens, torch.full_like(z, -3.0), z)
    ls = torch.tensor(-1.0, requires_grad=True)
    out = CensoredGaussianNLL(z_L=-3.0)(mu, z, ls, cens)
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(mu.grad).all() and torch.isfinite(ls.grad).all()


# =========================================================================
# THE STANDARDISATION GUARD
# =========================================================================

def test_guard_passes_on_correctly_standardised_input():
    check_censoring_consistency(torch.tensor([-2.0, -2.0, 0.3, 1.1]), -2.0,
                                torch.tensor([True, True, False, False]))


def test_guard_catches_raw_z_L_against_standardised_target():
    """The exact bug the module docstring names: z_L left in log10 km while the
    target was standardised."""
    _raises(ValueError, check_censoring_consistency,
            torch.tensor([-1.72, -1.72, 0.3, 1.1]), math.log10(FLOOR_KM),
            torch.tensor([True, True, False, False]))


def test_guard_catches_unclamped_asinh_target():
    """Censored rows scattered below z_L instead of sitting on it -- what an
    asinh target built without the floor clamp produces."""
    _raises(ValueError, check_censoring_consistency,
            torch.tensor([-2.0, -2.4, -2.1, 0.3]), -2.0,
            torch.tensor([True, True, True, False]))


def test_guard_is_a_noop_with_no_censored_rows():
    check_censoring_consistency(torch.randn(10), -99.0,
                                torch.zeros(10, dtype=torch.bool))


# =========================================================================
# GUARDS AND PLUMBING
# =========================================================================

def test_needs_sigma_attribute_distinguishes_from_mse():
    """How train_mlp routes.  nn.MSELoss has no such attribute."""
    assert getattr(CensoredGaussianNLL(z_L=0.0), 'needs_sigma', False) is True
    assert getattr(torch.nn.MSELoss(), 'needs_sigma', False) is False


def test_reductions():
    for red, shape in (('mean', ()), ('sum', ()), ('none', (7,))):
        out = CensoredGaussianNLL(z_L=0.0, reduction=red)(
            torch.zeros(7), torch.zeros(7), torch.tensor(0.0),
            torch.zeros(7, dtype=torch.bool))
        assert tuple(out.shape) == shape


def test_bad_inputs_raise():
    z = torch.zeros(3)
    _raises(ValueError, CensoredGaussianNLL(z_L=0.0, reduction='avg'),
            z, z, torch.tensor(0.0), torch.zeros(3, dtype=torch.bool))
    _raises(TypeError, CensoredGaussianNLL(z_L=0.0),
            z, z, torch.tensor(0.0), torch.zeros(3, dtype=torch.long))
    _raises(ValueError, CensoredGaussianNLL(z_L=0.0),
            z, z, torch.tensor(0.0), torch.zeros(4, dtype=torch.bool))
    for bad in (float('inf'), float('nan')):
        _raises(ValueError, gaussian_nll, z, z, torch.tensor(bad))


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
