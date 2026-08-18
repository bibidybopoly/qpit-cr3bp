"""Censored Gaussian (Tobit) likelihood.  Week 12.

THE LATENT MODEL.  z* | x ~ N(mu(x), sigma^2), observed z = max(z*, z_L).  The
labeller cannot resolve below FLOOR_KM (0.155 m, measured -- see
`sampling.py`), and `label_and_pack` clamps there, so for a row with
`at_floor` the honest statement is "d <= FLOOR", not "d = FLOOR".  MSE asserts
the latter and creates a point mass at the bottom of the target range that the
model must fit.  This asserts the former.

    uncensored   log sigma + (z - mu)^2 / (2 sigma^2) + 0.5 log(2 pi)
    censored     -log Phi((z_L - mu) / sigma)

M1 IS THE c == 0, FIXED-SIGMA CASE.  With no censored rows this reduces to the
Gaussian NLL, and with sigma held fixed that is MSE/(2 sigma^2) plus a constant.
The 2x2 therefore differs from the frozen Week 11 baseline in exactly the two
intended ways and nothing else.

WHAT THE GRADIENT DOES, AND WHY THAT PREDICTS M2b ~ M1.  With
t = (z_L - mu)/sigma and lambda = phi/Phi the inverse Mills ratio, a censored
row contributes d(NLL)/d(mu) = +lambda(t)/sigma.  Two limits:

    mu >> z_L   t -> -inf,  lambda(t) -> -t,  gradient -> (mu - z_L)/sigma^2
    mu << z_L   t -> +inf,  lambda(t) -> 0,   gradient -> 0

The first is IDENTICALLY the MSE gradient.  So Tobit differs from MSE only once
mu approaches the floor.  Week 11's dead-zone constant is 6.5 m against a
0.155 m floor -- 42x above -- so at the current operating point this loss is
numerically almost the same as the one it replaces, and M2b should land within
the seed spread of M1.  Tobit earns its keep only in combination with a target
that lets mu reach the floor; the interaction M2c - M2a is where to look, not
M2b - M1.  If M2b moves substantially on its own, that reasoning is wrong and
that is the interesting outcome.

z_L MUST BE STANDARDISED.  `models.train_mlp` optimises (z - y_mean)/y_std, so
mu and sigma live in standardised units and z_L must be carried through the same
affine map.  The likelihood is equivariant when it is: the censored term is
exactly invariant and the uncensored term shifts by a constant per row.  Left in
raw units against a standardised mu, every censored row contributes a large
constant penalty and a permanent downward gradient -- the run completes, reports
no error, and reads as "Tobit made it worse".  `check_censoring_consistency`
below is the guard, and it is cheap enough to call once per fit.

torch.where IS SAFE HERE.  Its backward multiplies the unused branch's gradient
by the mask, so a nan in an unused branch would poison the result.  Neither
branch produces nan for finite inputs: log_ndtr is finite for every finite t and
the quadratic is finite for sigma > 0.  That is a property of these two
expressions, not of torch.where, so it is asserted in the tests.

SIGMA IS NOT CLAMPED.  A hard clamp zeroes the gradient outside the bound and
freezes the parameter there permanently, which is a worse failure than the one
it prevents.  With a single global sigma and the initialisation recommended in
`models.train_mlp` (0.5 * log of M1's best val loss, since that loss is MSE on
standardised targets and its square root is the residual SD in sigma's own
units) the inflation pathology does not arise.  Non-finite sigma raises.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ['gaussian_nll', 'CensoredGaussianNLL', 'check_censoring_consistency']

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _reduce(x, reduction):
    if reduction == 'mean':
        return x.mean()
    if reduction == 'sum':
        return x.sum()
    if reduction == 'none':
        return x
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def gaussian_nll(mu, z_obs, log_sigma, reduction='mean'):
    """Plain Gaussian NLL.  The sigma-matched control for the 2x2.

    Written out independently rather than as a call into `CensoredGaussianNLL`
    with an empty mask, so that the test asserting the two agree compares two
    implementations and not one implementation with itself.
    """
    if not torch.isfinite(log_sigma).all():
        raise ValueError("log_sigma is not finite")
    sigma = torch.exp(log_sigma)
    r = (z_obs - mu) / sigma
    return _reduce(log_sigma + 0.5 * r ** 2 + _LOG_SQRT_2PI, reduction)


class CensoredGaussianNLL(nn.Module):
    """Left-censored Gaussian NLL at a fixed, already-standardised threshold.

    `needs_sigma` is how `models.train_mlp` tells this apart from
    `nn.MSELoss()` -- an explicit attribute rather than an isinstance check or
    an argument-count probe, both of which fail silently when a third loss is
    added later.
    """

    needs_sigma = True

    def __init__(self, z_L, reduction='mean'):
        super().__init__()
        z_L = float(z_L)
        if not math.isfinite(z_L):
            raise ValueError(f"z_L must be finite, got {z_L}")
        self.z_L = z_L
        self.reduction = reduction

    def forward(self, mu, z_obs, log_sigma, censored):
        if not torch.isfinite(log_sigma).all():
            raise ValueError("log_sigma is not finite")
        if censored.dtype != torch.bool:
            raise TypeError(f"censored must be bool, got {censored.dtype}")
        if censored.shape != mu.shape:
            raise ValueError(f"shape mismatch {censored.shape} vs {mu.shape}")

        sigma = torch.exp(log_sigma)

        r = (z_obs - mu) / sigma
        uncens = log_sigma + 0.5 * r ** 2 + _LOG_SQRT_2PI

        # log Phi via log_ndtr: accurate into the far left tail, where
        # log(cdf(t)) underflows to -inf around t ~ -8 and takes the gradient
        # with it.
        t = (self.z_L - mu) / sigma
        cens = -torch.special.log_ndtr(t)

        return _reduce(torch.where(censored, cens, uncens), self.reduction)


def check_censoring_consistency(z_obs, z_L, censored, atol=1e-5):
    """THE standardisation guard.  Call once per fit, not per batch.

    Every censored row carries the clamped label, and the clamp value IS z_L.
    So in whatever units the target is expressed, censored rows must satisfy
    z_obs == z_L exactly.  If z_L was left in raw units while z_obs was
    standardised (or the two used different y_mean / y_std), this fails
    immediately and loudly instead of surfacing as a bad result three days
    later.  It also catches an asinh target built without the floor clamp,
    which would leave censored rows scattered below z_L rather than on it.
    """
    if not censored.any():
        return
    z_c = z_obs[censored]
    dev = float((z_c - z_L).abs().max())
    if dev > atol:
        raise ValueError(
            f"censored rows do not sit on z_L: max deviation {dev:.6g} "
            f"(z_L={z_L:.6g}, censored z_obs range "
            f"[{float(z_c.min()):.6g}, {float(z_c.max()):.6g}]).  "
            "Check that z_L was standardised with the same y_mean/y_std as the "
            "target, and that the asinh target clamps at floor_km."
        )
