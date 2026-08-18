"""Target transforms for the distance label.  Week 12.

WHY THIS IS ITS OWN MODULE.  `features.py` is input-side and is the file that
carried the Week 11 leak; targets are output-side.  Keeping them apart keeps the
leak audit a question about one file.  `metrics.py` needs the inverses to move
predictions into a common space, so this module imports nothing local and can be
imported from anywhere without a cycle.

THE LOG PATH MUST MATCH THE DATASET.  `sampling.label_and_pack` writes
    log_d_km = log10(maximum(d_km, FLOOR_KM))
and `log10_forward` below reproduces that clamp exactly.  If it drifts, M1 is no
longer the frozen Week 11 baseline and the 2x2 loses its control.  FLOOR_KM is
NOT hardcoded here -- it is passed in, so the caller reads it from the dataset
rather than from a constant that can fall out of sync.

`forward` CLAMPS ON BOTH PATHS.  asinh is finite at zero and does not need a
clamp to be defined, but it needs one to be CONSISTENT: censored rows must land
on z_L in whichever target is running, or the two cells in a row of the 2x2 are
censored at different places and `losses.check_censoring_consistency` rejects
the asinh cells.  Feeding raw sub-floor distances into asinh would also be
fitting the reproducibility noise, which is the thing the floor exists to
declare unmeasurable.  The low-level `asinh_forward` stays unclamped so the
odd-function behaviour is still directly testable.

DELTA IS NOT A HYPERPARAMETER AND IS NOT DEFAULTED.  asinh(d/delta) has bounded
slope 1/sqrt(d^2 + delta^2) <= 1/delta, and the point of the transform is that
the bound bites at the scale where the network's own gradient budget runs out.
Setting delta = FLOOR_KM is the tempting choice and it is WRONG: the first
decade above the floor then spans asinh(10) - asinh(1) = 2.1168 against
ln(10) = 2.3026, a compression of 8%, and every decade above that is within 0.1%.
That is a transform that changes nothing anywhere the data is uncensored.
Week 11's measured breakdown scale is the dead-zone boundary at ~125 m, with the
model's own output floor at ~6.5 m; those two are the anchors to sweep.

DELTA IS ALSO A TRADE, AND AT 125 m IT IS A STEEP ONE.  log10(125 m / 0.155 m)
= 2.91 of the target's 5.81 decades, so that delta flattens half the dynamic
range -- and flattens exactly the half the model already fails in.  A model that
emits a constant across a region it is no longer asked to resolve will score
better without predicting better.  The arbiter is therefore the near-field
factor-of-N table in physical km (`metrics.factor_table`), which is invariant to
delta; R^2 on the transformed target is not.  Report the chosen delta with a
(delta/3, delta, 3*delta) band; the unit dependence of any log-like transform
defined at zero is not removable (Chen & Roth 2024, QJE 139(2), 891-936), only
anchorable.

SIGN CONVENTION.  asinh is odd, so a negative predicted z inverts to a negative
distance.  That is meaningful as model output -- "confidently at or inside the
surface" -- and meaningless as a physical distance.  `to_log10_km` clamps at the
floor before taking a log, which is the SAME operation the labeller applies to
the truth, so predictions and labels are censored identically.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'log10_forward', 'log10_inverse',
    'asinh_forward', 'asinh_inverse',
    'forward', 'inverse', 'to_log10_km', 'threshold_in_target_units',
    'TRANSFORMS',
]


def _check_scale(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


# =========================================================================
# SECTION 1 -- THE TWO TRANSFORMS
# =========================================================================

def log10_forward(d_km, floor_km):
    """log10(d), clamped at the floor.  Reproduces `sampling.label_and_pack`.

    `maximum` rather than `clip` to match the dataset call exactly; the two
    agree here, but the point is that this line is auditable against that one.
    """
    _check_scale(floor_km, 'floor_km')
    return np.log10(np.maximum(np.asarray(d_km, dtype=float), floor_km))


def log10_inverse(z):
    """10**z.  No clamp: the caller decides whether a sub-floor prediction is
    censored or an error, and `to_log10_km` is where that decision is made."""
    return 10.0 ** np.asarray(z, dtype=float)


def asinh_forward(d_km, delta_km):
    """asinh(d/delta).  Finite at d=0, slope bounded by 1/delta everywhere.

    UNCLAMPED by design -- `forward` applies the floor.  Called directly this
    accepts negative d and returns negative z, which is the odd symmetry the
    tests exercise and which `forward` deliberately does not expose.
    """
    _check_scale(delta_km, 'delta_km')
    return np.arcsinh(np.asarray(d_km, dtype=float) / delta_km)


def asinh_inverse(z, delta_km, z_max=300.0):
    """delta * sinh(z).  Exact inverse of `asinh_forward`.

    `z_max` guards the exponential: sinh overflows float64 near z ~ 710, and a
    diverging fit would otherwise turn into inf and then nan several functions
    downstream, where it is much harder to attribute.  Real targets on this
    dataset top out near asinh(100 km / delta), so 300 never binds on a sane run
    and clipping at it is a loud signal that something upstream diverged.
    """
    _check_scale(delta_km, 'delta_km')
    z = np.asarray(z, dtype=float)
    return delta_km * np.sinh(np.clip(z, -z_max, z_max))


# =========================================================================
# SECTION 2 -- REGISTRY, CLAMPED FORWARD, COMMON-SPACE CONVERSION
# =========================================================================

TRANSFORMS = ('log10', 'asinh')


def _check(name):
    if name not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {name!r}")


def forward(d_km, transform, floor_km, delta_km=None):
    """Label -> target units, floor-clamped on BOTH paths.

    The clamp is what guarantees every censored row lands on exactly
    `threshold_in_target_units(...)`, which is the invariant
    `losses.check_censoring_consistency` uses to catch a mis-standardised z_L.
    `delta_km` is required for 'asinh'.
    """
    _check(transform)
    _check_scale(floor_km, 'floor_km')
    d = np.maximum(np.asarray(d_km, dtype=float), floor_km)
    if transform == 'log10':
        return log10_forward(d, floor_km)
    if delta_km is None:
        raise ValueError("asinh requires an explicit delta_km; see module docstring")
    return asinh_forward(d, delta_km)


def inverse(z, transform, delta_km=None):
    """Target units -> km.  May return negative km under 'asinh'; that is the
    transform being odd, not a bug.  Use `to_log10_km` for anything reported."""
    _check(transform)
    if transform == 'log10':
        return log10_inverse(z)
    if delta_km is None:
        raise ValueError("asinh requires an explicit delta_km; see module docstring")
    return asinh_inverse(z, delta_km)


def to_log10_km(z, transform, floor_km, delta_km=None):
    """THE comparability function.  Any cell's predictions -> log10 km.

    Applies the labeller's own clamp to the inverted prediction, so a model that
    predicts below the floor is scored exactly as a label below the floor is
    stored.  Without this the four cells of the 2x2 are scored in different unit
    systems and R^2 across them means nothing.
    """
    return log10_forward(inverse(z, transform, delta_km), floor_km)


def threshold_in_target_units(floor_km, transform, delta_km=None):
    """z_L, the left-censoring threshold in the units the model optimises.

    MUST be standardised by the caller with the SAME y_mean / y_std that
    `models.train_mlp` applies to the target -- which is why `train_mlp` takes
    the raw value and does it itself.  The Tobit likelihood is equivariant under
    that affine map only if z_L is carried through it; left in raw units against
    a standardised mu it produces a large constant penalty and a permanent
    downward gradient on every censored row, and reads as a bad result.
    """
    return float(forward(floor_km, transform, floor_km, delta_km))
