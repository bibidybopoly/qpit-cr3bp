"""Tests for src/targets.py.  No CR3BP imports; runs standalone.

The load-bearing tests are `test_log10_forward_matches_dataset_clamp` -- if it
fails, M1 is not the Week 11 baseline and the 2x2 has no control -- and
`test_forward_clamps_on_the_asinh_path_too`, which is the invariant
`losses.check_censoring_consistency` relies on.  Everything else checks a claim
made in the module docstring, so a failure means the docstring is wrong, not
just the code.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from targets import (log10_forward, log10_inverse, asinh_forward, asinh_inverse,
                     forward, inverse, to_log10_km, threshold_in_target_units)

FLOOR_KM = 1.55e-4          # must equal sampling.FLOOR_KM
DELTAS = (1.55e-4, 6.5e-3, 0.125, 1.0)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


# =========================================================================
# THE DATASET CONTRACT
# =========================================================================

def test_log10_forward_matches_dataset_clamp():
    """Byte-for-byte agreement with `sampling.label_and_pack`'s target line."""
    rng = np.random.default_rng(0)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM) - 1, 2, 5000)
    assert np.array_equal(log10_forward(d, FLOOR_KM),
                          np.log10(np.maximum(d, FLOOR_KM)))


def test_log10_forward_clamps_below_floor():
    d = np.array([0.0, 1e-9, FLOOR_KM * 0.5, FLOOR_KM])
    assert np.all(log10_forward(d, FLOOR_KM) == np.log10(FLOOR_KM))


def test_at_floor_boundary_is_the_known_edge_case():
    """`at_floor` uses strict `<` while the clamp uses `maximum`, so a row at
    exactly FLOOR_KM gets the clamped target with at_floor False.  Documented,
    measure-zero, asserted here so it stays deliberate."""
    assert log10_forward(np.array([FLOOR_KM]), FLOOR_KM)[0] == np.log10(FLOOR_KM)
    assert not bool(FLOOR_KM < FLOOR_KM)


# =========================================================================
# THE CLAMP ON BOTH PATHS
# =========================================================================

def test_forward_clamps_on_the_asinh_path_too():
    """Every sub-floor row must land on exactly z_L in whichever target is
    running.  Without this the asinh cells scatter censored rows below z_L and
    `losses.check_censoring_consistency` rejects them."""
    for delta in DELTAS:
        z_L = threshold_in_target_units(FLOOR_KM, 'asinh', delta)
        z = forward(np.array([0.0, -1.0, FLOOR_KM * 1e-3, FLOOR_KM]),
                    'asinh', FLOOR_KM, delta)
        assert np.allclose(z, z_L, rtol=0.0, atol=1e-15)


def test_forward_clamp_leaves_above_floor_values_alone():
    rng = np.random.default_rng(9)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM) + 0.5, 2, 500)
    assert np.allclose(forward(d, 'asinh', FLOOR_KM, 1e-2),
                       asinh_forward(d, 1e-2), rtol=1e-15)


def test_asinh_forward_stays_unclamped_when_called_directly():
    """`forward` clamps; the primitive does not.  The odd-symmetry tests need
    the primitive, and conflating the two would hide which layer owns the
    floor."""
    assert asinh_forward(-1.0, 1e-3) == -asinh_forward(1.0, 1e-3)
    assert asinh_forward(0.0, 1e-3) == 0.0


# =========================================================================
# ROUND TRIP
# =========================================================================

def test_asinh_round_trip():
    rng = np.random.default_rng(1)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM), 2, 5000)
    for delta in DELTAS:
        back = asinh_inverse(asinh_forward(d, delta), delta)
        assert np.allclose(back, d, rtol=1e-12, atol=0.0)


def test_asinh_round_trip_at_and_below_zero():
    """asinh is odd and defined at zero -- the property log10 lacks."""
    d = np.array([-1.0, -1e-6, 0.0, 1e-6, 1.0])
    assert np.allclose(asinh_inverse(asinh_forward(d, 1e-3), 1e-3), d,
                       rtol=1e-12, atol=1e-300)


def test_log10_round_trip_above_floor():
    rng = np.random.default_rng(2)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM) + 0.1, 2, 5000)
    assert np.allclose(log10_inverse(log10_forward(d, FLOOR_KM)), d, rtol=1e-12)


# =========================================================================
# THE ASYMPTOTIC CLAIMS
# =========================================================================

def test_asinh_far_field_is_log_up_to_a_constant():
    """asinh(x) -> ln(2x); the residual falls as 1/(4x^2)."""
    for x in (1e3, 1e4, 1e6):
        assert abs(np.arcsinh(x) - np.log(2 * x)) < 1.0 / (3.0 * x ** 2)


def test_asinh_near_field_is_linear():
    """asinh(x) = x - x^3/6 + 3x^5/40 - ...  Checked against the NEXT term
    rather than a flat tolerance, which tests the series instead of a magic
    number -- a flat 1e-12 fails at x = 1e-2, where 3x^5/40 is 7.5e-12."""
    for x in (1e-4, 1e-3, 1e-2):
        assert abs(np.arcsinh(x) - (x - x ** 3 / 6.0)) <= 1.01 * 3 * x ** 5 / 40


def test_delta_equal_floor_is_the_documented_null():
    """The arithmetic behind 'delta = FLOOR is wrong': with delta = floor the
    first decade above the floor compresses by ~8% and later decades by ~0.1%,
    i.e. the transform does essentially nothing where the data is uncensored.
    This test exists so that reasoning is checked, not just asserted."""
    first = np.arcsinh(10.0) - np.arcsinh(1.0)
    second = np.arcsinh(100.0) - np.arcsinh(10.0)
    assert abs(first / np.log(10.0) - 0.919) < 5e-3       # ~8% compression
    assert abs(second / np.log(10.0) - 1.000) < 2e-3      # ~0.1%


def test_boundary_delta_flattens_half_the_range():
    """The cost of the anchor actually chosen: delta = 125 m sits 2.91 of the
    target's 5.81 decades above the floor.  If this ratio drifts, the arbiter
    column in `metrics.factor_table` is measuring a different thing."""
    assert abs(np.log10(0.125 / FLOOR_KM) - 2.907) < 5e-3
    assert abs(np.log10(100.0 / FLOOR_KM) - 5.810) < 5e-3


# =========================================================================
# THE BOUNDED-SLOPE CLAIM -- THE REASON FOR THE TRANSFORM
# =========================================================================

def test_asinh_slope_is_bounded_by_one_over_delta():
    """|dz/dd| = 1/sqrt(d^2 + delta^2) <= 1/delta, checked against the analytic
    form and against the bound.  This is the property log10 does not have."""
    d = 10.0 ** np.linspace(np.log10(FLOOR_KM), 2, 400)
    h = d * 1e-7
    for delta in (1e-4, 1e-2, 1.0):
        num = (asinh_forward(d + h, delta) - asinh_forward(d - h, delta)) / (2 * h)
        assert np.allclose(num, 1.0 / np.sqrt(d ** 2 + delta ** 2), rtol=1e-6)
        assert np.all(np.abs(num) <= 1.0 / delta + 1e-9)


def test_log10_slope_is_unbounded_near_zero():
    """The failure mode P3 predicts: grad(log d) ~ 1/(d ln10) diverges, so the
    bound the network can supply is exceeded at small d for ANY finite budget."""
    d = np.array([1e-2, 1e-4, 1e-6])
    slope = 1.0 / (d * np.log(10.0))
    assert slope[-1] > 100 * slope[0]


def test_asinh_is_strictly_monotone():
    d = 10.0 ** np.linspace(np.log10(FLOOR_KM), 2, 2000)
    assert np.all(np.diff(asinh_forward(d, 1e-3)) > 0)


# =========================================================================
# COMMON SPACE AND THRESHOLD
# =========================================================================

def test_to_log10_km_is_identity_on_the_log10_path():
    rng = np.random.default_rng(3)
    z = rng.uniform(np.log10(FLOOR_KM), 2, 1000)
    assert np.allclose(to_log10_km(z, 'log10', FLOOR_KM), z, rtol=1e-12)


def test_to_log10_km_agrees_across_transforms():
    """Same distances, two parameterisations, one reported space."""
    rng = np.random.default_rng(4)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM) + 0.1, 2, 2000)
    a = to_log10_km(forward(d, 'log10', FLOOR_KM), 'log10', FLOOR_KM)
    b = to_log10_km(forward(d, 'asinh', FLOOR_KM, 1e-3), 'asinh', FLOOR_KM, 1e-3)
    assert np.allclose(a, b, rtol=1e-10)


def test_negative_asinh_prediction_censors_rather_than_nans():
    """An odd transform lets the model predict negative distance.  That must
    come back as the floor, exactly as a sub-floor label is stored -- not nan."""
    out = to_log10_km(np.array([-5.0, -1.0, 0.0]), 'asinh', FLOOR_KM, 1e-3)
    assert np.all(np.isfinite(out)) and np.all(out == np.log10(FLOOR_KM))


def test_threshold_in_target_units():
    assert abs(threshold_in_target_units(FLOOR_KM, 'log10')
               - np.log10(FLOOR_KM)) < 1e-12
    # delta = floor gives the tidy asinh(1); recorded because it is the value
    # that looks right and is not the one to use.
    assert abs(threshold_in_target_units(FLOOR_KM, 'asinh', FLOOR_KM)
               - np.log(1.0 + np.sqrt(2.0))) < 1e-12
    assert abs(threshold_in_target_units(FLOOR_KM, 'asinh', 1e-2)
               - np.arcsinh(FLOOR_KM / 1e-2)) < 1e-12


# =========================================================================
# GUARDS
# =========================================================================

def test_asinh_requires_explicit_delta():
    _raises(ValueError, forward, np.array([1.0]), 'asinh', FLOOR_KM)
    _raises(ValueError, inverse, np.array([1.0]), 'asinh')


def test_bad_transform_name_raises():
    _raises(ValueError, forward, np.array([1.0]), 'log', FLOOR_KM)


def test_nonpositive_scales_raise():
    for bad in (0.0, -1.0, np.nan, np.inf):
        _raises(ValueError, log10_forward, np.array([1.0]), bad)
        _raises(ValueError, asinh_forward, np.array([1.0]), bad)
        _raises(ValueError, forward, np.array([1.0]), 'asinh', bad, 1e-3)


def test_sinh_guard_does_not_bind_on_realistic_targets():
    """100 km at the smallest defensible delta is nowhere near the clip."""
    assert asinh_forward(100.0, FLOOR_KM) < 20.0


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
