"""Tests for the observables, window bound, calibration and label rule in
src/labeling.py.

Run: pytest tests/test_labeling_observables.py -v
     (or python tests/test_labeling_observables.py)

`tests/test_labeling.py` certifies the two NAFF indicators against analytic
signals and the Chirikov map, which is the right way round: the indicator has
to be trustworthy before it is pointed at the tori.  But it covers only
`naff_frequency` and `frequency_drift` — 9 of the module's 11 public functions
were untested, including everything between the indicator and the label:

    curve_plane -> stroboscopic_signal -> frequency_drift -> qp_label
                                       calibrate_thresholds

Those are the steps that decide the gap experiment's outcome.  A wrong
projection plane, a threshold calibrated off the wrong rows, or a disjunctive
label rule all produce a plausible pass rate rather than an error.

Everything here is analytic or synthetic — no torus is integrated — so the file
is fast and independent of the CR3BP.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import labeling as L                                            # noqa: E402
from labeling import (_hann, naff_frequency, curve_plane,        # noqa: E402
                      stroboscopic_signal, canonical_signals,
                      deviation_growth, max_usable_periods,
                      calibrate_thresholds, qp_label)
from qpit import node_angles                                     # noqa: E402


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def _curve_with_plane(a, b, N=31, centre=None, extra_k2=0.0):
    """A curve whose k=1 POSITION mode is exactly span{a, b}.

    For v(theta) = a cos(theta) + b sin(theta) the k=+1 DFT coefficient is
    a/2 - i b/2, so `curve_plane` sees Re = a/2 and Im = -b/2.
    """
    th = node_angles(N)
    U = np.zeros((N, 6))
    U[:, :3] = np.outer(np.cos(th), a) + np.outer(np.sin(th), b)
    if centre is not None:
        U[:, :3] += np.asarray(centre, float)
    if extra_k2:
        U[:, :3] += extra_k2 * np.outer(np.cos(2 * th), np.array([1.0, 1.0, 1.0]))
    U[:, 3:] = 0.37                       # velocities must be ignored
    return U


# ---------------------------------------------------------------------------
# 1. The window — a claim the module makes about itself
# ---------------------------------------------------------------------------

def test_hann_window_integrates_to_the_window_length():
    """`_hann`'s docstring says "Integrates to T over the window ... checked in
    test_labeling".  It was NOT checked anywhere; this is that check.

    sum(w) == n is what makes (1/T) * sum(f*w) * dt an unbiased amplitude:
    with T = n*dt the expression is mean(f*w), which returns A exactly for a
    constant signal only if the window has unit mean.
    """
    for n in (16, 101, 4096):
        w = _hann(n)
        assert w.shape == (n,)
        assert abs(w.sum() / n - 1.0) < 1e-12, "window mean is not 1"
        assert np.all(w >= -1e-15), "Hann window must be non-negative"
        assert abs(w[0]) < 1e-12, "window must vanish at the left edge"
        # peaks at 2 in the middle; an odd n has no sample exactly at the
        # centre, so the discrete max approaches 2 without reaching it
        assert w.max() <= 2.0 + 1e-12
        assert w.max() > 2.0 - 10.0 / n

    # the unbiasedness the docstring actually claims, for a constant signal
    n = 2048
    w = _hann(n)
    A = 3.25
    assert abs(np.mean(np.full(n, A) * w) - A) < 1e-12


# ---------------------------------------------------------------------------
# 2. The projection plane
# ---------------------------------------------------------------------------

def test_curve_plane_recovers_a_known_orthonormal_pair():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    e1, e2, cond = curve_plane(_curve_with_plane(a, b))
    assert abs(np.linalg.norm(e1) - 1) < 1e-12
    assert abs(np.linalg.norm(e2) - 1) < 1e-12
    assert abs(e1 @ e2) < 1e-12
    assert abs(cond - 1.0) < 1e-12, "orthogonal Re/Im must give conditioning 1"
    assert np.allclose(np.abs(e1), np.abs(a), atol=1e-12)
    assert np.allclose(np.abs(e2), np.abs(b), atol=1e-12)


def test_curve_plane_spans_the_k1_mode_at_any_orientation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.standard_normal(3), rng.standard_normal(3)
        e1, e2, cond = curve_plane(_curve_with_plane(a, b))
        # both raw directions must lie in span{e1, e2}
        P = np.eye(3) - np.outer(e1, e1) - np.outer(e2, e2)
        assert np.linalg.norm(P @ a) < 1e-10
        assert np.linalg.norm(P @ b) < 1e-10
        assert abs(e1 @ e2) < 1e-12
        assert 0.0 < cond <= 1.0 + 1e-12


def test_curve_plane_is_blind_to_the_centre_and_to_velocity():
    """The plane must depend on the k=1 SHAPE only.  A centre offset is the
    k=0 mode, and velocities are excluded on purpose -- mixing them would
    weight the projection by an arbitrary choice of units."""
    a, b = np.array([1.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.3])
    base = curve_plane(_curve_with_plane(a, b))
    shifted = curve_plane(_curve_with_plane(a, b, centre=[1.15, -0.4, 0.02]))
    for x, y in zip(base[:2], shifted[:2]):
        assert np.allclose(x, y, rtol=0, atol=1e-12)

    U = _curve_with_plane(a, b)
    U2 = U.copy(); U2[:, 3:] *= -17.0
    for x, y in zip(base[:2], curve_plane(U2)[:2]):
        assert np.allclose(x, y, rtol=0, atol=1e-12)


def test_curve_plane_conditioning_is_the_sine_of_the_raw_angle():
    """`conditioning` is what tells the caller to fall back, so it has to mean
    what the docstring says rather than merely be small when things are bad."""
    a = np.array([1.0, 0.0, 0.0])
    for ang in (np.pi / 2, np.pi / 3, np.pi / 6, 0.05):
        b = np.array([np.cos(ang), np.sin(ang), 0.0])
        _, _, cond = curve_plane(_curve_with_plane(a, b))
        assert abs(cond - abs(np.sin(ang))) < 1e-9, (ang, cond)


def test_curve_plane_refuses_a_degenerate_mode_instead_of_guessing():
    a = np.array([1.0, 0.0, 0.0])
    # no k=1 position content at all
    _raises(ValueError, curve_plane,
            _curve_with_plane(np.zeros(3), np.zeros(3), extra_k2=1.0))
    # k=1 mode nearly linear: Re and Im parallel
    _raises(ValueError, curve_plane, _curve_with_plane(a, a * (1 + 1e-16)))


# ---------------------------------------------------------------------------
# 3. The stroboscopic signal — where rho comes from
# ---------------------------------------------------------------------------

def test_stroboscopic_signal_projects_onto_the_plane_and_drops_the_centre():
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    centre = np.array([1.15, -0.3, 0.02, 0.0, 0.0, 0.0])
    X = np.zeros((5, 6))
    X[:, :3] = centre[:3] + np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0],
                                      [0, -1, 0], [2, 3, 9]])
    X[:, 3:] = 99.0                                   # must be ignored
    f = stroboscopic_signal(X, centre, e1, e2)
    assert np.allclose(f, [1, 1j, -1, -1j, 2 + 3j], rtol=0, atol=1e-12)


def test_stroboscopic_signal_of_a_rotating_point_is_a_pure_tone_at_rho():
    """The end-to-end claim the primary score rests on: sampled at multiples
    of T the motion is theta_n = theta_0 + n*rho, so the projected signal is a
    single tone and NAFF returns rho itself.

    A centre that was not subtracted would put the peak at zero frequency --
    which is the failure the docstring warns about, and it is invisible unless
    the recovered frequency is compared to the rho that went in.
    """
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    centre = np.array([1.15, -0.3, 0.02, 0.0, 0.0, 0.0])
    for rho in (0.4, 1.842, 2.9):
        n = 4000
        ph = rho * np.arange(n)
        X = np.zeros((n, 6))
        X[:, 0] = centre[0] + 0.01 * np.cos(ph)
        X[:, 1] = centre[1] + 0.01 * np.sin(ph)
        X[:, 2] = centre[2]
        f = stroboscopic_signal(X, centre, e1, e2)
        w, amp = naff_frequency(f, 1.0)      # returns a TUPLE, not a dict
        assert abs(np.mod(w, 2 * np.pi) - rho) < 1e-6, (rho, w)
        # and the recovered amplitude is the circle radius, unbiased by the
        # window -- which is what `_hann`'s unit mean buys
        assert abs(amp - 0.01) < 1e-6, amp


# ---------------------------------------------------------------------------
# 4. Canonical signals
# ---------------------------------------------------------------------------

def test_canonical_signals_use_the_rotating_frame_momenta():
    """px = vx - y, py = vy + x, pz = vz.  Dropping the frame terms silently
    changes which frequencies the dense cross-check sees."""
    rng = np.random.default_rng(1)
    n = 500
    X = rng.standard_normal((n, 6))
    out = canonical_signals(X)
    x, y, z, vx, vy, vz = X.T
    for sig, (q, p) in zip(out, [(x, vx - y), (y, vy + x), (z, vz)]):
        qn = (q - q.mean()) / q.std()
        pn = (p - p.mean()) / p.std()
        assert np.allclose(sig.real, qn, rtol=0, atol=1e-12)
        assert np.allclose(sig.imag, pn, rtol=0, atol=1e-12)


def test_canonical_signals_are_mean_removed_and_variance_matched():
    """A q + i*p with mismatched amplitudes is a MIXTURE of e^{+iwt} and
    e^{-iwt} rather than a clean tone, which raises the leakage floor."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 6)) * np.array([1, 1e3, 1, 1e-3, 1, 1])
    for s in canonical_signals(X):
        assert abs(s.real.mean()) < 1e-12 and abs(s.imag.mean()) < 1e-12
        assert abs(s.real.std() - 1.0) < 1e-12
        assert abs(s.imag.std() - 1.0) < 1e-12


def test_canonical_signals_return_None_for_a_degenerate_pair():
    """z == 0 on a planar orbit: returning None is what lets qp_diagnostics
    drop that pair instead of dividing by zero and propagating nan into the
    max over pairs, which would make every planar orbit unlabelled."""
    n = 300
    th = np.linspace(0, 20 * np.pi, n)
    X = np.zeros((n, 6))
    X[:, 0], X[:, 1] = np.cos(th), np.sin(th)
    X[:, 3], X[:, 4] = -np.sin(th), np.cos(th)
    out = canonical_signals(X)
    assert out[2] is None, "planar orbit should give no z-pair signal"
    assert out[0] is not None and out[1] is not None
    assert len([s for s in out if s is not None]) == 2


def test_canonical_signal_scaling_cannot_move_a_frequency():
    """"a linear change of the observable ... cannot change which frequencies
    are present, only their relative weight" -- pin that."""
    n, w0 = 3000, 0.317
    t = np.arange(n)
    X = np.zeros((n, 6))
    X[:, 0], X[:, 3] = np.cos(w0 * t), -np.sin(w0 * t)
    X[:, 1], X[:, 4] = np.sin(w0 * t), np.cos(w0 * t)
    X[:, 2], X[:, 5] = 0.5 * np.cos(w0 * t), -0.5 * np.sin(w0 * t)
    a = canonical_signals(X)[2]
    Xs = X.copy(); Xs[:, 2] *= 1e4; Xs[:, 5] *= 1e4
    b = canonical_signals(Xs)[2]
    assert np.allclose(a, b, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Deviation growth
# ---------------------------------------------------------------------------

def test_deviation_growth_is_about_one_for_a_bounded_orbit():
    th = np.linspace(0, 40 * np.pi, 401)
    X = np.zeros((401, 6))
    X[:, 0], X[:, 1] = np.cos(th), np.sin(th)
    ratio, mx = deviation_growth(X, np.zeros(6))
    assert abs(ratio - 1.0) < 1e-9
    assert abs(mx - 1.0) < 1e-9


def test_deviation_growth_detects_exponential_escape():
    n = 200
    X = np.zeros((n, 6))
    X[:, 0] = np.exp(np.arange(n) * np.log(1.5))
    ratio, mx = deviation_growth(X, np.zeros(6))
    assert ratio > 1e10
    assert mx == X[-1, 0]


def test_deviation_growth_uses_positions_only_and_splits_at_the_midpoint():
    n = 100
    X = np.zeros((n, 6))
    X[:, 0] = 1.0
    X[n // 2:, 0] = 4.0                     # steps up exactly at the split
    X[:, 3:] = 1e6                          # velocities must not count
    ratio, mx = deviation_growth(X, np.zeros(6))
    assert abs(ratio - 4.0) < 1e-12
    assert abs(mx - 4.0) < 1e-12


def test_deviation_growth_splits_exactly_in_half():
    """A step function cannot tell a 1/2 split from a 1/3 split -- both put the
    step in the late window and both return 4.  A linear ramp can: the early
    max IS the split index, so the ratio reads it off directly.

    The split matters because the two halves are what "late over early" means;
    an uneven split biases every ratio in the dataset in the same direction.
    """
    n = 100
    X = np.zeros((n, 6))
    X[:, 0] = np.arange(1, n + 1)          # d[j] = j + 1, strictly increasing
    ratio, mx = deviation_growth(X, np.zeros(6))
    assert abs(ratio - n / (n // 2)) < 1e-12, ratio     # 100/50 = 2.0 at h=n//2
    assert abs(mx - n) < 1e-12
    # a third-split would read 100/33 = 3.03, a two-thirds split 100/67 = 1.49
    assert abs(ratio - n / (n // 3)) > 0.5
    assert abs(ratio - n / (2 * n // 3)) > 0.4


def test_deviation_growth_survives_a_zero_early_half():
    X = np.zeros((10, 6))
    X[5:, 0] = 1.0
    ratio, _ = deviation_growth(X, np.zeros(6))
    assert np.isfinite(ratio) and ratio > 0


# ---------------------------------------------------------------------------
# 6. The shadowing bound — a hard ceiling, not a compute budget
# ---------------------------------------------------------------------------

def test_max_usable_periods_matches_its_formula():
    for R, eps, safety in ((1212.0, 1e-12, 0.5), (146.4, 1e-12, 0.5),
                           (10.0, 1e-10, 1.0), (2.0, 1e-14, 0.25)):
        assert abs(max_usable_periods(R, eps, safety)
                   - safety * np.log(1 / eps) / np.log(R)) < 1e-12


def test_max_usable_periods_is_infinite_for_a_stable_torus():
    for R in (1.0, 0.5, 0.0):
        assert max_usable_periods(R) == np.inf


def test_max_usable_periods_falls_as_the_torus_gets_more_unstable():
    prev = np.inf
    for R in (1.5, 10.0, 146.4, 1212.0, 1e5):
        n = max_usable_periods(R)
        assert n < prev
        prev = n


def test_default_safety_factor_halves_the_documented_bound_KNOWN_DEFECT():
    """DOCUMENTATION MISMATCH, pinned so it is not mistaken for a bug.

    NAMED `_KNOWN_DEFECT` (2026-09-11).  It was a pin by content but not by
    name, so `grep -r "KNOWN_DEFECT\|KNOWN_GAP" tests/` missed it and the
    register in docs/CUTS_AND_DISPOSITIONS.md concluded there was one pin
    where there are four.  A pin nobody can enumerate is not a pin.

    The docstring gives the formula as n_max = log(1/eps)/log(R_u) and then
    says "Measured on the L2 halo family at C ~ 3.15: R_u = 1212, giving
    n_max = 3.9 periods at rtol 1e-12".  That 3.9 is the formula WITHOUT the
    safety factor.  The function defaults to safety=0.5, so it returns 1.95 --
    and the prose never mentions `safety`.  Anyone sizing a NAFF window from
    the docstring gets twice what the code allows.
    """
    bare = np.log(1 / 1e-12) / np.log(1212.0)
    assert abs(bare - 3.89) < 0.01, "the docstring's 3.9 no longer reproduces"
    assert abs(max_usable_periods(1212.0) - 0.5 * bare) < 1e-12
    assert abs(max_usable_periods(1212.0, safety=1.0) - bare) < 1e-12


# ---------------------------------------------------------------------------
# 7. Calibration — the step that makes the thresholds defensible
# ---------------------------------------------------------------------------

def _diag(ds, dd, dev=1.0, ok=True, reason=''):
    return dict(ok=ok, reason=reason, drift_strobe=ds, drift_dense=dd,
                dev_ratio=dev)


def test_thresholds_are_the_control_quantile_times_the_margin():
    ctrl = [_diag(1e-9 * i, 2e-9 * i) for i in range(1, 101)]
    thr = calibrate_thresholds(ctrl, quantile=99.0, margin=10.0)
    exp_s = np.percentile([d['drift_strobe'] for d in ctrl], 99.0) * 10.0
    exp_d = np.percentile([d['drift_dense'] for d in ctrl], 99.0) * 10.0
    assert abs(thr['drift_strobe'] - exp_s) < 1e-18
    assert abs(thr['drift_dense'] - exp_d) < 1e-18
    assert thr['n_control'] == 100
    assert thr['quantile'] == 99.0 and thr['margin'] == 10.0


def test_calibration_ignores_unusable_control_rows():
    """A control row that failed to integrate, or escaped to a nan drift, must
    not enter the quantile -- including it would drag the threshold to nan or
    to a value set by the failure rather than by the noise floor."""
    good = [_diag(1e-9, 2e-9) for _ in range(50)]
    bad = ([_diag(np.nan, np.nan) for _ in range(20)]
           + [_diag(1.0, 1.0, ok=False) for _ in range(20)]
           + [_diag(np.inf, np.inf) for _ in range(5)])
    thr = calibrate_thresholds(good + bad)
    assert abs(thr['drift_strobe'] - 1e-8) < 1e-20
    assert abs(thr['drift_dense'] - 2e-8) < 1e-20
    assert thr['n_control'] == 95, "n_control counts rows supplied, not rows used"


def test_calibration_refuses_to_invent_a_threshold_from_nothing():
    _raises(ValueError, calibrate_thresholds, [])
    _raises(ValueError, calibrate_thresholds,
            [_diag(np.nan, np.nan) for _ in range(10)])
    _raises(ValueError, calibrate_thresholds,
            [_diag(1.0, 1.0, ok=False) for _ in range(10)])


def test_dev_ratio_threshold_is_hard_coded_not_calibrated_KNOWN_GAP():
    """PINS AN INCONSISTENCY between the module's claim and its code.

    The module docstring says "thresholds are CALIBRATED from a positive
    control, never hard-coded (see calibrate_thresholds)".  Two of the three
    are.  `dev_ratio` is the literal 3.0, independent of the control, of the
    window length and of the tolerances -- the same quantities the docstring
    argues a threshold must be re-measured against.

    Not necessarily wrong: a ratio of 3 is a wide margin for something that is
    ~1 on a torus and astronomically large off it.  But it is not calibrated,
    and the docstring says all of them are.  Pinned so the claim and the code
    are reconciled deliberately rather than by accident.
    """
    a = calibrate_thresholds([_diag(1e-9, 1e-9, dev=1.0) for _ in range(30)])
    b = calibrate_thresholds([_diag(1e-3, 1e-3, dev=2.9) for _ in range(30)],
                             quantile=50.0, margin=1.0)
    assert a['dev_ratio'] == 3.0
    assert b['dev_ratio'] == 3.0, "dev_ratio moved -- it may now be calibrated"
    assert a['drift_strobe'] != b['drift_strobe'], "the other two DO respond"


# ---------------------------------------------------------------------------
# 8. The label rule
# ---------------------------------------------------------------------------

THR = dict(drift_strobe=1e-8, drift_dense=1e-7, dev_ratio=3.0)


def test_label_is_None_when_the_measurement_failed():
    """None is 'unmeasured', NOT 'negative'.  Collapsing the two would count
    every integration failure as evidence against quasi-periodicity and bias
    the treatment arm downward."""
    assert qp_label(_diag(0.0, 0.0, ok=False, reason='integration_failed'),
                    THR) is None
    assert qp_label(_diag(0.0, 0.0, ok=False, reason='nonfinite'), THR) is None


def test_escape_is_a_negative_not_a_failure():
    d = _diag(np.nan, np.nan, dev=1e12, reason='escaped')
    assert qp_label(d, THR) is False


def test_label_is_conjunctive_every_indicator_must_pass():
    """Each indicator covers the other's blind spot, so ANY of them failing
    has to sink the label.  A disjunctive rule would let a bounded-chaos orbit
    through on its deviation ratio alone."""
    assert qp_label(_diag(1e-9, 1e-9, 1.0), THR) is True
    assert qp_label(_diag(1e-7, 1e-9, 1.0), THR) is False      # strobe fails
    assert qp_label(_diag(1e-9, 1e-6, 1.0), THR) is False      # dense fails
    assert qp_label(_diag(1e-9, 1e-9, 9.9), THR) is False      # deviation fails
    assert qp_label(_diag(1e-7, 1e-6, 9.9), THR) is False


def test_label_boundary_is_inclusive():
    """<= not <: a score exactly at the threshold passes.  The thresholds come
    from a percentile of the control, so the boundary is attainable."""
    exact = _diag(THR['drift_strobe'], THR['drift_dense'], THR['dev_ratio'])
    assert qp_label(exact, THR) is True
    just_over = _diag(np.nextafter(THR['drift_strobe'], 1),
                      THR['drift_dense'], THR['dev_ratio'])
    assert qp_label(just_over, THR) is False


def test_label_returns_real_bools_not_numpy_scalars():
    """The label goes into a dataset column and into `is`-comparisons above;
    a np.bool_ would compare False against `True` under `is`."""
    out = qp_label(_diag(np.float64(1e-9), np.float64(1e-9), np.float64(1.0)), THR)
    assert out is True and isinstance(out, bool)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
