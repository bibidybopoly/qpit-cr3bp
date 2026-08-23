"""Tests for the target readout, label rule and sweep orchestration in
src/torus_probe.py.

Run: pytest tests/test_torus_probe_targets.py -v
     (or python tests/test_torus_probe_targets.py)

`tests/test_torus_probe.py` covers the geometry (`curve_distance`,
`curve_local_distance`, `regression_target`) and the probe itself.  Untested
were the five functions between the probe and the experiment:

    sample_probe        per-point targets, and the 1.1433 overshoot correction
    label_from_probe    the conjunctive readout
    calibrate_probe     the positive control that sets the thresholds
    probe_gap           the treatment arm
    calibrate_censor    where the probe stops measuring distance

`sample_probe` and `label_from_probe` are exercised for real -- the first is
pure geometry over two curves, the second pure logic.  `calibrate_probe` and
`probe_gap` are ORCHESTRATION over `probe_curve`, which is Newton on a torus and
far too slow for a unit test, so they run against a stub that records its calls.
What is being checked there is which candidates get built and how the results
are reduced, not whether Newton converges -- `probe_curve` has its own coverage.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import torus_probe as TP                                        # noqa: E402
from torus_probe import (regression_target, sample_probe,        # noqa: E402
                         label_from_probe, calibrate_probe, probe_gap)
from qpit import node_angles, evaluate_curve, curve_amplitude    # noqa: E402


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def _curve(N=31, A=1e-3, centre=(1.0438, 0.0, -0.0756), phase=0.0):
    """A closed curve of known amplitude. Geometry only, not an invariant torus
    -- nothing below asks it to be one."""
    th = node_angles(N) + phase
    U = np.zeros((N, 6))
    U[:, 0] = centre[0] + A * np.cos(th)
    U[:, 1] = centre[1] + A * np.sin(th)
    U[:, 2] = centre[2] + 0.3 * A * np.sin(th)
    U[:, 4] = 0.3922
    return U


def _record(A=1e-3, T=3.13, rho=1.842, C=3.0603, **kw):
    return dict(U0=_curve(A=A, **kw), T=T, rho=rho, C=C, amplitude=A)


def _stub_probe():
    """Replace `probe_curve` with a recorder. Returns (calls, original)."""
    calls, real = [], TP.probe_curve

    def stub(U_cand, T_cand, rho_cand, mu, C_target=None, **kw):
        calls.append(dict(U=np.asarray(U_cand, float).copy(), T=float(T_cand),
                          rho=float(rho_cand), C_target=C_target))
        return dict(converged=True, code='ok', dist_max=1e-9, damp_rel=1e-3,
                    res_inf=1e-13, U_cand=np.asarray(U_cand, float),
                    result=dict(U0=np.asarray(U_cand, float), T=float(T_cand),
                                rho=float(rho_cand), amplitude=1e-3))
    TP.probe_curve = stub
    return calls, real


# ---------------------------------------------------------------------------
# 1. Per-point targets
# ---------------------------------------------------------------------------

def test_sample_probe_returns_one_target_per_theta():
    """The whole point: one Newton solve yields len(thetas) distinct labels
    instead of len(thetas) copies of the curve's dist_max."""
    U_cand = _curve(A=1.0e-3)
    U_true = _curve(A=1.0e-3 + 2e-6)
    p = dict(converged=True, U_cand=U_cand, result=dict(U0=U_true))
    th = np.linspace(0, 2 * np.pi, 9, endpoint=False)

    out = sample_probe(p, th, n_grid=512)
    assert out['d6'].shape == (9,)
    assert out['states'].shape == (9, 6)
    assert np.array_equal(out['theta'], th)
    assert np.all(np.isfinite(out['d6']))
    assert len(np.unique(np.round(out['d6'], 15))) > 1, (
        "every theta returned the same distance -- that is dist_max, not a "
        "per-point label")
    assert np.allclose(out['states'], evaluate_curve(U_cand, th))


def test_sample_probe_refuses_an_unusable_probe_instead_of_returning_zeros():
    """Returning None is what keeps a failed solve out of the dataset; a row of
    zeros would read as 'exactly on a torus'."""
    U = _curve()
    good = dict(converged=True, U_cand=U, result=dict(U0=U))
    assert sample_probe(good, [0.0]) is not None
    assert sample_probe(dict(good, converged=False), [0.0]) is None
    assert sample_probe(dict(good, result=None), [0.0]) is None
    assert sample_probe({'converged': True, 'result': {'U0': U}}, [0.0]) is None


def test_calibration_divides_the_overshoot_out_and_stays_visible():
    """min-norm Newton slides along the family nullspace, so it lands past the
    true nearest torus and the raw distance OVERSTATES by ~1.1433.  The default
    of 1.0 must leave the raw number alone -- the correction is meant to be
    visible in the dataset, not baked in."""
    U_cand, U_true = _curve(A=1e-3), _curve(A=1e-3 + 5e-6)
    p = dict(converged=True, U_cand=U_cand, result=dict(U0=U_true))
    th = np.linspace(0, 2 * np.pi, 5, endpoint=False)

    raw = sample_probe(p, th)
    assert raw['calibration'] == 1.0
    assert np.array_equal(raw['d6_cal'], raw['d6'])

    cal = sample_probe(p, th, calibration=1.1433)
    assert cal['calibration'] == 1.1433
    assert np.allclose(cal['d6_cal'], raw['d6'] / 1.1433, rtol=0, atol=1e-18)
    assert np.array_equal(cal['d6'], raw['d6']), "raw d6 must survive untouched"
    assert np.all(cal['d6_cal'] < raw['d6_cal'])


def test_sample_probe_target_agrees_with_regression_target():
    """log_d6 and censored must be exactly what regression_target produces from
    the CALIBRATED distance -- not from the raw one."""
    U_cand, U_true = _curve(A=1e-3), _curve(A=1e-3 + 1e-5)
    p = dict(converged=True, U_cand=U_cand, result=dict(U0=U_true))
    th = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    out = sample_probe(p, th, floor=1e-12, censor=1e-6, calibration=1.1433)

    lg, cen = regression_target(out['d6_cal'], 1e-12, 1e-6)
    assert np.array_equal(out['log_d6'], lg)
    assert np.array_equal(out['censored'], cen)


def test_sample_probe_reports_the_reparameterisation_wrapped_to_pi():
    """theta_foot - theta shows the reparameterisation directly; a systematic
    offset is exactly what makes node-wise distance overstate.  It has to be
    wrapped, or a foot at 0.01 against a theta at 6.27 reads as a jump of 2pi
    instead of the small shift it is."""
    U_cand = _curve(A=1e-3)
    U_true = _curve(A=1e-3, phase=0.05)          # same curve, shifted phase
    p = dict(converged=True, U_cand=U_cand, result=dict(U0=U_true))
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    out = sample_probe(p, th, n_grid=2048)

    assert np.all(out['dtheta'] > -np.pi - 1e-12)
    assert np.all(out['dtheta'] <= np.pi + 1e-12)
    assert np.abs(out['dtheta']).max() < 1.0, "wrapping failed at the seam"
    # SIGN: U_true(theta) == U_cand(theta + 0.05), so the point at theta on the
    # candidate has its foot at theta - 0.05 on the true curve and dtheta is
    # NEGATIVE.  Pinned because a flipped sign would read as the candidate
    # lagging when it leads, and the offset is meant to be diagnostic.
    assert abs(np.median(out['dtheta']) + 0.05) < 0.02, np.median(out['dtheta'])
    assert np.ptp(out['dtheta']) < 1e-6, "the offset should be uniform in theta"


# ---------------------------------------------------------------------------
# 2. The label rule
# ---------------------------------------------------------------------------

THR = dict(dist_max=1e-8, damp_rel=1e-2, res_inf=1e-11)


def _probe(dist=1e-9, damp=1e-3, res=1e-13, converged=True):
    return dict(converged=converged, dist_max=dist, damp_rel=damp, res_inf=res)


def test_probe_label_requires_convergence_first():
    assert label_from_probe(_probe(converged=False), THR) is False
    # and a non-converged probe is False even with perfect scores
    assert label_from_probe(_probe(1e-30, 1e-30, 1e-30, converged=False),
                            THR) is False


def test_probe_label_is_conjunctive():
    """damp_rel guards the case where Newton converged to a genuine torus of a
    very different amplitude.  Dropping it would turn the probe into 'does a
    torus exist at this energy', which is trivially yes everywhere."""
    assert label_from_probe(_probe(), THR) is True
    assert label_from_probe(_probe(dist=1e-7), THR) is False
    assert label_from_probe(_probe(damp=1.0), THR) is False
    assert label_from_probe(_probe(res=1e-9), THR) is False


def test_probe_label_boundary_is_inclusive_and_returns_a_real_bool():
    exact = _probe(THR['dist_max'], THR['damp_rel'], THR['res_inf'])
    out = label_from_probe(exact, THR)
    assert out is True and isinstance(out, bool)
    assert label_from_probe(_probe(np.nextafter(THR['dist_max'], 1)),
                            THR) is False


# ---------------------------------------------------------------------------
# 3. The positive control
# ---------------------------------------------------------------------------

def test_control_members_are_perturbed_before_being_probed():
    """PERTURB IS NOT OPTIONAL.  Re-probing a member from its own converged U0
    is degenerate -- Newton exits after one iteration with dist_max identically
    0.0, and a threshold of zero makes every candidate negative.  Measured on
    L2 halo #29: the first run of this function returned 0.000e+00.
    """
    calls, real = _stub_probe()
    try:
        mem = [_record(), _record(A=2e-3)]
        calibrate_probe(mem, 0.012, perturb=1e-6, seed=0)
    finally:
        TP.probe_curve = real

    assert len(calls) == 2
    for c, m in zip(calls, mem):
        assert not np.array_equal(c['U'], m['U0']), "member was probed unperturbed"
        # positions moved, velocities did not
        assert np.abs(c['U'][:, 3:] - m['U0'][:, 3:]).max() == 0.0
        moved = np.abs(c['U'][:, :3] - m['U0'][:, :3]).max()
        assert 0 < moved < 1e-4 * curve_amplitude(m['U0']) * 1e4
        assert c['C_target'] == m['C']


def test_control_perturbation_scales_with_curve_amplitude():
    """A fixed absolute perturbation would be enormous for a 17 km torus and
    negligible for a 500 km one, so the control would not be comparable across
    the family."""
    sizes = {}
    for A in (1e-4, 1e-3, 1e-2):
        calls, real = _stub_probe()
        try:
            calibrate_probe([_record(A=A)], 0.012, perturb=1e-6, seed=0)
        finally:
            TP.probe_curve = real
        sizes[A] = np.abs(calls[0]['U'][:, :3] - _record(A=A)['U0'][:, :3]).max()
    assert sizes[1e-3] > 5 * sizes[1e-4]
    assert sizes[1e-2] > 5 * sizes[1e-3]


def test_control_thresholds_are_the_quantile_times_the_margin_with_hard_floors():
    calls, real = _stub_probe()
    try:
        thr = calibrate_probe([_record() for _ in range(5)], 0.012,
                              quantile=95.0, margin=10.0, floor=1e-9)
    finally:
        TP.probe_curve = real
    # stub returns dist_max 1e-9 -> 1e-9 * 10 = 1e-8, above the 1e-9 floor
    assert abs(thr['dist_max'] - 1e-8) < 1e-20
    assert abs(thr['damp_rel'] - 1e-2) < 1e-16
    assert thr['res_inf'] >= 1e-11, "res_inf floor must never be undercut"
    assert thr['n_control'] == 5 and thr['n_failed'] == 0
    assert len(thr['probes']) == 5


def test_control_floor_stops_a_degenerate_control_producing_an_impossible_threshold():
    """"a control that still comes back degenerate cannot silently yield a
    threshold no candidate can meet"."""
    calls, real = _stub_probe()
    try:
        TP.probe_curve = lambda *a, **k: dict(
            converged=True, dist_max=0.0, damp_rel=0.0, res_inf=0.0)
        thr = calibrate_probe([_record() for _ in range(3)], 0.012, floor=1e-9)
    finally:
        TP.probe_curve = real
    assert thr['dist_max'] == 1e-9
    assert thr['damp_rel'] == 1e-9
    assert thr['res_inf'] == 1e-11


def test_control_raises_rather_than_returning_a_meaningless_threshold():
    real = TP.probe_curve
    try:
        TP.probe_curve = lambda *a, **k: dict(converged=False, code='diverged')
        _raises(ValueError, calibrate_probe, [_record()], 0.012)
    finally:
        TP.probe_curve = real


def test_control_counts_failures_without_letting_them_set_the_threshold():
    real = TP.probe_curve
    state = {'n': 0}

    def flaky(*a, **k):
        state['n'] += 1
        if state['n'] % 2:
            return dict(converged=True, dist_max=1e-9, damp_rel=1e-3, res_inf=1e-13)
        return dict(converged=False, code='diverged', dist_max=1e3,
                    damp_rel=1e3, res_inf=1e3)
    try:
        TP.probe_curve = flaky
        thr = calibrate_probe([_record() for _ in range(6)], 0.012, margin=10.0)
    finally:
        TP.probe_curve = real
    assert thr['n_control'] == 3 and thr['n_failed'] == 3
    assert abs(thr['dist_max'] - 1e-8) < 1e-20, "a diverged probe set the threshold"


# ---------------------------------------------------------------------------
# 4. The treatment arm
# ---------------------------------------------------------------------------

def test_gap_sweep_builds_one_candidate_per_alpha_with_interpolated_T_and_rho():
    a = _record(A=1e-3, T=3.10, rho=1.80, C=3.06)
    b = _record(A=2e-3, T=3.20, rho=1.90, C=3.06)
    alphas = [0.25, 0.5, 0.75]
    calls, real = _stub_probe()
    try:
        rows = probe_gap(a, b, 0.012, alphas, reproject=False)
    finally:
        TP.probe_curve = real

    assert len(rows) == 3 and len(calls) == 3
    for row, c, al in zip(rows, calls, alphas):
        assert row['alpha'] == al
        assert abs(c['T'] - ((1 - al) * a['T'] + al * b['T'])) < 1e-12
        assert abs(c['rho'] - ((1 - al) * a['rho'] + al * b['rho'])) < 1e-12
        assert abs(c['C_target'] - 0.5 * (a['C'] + b['C'])) < 1e-15


def test_gap_sweep_targets_the_mean_energy_of_the_pair():
    """The same rule the sampler uses: a candidate at either member's own C
    would not sit in the gap, it would sit on a member's energy level."""
    a, b = _record(C=3.00), _record(C=3.10)
    calls, real = _stub_probe()
    try:
        probe_gap(a, b, 0.012, [0.5], reproject=False)
    finally:
        TP.probe_curve = real
    assert abs(calls[0]['C_target'] - 3.05) < 1e-15


def test_gap_sweep_candidate_is_the_linear_interpolant_of_the_two_curves():
    a, b = _record(A=1e-3), _record(A=3e-3)
    calls, real = _stub_probe()
    try:
        probe_gap(a, b, 0.012, [0.0, 1.0], reproject=False)
    finally:
        TP.probe_curve = real
    th = node_angles(a['U0'].shape[0])
    assert np.allclose(calls[0]['U'], evaluate_curve(a['U0'], th), atol=1e-12)
    assert np.allclose(calls[1]['U'], evaluate_curve(b['U0'], th), atol=1e-12)


def test_gap_sweep_applies_sigma_to_the_second_member_only():
    a, b = _record(A=1e-3), _record(A=1e-3)
    calls, real = _stub_probe()
    try:
        probe_gap(a, b, 0.012, [1.0], sigma=0.4, reproject=False)
    finally:
        TP.probe_curve = real
    th = node_angles(a['U0'].shape[0])
    assert np.allclose(calls[0]['U'], evaluate_curve(b['U0'], th + 0.4), atol=1e-12)


def test_gap_sweep_reprojects_and_records_the_correction():
    """"a large correction means the pair is not actually adjacent" -- so the
    magnitude has to be recorded, not just applied."""
    a, b = _record(A=1e-3, C=3.06), _record(A=2e-3, C=3.06)
    calls, real = _stub_probe()
    try:
        rows = probe_gap(a, b, 0.012, [0.5], reproject=True)
    finally:
        TP.probe_curve = real
    assert 'reproj_rel' in rows[0]
    assert np.isfinite(rows[0]['reproj_rel']) and rows[0]['reproj_rel'] >= 0.0

    calls2, real = _stub_probe()
    try:
        rows2 = probe_gap(a, b, 0.012, [0.5], reproject=False)
    finally:
        TP.probe_curve = real
    assert rows2[0]['reproj_rel'] == 0.0
    assert not np.allclose(calls[0]['U'], calls2[0]['U']), (
        "reproject=True produced the same candidate as reproject=False")


def test_gap_sweep_reports_the_forbidden_region_as_a_row_not_an_exception():
    """A candidate that cannot be reprojected onto the target energy has to
    come back as an un-converged row, or one bad alpha aborts the sweep and the
    remaining alphas are silently missing from the experiment."""
    a, b = _record(C=3.06), _record(A=2e-3, C=3.06)
    a = dict(a, C=1e6)                       # unreachable target energy
    b = dict(b, C=1e6)
    calls, real = _stub_probe()
    try:
        rows = probe_gap(a, b, 0.012, [0.25, 0.5], reproject=True)
    finally:
        TP.probe_curve = real
    assert len(rows) == 2, "the sweep stopped early"
    for r in rows:
        assert r['converged'] is False
        assert r['code'] == 'forbidden_region'
        assert np.isnan(r['reproj_rel'])
    assert len(calls) == 0, "a forbidden candidate was still handed to Newton"


def test_gap_sweep_accepts_a_scalar_alpha():
    a, b = _record(), _record(A=2e-3)
    calls, real = _stub_probe()
    try:
        rows = probe_gap(a, b, 0.012, 0.5, reproject=False)
    finally:
        TP.probe_curve = real
    assert len(rows) == 1 and rows[0]['alpha'] == 0.5


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
