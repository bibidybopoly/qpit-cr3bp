"""Tests for the last untested public names — the ones that need a real
integration, a real Newton solve or a real training run.

Run: pytest tests/test_slow_paths.py -v  (or python tests/test_slow_paths.py)

    orbits.compute_halo_family        pseudo-arclength in 3D
    orbits.sweep_rotation_numbers     the centre pair along a family
    labeling.propagate_sampled        the dense + stroboscopic sample pair
    labeling.qp_diagnostics           the full continuous score set
    torus_probe.calibrate_censor      where the probe stops measuring distance
    ablation.run_cell / run_2x2       one cell, and the 2x2
    models.train_multi_seed           the multi-seed fit

SLOW BY NECESSITY, roughly 30 s.  `calibrate_censor` is the bulk of it: it is
Newton per displacement, so it runs on 5 points rather than the default 17.

Where a number is orbit-specific (the Newton overshoot plateau, ||V_pos||) the
test pins the STRUCTURE the docstring argues for, not the value measured on a
different family at a different N — those move with resolution and amplitude.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import constants as C                                            # noqa: E402
import qpit                                                      # noqa: E402
import orbits                                                    # noqa: E402
import labeling as L                                             # noqa: E402
import torus_probe as TP                                         # noqa: E402
import ablation as A                                             # noqa: E402
from cr3bp import MU_EARTH_MOON as MU                            # noqa: E402
from models import MLPConfig, train_multi_seed                   # noqa: E402

HALO = dict(state0=list(C.HALO29_STATE), period=C.HALO29_PERIOD)
HALO_X = np.array([C.HALO29_STATE[0], C.HALO29_STATE[2],
                   C.HALO29_STATE[4], C.HALO29_PERIOD / 2])

_CACHE = {}


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def halo_family(n=5):
    if ('fam', n) not in _CACHE:
        _CACHE[('fam', n)] = orbits.compute_halo_family(
            HALO_X, 0.004, n, MU, verbose=False)
    return _CACHE[('fam', n)]


def torus():
    if 'torus' not in _CACHE:
        U0, T0, r0 = qpit.seed_from_halo(HALO, MU, 15, 1e-4)
        o = qpit.converge_torus(U0, T0, r0, MU, tol=1e-12, itmax=15,
                                verbose=False)
        assert o['converged'], o['code']
        _CACHE['torus'] = (o['U0'], o['T'], o['rho'])
    return _CACHE['torus']


# ---------------------------------------------------------------------------
# 1. The halo family
# ---------------------------------------------------------------------------

def test_halo_family_members_are_all_perpendicular_crossings():
    """Every member must satisfy the 3-component constraint [y, vx, vz] = 0 at
    T_half, not just the first.  Continuation corrects each from the previous,
    so drift accumulates silently while the family still looks smooth."""
    fam = halo_family()
    assert len(fam) == 5
    for i, o in enumerate(fam):
        F, _, _ = orbits.halo_periodicity_constraint(o['X'], MU)
        assert np.abs(F).max() < 1e-9, (i, np.abs(F).max())


def test_halo_family_records_carry_a_consistent_state_and_period():
    fam = halo_family()
    for o in fam:
        assert o['period'] == 2 * o['T_half']
        assert np.allclose(o['state0'],
                           [o['x0'], 0.0, o['z0'], 0.0, o['vy0'], 0.0])
        assert np.allclose(o['X'], [o['x0'], o['z0'], o['vy0'], o['T_half']])


def test_halo_family_starts_at_the_orbit_it_was_given():
    fam = halo_family()
    assert np.allclose(fam[0]['X'], HALO_X, rtol=0, atol=1e-12)
    assert abs(fam[0]['jacobi'] - C.HALO29_C) < 1e-6


def test_halo_family_marches_and_C_varies_smoothly():
    fam = halo_family()
    Cv = np.array([o['jacobi'] for o in fam])
    z0 = np.array([o['z0'] for o in fam])
    assert np.all(np.diff(Cv) > 0) or np.all(np.diff(Cv) < 0), Cv
    assert np.abs(np.diff(Cv, 2)).max() < 0.5 * np.abs(np.diff(Cv)).max()
    assert np.ptp(z0) > 0, "the family did not move"
    # consecutive members sit ds apart in the free variables
    X = np.array([o['X'] for o in fam])
    assert np.allclose(np.linalg.norm(np.diff(X, axis=0), axis=1), 0.004,
                       rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. Rotation numbers along the family
# ---------------------------------------------------------------------------

def test_rotation_number_sweep_reproduces_the_recorded_value_for_halo_29():
    """rho/2pi = 0.293229 is the identifying number for the Week 8 target --
    the docstring says the identifying pair is (C, rho/2pi), NOT the index, so
    this is the value that has to keep reproducing."""
    sw = orbits.sweep_rotation_numbers(halo_family(), MU, verbose=False)
    for k in ('index', 'C', 'theta', 'rho', 'anchor'):
        assert k in sw, k
    assert abs(sw['rho'][0] - 0.293229) < 1e-5, sw['rho'][0]
    assert abs(sw['C'][0] - C.HALO29_C) < 1e-6


def test_rotation_number_sweep_is_aligned_with_the_family_it_was_given():
    fam = halo_family()
    sw = orbits.sweep_rotation_numbers(fam, MU, verbose=False)
    assert len(sw['rho']) == len(fam)
    assert np.array_equal(sw['index'], np.arange(len(fam)))
    assert np.allclose(sw['C'], [o['jacobi'] for o in fam], atol=1e-9)
    assert 0 <= sw['anchor'] < len(fam)


def test_rotation_number_is_the_normalised_form_of_theta():
    """`rho` is reported NORMALISED (theta/2pi) while `theta` is in radians --
    and `qpit.seed_from_halo` wants NEITHER, it takes the eigenvalue argument
    in radians directly.  Mixing them is a factor of 2pi in the seed."""
    sw = orbits.sweep_rotation_numbers(halo_family(), MU, verbose=False)
    assert np.allclose(sw['rho'], sw['theta'] / (2 * np.pi), rtol=1e-12)
    d = np.diff(sw['theta'])
    assert np.all(d < 0) or np.all(d > 0), "the sweep is not monotone"


def test_theta_is_genuinely_unwrapped_across_a_family_that_crosses_zero():
    """The unwrapping claim, exercised rather than assumed.

    A 5-member family never leaves (0, 2pi), so every assertion about wrapping
    is vacuous on it -- verified by mutation: replacing `th` with
    `np.mod(th, 2*pi)` passes the short-family tests untouched.  A 40-member
    family at ds=0.01 walks theta from 1.84 down THROUGH zero to -0.59, which
    is where wrapped and unwrapped genuinely differ: a wrapped sweep jumps to
    5.70 at the crossing.

    It matters because `theta` is the quantity followed continuously along the
    family (by eigenvector overlap), and a 2pi jump in it is indistinguishable
    from the centre pair being lost.

    Costs ~4 s -- the reason the rest of this section uses the short family.
    """
    if ('long', 40) not in _CACHE:
        _CACHE[('long', 40)] = orbits.compute_halo_family(
            HALO_X, 0.01, 40, MU, verbose=False)
    fam = _CACHE[('long', 40)]
    sw = orbits.sweep_rotation_numbers(fam, MU, verbose=False)
    th = sw['theta']

    assert th.min() < 0.0, ("the family did not cross zero -- this test no "
                            "longer exercises unwrapping", th.min())
    assert np.abs(np.diff(th)).max() < np.pi, "theta jumped: it is wrapped"
    assert np.all(np.diff(th) < 0), "the sweep stopped being monotone"
    # and rho follows theta through zero rather than being folded into (0,1)
    assert sw['rho'].min() < 0.0
    assert np.allclose(sw['rho'], th / (2 * np.pi), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Sampled propagation
# ---------------------------------------------------------------------------

def test_stroboscopic_grid_is_exact_multiples_of_T():
    """"placed in t_eval directly, not picked out of the dense grid by
    nearest-neighbour, which would inject a timing error of dt/2 into rho and
    swamp the drift signal."  A dt/2 error at n_per_period=64 is ~1% of a
    period -- orders above the 1e-12 drift the indicator has to resolve."""
    x0 = np.array(C.HALO29_STATE)
    T = C.HALO29_PERIOD
    n_per, n_p = 64, 6
    t_d, X_d, t_s, X_s = L.propagate_sampled(x0, T, MU, n_p, n_per)

    assert np.allclose(t_s, T * np.arange(n_p + 1), rtol=0, atol=1e-12)
    assert np.abs(t_s - T * np.arange(n_p + 1)).max() == 0.0
    assert t_d.size == n_p * n_per + 1
    assert X_d.shape == (t_d.size, 6) and X_s.shape == (n_p + 1, 6)
    assert np.allclose(X_s[0], x0, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(X_s))


def test_dense_and_stroboscopic_come_from_one_integration():
    """Both sample sets are read out of a single solve, so where the two grids
    coincide they must agree EXACTLY -- not to integrator tolerance."""
    x0 = np.array(C.HALO29_STATE)
    T = C.HALO29_PERIOD
    t_d, X_d, t_s, X_s = L.propagate_sampled(x0, T, MU, 4, 32)
    for k, ts in enumerate(t_s):
        j = int(np.argmin(np.abs(t_d - ts)))
        if abs(t_d[j] - ts) < 1e-12:
            assert np.array_equal(X_d[j], X_s[k]), k


# ---------------------------------------------------------------------------
# 4. The full diagnostic set
# ---------------------------------------------------------------------------

def test_diagnostics_record_the_shadowing_bound_they_were_given():
    U, T, r = torus()
    e1, e2, _ = L.curve_plane(U)
    centre = U.mean(axis=0)
    d = L.qp_diagnostics(U[0], T, MU, centre, e1, e2, n_periods=16,
                         n_per_period=32, R_u=146.4)
    assert abs(d['n_max'] - L.max_usable_periods(146.4)) < 1e-12
    assert d['window_ok'] is False, "16 periods is far past the 2.8-period bound"
    assert d['n_periods'] == 16 and d['n_per_period'] == 32
    for k in ('ok', 'reason', 'drift_strobe', 'rho_est', 'drift_dense',
              'dev_ratio', 'max_dev'):
        assert k in d, k


def test_a_window_long_enough_for_NAFF_has_already_escaped():
    """The tension the module documents, reproduced on a real torus.

    The shadowing bound at R_u = 146 allows ~2.8 stroboscopic periods.  NAFF
    needs >= 16 samples per half, i.e. >= 32 periods.  There is no window that
    satisfies both: by the time the signal is long enough to measure a
    frequency, the trajectory has left the torus it started on.

    That is exactly "Confirmed empirically: at n_periods=60 all three
    experimental arms escaped, INCLUDING the positive control" — and escape is
    recorded as a real measurement (ok=True), not a failure.
    """
    U, T, r = torus()
    e1, e2, _ = L.curve_plane(U)
    centre = U.mean(axis=0)
    d = L.qp_diagnostics(U[0], T, MU, centre, e1, e2, n_periods=40,
                         n_per_period=32, escape_radius=1.0, R_u=146.4)
    assert d['ok'] is True, "escape is a measurement, not a failure"
    assert d['reason'] == 'escaped'
    assert d['max_dev'] > 1.0
    assert np.isnan(d['drift_strobe']), "no drift should be reported after escape"
    assert d['window_ok'] is False


def test_a_short_window_raises_instead_of_reporting_a_reason_KNOWN_GAP():
    """PINS A ROBUSTNESS GAP.

    `qp_diagnostics` promises never to return a label and to record a `reason`
    for anything it could not measure -- it does that for 'integration_failed'
    and 'nonfinite'.  It does NOT for a window too short to run NAFF: below 32
    stroboscopic periods the call dies inside `naff_frequency` with
    "need at least 16 samples".

    That is precisely the regime the shadowing bound permits (n_max = 2.8 here),
    so the only physically defensible window is the one with no graceful path.
    Nothing in the dataset depends on it -- the Week 10 run used the long
    window and recorded 'escaped' -- but a caller who followed the shadowing
    advice would get an exception rather than a row.
    """
    U, T, r = torus()
    e1, e2, _ = L.curve_plane(U)
    centre = U.mean(axis=0)
    _raises(ValueError, L.qp_diagnostics, U[0], T, MU, centre, e1, e2,
            n_periods=4, n_per_period=32, R_u=146.4)


def test_diagnostics_report_escape_without_an_R_u():
    U, T, r = torus()
    e1, e2, _ = L.curve_plane(U)
    d = L.qp_diagnostics(U[0], T, MU, U.mean(axis=0), e1, e2, n_periods=40,
                         n_per_period=32)
    assert d['n_max'] == np.inf and d['window_ok'] is True
    assert d['ok'] is True


# ---------------------------------------------------------------------------
# 5. Where the probe stops measuring distance
# ---------------------------------------------------------------------------

def censor_sweep():
    if 'censor' not in _CACHE:
        U, T, r = torus()
        mem = dict(U0=U, T=T, rho=r, C=qpit.mean_jacobi(U, MU))
        DG = qpit.stability_matrix(U, T, r, MU)
        # three points well inside the faithful regime and three well past
        # the basin edge (which sits near 2e-7).  A sweep with only ONE broken
        # point cannot distinguish a plateau taken from the small-d end from
        # one taken from the large-d end -- the median absorbs the outlier.
        _CACHE['censor'] = TP.calibrate_censor(
            mem, MU, displacements=np.array([1e-9, 3e-9, 1e-8,
                                             1e-6, 3e-6, 1e-5]), DG=DG)
    return _CACHE['censor']


def test_censor_sweep_finds_a_plateau_then_a_break():
    """The ratio d_returned/d is an ABSOLUTE calibration, not a linearity
    check: it should sit on a plateau while Newton stays in the basin of the
    nearby torus, then jump by orders of magnitude when it leaves."""
    out = censor_sweep()
    ok = out['converged']
    assert ok.sum() >= 4
    ratio = out['ratio']
    plateau = out['plateau']
    assert plateau > 1.0, "Newton should OVERSHOOT the nearest torus"
    assert plateau < 2.0, plateau
    near = ratio[:3]
    assert np.abs(near - plateau).max() < 0.05 * plateau, near

    # Past the basin edge a probe either returns a much larger ratio (Newton
    # found a DIFFERENT torus, so `dist` records how far it travelled) or fails
    # to converge at all.  Both mean "not measuring distance"; neither may be
    # read as a distance.  Measured here: 42.9, no-convergence, 10.4.
    far, far_ok = ratio[-3:], ok[-3:]
    assert not np.all(far_ok), "expected at least one probe to fail past the edge"
    assert np.all(far[far_ok] > 5 * plateau), ("no break found -- the sweep "
                                               "does not bracket the edge", ratio)

    # The plateau must be read off the SMALL-d end.  Taking it from the large-d
    # end would calibrate against a lower bound -- and a median over the far
    # rows is NOT obviously wrong-looking, so pin it explicitly.
    assert abs(plateau - np.median(near)) < 1e-9, (plateau, near)
    assert np.nanmedian(far) > 5 * plateau, (plateau, far)


def test_censor_is_half_the_last_faithful_displacement():
    out = censor_sweep()
    assert out['censor'] == out['last_faithful'] / 2.0
    assert out['last_faithful'] <= out['displacement'].max()
    assert out['tol'] == 0.05


def test_position_only_distance_cannot_pass_the_calibration():
    """"a position-only metric cannot pass this test even in the faithful
    regime: the bundle vector is a unit 6-vector whose position part has norm
    below 1, so d3/d saturates at ||V_pos|| rather than 1."

    This is the argument for making the regression target d6 rather than d3,
    and it is checkable directly.
    """
    out = censor_sweep()
    ok = out['converged']
    d3_ratio = (out['d3'] / out['displacement'])[ok][:3]
    assert out['V_pos_norm'] < 1.0
    assert np.all(d3_ratio < out['plateau']), (d3_ratio, out['plateau'])
    assert abs(np.median(d3_ratio) - out['V_pos_norm']) < 0.1, (
        np.median(d3_ratio), out['V_pos_norm'])


def test_censor_sweep_reports_the_growth_factor_it_displaced_along():
    out = censor_sweep()
    assert abs(out['R_u'] - 146.4) < 1.0


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

def toy(n=400, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    w = rng.standard_normal(n_feat)
    d = 10.0 ** (X @ w * 0.3 + rng.normal(0, 0.05, n)) * 1e-2
    return X, d


TINY = MLPConfig(hidden=(8,), batch_size=64, max_epochs=6, min_epochs=1,
                 patience=99, lr=1e-2)


def test_train_multi_seed_returns_one_prediction_row_per_seed():
    X, d = toy()
    y = np.log10(d)
    tr, va, te = slice(0, 240), slice(240, 320), slice(320, None)
    seeds = (0, 1, 2)
    P, H = train_multi_seed(X[tr], y[tr], X[va], y[va], X[te], TINY, seeds=seeds)

    assert P.shape == (len(seeds), X[te].shape[0])
    assert len(H) == len(seeds)
    assert np.all(np.isfinite(P))
    for h in H:
        for k in ('best_epoch', 'best_val_loss', 'config', 'early_stopped'):
            assert k in h, k
    # different seeds must give different fits, or the seeding does nothing
    assert not np.allclose(P[0], P[1])


def test_train_multi_seed_is_reproducible_from_the_seed():
    X, d = toy()
    y = np.log10(d)
    tr, va, te = slice(0, 240), slice(240, 320), slice(320, None)
    P1, _ = train_multi_seed(X[tr], y[tr], X[va], y[va], X[te], TINY, seeds=(0,))
    P2, _ = train_multi_seed(X[tr], y[tr], X[va], y[va], X[te], TINY, seeds=(0,))
    assert np.array_equal(P1, P2), "same seed gave a different fit"


def test_train_multi_seed_records_the_config_it_ran_with():
    """`hist['config']` is what makes a frozen result recoverable from a results
    file -- it is why `log_sigma_init` lives on the config rather than being a
    call argument."""
    X, d = toy()
    y = np.log10(d)
    _, H = train_multi_seed(X[:240], y[:240], X[240:320], y[240:320],
                            X[320:], TINY, seeds=(0,))
    cfg = H[0]['config']
    assert cfg['max_epochs'] == TINY.max_epochs
    assert tuple(cfg['hidden']) == tuple(TINY.hidden)
    assert cfg['lr'] == TINY.lr


# ---------------------------------------------------------------------------
# 7. The ablation cells
# ---------------------------------------------------------------------------

def _split():
    X, d = toy()
    return (X[:240], X[240:320], X[320:], d[:240], d[240:320], d[320:])


def test_run_cell_scores_a_single_cell_across_seeds():
    Xtr, Xva, Xte, dtr, dva, dte = _split()
    ev, pred = A.run_cell('M1', 'log10', 'mse', Xtr, Xva, Xte, dtr, dva, dte,
                          C.FLOOR_KM, cfg=TINY, seeds=(0, 1), verbose=False)
    assert ev['cell'] == 'M1'
    assert ev['transform'] == 'log10' and ev['loss'] == 'mse'
    assert ev['n_seeds'] == 2 and ev['n_test'] == Xte.shape[0]
    assert len(ev['per_seed']) == 2 and len(ev['histories']) == 2
    assert pred.shape[-1] == Xte.shape[0]
    s = ev['summary']
    for k in ('r2_measured', 'r2_measured_sd', 'rmse_measured', 'bias_near'):
        assert k in s and np.isfinite(s[k]), k
    assert s['r2_measured_sd'] >= 0.0


def test_run_cell_is_scored_on_measured_rows_only():
    """A clamped label is not a point observation, and scoring against it
    rewards predicting the clamp.  The summary keys say `_measured` for that
    reason, and the censored fraction has to be reported alongside."""
    Xtr, Xva, Xte, dtr, dva, dte = _split()
    dte = dte.copy()
    dte[:40] = C.FLOOR_KM * 0.1                # force a censored block
    ev, _ = A.run_cell('M1', 'log10', 'mse', Xtr, Xva, Xte, dtr, dva, dte,
                       C.FLOOR_KM, cfg=TINY, seeds=(0,), verbose=False)
    assert abs(ev['censored_frac'] - 40 / dte.size) < 1e-12
    assert np.isfinite(ev['summary']['r2_measured'])


def test_run_2x2_produces_the_four_named_cells():
    """M1 log10+mse is the frozen Week 11 baseline reproduced; the other three
    turn on one change each, so the two effects stay attributable."""
    Xtr, Xva, Xte, dtr, dva, dte = _split()
    res, preds = A.run_2x2(Xtr, Xva, Xte, dtr, dva, dte, C.FLOOR_KM, 1e-2,
                           cfg=TINY, seeds=(0,), verbose=False)
    assert sorted(res) == ['M1', 'M2a', 'M2b', 'M2c']
    assert sorted(preds) == ['M1', 'M2a', 'M2b', 'M2c']
    assert res['M1']['transform'] == 'log10' and res['M1']['loss'] == 'mse'
    # exactly one change per off-diagonal cell, both on the corner
    changed = {k: (res[k]['transform'] != 'log10', res[k]['loss'] != 'mse')
               for k in res}
    assert changed['M1'] == (False, False)
    assert sum(changed['M2a']) == 1 and sum(changed['M2b']) == 1
    assert changed['M2a'] != changed['M2b']
    assert changed['M2c'] == (True, True)
    for k in res:
        assert np.isfinite(res[k]['summary']['r2_measured']), k

    # delta belongs to the TARGET, so it is carried only by the asinh cells.
    # A delta recorded against a log10 cell would be a number with no meaning
    # attached to a frozen result -- and delta is swept, so it would look like
    # a free parameter the baseline also depended on.
    assert res['M1']['delta_km'] is None
    assert res['M2b']['delta_km'] is None
    assert res['M2a']['delta_km'] == 1e-2
    assert res['M2c']['delta_km'] == 1e-2


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
