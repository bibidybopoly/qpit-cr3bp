"""Tests for src/orbits.py — the correctors, continuation and Floquet machinery.

Run: pytest tests/test_orbits.py -v   (or python tests/test_orbits.py)

This module had NO test file and was never imported by any test, because until
now importing it ran an exploratory script: ~30 lines of print plus a real
integration, at import time.  Notebooks worked around that rather than fixing
it.  These tests exist as much to hold the module importable as to check the
maths.

Everything is pinned to physical invariants -- symplecticity, the null space of
the constraint Jacobian, agreement between the analytic STM Jacobian and finite
differences -- rather than to recorded numbers, so a change that is merely a
different discretisation passes and a change that breaks the dynamics fails.
"""

import os
import subprocess
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

import orbits                                              # noqa: E402
from orbits import (y_axis_crossing, correct_lyapunov_orbit,   # noqa: E402
                    periodicity_constraint, family_tangent, predict_step,
                    pseudo_arclength_correct,
                    compute_family_pseudo_arclength,
                    monodromy_and_multipliers, compute_manifold,
                    halo_periodicity_constraint)
from cr3bp import MU_EARTH_MOON, find_lagrange_points       # noqa: E402
import constants as C                            # noqa: E402

MU = MU_EARTH_MOON

#: The small L1 Lyapunov orbit the notebooks start from, converged to
#: |vx| < 1e-13 at the perpendicular crossing.  Same orbit as
#: `pseudo_arclength_continuation.ipynb` cell 0.  Frozen here so the tests do
#: not depend on the corrector they are testing.
LYAP_X = np.array([0.835915125772357,      # x0
                   0.008434885455359,      # vy0
                   1.345895046411205])     # T_half

#: L2 halo #29 -- the Week 8 target, as [x0, z0, vy0, T_half].
#: Same orbit as tests/test_qpit_week8.py's HALO and test_features.py's HALO29.
HALO_X = np.array([C.HALO29_STATE[0], C.HALO29_STATE[2],
                   C.HALO29_STATE[4], C.HALO29_PERIOD / 2])


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


# ---------------------------------------------------------------------------
# 0. The module must stay importable
# ---------------------------------------------------------------------------

def test_import_is_silent_and_has_no_side_effects():
    """Importing orbits must print nothing and compute nothing.

    This is a regression test, not a style check.  The module used to run its
    L1-Lyapunov derivation at import: `halo_families.ipynb` had to wrap the
    import in `redirect_stdout(os.devnull)` and `qpit_solver.ipynb` carried the
    comment "expect ~30 lines of noise".  A module that does work on import
    cannot be imported by a test suite, which is exactly why this file did not
    exist.  If the derivation ever escapes `if __name__ == '__main__'` again,
    this fails immediately.
    """
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, '-c', 'import orbits'],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout == '', f"import printed:\n{r.stdout}"


def test_derivation_is_preserved_and_still_runs_as_a_script():
    """`python src/orbits.py` must still produce the derivation it always did."""
    r = subprocess.run([sys.executable, os.path.join(SRC, 'orbits.py')],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert 'Jacobian at L1:' in r.stdout
    assert 'Oscillation frequency omega' in r.stdout
    assert 'residual to correct' in r.stdout


# ---------------------------------------------------------------------------
# 1. The event function
# ---------------------------------------------------------------------------

def test_y_axis_crossing_returns_y_and_is_configured_as_a_terminal_event():
    state = [0.83, 0.017, -0.4, 0.2]
    assert y_axis_crossing(1.234, state, MU) == state[1]
    # solve_ivp reads these attributes off the function object; losing them
    # silently turns the corrector's half-period search into a full integration
    assert y_axis_crossing.terminal is True
    assert y_axis_crossing.direction == -1


# ---------------------------------------------------------------------------
# 2. The single-shooting corrector
# ---------------------------------------------------------------------------

def test_corrector_reaches_a_perpendicular_crossing():
    """vx at the half-period crossing is what the corrector drives to zero."""
    lps = find_lagrange_points(MU)
    x0 = lps['L1'][0] - 0.001
    vy0_guess = (0.885435 / -0.105758) * -0.001      # the linearised seed
    vy0, t_cross, state_cross = correct_lyapunov_orbit(
        x0, vy0_guess, MU, verbose=False)

    assert abs(state_cross[2]) < 1e-11, state_cross[2]   # vx at crossing
    assert abs(state_cross[1]) < 1e-9                    # y at crossing
    assert t_cross > 0.5
    # and it must land on the frozen orbit the notebooks continue from
    assert abs(vy0 - LYAP_X[1]) < 1e-9
    assert abs(t_cross - LYAP_X[2]) < 1e-9


def test_corrector_rejects_a_degenerate_crossing():
    """The `min_crossing_time` guard must actually fire.

    This is the guard that `differential_corrector.ipynb` cell 5 does NOT have
    -- it vendors an older copy of this function.  Raising the threshold above
    the true crossing time (~1.346) is a deterministic way to trip it without
    hunting for a pathological guess.
    """
    lps = find_lagrange_points(MU)
    x0 = lps['L1'][0] - 0.001
    _raises(RuntimeError, correct_lyapunov_orbit, x0, LYAP_X[1], MU,
            min_crossing_time=5.0, verbose=False)


def test_corrector_gives_up_rather_than_returning_a_bad_orbit():
    _raises(RuntimeError, correct_lyapunov_orbit,
            LYAP_X[0], LYAP_X[1], MU, tol=1e-18, max_iter=2, verbose=False)


# ---------------------------------------------------------------------------
# 3. The periodicity constraint and its Jacobian
# ---------------------------------------------------------------------------

def test_periodicity_constraint_vanishes_on_a_converged_orbit():
    F, dF, final_state = periodicity_constraint(LYAP_X, MU)
    assert np.abs(F).max() < 1e-10, F
    assert dF.shape == (2, 3)
    assert final_state.shape == (4,)


def test_periodicity_jacobian_matches_finite_differences():
    """The analytic Jacobian is assembled from the STM; check it independently.

    This is the load-bearing test for the whole continuation scheme: every
    tangent, prediction and Newton step downstream is built from this matrix.
    An STM index slip (Phi[1,3] vs Phi[3,1], say) still produces a plausible
    2x3 matrix and a family that quietly wanders off the true branch.
    """
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    h = 1e-6
    fd = np.zeros((2, 3))
    for j in range(3):
        Xp, Xm = LYAP_X.copy(), LYAP_X.copy()
        Xp[j] += h
        Xm[j] -= h
        Fp, _, _ = periodicity_constraint(Xp, MU)
        Fm, _, _ = periodicity_constraint(Xm, MU)
        fd[:, j] = (Fp - Fm) / (2 * h)

    err = np.abs(dF - fd).max() / max(1.0, np.abs(fd).max())
    assert err < 1e-6, f"analytic vs finite-difference Jacobian differs by {err:.2e}\n{dF}\n{fd}"


# ---------------------------------------------------------------------------
# 4. Tangent, prediction, arclength correction
# ---------------------------------------------------------------------------

def test_family_tangent_is_a_unit_vector_in_the_null_space():
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    t = family_tangent(dF)
    assert abs(np.linalg.norm(t) - 1.0) < 1e-12
    assert np.abs(dF @ t).max() < 1e-9, "tangent leaves the constraint surface"


def test_family_tangent_keeps_the_previous_direction():
    """The null space is defined only up to sign; continuation depends on the
    sign staying put, or the family walks back over itself."""
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    t = family_tangent(dF)
    assert np.allclose(family_tangent(dF, previous_tangent=t), t)
    assert np.allclose(family_tangent(dF, previous_tangent=-t), -t)


def test_predict_step_moves_exactly_ds_along_the_tangent():
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    t = family_tangent(dF)
    ds = 1e-3
    Xp = predict_step(LYAP_X, t, ds)
    assert abs(np.linalg.norm(Xp - LYAP_X) - ds) < 1e-12
    assert np.allclose(Xp - LYAP_X, ds * t)


def test_pseudo_arclength_correction_lands_on_the_family_at_arclength_ds():
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    t = family_tangent(dF)
    ds = 1e-3
    X_new = pseudo_arclength_correct(predict_step(LYAP_X, t, ds),
                                     LYAP_X, t, ds, MU)
    F_new, _, _ = periodicity_constraint(X_new, MU)
    assert np.abs(F_new).max() < 1e-9, "corrected point is not on the family"
    arc = np.dot(X_new - LYAP_X, t)
    assert abs(arc - ds) < 1e-9, f"arclength constraint not met: {arc} vs {ds}"
    assert not np.allclose(X_new, LYAP_X), "correction collapsed back to the anchor"


def test_pseudo_arclength_correction_reports_failure():
    _, dF, _ = periodicity_constraint(LYAP_X, MU)
    t = family_tangent(dF)
    _raises(RuntimeError, pseudo_arclength_correct,
            predict_step(LYAP_X, t, 1e-3), LYAP_X, t, 1e-3, MU,
            tol=1e-18, max_iter=2)


# ---------------------------------------------------------------------------
# 5. Continuation
# ---------------------------------------------------------------------------

def test_family_members_are_all_genuine_periodic_orbits():
    """Every member must satisfy the constraint, not just the first.

    Continuation can drift: each orbit is corrected from the previous one, so a
    slow accumulation of error produces a family that still looks smooth while
    no longer solving F = 0.  Checking the last member is the point.
    """
    fam = compute_family_pseudo_arclength(LYAP_X, ds=1e-3, n_orbits=6, mu=MU,
                                          verbose=False)
    assert len(fam) == 6
    for i, orb in enumerate(fam):
        F, _, _ = periodicity_constraint(orb['X'], MU)
        assert np.abs(F).max() < 1e-9, f"orbit {i} is off the family: {F}"
        assert orb['period'] == 2 * orb['T_half']
        assert orb['state0'] == [orb['x0'], 0.0, 0.0, orb['vy0']]


def test_family_marches_monotonically_and_C_varies_smoothly():
    fam = compute_family_pseudo_arclength(LYAP_X, ds=1e-3, n_orbits=6, mu=MU,
                                          verbose=False)
    x0s = np.array([o['x0'] for o in fam])
    C = np.array([o['jacobi'] for o in fam])
    steps = np.diff(x0s)
    assert np.all(steps < 0) or np.all(steps > 0), f"family reversed: {steps}"
    # C is smooth along a family: second differences stay far below first ones
    assert np.abs(np.diff(C, 2)).max() < 0.1 * np.abs(np.diff(C)).max()
    # consecutive orbits sit ds apart in the full free-variable space
    d = np.linalg.norm(np.diff(np.array([o['X'] for o in fam]), axis=0), axis=1)
    assert np.allclose(d, 1e-3, rtol=1e-3), d


# ---------------------------------------------------------------------------
# 6. Floquet
# ---------------------------------------------------------------------------

def test_monodromy_is_symplectic_with_a_reciprocal_spectrum():
    """CR3BP monodromy is symplectic, so det = 1 and the multipliers come in
    reciprocal pairs with a trivial pair at 1.  This catches a period that is
    actually a half-period, and an STM assembled in the wrong order."""
    state0 = [LYAP_X[0], 0.0, 0.0, LYAP_X[1]]
    M, mult = monodromy_and_multipliers(state0, 2 * LYAP_X[2], MU)

    assert M.shape == (4, 4)
    assert abs(np.linalg.det(M) - 1.0) < 1e-6, np.linalg.det(M)
    assert abs(np.prod(mult).real - 1.0) < 1e-6

    mods = np.sort(np.abs(mult))
    assert abs(mods[0] * mods[-1] - 1.0) < 1e-5, "no reciprocal hyperbolic pair"
    assert mods[-1] > 10.0, "L1 Lyapunov orbits are strongly unstable"
    # the trivial pair sits on the unit circle
    unit = np.abs(np.abs(mult) - 1.0) < 1e-4
    assert unit.sum() >= 2, f"expected a trivial unit pair, got {np.abs(mult)}"


def test_manifold_strands_have_the_right_shape_and_leave_the_orbit():
    state0 = [LYAP_X[0], 0.0, 0.0, LYAP_X[1]]
    period = 2 * LYAP_X[2]
    strands = compute_manifold(state0, period, MU, stable=False, n_points=4,
                               epsilon=1e-6, n_strand_points=25)
    assert len(strands) == 4
    for s in strands:
        assert s.shape == (25, 4)
        assert np.all(np.isfinite(s))

    # an unstable strand must depart: the last point is further from the start
    # of the strand than the first, by a wide margin
    for s in strands:
        assert np.linalg.norm(s[-1] - s[0]) > 1e-4


# ---------------------------------------------------------------------------
# 7. The 3D (halo) constraint
# ---------------------------------------------------------------------------

def test_halo_constraint_vanishes_on_the_week8_target():
    F, dF, final_state = halo_periodicity_constraint(HALO_X, MU)
    assert np.abs(F).max() < 1e-9, F
    assert dF.shape == (3, 4)
    assert final_state.shape == (6,)


def test_halo_jacobian_matches_finite_differences():
    """Same argument as the planar case, one dimension up.  The 3D free-variable
    ordering [x0, z0, vy0, T] maps to STM columns 0, 2, 4 -- a mapping that is
    easy to get wrong and impossible to notice from the residual alone."""
    _, dF, _ = halo_periodicity_constraint(HALO_X, MU)
    h = 1e-7
    fd = np.zeros((3, 4))
    for j in range(4):
        Xp, Xm = HALO_X.copy(), HALO_X.copy()
        Xp[j] += h
        Xm[j] -= h
        Fp, _, _ = halo_periodicity_constraint(Xp, MU)
        Fm, _, _ = halo_periodicity_constraint(Xm, MU)
        fd[:, j] = (Fp - Fm) / (2 * h)

    err = np.abs(dF - fd).max() / max(1.0, np.abs(fd).max())
    assert err < 1e-5, f"analytic vs finite-difference differs by {err:.2e}\n{dF}\n{fd}"


def test_halo_tangent_and_arclength_step_work_on_the_3d_constraint():
    """The PAC engine is shared between 2D and 3D via `constraint_fn`; this
    pins that the 3x4 path through it actually solves."""
    _, dF, _ = halo_periodicity_constraint(HALO_X, MU)
    t = family_tangent(dF)
    assert t.shape == (4,)
    assert np.abs(dF @ t).max() < 1e-7

    ds = 1e-4
    X_new = pseudo_arclength_correct(
        predict_step(HALO_X, t, ds), HALO_X, t, ds, MU,
        constraint_fn=halo_periodicity_constraint)
    F_new, _, _ = halo_periodicity_constraint(X_new, MU)
    assert np.abs(F_new).max() < 1e-8, F_new
    assert abs(np.dot(X_new - HALO_X, t) - ds) < 1e-8


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
