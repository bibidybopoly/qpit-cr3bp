"""Tests for the Newton system, tangent, corrector, continuation and bundles
in src/qpit.py — the solver core.

Run: pytest tests/test_qpit_continuation.py -v
     (or python tests/test_qpit_continuation.py)

SLOW BY NECESSITY.  Everything here needs a genuinely converged torus, and
continuation needs several of them.  One torus is built at module scope
(N=15, ~1.5 s) and reused; the file runs in well under a minute.  It is the
only test file in the suite that integrates, so `-k "not continuation"` skips
essentially all of the runtime.

These six functions were the last untested public names in qpit.py and they are
the ones the research output actually comes from: `continue_torus_family`
produced every family in data/families/, and `torus_bundle` produced the
negative control of the gap experiment.  The tests are structural identities
(rank, nullity, orthogonality, the bundle eigen-relation) rather than recorded
numbers, so they survive a change of tolerances or step size.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import constants as C                                            # noqa: E402
import qpit                                                      # noqa: E402
from qpit import (assemble_system, torus_tangent,                # noqa: E402
                  correct_along_family, continue_torus_family,
                  torus_bundle, torus_manifold, stability_matrix,
                  stability_index, invariance_jacobian,
                  invariance_residual, mean_jacobi, curve_amplitude,
                  pack_X, unpack_X, flatten_curve, unflatten_curve,
                  longitudinal_tangent, latitudinal_tangent, newton_step,
                  node_angles)
from cr3bp_3d import MU_EARTH_MOON as MU                          # noqa: E402

HALO = dict(state0=list(C.HALO29_STATE), period=C.HALO29_PERIOD)
N = 15

_CACHE = {}


def torus():
    """One converged torus, built once. ~1.5 s."""
    if 'T' not in _CACHE:
        U0, T0, r0 = qpit.seed_from_halo(HALO, MU, N, 1e-4)
        out = qpit.converge_torus(U0, T0, r0, MU, tol=1e-12, itmax=15,
                                  verbose=False)
        assert out['converged'], out['code']
        _CACHE['T'] = (out['U0'], out['T'], out['rho'])
    return _CACHE['T']


def DG_cached():
    if 'DG' not in _CACHE:
        U, T, r = torus()
        _CACHE['DG'] = stability_matrix(U, T, r, MU)
    return _CACHE['DG']


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


# ---------------------------------------------------------------------------
# 1. The assembled system
# ---------------------------------------------------------------------------

def test_system_has_the_documented_row_and_column_layout():
    U, T, r = torus()
    J, rvec, info = assemble_system(U, T, r, MU, pin='C')
    assert J.shape == (6 * N + 3, 6 * N + 2)      # 6N invariance + 3 scalars
    assert rvec.shape == (6 * N + 3,)
    for k in ('res_inf', 'res_2', 'phase0', 'phase1', 'pin_err', 'arclen_err'):
        assert k in info
    assert info['arclen_err'] is None             # no arclength row requested

    t, _ = torus_tangent(U, T, r, MU, pin='C')
    J2, rv2, info2 = assemble_system(U, T, r, MU, pin='C', tangent=t,
                                     X_prev=pack_X(U, T, r), ds=1e-6)
    assert J2.shape == (6 * N + 4, 6 * N + 2)     # the arclength row is added
    assert rv2.shape == (6 * N + 4,)
    assert info2['arclen_err'] is not None
    assert np.array_equal(J2[-1], t), "the last row must BE the tangent"


def test_dropping_rho_drops_the_last_column():
    U, T, r = torus()
    J, _, _ = assemble_system(U, T, r, MU, pin='C', free_rho=False)
    assert J.shape == (6 * N + 3, 6 * N + 1)


def test_rank_climbs_exactly_as_the_constraint_rows_are_added():
    """The whole design argument, as a rank ladder:

        invariance alone            nullity 4  (2 phases + 2 physical)
        + 2 phase rows + 1 pin      nullity 1  (the family direction)
        + arclength                 nullity 0  (isolated)

    The 6N+1 -> 6N+2 jump is what the docstring calls "the cheapest
    confirmation that the arclength row is doing its job".
    """
    U, T, r = torus()
    JU, JT, Jr, _ = invariance_jacobian(U, T, r, MU)
    block = np.hstack([JU, JT[:, None], Jr[:, None]])
    sv_b = np.linalg.svd(block, compute_uv=False)
    # a (6N) x (6N+2) block of rank 6N-2 leaves 4 directions unconstrained
    assert np.linalg.matrix_rank(block, tol=sv_b[0] * 1e-8) == 6 * N - 2

    J1, _, _ = assemble_system(U, T, r, MU, pin='C')
    sv1 = np.linalg.svd(J1, compute_uv=False)
    assert sv1[-2] / sv1[-1] > 1e3, "nullspace is not cleanly 1-D"
    assert np.linalg.matrix_rank(J1, tol=sv1[0] * 1e-8) == 6 * N + 1

    t, _ = torus_tangent(U, T, r, MU, pin='C')
    J2, _, _ = assemble_system(U, T, r, MU, pin='C', tangent=t,
                               X_prev=pack_X(U, T, r), ds=1e-6)
    sv2 = np.linalg.svd(J2, compute_uv=False)
    assert np.linalg.matrix_rank(J2, tol=sv2[0] * 1e-8) == 6 * N + 2
    assert sv2[-1] / sv1[-1] > 1e3, "the arclength row did not lift the nullspace"


def test_system_reproduces_newton_step_exactly():
    """The docstring says: "With pin='C' and tangent=None this reproduces
    newton_step's matrix and right-hand side exactly, bit for bit; tests assert
    it."  No test did.  This is that test.

    newton_step does not return its matrix, but it returns the singular values
    of it — which pin the matrix up to rotation — plus every scalar the two
    share.  `newton_step` is deliberately left untouched so every Week 8 number
    stays reproducible; that only holds while the two paths agree.
    """
    U, T, r = torus()
    J, rvec, info = assemble_system(U, T, r, MU, pin='C')
    _, _, _, ninfo = newton_step(U, T, r, MU)

    # Every scalar the two paths share is EXACTLY equal -- those come from
    # identical code, so exact equality is the right bar and it holds.
    assert info['res_inf'] == ninfo['res_inf']
    assert info['res_2'] == ninfo['res_2']
    assert info['phase0'] == ninfo['phase0']
    assert info['phase1'] == ninfo['phase1']
    assert info['pin_err'] == ninfo['dC']

    # The singular values agree to ~1e-15 relative but are NOT bit-identical,
    # and that is not evidence against the docstring's "bit for bit" claim about
    # the MATRIX: newton_step's `sv` comes out of lstsq (LAPACK gelsd) while
    # this one comes from svd (gesdd).  Two different routines on the same
    # matrix differ in the last ulp.  The near-zero tail is looser, as it must
    # be -- it is a rendering of zero, not a value.
    sv = np.linalg.svd(J, compute_uv=False)
    assert sv.shape == ninfo['sv'].shape
    rel = np.abs(sv[:-1] - ninfo['sv'][:-1]) / np.maximum(sv[:-1], 1e-300)
    assert rel.max() < 1e-10, rel.max()
    assert sv[-1] < 1e-12 * sv[0] and ninfo['sv'][-1] < 1e-10 * sv[0]


def test_residual_info_agrees_with_invariance_residual_to_the_augmented_floor():
    """NOT bit-identical, and the reason is the accuracy floor Week 8 recorded.

    `assemble_system` takes F from `invariance_jacobian`, which propagates the
    42-dim augmented system; `invariance_residual` propagates the 6-dim state
    alone.  DOP853 sizes its steps from the error estimate over ALL integrated
    components, so adding the STM changes the step sequence and shifts the final
    state by ~1e-10.  Measured here: 8.9e-13 against 6.6e-11.

    That is the same floor `test_qpit_integration.py` documents, and it is why
    convergence should not be asked for tighter than about 1e-11.
    """
    U, T, r = torus()
    _, _, info = assemble_system(U, T, r, MU, pin='C')
    direct = float(np.abs(invariance_residual(U, T, r, MU)).max())
    assert info['res_inf'] < 1e-10
    assert direct < 1e-10
    assert abs(info['res_inf'] - direct) < 1e-10, (info['res_inf'], direct)
    assert info['res_2'] >= info['res_inf']


def test_pin_target_defaults_to_the_current_value_of_the_pinned_quantity():
    """A pin whose target is read off the state pins zero error at the start —
    which is what makes 'START FROM A CONVERGED TORUS' load-bearing."""
    U, T, r = torus()
    for pin in ('C', 'rho', 'T'):
        _, _, info = assemble_system(U, T, r, MU, pin=pin)
        assert abs(info['pin_err']) < 1e-12, (pin, info['pin_err'])
    # and an explicit target shows up as a non-zero pin error
    _, _, info = assemble_system(U, T, r, MU, pin='rho', pin_target=r + 1e-4)
    assert abs(info['pin_err']) > 1e-6


# ---------------------------------------------------------------------------
# 2. The family tangent
# ---------------------------------------------------------------------------

def test_tangent_is_a_unit_nullvector_of_the_constrained_system():
    U, T, r = torus()
    t, diag = torus_tangent(U, T, r, MU, pin='rho')
    assert t.shape == (6 * N + 2,)
    assert abs(np.linalg.norm(t) - 1.0) < 1e-12

    J, _, _ = assemble_system(U, T, r, MU, pin='rho')
    assert np.abs(J @ t).max() / np.abs(J).max() < 1e-10
    assert diag['sv_null'] < 1e-8 * diag['sv_next']
    assert diag['gap'] > 1e3
    assert 'warning' not in diag


def test_tangent_is_orthogonal_to_both_phase_symmetries():
    """The reason the tangent is taken from the CONSTRAINED system.

    The invariance block alone has nullity 4, two of which are the phase
    symmetries.  A tangent drawn from there would generically be a mixture,
    and the predictor would step along a relabelling of the same curve —
    amplitude unchanged, residual small, "a family that goes nowhere while
    looking healthy".
    """
    U, T, r = torus()
    t, _ = torus_tangent(U, T, r, MU, pin='rho')
    tU = flatten_curve(unflatten_curve(t[:U.size]))

    for name, tan in (('longitudinal', longitudinal_tangent(U, T, r, MU)),
                      ('latitudinal', latitudinal_tangent(U))):
        v = flatten_curve(tan)
        v = v / np.linalg.norm(v)
        overlap = abs(float(np.dot(tU, v))) / max(np.linalg.norm(tU), 1e-300)
        assert overlap < 1e-6, (name, overlap)


def test_tangent_orientation_increases_amplitude_by_default():
    """The small-amplitude end of every family here is the periodic orbit
    itself, so an unoriented tangent that happened to point inward would march
    the continuation straight back into the halo."""
    U, T, r = torus()
    t, diag = torus_tangent(U, T, r, MU, pin='rho')
    assert diag['dA_ds'] > 0, diag['dA_ds']

    # a finite step along +t really does grow the curve
    A0 = curve_amplitude(U)
    h = 0.01 * A0 / np.max(np.linalg.norm(unflatten_curve(t[:U.size])[:, :3], axis=1))
    assert curve_amplitude(U + h * unflatten_curve(t[:U.size])) > A0


def test_tangent_sign_follows_the_previous_step():
    U, T, r = torus()
    t, _ = torus_tangent(U, T, r, MU, pin='rho')
    same, d_same = torus_tangent(U, T, r, MU, pin='rho', prev_tangent=t)
    flip, d_flip = torus_tangent(U, T, r, MU, pin='rho', prev_tangent=-t)
    assert np.allclose(same, t, rtol=0, atol=1e-12)
    assert np.allclose(flip, -t, rtol=0, atol=1e-12)
    assert d_flip['dA_ds'] == -d_same['dA_ds']


def test_tangent_warns_when_the_nullspace_is_not_cleanly_one_dimensional():
    """`gap_warn` is the bifurcation guard; it has to actually fire."""
    U, T, r = torus()
    _, diag = torus_tangent(U, T, r, MU, pin='rho', gap_warn=1e30)
    assert 'warning' in diag
    assert 'bifurcation' in diag['warning']


def test_the_pin_zeroes_the_component_it_pins():
    """The pin chooses WHICH one-parameter family is marched along.  Near this
    torus the two directions are almost parallel (|cos| = 0.9999994), so
    comparing them wholesale is not the discriminating test -- the sharp one is
    that pin='rho' drives the tangent's rho-component to zero, because a family
    at constant rho cannot move rho, while pin='C' leaves it free.
    """
    U, T, r = torus()
    t_rho, _ = torus_tangent(U, T, r, MU, pin='rho')
    t_C, _ = torus_tangent(U, T, r, MU, pin='C')

    assert abs(t_rho[-1]) < 1e-12, ("pin='rho' left a rho component", t_rho[-1])
    assert abs(t_C[-1]) > 1e-6, ("pin='C' should leave rho free", t_C[-1])
    assert not np.allclose(t_rho, t_C, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. The arclength-closed corrector
# ---------------------------------------------------------------------------

def test_corrector_returns_a_converged_torus_from_a_perturbed_start():
    U, T, r = torus()
    rng = np.random.default_rng(0)
    U_bad = U + 1e-8 * curve_amplitude(U) * rng.standard_normal(U.shape)

    out = correct_along_family(U_bad, T, r, MU, U, 'rho', r, verbose=False)
    assert out['converged'], out.get('code')
    F = invariance_residual(out['U0'], out['T'], out['rho'], MU)
    assert np.abs(F).max() < 1e-10


def test_corrector_holds_the_pinned_quantity():
    U, T, r = torus()
    rng = np.random.default_rng(1)
    U_bad = U + 1e-8 * curve_amplitude(U) * rng.standard_normal(U.shape)
    out = correct_along_family(U_bad, T, r, MU, U, 'rho', r, verbose=False)
    assert out['converged']
    assert abs(out['rho'] - r) < 1e-9, "pin='rho' did not hold rho"

    C0 = mean_jacobi(U, MU)
    out2 = correct_along_family(U_bad, T, r, MU, U, 'C', C0, verbose=False)
    assert out2['converged']
    assert abs(mean_jacobi(out2['U0'], MU) - C0) < 1e-10


def test_corrector_meets_the_arclength_constraint_when_given_one():
    """This is the step continuation actually takes: predict along t by ds,
    then correct back to the family WHILE staying ds along t."""
    U, T, r = torus()
    t, _ = torus_tangent(U, T, r, MU, pin='rho')
    X_prev = pack_X(U, T, r)
    ds = 0.02 * curve_amplitude(U) * np.sqrt(N)
    Xp = X_prev + ds * t
    Up, Tp, rp = unpack_X(Xp)

    out = correct_along_family(Up, Tp, rp, MU, U, 'rho', r, tangent=t,
                               X_prev=X_prev, ds=ds, verbose=False)
    assert out['converged'], out.get('code')
    X_new = pack_X(out['U0'], out['T'], out['rho'])
    assert abs(float(np.dot(X_new - X_prev, t)) - ds) < 1e-9 * ds
    assert np.abs(invariance_residual(out['U0'], out['T'],
                                      out['rho'], MU)).max() < 1e-10
    assert curve_amplitude(out['U0']) > curve_amplitude(U)


# ---------------------------------------------------------------------------
# 4. Continuation
# ---------------------------------------------------------------------------

def _family(n_steps=3, pin='rho'):
    key = ('fam', n_steps, pin)
    if key not in _CACHE:
        U, T, r = torus()
        _CACHE[key] = continue_torus_family(U, T, r, MU, pin=pin,
                                            n_steps=n_steps, verbose=False)
    return _CACHE[key]


def test_continuation_returns_the_starting_torus_as_member_zero():
    U, T, r = torus()
    fam = _family()
    m0 = fam['members'][0]
    assert np.array_equal(m0['U0'], U)
    assert m0['T'] == T and m0['rho'] == r
    assert m0['ds'] == 0.0 and m0['n_iter'] == 0


def test_every_member_is_a_converged_torus():
    """Continuation corrects each member from the previous one, so error can
    accumulate silently while the family still looks smooth.  Checking the LAST
    member is the point."""
    fam = _family()
    assert len(fam['members']) >= 2
    for i, m in enumerate(fam['members']):
        F = invariance_residual(m['U0'], m['T'], m['rho'], MU)
        assert np.abs(F).max() < 1e-9, (i, np.abs(F).max())
        assert m['res_inf'] < 1e-9
        assert m['tail'] < 1e-6, (i, m['tail'])


def test_continuation_holds_the_pin_across_the_whole_family():
    """pin='rho' means every member shares the rotation number; that is what
    makes the family one-parameter in amplitude rather than a two-parameter
    smear."""
    fam = _family(pin='rho')
    assert fam['pin'] == 'rho'
    rhos = np.array([m['rho'] for m in fam['members']])
    assert np.abs(rhos - fam['pin_target']).max() < 1e-9, rhos


def test_continuation_marches_monotonically_in_amplitude():
    fam = _family()
    A = np.array([m['amplitude'] for m in fam['members']])
    assert np.all(np.diff(A) > 0), A
    # and C varies smoothly: second differences well below first
    Cv = np.array([m['C'] for m in fam['members']])
    if Cv.size >= 3:
        assert np.abs(np.diff(Cv, 2)).max() <= np.abs(np.diff(Cv)).max()


def test_continuation_steps_are_scaled_to_the_curve_amplitude():
    """ds = ds_frac * A * sqrt(N) makes each node move by about ds_frac * A,
    which is why the same ds_frac works across three orders of magnitude of
    family amplitude and never needs retuning."""
    U, T, r = torus()
    fam = _family()
    ds1 = fam['members'][1]['ds']
    expect = 0.10 * curve_amplitude(U) * np.sqrt(N)
    assert 0.4 * expect < ds1 <= 1.01 * expect, (ds1, expect)

    # consecutive members really are about ds apart in X
    X = [pack_X(m['U0'], m['T'], m['rho']) for m in fam['members'][:2]]
    assert abs(np.linalg.norm(X[1] - X[0]) - ds1) < 0.2 * ds1


def test_continuation_reports_why_it_stopped():
    fam = _family()
    assert fam['terminated'] == 'n_steps'
    assert len(fam['members']) == 4          # the seed plus n_steps


def test_continuation_records_the_fields_downstream_code_reads():
    """densify, sampling and torus_field all consume these member records."""
    for m in _family()['members']:
        for k in ('U0', 'T', 'rho', 'C', 'amplitude', 'N', 'res_inf',
                  'offgrid', 'tail', 'n_iter', 'ds'):
            assert k in m, k
        assert m['U0'].shape == (m['N'], 6)
        assert abs(m['C'] - mean_jacobi(m['U0'], MU)) < 1e-12
        assert abs(m['amplitude'] - curve_amplitude(m['U0'])) < 1e-15


# ---------------------------------------------------------------------------
# 5. The bundle
# ---------------------------------------------------------------------------

def _dominant_projector(DG):
    """Projector onto the real span of the dominant eigenvalue circle."""
    ev, evec = np.linalg.eig(DG)
    top = np.abs(ev) > 0.5 * np.abs(ev).max()
    Q, _ = np.linalg.qr(np.hstack([evec[:, top].real, evec[:, top].imag]))
    return Q @ Q.T, int(top.sum()), float(np.abs(ev)[top].mean())


def test_the_dominant_eigenvalues_form_a_circle_of_N_equal_moduli():
    """The structure the docstring derives: if w solves the bundle equation
    then e^{ik theta} w solves it with lambda e^{-ik rho}, so each real bundle
    contributes N eigenvalues of EQUAL MODULUS.  Measured: 15 of them at
    146.35, against 90 eigenvalues total."""
    DG = DG_cached()
    _, n_top, modulus = _dominant_projector(DG)
    assert n_top == N, (n_top, N)
    mod = np.sort(np.abs(np.linalg.eigvals(DG)))[::-1]
    assert np.ptp(mod[:N]) / mod[0] < 1e-5, "the top circle is not degenerate"
    assert mod[N] < 2.0, "the next circle should be the trivial one at ~1"
    assert abs(modulus - 146.4) < 1.0, modulus


def test_bundle_converges_into_the_dominant_subspace_not_onto_an_eigenvector():
    """WHAT THE POWER ITERATION ACTUALLY DOES.

    The docstring says convergence runs "at a rate set by the modulus gap,
    which here is R_u : 1 = 146 : 1, so 12 iterations leave 1e-26".  That rate
    is right for reaching the dominant CIRCLE -- and the iterate does land in
    that 15-D invariant subspace to ~1e-8.

    It is not right for the returned vector.  Within the circle all N
    eigenvalues share a modulus, so the ratio governing convergence to a single
    eigenvector is 1:1 and the iterate keeps rotating inside the subspace
    forever.  Measured: cos(DG v, v) sits around 0.05-0.20, nowhere near 1, and
    does not improve with more iterations.

    This is not a correctness problem -- any direction in the unstable bundle
    grows at ~R per period, which is all the negative control and the manifold
    globalisation need.  But `V` is not a canonical vector, and the docstring's
    1e-26 should not be read as applying to it.
    """
    U, T, r = torus()
    DG = DG_cached()
    P, _, modulus = _dominant_projector(DG)

    for n in (8, 12, 20):
        V, R = torus_bundle(U, T, r, MU, n_power=n, DG=DG)
        v = flatten_curve(V)
        v = v / np.linalg.norm(v)
        assert np.linalg.norm(P @ v) / np.linalg.norm(v) > 1 - 1e-6, n
        assert abs(np.linalg.norm(DG @ v) - modulus) / modulus < 1e-3, n

    v12 = flatten_curve(torus_bundle(U, T, r, MU, n_power=12, DG=DG)[0])
    v12 /= np.linalg.norm(v12)
    w = DG @ v12
    assert abs(float(w @ v12)) / np.linalg.norm(w) < 0.9, (
        "the iterate has become an eigenvector -- if the circle degeneracy was "
        "broken, this whole test and its docstring need revisiting")


def test_bundle_directions_are_stable_across_the_iteration_count():
    """The reassuring half of the previous test.

    R oscillates at ~1e-4 because the iterate keeps rotating inside the
    degenerate circle -- but the per-node DIRECTIONS, which is what `V` is and
    the only thing the negative control and the manifold globalisation consume,
    are stable to ~5e-7 from n_power=2 all the way to 100.  Row normalisation
    absorbs the within-circle rotation.

    That is what makes a regenerated dataset reproducible: the displacement
    directions do not depend on the iteration count, even though the reported
    growth factor wobbles in its fourth digit.
    """
    U, T, r = torus()
    DG = DG_cached()
    ref, _ = torus_bundle(U, T, r, MU, n_power=12, DG=DG)
    for n in (2, 4, 8, 20, 50):
        V, _ = torus_bundle(U, T, r, MU, n_power=n, DG=DG)
        align = np.abs(np.sum(V * ref, axis=1))
        assert align.min() > 1 - 1e-5, (n, align.min())


def test_bundle_rows_are_unit_vectors():
    U, T, r = torus()
    V, _ = torus_bundle(U, T, r, MU, DG=DG_cached())
    assert V.shape == (N, 6)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, rtol=0, atol=1e-12)


def test_bundle_growth_factor_reproduces_the_stability_index():
    """"which should reproduce the radius behind stability_index, giving an
    independent check on both" -- computed by completely different routes
    (power iteration vs eigensolve).  They agree to ~1e-4, which is the
    oscillation amplitude of R within the degenerate circle, not 1e-26."""
    U, T, r = torus()
    DG = DG_cached()
    _, R = torus_bundle(U, T, r, MU, which='unstable', DG=DG)
    nu = stability_index(DG)
    assert abs(nu - 0.5 * (R + 1.0 / R)) / nu < 1e-3
    assert R > 100, "L2 halo #29's torus should be strongly unstable"
    assert abs(R - 146.4) < 1.0, R


def test_stable_bundle_is_the_reciprocal_of_the_unstable_one():
    """Reciprocal to ~1e-4, for the same reason: both radii are read off an
    iterate that is still rotating inside its circle."""
    U, T, r = torus()
    DG = DG_cached()
    _, Ru = torus_bundle(U, T, r, MU, which='unstable', DG=DG)
    _, Rs = torus_bundle(U, T, r, MU, which='stable', DG=DG)
    assert abs(Ru * Rs - 1.0) < 1e-3, (Ru, Rs)
    assert Rs < 1.0 < Ru


def test_stable_bundle_actually_contracts_under_the_forward_map():
    """The reciprocity check alone does NOT test this.

    `torus_bundle` returns `1.0 / R` for which='stable', so a version that
    forgot to invert DG before iterating would still report a reciprocal pair
    -- the number would be right and the DIRECTION would be the unstable one.
    Verified by mutation: replacing `np.linalg.inv(DG)` with `DG` leaves every
    reciprocity assertion passing.

    The discriminating statement is about the vector: the stable bundle must
    CONTRACT under the forward stroboscopic map, by the same factor the
    unstable one grows.
    """
    U, T, r = torus()
    DG = DG_cached()
    Vu, Ru = torus_bundle(U, T, r, MU, which='unstable', DG=DG)
    Vs, Rs = torus_bundle(U, T, r, MU, which='stable', DG=DG)

    vu = flatten_curve(Vu); vu /= np.linalg.norm(vu)
    vs = flatten_curve(Vs); vs /= np.linalg.norm(vs)

    grow = np.linalg.norm(DG @ vu)
    shrink = np.linalg.norm(DG @ vs)
    assert grow > 100.0, grow
    assert shrink < 0.02, ("the stable bundle grows under the forward map -- "
                           "it is the unstable direction", shrink)
    assert abs(grow * shrink - 1.0) < 1e-3, (grow, shrink)
    # and the two bundles are genuinely different directions
    assert abs(float(vu @ vs)) < 0.9


def test_bundle_is_deterministic_for_a_fixed_n_power():
    """The seeded RNG makes repeated calls identical, which is what keeps a
    regenerated negative control matching the one that was analysed -- provided
    n_power is also unchanged (see the KNOWN_GAP test above)."""
    U, T, r = torus()
    DG = DG_cached()
    a, Ra = torus_bundle(U, T, r, MU, DG=DG)
    b, Rb = torus_bundle(U, T, r, MU, DG=DG)
    assert np.array_equal(a, b) and Ra == Rb


# ---------------------------------------------------------------------------
# 6. The manifold
# ---------------------------------------------------------------------------

def test_manifold_returns_one_strand_per_node():
    U, T, r = torus()
    V, _ = torus_bundle(U, T, r, MU, DG=DG_cached())
    strands = torus_manifold(U, T, r, MU, V, d=1e-6, n_periods=1.0, n_out=40)
    assert len(strands) == N
    for s in strands:
        s = np.asarray(s)
        assert s.shape == (40, 6)
        assert np.all(np.isfinite(s))


def test_unstable_strands_depart_from_the_torus():
    """d = 1e-6 grows by R per stroboscopic period, so after one period the
    displacement should be visibly larger than it started."""
    U, T, r = torus()
    V, R = torus_bundle(U, T, r, MU, DG=DG_cached())
    strands = torus_manifold(U, T, r, MU, V, d=1e-6, n_periods=1.0, n_out=40)
    grew = 0
    for j, s in enumerate(strands):
        s = np.asarray(s)
        d0 = np.linalg.norm(s[0] - U[j])
        d1 = min(np.linalg.norm(U - s[-1], axis=1))
        if d1 > 5 * d0:
            grew += 1
    assert grew >= N // 2, f"only {grew}/{N} strands departed"


def test_manifold_sides_leave_in_opposite_directions():
    """side=+1 and side=-1 are the two branches of the same manifold; they must
    start on opposite sides of the curve."""
    U, T, r = torus()
    V, _ = torus_bundle(U, T, r, MU, DG=DG_cached())
    plus = torus_manifold(U, T, r, MU, V, d=1e-6, n_periods=0.25, n_out=10,
                          side=+1)
    minus = torus_manifold(U, T, r, MU, V, d=1e-6, n_periods=0.25, n_out=10,
                           side=-1)
    for j in range(N):
        a = np.asarray(plus[j])[0] - U[j]
        b = np.asarray(minus[j])[0] - U[j]
        assert float(np.dot(a, b)) < 0, j
        assert abs(np.linalg.norm(a) - np.linalg.norm(b)) < 1e-15


def test_manifold_displacement_size_is_what_was_asked_for():
    U, T, r = torus()
    V, _ = torus_bundle(U, T, r, MU, DG=DG_cached())
    for d in (1e-7, 1e-6, 1e-5):
        s = torus_manifold(U, T, r, MU, V, d=d, n_periods=0.05, n_out=5)
        start = np.asarray(s[0])[0]
        assert abs(np.linalg.norm(start - U[0]) - d) < 1e-3 * d


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
