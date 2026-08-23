"""Week 8 QPIT tests.

Run: pytest tests/test_qpit_week8.py -v   (or python tests/test_qpit_week8.py)

No pytest dependency: the module-scoped `halo` fixture this file used to carry
was the only `import pytest` in the suite, and it made the file the only one
that could not run under the standalone runner.  A plain module-level constant
is exactly equivalent -- the fixture returned a literal and nothing mutates it.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import constants as C                            # noqa: E402

from cr3bp_3d import MU_EARTH_MOON as MU, halo_stability   # noqa: E402
import qpit                                                # noqa: E402


# ---------------------------------------------------------------------------
# The Week 8 target orbit
# ---------------------------------------------------------------------------

#: L2 halo #29 -- the Week 8 target.
#:
#: C = 3.060334, rho/2pi = 0.293229, hyperbolic multiplier 146.4.
#:
#: Hardcoded rather than rebuilt from orbits.compute_halo_family so the suite
#: stays fast and independent of continuation settings. Regenerated from a
#: Richardson L2 seed (Az = 0.03) corrected by halo_corrector, then pseudo-
#: arclength continuation at ds = 0.004; the target came out at index 75 of
#: that family, so the INDEX is not portable but the orbit is. Southern
#: branch (z0 < 0); the family is symmetric in z.
#:
#: If these ever need refreshing, the identifying pair is (C, rho/2pi), not
#: the index.  Read-only: `qpit.seed_from_halo` and `halo_stability` both
#: only read from it, so one shared dict is safe.
HALO = dict(state0=list(C.HALO29_STATE), period=C.HALO29_PERIOD)


# ---------------------------------------------------------------------------
# 1. Jacobi gradient agrees with the vector field
# ---------------------------------------------------------------------------

def test_jacobi_gradient_orthogonal_to_flow():
    """dC/dx . f(x) == 0 for arbitrary states.

    C is conserved along the flow, so this identity must hold exactly. It is
    convention-proof: it fails if jacobi_gradient and cr3bp_rhs_3d disagree on
    where mu sits, on the sign of the Coriolis block, or on the form of omega.
    One assertion covers all of it.
    """
    rng = np.random.default_rng(20260805)
    for _ in range(20):
        x = rng.normal(size=6) * 0.5 + np.array([0.9, 0.1, 0.1, 0.0, 0.3, 0.0])
        val = qpit.jacobi_gradient(x, MU) @ qpit.vector_field(x, MU)
        assert abs(val) < 1e-12, f"dC/dx . f = {val:.3e} at {x}"


def test_jacobi_gradient_matches_finite_difference():
    rng = np.random.default_rng(7)
    x = rng.normal(size=6) * 0.3 + np.array([1.05, 0.0, 0.08, 0.0, 0.38, 0.0])
    g = qpit.jacobi_gradient(x, MU)
    h = 1e-6
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (qpit.jacobi_constant(xp, MU) - qpit.jacobi_constant(xm, MU)) / (2 * h)
        assert abs(fd - g[i]) < 1e-6 * max(1.0, abs(g[i]))


# ---------------------------------------------------------------------------
# 2. Seed sign convention
# ---------------------------------------------------------------------------

def test_seed_residual_is_second_order():
    """The derived seed leaves an O(eps^2) residual, the flipped one O(eps).

    This is the test that catches a seed/residual convention mismatch. Run at
    small eps: the quadratic coefficient is ~1e3, so the two regimes only
    separate below eps ~ 1e-3.
    """
    state0 = np.asarray(HALO['state0'], float)
    T0 = HALO['period']
    c = halo_stability(state0, T0, MU)['center']
    v = qpit.canonical_eigenvector_phase(c['vec'])
    rho = qpit.seed_rho(c['mu'])

    N = 15
    th = qpit.node_angles(N)

    def res(sign, eps):
        U = state0[None, :] + eps * (
            np.cos(th)[:, None] * np.real(v)[None, :]
            + sign * np.sin(th)[:, None] * np.imag(v)[None, :])
        return np.abs(qpit.invariance_residual(U, T0, rho, MU)).max()

    # eps steps by 10x; quadratic -> ratio ~100, linear -> ratio ~10
    good = res(-1, 1e-4) / res(-1, 1e-5)
    bad = res(+1, 1e-4) / res(+1, 1e-5)
    assert good > 50.0, f"derived seed not quadratic (ratio {good:.1f})"
    assert bad < 20.0, f"flipped seed unexpectedly quadratic (ratio {bad:.1f})"

    # and seed_curve must be using the derived sign
    U_mod = qpit.seed_curve(state0, v, N, 1e-4)
    U_ref = state0[None, :] + 1e-4 * (
        np.cos(th)[:, None] * np.real(v)[None, :]
        - np.sin(th)[:, None] * np.imag(v)[None, :])
    assert np.allclose(U_mod, U_ref), "seed_curve sign does not match the derivation"


# ---------------------------------------------------------------------------
# 3. Rank deficiency of the invariance block
# ---------------------------------------------------------------------------

def test_invariance_block_rank_deficiency_is_two():
    """Exactly two singular values collapse at a converged torus.

    Structural prediction: the solution set is 4-dimensional in [U0, T, rho]
    (2 phase symmetries + 2 physical parameters), so the 6N x (6N+2) block has
    rank 6N - 2. Verified by the size of the gap in the spectrum.
    """
    N = 15
    U0, T, rho = qpit.seed_from_halo(HALO, MU, N, 1e-4)
    out = qpit.converge_torus(U0, T, rho, MU, tol=1e-12, itmax=15, verbose=False)
    assert out['converged'], out['code']

    JU, JT, Jr, _ = qpit.invariance_jacobian(out['U0'], out['T'], out['rho'], MU)
    sv = np.linalg.svd(np.hstack([JU, JT[:, None], Jr[:, None]]), compute_uv=False)
    gap = sv[-3] / sv[-2]
    assert gap > 100.0, (
        f"expected a large gap after two small singular values, got {gap:.2e}; "
        f"rank deficiency is not 2")


# ---------------------------------------------------------------------------
# 4. Convergence and the properties of the result
# ---------------------------------------------------------------------------

def test_first_torus_converges_and_is_a_torus():
    N = 31
    U0, T, rho = qpit.seed_from_halo(HALO, MU, N, 1e-4)
    out = qpit.converge_torus(U0, T, rho, MU, tol=1e-12, itmax=15, verbose=False)

    assert out['converged'], out['code']
    assert out['n_iter'] <= 6, "convergence should be quadratic from a good seed"

    # off-grid: invariance must hold between the nodes too, not only at them
    og = qpit.offgrid_residual(out['U0'], out['T'], out['rho'], MU)
    assert og < 1e-9, f"off-grid residual {og:.3e}: N too small"

    # a torus is an invariant set of an autonomous system, so C is constant on
    # it. Only the MEAN is constrained, so per-node agreement is a real test.
    Cn = np.array([qpit.jacobi_constant(out['U0'][j], MU) for j in range(N)])
    assert Cn.max() - Cn.min() < 1e-11

    # amplitude must survive: collapse to the periodic orbit is the failure
    assert out['amplitude'] > 0.5 * qpit.curve_amplitude(U0)


def test_wrong_sign_seed_collapses():
    """The flipped seed falls into v == x_halo, which solves the invariance
    equation for any rho. Amplitude, not residual, is what reveals it."""
    state0 = np.asarray(HALO['state0'], float)
    T0 = HALO['period']
    c = halo_stability(state0, T0, MU)['center']
    v = qpit.canonical_eigenvector_phase(c['vec'])
    rho = qpit.seed_rho(c['mu'])
    N = 15
    th = qpit.node_angles(N)
    U_bad = state0[None, :] + 1e-4 * (
        np.cos(th)[:, None] * np.real(v)[None, :]
        + np.sin(th)[:, None] * np.imag(v)[None, :])

    out = qpit.converge_torus(U_bad, T0, rho, MU, tol=1e-12, itmax=15, verbose=False)
    assert not out['converged']
    assert out['amplitude'] < 0.5 * qpit.curve_amplitude(U_bad)


def test_pinning_rho_leaves_a_cubic_floor():
    """Holding rho at the linear value over-constrains the system.

    rho at the seed is the zero-amplitude rotation number; the real torus is
    shifted by O(eps^2), so pinning it asks for a torus that does not exist.
    The stalled residual is N-independent, which is what distinguishes an
    inconsistent constraint from a Fourier truncation error.
    """
    floors = []
    for N in (15, 31):
        U0, T, rho = qpit.seed_from_halo(HALO, MU, N, 1e-3)
        out = qpit.converge_torus(U0, T, rho, MU, tol=1e-13, itmax=10,
                                  free_rho=False, verbose=False)
        floors.append(out['history'][-1]['res_inf'])

    assert min(floors) > 1e-10, "expected a visible floor when rho is pinned"
    assert abs(floors[0] - floors[1]) < 0.5 * floors[0], (
        "floor moved with N — that would make it truncation, not inconsistency")

    # and freeing rho must clear it
    U0, T, rho = qpit.seed_from_halo(HALO, MU, 31, 1e-3)
    free = qpit.converge_torus(U0, T, rho, MU, tol=1e-12, itmax=15, verbose=False)
    assert free['converged']
    assert free['history'][-1]['res_inf'] < 1e-11


# ---------------------------------------------------------------------------
# 5. Spectral machinery
# ---------------------------------------------------------------------------

def test_evaluate_curve_reproduces_nodes():
    N = 15
    U0, _, _ = qpit.seed_from_halo(HALO, MU, N, 1e-3)
    back = qpit.evaluate_curve(U0, qpit.node_angles(N))
    assert np.allclose(back, U0, atol=1e-12)


def test_rotation_operator_matches_interpolation():
    """R(rho) U and evaluate_curve(U, theta + rho) must agree exactly."""
    N = 15
    U0, _, _ = qpit.seed_from_halo(HALO, MU, N, 1e-3)
    rho = 0.7
    lhs = qpit.rotation_operator(N, rho) @ U0
    rhs = qpit.evaluate_curve(U0, qpit.node_angles(N) + rho)
    assert np.allclose(lhs, rhs, atol=1e-12)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
