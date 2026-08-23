"""
Integration checks: qpit.py against the real cr3bp_3d.py / orbits.py.

DESTINATION: tests/test_qpit_integration.py

Unlike tests/test_qpit.py, these DO propagate trajectories, so they are slow
(seconds, not milliseconds) and they exercise the adapter wiring.  Scope is
deliberately limited to verifying that the residual and Jacobian are what
they claim to be.  Driving Newton to convergence is Week 8.

Run:  python tests/test_qpit_integration.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import qpit  # noqa: E402
from cr3bp_3d import (  # noqa: E402
    MU_EARTH_MOON, richardson_halo_seed, halo_corrector,
)

MU = MU_EARTH_MOON
N = 9          # small on purpose; FD cost scales as 6N
EPS = 1e-5
# Two step sizes, because the two blocks have different noise floors.
# State perturbations act on O(1) quantities; T and rho act through a
# residual that is only O(1e-6) near the seed, so they need a coarser h to
# stay out of subtractive cancellation.  See invariance_jacobian docstring.
H_FD_STATE = 1e-6
H_FD_PARAM = 1e-5

# Threshold is loose on purpose.  A transposed Kronecker order or a wrong
# STM block would give O(1) relative error, not 1e-3, so this is decisive
# for what it is testing without being a tuned-tolerance trap.
FD_TOL = 1e-3


def _reference_orbit(Az=0.03, point='L2'):
    """A converged L2 halo with a clean center pair, in family-record form."""
    seed, _, _ = richardson_halo_seed(Az, MU, point=point, northern=True)
    s0, T, converged = halo_corrector(seed, MU)
    assert converged, "halo corrector did not converge"
    return {'state0': list(s0), 'period': T}


def test_adapter_vector_field_matches_rhs():
    """Wrapper must return an ARRAY, not the list cr3bp_rhs_3d hands back."""
    from cr3bp_3d import cr3bp_rhs_3d
    x = np.array([1.15, 0.0, 0.02, 0.0, 0.2, 0.0])
    got = qpit.vector_field(x, MU)
    assert isinstance(got, np.ndarray) and got.shape == (6,)
    assert np.allclose(got, np.asarray(cr3bp_rhs_3d(0.0, x, MU), float))


def test_adapter_propagate_agrees_with_stm_version():
    """The two propagators must not silently disagree on the state."""
    orbit = _reference_orbit()
    x0 = np.asarray(orbit['state0'], float)
    T = orbit['period']
    xf_plain = qpit.propagate(x0, T, MU)
    xf_stm, Phi = qpit.propagate_stm(x0, T, MU)
    assert Phi.shape == (6, 6)
    assert np.abs(xf_plain - xf_stm).max() < 1e-9


def test_propagate_stm_recovers_periodicity():
    """Propagating a converged halo for one period must return the start."""
    orbit = _reference_orbit()
    x0 = np.asarray(orbit['state0'], float)
    xf, _ = qpit.propagate_stm(x0, orbit['period'], MU)
    assert np.abs(xf - x0).max() < 1e-9


def test_seed_from_halo_shapes_and_units():
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    assert U0.shape == (N, 6)
    assert abs(T0 - orbit['period']) < 1e-14
    # rho must be RADIANS: a small-amplitude L2 halo sits near the planar
    # bifurcation, so rho is small but strictly positive and well under pi.
    assert 0.0 < rho0 < np.pi
    # node mean returns the underlying periodic orbit
    assert np.abs(U0.mean(axis=0) - np.asarray(orbit['state0'], float)).max() < 1e-12


def test_seed_phase_is_reproducible():
    """Canonical phase must defeat numpy's arbitrary eigenvector phase.

    Without this, repeated runs give differently-phased curves from the same
    halo -- fatal for reproducible dataset generation later.
    """
    orbit = _reference_orbit()
    a, _, _ = qpit.seed_from_halo(orbit, MU, N, EPS, canonical_phase=True)
    b, _, _ = qpit.seed_from_halo(orbit, MU, N, EPS, canonical_phase=True)
    assert np.abs(a - b).max() < 1e-14


def test_residual_small_but_nonzero_at_seed():
    """The linear seed is close to a torus but is not one.

    Zero residual would mean the seed is already exact (it is not); a large
    residual would mean the seed or rho is wrong.
    """
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    F = qpit.invariance_residual(U0, T0, rho0, MU)
    nrm = np.linalg.norm(F)
    assert 1e-12 < nrm < 1e-2, f"|F| = {nrm:.3e} outside expected band"


def test_residual_shrinks_with_amplitude():
    """The seed is a LINEAR approximation, so its error must fall with eps.

    This is the real check that the seed formula and rho are consistent: a
    sign error in either would break the scaling.
    """
    orbit = _reference_orbit()
    norms = []
    for eps in (1e-4, 1e-5, 1e-6):
        U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, eps)
        norms.append(np.linalg.norm(qpit.invariance_residual(U0, T0, rho0, MU)))
    assert norms[0] > norms[1] > norms[2], f"residual not shrinking: {norms}"


def test_jacobian_dU0_against_finite_difference():
    """Confirms the (R kron I6) @ Phi_tilde ordering."""
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    J_U0, _, _, _ = qpit.invariance_jacobian(U0, T0, rho0, MU)
    rng = np.random.default_rng(0)
    for c in rng.choice(6 * N, 4, replace=False):
        up, um = U0.reshape(-1).copy(), U0.reshape(-1).copy()
        up[c] += H_FD_STATE
        um[c] -= H_FD_STATE
        fd = (qpit.invariance_residual(up.reshape(N, 6), T0, rho0, MU)
              - qpit.invariance_residual(um.reshape(N, 6), T0, rho0, MU)) / (2 * H_FD_STATE)
        rel = np.abs(fd - J_U0[:, c]).max() / max(np.abs(fd).max(), 1e-14)
        assert rel < FD_TOL, f"column {c}: rel err {rel:.2e}"


def test_jacobian_dT_and_drho_against_finite_difference():
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    _, J_T, J_rho, _ = qpit.invariance_jacobian(U0, T0, rho0, MU)
    fdT = (qpit.invariance_residual(U0, T0 + H_FD_PARAM, rho0, MU)
           - qpit.invariance_residual(U0, T0 - H_FD_PARAM, rho0, MU)) / (2 * H_FD_PARAM)
    assert np.abs(fdT - J_T).max() / np.abs(fdT).max() < FD_TOL
    fdr = (qpit.invariance_residual(U0, T0, rho0 + H_FD_PARAM, MU)
           - qpit.invariance_residual(U0, T0, rho0 - H_FD_PARAM, MU)) / (2 * H_FD_PARAM)
    assert np.abs(fdr - J_rho).max() / np.abs(fdr).max() < FD_TOL


def test_fd_error_is_cancellation_not_bug():
    """Documents the trap: FD error must GROW as h shrinks below ~1e-6.

    If a future change made the rho column genuinely wrong, the error would
    be flat in h instead of scaling as 1/h, and this test would fail.
    """
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    _, _, J_rho, _ = qpit.invariance_jacobian(U0, T0, rho0, MU)
    errs = []
    for h in (H_FD_PARAM, 1e-7, 1e-9):
        fd = (qpit.invariance_residual(U0, T0, rho0 + h, MU)
              - qpit.invariance_residual(U0, T0, rho0 - h, MU)) / (2 * h)
        errs.append(np.abs(fd - J_rho).max() / np.abs(fd).max())
    assert errs[0] < FD_TOL, "coarse-h error should be small"
    assert errs[2] > 10 * errs[0], "error should grow as h shrinks (roundoff)"


def test_stability_matrix_shape_and_unit_eigenvalue():
    """DG must carry an eigenvalue near 1 -- the torus tangent direction."""
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    DG = qpit.stability_matrix(U0, T0, rho0, MU)
    assert DG.shape == (6 * N, 6 * N)
    ev = np.linalg.eigvals(DG)
    assert np.abs(np.abs(ev) - 1.0).min() < 1e-6


def test_newton_step_reduces_residual():
    """ONE step only -- convergence is Week 8, this just checks direction."""
    orbit = _reference_orbit()
    U0, T0, rho0 = qpit.seed_from_halo(orbit, MU, N, EPS)
    before = np.linalg.norm(qpit.invariance_residual(U0, T0, rho0, MU))
    # newton_step's 4th return became an INFO DICT in Week 8; it was a bare
    # float |F| in Week 7, and this call site was never updated, so the
    # comparison below raised TypeError instead of testing anything.  `before`
    # is a 2-norm, so the matching key is res_2 -- res_inf is what
    # converge_torus declares convergence on and is a different number.
    U1, T1, r1, info = qpit.newton_step(U0, T0, rho0, MU)
    reported = info['res_2']
    after = np.linalg.norm(qpit.invariance_residual(U1, T1, r1, MU))

    # NOT bit-identical, and the reason matters.  newton_step gets its F from
    # invariance_jacobian, which propagates the 42-dim augmented system;
    # invariance_residual propagates the 6-dim state alone.  DOP853's adaptive
    # controller sizes its steps from the error estimate over ALL integrated
    # components, so adding the STM changes the step sequence and shifts the
    # final state by ~1e-10.  That is an accuracy floor on the residual, and
    # it will matter when Week 8 sets a convergence tolerance -- do not ask
    # for tighter than about 1e-11 without addressing it.
    assert abs(reported - before) / before < 1e-3

    assert after < before, f"residual grew: {before:.3e} -> {after:.3e}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
