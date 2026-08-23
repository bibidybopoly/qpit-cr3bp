"""Tests for the CR3BP effective potential, Jacobians and augmented RHS.

Run: pytest tests/test_cr3bp_jacobians.py -v
     (or python tests/test_cr3bp_jacobians.py)

Six public functions were untested — `effective_potential`, `cr3bp_jacobian`,
`cr3bp_rhs_with_stm` and their 3D twins, plus `jacobi_constant_3d`.  They are
pure functions of a state, so nothing here needs a torus or an integration.

They are also load-bearing.  `cr3bp_jacobian` IS the variational equation:
every STM, every differential corrector, every monodromy matrix and every
Floquet multiplier in this project is computed from it.  The existing STM tests
in test_cr3bp.py / test_cr3bp_3d.py check the STM against finite differences of
the FLOW, which would catch a badly wrong Jacobian — but a sign slip in a single
Hessian entry can survive that at the tolerances used, and would then show up as
a corrector that converges slightly more slowly to a slightly wrong orbit.

The decisive tests here are the direct ones: the analytic Jacobian against a
finite difference of the RHS at the same point, and the Hessian block against a
numerical Hessian of `effective_potential` — which is the pair most likely to
drift apart, since one was derived by hand from the other.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import constants as C                                            # noqa: E402
from cr3bp import (MU_EARTH_MOON, effective_potential,            # noqa: E402
                   cr3bp_rhs, cr3bp_jacobian, cr3bp_rhs_with_stm,
                   jacobi_constant, find_lagrange_points)
from cr3bp_3d import (cr3bp_rhs_3d, cr3bp_jacobian_3d,            # noqa: E402
                      cr3bp_rhs_with_stm_3d, jacobi_constant_3d)

MU = MU_EARTH_MOON

#: States away from both primaries, where the potential is smooth enough for
#: finite differences to be meaningful.
STATES_2D = [
    np.array([0.8369, 0.0, 0.0, 0.0]),          # L1, at rest
    np.array([0.836, 0.017, -0.04, 0.31]),      # a Lyapunov-ish point
    np.array([-0.5, 0.7, 0.2, -0.3]),           # far field
    np.array([1.1557, 0.0, 0.0, 0.1]),          # L2
]
STATES_3D = [
    np.array(C.HALO29_STATE),
    np.array([0.82, 0.05, 0.1, -0.02, 0.15, 0.05]),
    np.array([-0.4, 0.6, -0.3, 0.1, -0.2, 0.05]),
]


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def _fd_jacobian(f, x, mu, h=1e-6):
    """Central-difference Jacobian of f(0, x, mu) with respect to x."""
    x = np.asarray(x, float)
    n = x.size
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (np.asarray(f(0.0, x + e, mu), float)
                   - np.asarray(f(0.0, x - e, mu), float)) / (2 * h)
    return J


# ---------------------------------------------------------------------------
# 1. The effective potential
# ---------------------------------------------------------------------------

def test_effective_potential_vectorises_over_a_grid():
    """The docstring promises grid evaluation; hill_regions.ipynb relies on it."""
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)
    O = effective_potential(X, Y, MU)
    assert O.shape == X.shape
    assert np.isfinite(O[np.isfinite(O)]).all()
    assert abs(effective_potential(0.5, 0.3, MU)
               - O[0, 0] * 0 - effective_potential(np.array(0.5),
                                                   np.array(0.3), MU)) < 1e-15


def test_effective_potential_is_symmetric_in_y():
    """Omega depends on y only through y^2 -- the symmetry every halo family
    in this project is continued through."""
    for xv in (-0.5, 0.3, 0.8369, 1.2):
        for yv in (0.1, 0.4, 1.3):
            assert abs(effective_potential(xv, yv, MU)
                       - effective_potential(xv, -yv, MU)) < 1e-15


def test_gradient_of_the_potential_vanishes_at_the_lagrange_points():
    """This is what DEFINES them, and it ties `effective_potential` to
    `find_lagrange_points`: if either drifts, they stop agreeing."""
    lps = find_lagrange_points(MU)
    h = 1e-7
    for name, (xL, yL) in lps.items():
        gx = (effective_potential(xL + h, yL, MU)
              - effective_potential(xL - h, yL, MU)) / (2 * h)
        gy = (effective_potential(xL, yL + h, MU)
              - effective_potential(xL, yL - h, MU)) / (2 * h)
        assert abs(gx) < 1e-5, (name, gx)
        assert abs(gy) < 1e-5, (name, gy)


def test_potential_reproduces_the_jacobi_constant_convention():
    """C = 2*Omega - v^2, with the mu(1-mu)/2 term OMITTED.  Pinning it here as
    well as in test_features.py ties the 4-D route to the potential itself."""
    for s in STATES_2D:
        x, y, vx, vy = s
        C_direct = jacobi_constant(s, MU)
        C_from_omega = 2 * effective_potential(x, y, MU) - (vx ** 2 + vy ** 2)
        assert abs(C_direct - C_from_omega) < 1e-14, s


def test_potential_grows_without_bound_approaching_a_primary():
    for d in (1e-2, 1e-4, 1e-6):
        near_earth = effective_potential(-MU + d, 0.0, MU)
        near_moon = effective_potential(1 - MU + d, 0.0, MU)
        assert near_earth > 1.0 / d * 0.5
        assert near_moon > 1.0 / d * 0.005


# ---------------------------------------------------------------------------
# 2. The planar Jacobian
# ---------------------------------------------------------------------------

def test_planar_jacobian_matches_a_finite_difference_of_the_rhs():
    """The decisive check.  A = df/dx by construction, so it must reproduce a
    numerical derivative of `cr3bp_rhs` at the same point."""
    for s in STATES_2D:
        A = cr3bp_jacobian(s, MU)
        fd = _fd_jacobian(cr3bp_rhs, s, MU)
        scale = max(1.0, np.abs(fd).max())
        assert np.abs(A - fd).max() / scale < 1e-6, (s, np.abs(A - fd).max())


def test_planar_jacobian_has_the_documented_block_structure():
    for s in STATES_2D:
        A = cr3bp_jacobian(s, MU)
        assert A.shape == (4, 4)
        assert np.array_equal(A[0:2, 0:2], np.zeros((2, 2)))      # no dv/dx
        assert np.array_equal(A[0:2, 2:4], np.eye(2))             # dx/dt = v
        assert np.array_equal(A[2:4, 2:4], np.array([[0.0, 2.0],  # Coriolis
                                                     [-2.0, 0.0]]))


def test_planar_jacobian_hessian_block_is_symmetric():
    """The lower-left block is the Hessian of a scalar potential, so
    Omega_xy == Omega_yx exactly.  An asymmetric entry means one of the two
    hand-derived cross terms is wrong."""
    for s in STATES_2D:
        A = cr3bp_jacobian(s, MU)
        assert A[2, 1] == A[3, 0], s


def test_planar_hessian_block_matches_a_numerical_hessian_of_the_potential():
    """`cr3bp_jacobian`'s Hessian entries were derived by hand FROM
    `effective_potential`.  Nothing keeps them in step, so check it."""
    h = 1e-5
    for s in STATES_2D:
        x, y = s[0], s[1]
        O = lambda a, b: effective_potential(a, b, MU)      # noqa: E731
        Oxx = (O(x + h, y) - 2 * O(x, y) + O(x - h, y)) / h ** 2
        Oyy = (O(x, y + h) - 2 * O(x, y) + O(x, y - h)) / h ** 2
        Oxy = (O(x + h, y + h) - O(x + h, y - h)
               - O(x - h, y + h) + O(x - h, y - h)) / (4 * h ** 2)
        A = cr3bp_jacobian(s, MU)
        scale = max(1.0, abs(Oxx), abs(Oyy))
        assert abs(A[2, 0] - Oxx) / scale < 1e-5, (s, A[2, 0], Oxx)
        assert abs(A[3, 1] - Oyy) / scale < 1e-5, (s, A[3, 1], Oyy)
        assert abs(A[2, 1] - Oxy) / scale < 1e-5, (s, A[2, 1], Oxy)


def test_planar_jacobian_is_trace_free():
    """The flow is volume preserving, so tr(A) = 0 exactly -- the Coriolis
    block contributes nothing on the diagonal and the potential block sits
    off it."""
    for s in STATES_2D:
        assert abs(np.trace(cr3bp_jacobian(s, MU))) < 1e-15


def test_planar_jacobian_ignores_velocity():
    """A depends on POSITION only; the velocity enters the flow but not its
    linearisation. If it did, the STM would stop being a function of the
    trajectory's positions alone."""
    s = STATES_2D[1].copy()
    A1 = cr3bp_jacobian(s, MU)
    s2 = s.copy(); s2[2:] = [-9.0, 4.0]
    assert np.array_equal(A1, cr3bp_jacobian(s2, MU))


# ---------------------------------------------------------------------------
# 3. The 3D Jacobian, and its agreement with the planar one
# ---------------------------------------------------------------------------

def test_spatial_jacobian_matches_a_finite_difference_of_the_rhs():
    for s in STATES_3D:
        A = cr3bp_jacobian_3d(s, MU)
        fd = _fd_jacobian(cr3bp_rhs_3d, s, MU)
        scale = max(1.0, np.abs(fd).max())
        assert np.abs(A - fd).max() / scale < 1e-6, (s, np.abs(A - fd).max())


def test_spatial_jacobian_block_structure_and_hessian_symmetry():
    for s in STATES_3D:
        A = cr3bp_jacobian_3d(s, MU)
        assert A.shape == (6, 6)
        assert np.array_equal(A[0:3, 0:3], np.zeros((3, 3)))
        assert np.array_equal(A[0:3, 3:6], np.eye(3))
        assert np.array_equal(A[3:6, 3:6], np.array([[0.0, 2.0, 0.0],
                                                     [-2.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0]]))
        H = A[3:6, 0:3]
        assert np.array_equal(H, H.T), "Hessian block is not symmetric"
        assert abs(np.trace(A)) < 1e-15


def test_spatial_jacobian_reduces_to_the_planar_one_in_the_plane():
    """z = vz = 0 is an invariant subspace, so the 6x6 restricted to
    (x, y, vx, vy) must BE the 4x4.  The two were derived separately; this is
    the only thing that keeps them from diverging."""
    for s2 in STATES_2D:
        x, y, vx, vy = s2
        s3 = np.array([x, y, 0.0, vx, vy, 0.0])
        A3 = cr3bp_jacobian_3d(s3, MU)
        A2 = cr3bp_jacobian(s2, MU)
        idx = np.ix_([0, 1, 3, 4], [0, 1, 3, 4])
        assert np.allclose(A3[idx], A2, rtol=0, atol=1e-13), s2
        # and the out-of-plane block decouples: no z coupling in the plane
        assert abs(A3[5, 0]) < 1e-15 and abs(A3[5, 1]) < 1e-15
        assert abs(A3[3, 2]) < 1e-15 and abs(A3[4, 2]) < 1e-15


def test_out_of_plane_hessian_entry_has_no_rotation_term():
    """Omega_zz carries no leading 1: the frame rotates about z, so the
    centrifugal term is (x^2 + y^2)/2 and contributes nothing to d2/dz2.
    Getting this wrong shifts every vertical Floquet exponent."""
    for s in STATES_3D:
        x, y, z = s[:3]
        r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
        r2 = np.sqrt((x - 1 + MU) ** 2 + y ** 2 + z ** 2)
        Ozz = (-(1 - MU) * (1 / r1 ** 3 - 3 * z ** 2 / r1 ** 5)
               - MU * (1 / r2 ** 3 - 3 * z ** 2 / r2 ** 5))
        assert abs(cr3bp_jacobian_3d(s, MU)[5, 2] - Ozz) < 1e-12, s


# ---------------------------------------------------------------------------
# 4. The augmented right-hand sides
# ---------------------------------------------------------------------------

def test_augmented_rhs_carries_the_state_and_the_variational_equation():
    """aug[0:4] must be the plain RHS and aug[4:20] must be A @ Phi -- the
    variational equation. Anything else and every STM in the project is wrong
    in a way no single-trajectory test would reveal."""
    rng = np.random.default_rng(0)
    for s in STATES_2D:
        Phi = rng.standard_normal((4, 4))
        aug = np.concatenate([s, Phi.flatten()])
        out = cr3bp_rhs_with_stm(0.0, aug, MU)
        assert out.shape == (20,)
        assert np.allclose(out[:4], cr3bp_rhs(0.0, s, MU), rtol=0, atol=1e-15)
        assert np.allclose(out[4:].reshape(4, 4),
                           cr3bp_jacobian(s, MU) @ Phi, rtol=0, atol=1e-14)


def test_augmented_rhs_at_identity_is_the_jacobian_itself():
    for s in STATES_2D:
        aug = np.concatenate([s, np.eye(4).flatten()])
        out = cr3bp_rhs_with_stm(0.0, aug, MU)
        assert np.allclose(out[4:].reshape(4, 4), cr3bp_jacobian(s, MU),
                           rtol=0, atol=1e-15)


def test_spatial_augmented_rhs_carries_the_variational_equation():
    rng = np.random.default_rng(1)
    for s in STATES_3D:
        Phi = rng.standard_normal((6, 6))
        aug = np.concatenate([s, Phi.flatten()])
        out = cr3bp_rhs_with_stm_3d(0.0, aug, MU)
        assert out.shape == (42,)
        assert np.allclose(out[:6], cr3bp_rhs_3d(0.0, s, MU), rtol=0, atol=1e-15)
        assert np.allclose(out[6:].reshape(6, 6),
                           cr3bp_jacobian_3d(s, MU) @ Phi, rtol=0, atol=1e-14)


def test_augmented_rhs_is_row_major_not_column_major():
    """Phi is flattened and reshaped with numpy's default C order throughout.
    A Fortran-order slip transposes every STM, which still integrates and still
    has determinant 1 -- and gives the wrong sensitivities to every corrector."""
    s = STATES_2D[1]
    Phi = np.arange(16, dtype=float).reshape(4, 4)
    out = cr3bp_rhs_with_stm(0.0, np.concatenate([s, Phi.flatten()]), MU)
    A = cr3bp_jacobian(s, MU)
    assert np.allclose(out[4:], (A @ Phi).flatten(), rtol=0, atol=1e-13)
    assert not np.allclose(out[4:], (A @ Phi.T).flatten(), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. The 3D Jacobi constant
# ---------------------------------------------------------------------------

def test_jacobi_3d_agrees_with_the_planar_version_in_the_plane():
    """Two independent implementations of the same convention. If they ever
    disagree, a planar orbit's energy depends on which module read it."""
    for s2 in STATES_2D:
        x, y, vx, vy = s2
        s3 = np.array([x, y, 0.0, vx, vy, 0.0])
        assert abs(jacobi_constant_3d(s3, MU) - jacobi_constant(s2, MU)) < 1e-14


def test_jacobi_3d_omits_the_mu_term_and_matches_the_recorded_halo_value():
    s = np.array(C.HALO29_STATE)
    Cv = jacobi_constant_3d(s, MU)
    assert abs(Cv - C.HALO29_C) < 1e-6, Cv
    with_term = Cv + MU * (1 - MU) / 2
    assert abs(with_term - C.HALO29_C) > 1e-3, "the omitted term is undetectable"


def test_jacobi_3d_depends_on_speed_only_through_its_square():
    rng = np.random.default_rng(2)
    for s in STATES_3D:
        flipped = s.copy(); flipped[3:] *= -1
        assert abs(jacobi_constant_3d(s, MU)
                   - jacobi_constant_3d(flipped, MU)) < 1e-14
        # an arbitrary rotation of the velocity vector preserves |v|
        v = s[3:]
        th = 0.7
        rot = np.array([[np.cos(th), -np.sin(th), 0],
                        [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        s_rot = s.copy(); s_rot[3:] = rot @ v
        assert abs(jacobi_constant_3d(s, MU)
                   - jacobi_constant_3d(s_rot, MU)) < 1e-14


def test_jacobi_3d_is_conserved_along_the_flow():
    """dC/dt = 0 is the defining property, and it is convention-proof: it fails
    if `jacobi_constant_3d` and `cr3bp_rhs_3d` disagree about where mu sits."""
    h = 1e-6
    for s in STATES_3D:
        f = np.asarray(cr3bp_rhs_3d(0.0, s, MU), float)
        dC = (jacobi_constant_3d(s + h * f, MU)
              - jacobi_constant_3d(s - h * f, MU)) / (2 * h)
        assert abs(dC) < 1e-6, (s, dC)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
