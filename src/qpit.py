from __future__ import annotations

import numpy as np

# =========================================================================
# ADAPTER BLOCK -- WIRED TO cr3bp_3d.py
# =========================================================================
# Thin wrappers only.  Three adaptations are needed and each is a real
# difference, not cosmetic:
#
#   1. cr3bp_rhs_3d has signature (t, state, mu) for solve_ivp; this module
#      wants (x, mu).  The system is autonomous, so t is passed as 0.0.
#   2. cr3bp_rhs_3d returns a Python LIST.  Left unconverted it silently
#      breaks the array arithmetic in the Jacobian assembly.
#   3. cr3bp_3d has no plain state-only propagator, so `propagate` is built
#      here with the same integrator and tolerances as integrate_with_stm_3d.

from scipy.integrate import solve_ivp as _solve_ivp   # noqa: E402

from cr3bp_3d import (                                # noqa: E402
    MU_EARTH_MOON,
    cr3bp_rhs_3d as _cr3bp_rhs_3d,
    integrate_with_stm_3d as _integrate_with_stm_3d,
    halo_stability as _halo_stability,
    jacobi_constant_3d as _jacobi_constant_3d,
)

_INTEGRATOR = dict(method='DOP853', rtol=1e-12, atol=1e-12)


def vector_field(x, mu):
    """CR3BP xdot in the rotating frame, shape (6,)."""
    return np.asarray(_cr3bp_rhs_3d(0.0, np.asarray(x, float), mu), dtype=float)


def propagate(x0, t, mu):
    """State after time t, shape (6,)."""
    sol = _solve_ivp(_cr3bp_rhs_3d, [0.0, float(t)], np.asarray(x0, float),
                     args=(mu,), **_INTEGRATOR)
    if not sol.success:
        raise RuntimeError(f"propagate failed: {sol.message}")
    return sol.y[:, -1]


def propagate_stm(x0, t, mu):
    """(state after time t, STM Phi(t,0)) -- shapes (6,) and (6,6)."""
    sol = _integrate_with_stm_3d(np.asarray(x0, float), [0.0, float(t)], mu)
    if not sol.success:
        raise RuntimeError(f"propagate_stm failed: {sol.message}")
    yf = sol.y[:, -1]
    return yf[:6], yf[6:].reshape(6, 6)


def _require_adapters() -> None:
    """Retained so the guard survives if the adapter is ever unwired."""
    missing = [n for n, f in (("vector_field", vector_field),
                              ("propagate", propagate),
                              ("propagate_stm", propagate_stm)) if f is None]
    if missing:
        raise RuntimeError(
            f"qpit.py adapter not wired: {', '.join(missing)}. "
            "See the ADAPTER BLOCK at the top of src/qpit.py."
        )


# =========================================================================
# SECTION 1 -- ANGLES, WAVENUMBERS, UNIT CONVERSION
# =========================================================================

def check_odd(N: int) -> None:
    if N % 2 == 0:
        raise ValueError(f"N must be odd (got {N}); see convention 2.")


def node_angles(N: int) -> np.ndarray:
    """theta_j = 2*pi*j/N, shape (N,)."""
    check_odd(N)
    return 2.0 * np.pi * np.arange(N) / N


def wavenumbers(N: int) -> np.ndarray:
    """Symmetric integer wavenumbers -(N-1)/2 ... (N-1)/2, shape (N,).

    Returned in SYMMETRIC order, not numpy.fft order.  Row i of dft_matrix
    corresponds to wavenumbers(N)[i].
    """
    check_odd(N)
    h = (N - 1) // 2
    return np.arange(-h, h + 1)


def rho_from_normalised(rho_norm: float) -> float:
    """Convert a rotation number expressed as a fraction of 2*pi to radians.

    Your Floquet tables use the normalised form (period doubling at 0.5).
    Every published GMOS formula uses radians.  Convert here, once.
    """
    return 2.0 * np.pi * rho_norm


def rho_to_normalised(rho: float) -> float:
    """Inverse of rho_from_normalised, for reporting against Floquet tables."""
    return rho / (2.0 * np.pi)


# =========================================================================
# SECTION 2 -- DFT, ROTATION, AND SPECTRAL DIFFERENTIATION
# =========================================================================
# These are the exactness core of GMOS.  Rotating the curve is a phase
# multiplication in Fourier space -- no interpolation, so no discretisation
# error enters the residual.  All three operators act on the NODE index
# only; the 6 state components never mix.

def dft_matrix(N: int) -> np.ndarray:
    """Forward DFT, shape (N, N) complex.  C = D @ v, with the 1/N here.

    D[i, j] = (1/N) * exp(-1j * k_i * theta_j)
    """
    k = wavenumbers(N)[:, None]        # (N, 1)
    th = node_angles(N)[None, :]       # (1, N)
    return np.exp(-1j * k * th) / N


def idft_matrix(N: int) -> np.ndarray:
    """Inverse DFT, shape (N, N) complex.  v = Dinv @ C.

    Dinv[j, i] = exp(+1j * k_i * theta_j)
    """
    k = wavenumbers(N)[None, :]        # (1, N)
    th = node_angles(N)[:, None]       # (N, 1)
    return np.exp(1j * k * th)


def rotation_operator(N: int, rho: float) -> np.ndarray:
    """R(rho): shifts a discretised curve by +rho.  Shape (N, N), REAL.

    Derivation: v(theta + rho) = sum_k (C_k e^{i k rho}) e^{i k theta}, so
    the shift is diag(e^{i k rho}) in Fourier space.

    Realness is not an accident.  v is real and the wavenumbers are
    symmetric, so the k and -k contributions are complex conjugates and the
    imaginary parts cancel exactly.  A non-negligible imaginary residue
    means the wavenumber indexing is wrong -- hence the assertion.

    For the residual you want R(-rho); pass a negative rho.
    """
    k = wavenumbers(N)
    Q = np.diag(np.exp(1j * k * rho))
    R = idft_matrix(N) @ Q @ dft_matrix(N)
    imag = np.abs(R.imag).max()
    if imag > 1e-10:
        raise AssertionError(
            f"rotation_operator: imaginary residue {imag:.2e} exceeds 1e-10. "
            "Wavenumber ordering or DFT normalisation is wrong."
        )
    return R.real


def d_rotation_d_rho(N: int, rho: float) -> np.ndarray:
    """d/drho of rotation_operator(N, rho).  Shape (N, N), REAL.

    d/drho diag(e^{i k rho}) = diag(i k e^{i k rho}).  Needed for the
    dF/drho column of the Jacobian.
    """
    k = wavenumbers(N)
    Q = np.diag(1j * k * np.exp(1j * k * rho))
    dR = idft_matrix(N) @ Q @ dft_matrix(N)
    imag = np.abs(dR.imag).max()
    if imag > 1e-10:
        raise AssertionError(
            f"d_rotation_d_rho: imaginary residue {imag:.2e} exceeds 1e-10."
        )
    return dR.real


def spectral_derivative_operator(N: int) -> np.ndarray:
    """d/dtheta_1 as an (N, N) REAL matrix, exact for band-limited curves.

    Differentiating the Fourier series multiplies C_k by i*k.  Used for the
    latitudinal phase-constraint tangent -- no finite differences needed.
    """
    k = wavenumbers(N)
    Q = np.diag(1j * k)
    Dth = idft_matrix(N) @ Q @ dft_matrix(N)
    imag = np.abs(Dth.imag).max()
    if imag > 1e-10:
        raise AssertionError(
            f"spectral_derivative_operator: imaginary residue {imag:.2e}."
        )
    return Dth.real


def apply_node_operator(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply an (N, N) node-space operator to a curve U of shape (N, 6).

    Equivalent to (M kron I6) acting on flatten_curve(U), but without
    forming the 6N x 6N Kronecker product.
    """
    return M @ U


# =========================================================================
# SECTION 3 -- FLATTENING
# =========================================================================

def flatten_curve(U: np.ndarray) -> np.ndarray:
    """(N, 6) -> (6N,), row-major.  Entry 6*j + a is component a of node j."""
    return U.reshape(-1)


def unflatten_curve(u: np.ndarray) -> np.ndarray:
    """(6N,) -> (N, 6), inverse of flatten_curve."""
    return u.reshape(-1, 6)


# =========================================================================
# SECTION 4 -- SEED CONSTRUCTION FROM A HALO ORBIT
# =========================================================================

def seed_curve(x_halo: np.ndarray,
               v_center: np.ndarray,
               N: int,
               epsilon: float) -> np.ndarray:
    """Linear approximation to the invariant curve near a periodic orbit.

    U0[j] = x_halo + epsilon * ( Re(v_C) cos(theta_j) - Im(v_C) sin(theta_j) )
          = x_halo + epsilon * Re[ e^{i theta_j} v_C ]

    SIGN -- DERIVED, NOT GUESSED (Week 8; the Week 7 version had a PLUS here
    and it was wrong for this file's residual).  Derivation:

      rotation_operator(N, rho) applies diag(e^{i k rho}), so
      (R(rho) v)(theta) = v(theta + rho).  invariance_residual uses R(-rho),
      so F = 0 reads  phi_T(v(theta - rho)) = v(theta), i.e.

          phi_T(v(theta)) = v(theta + rho)                             (*)

      Let M be the monodromy at x_halo and M v_C = e^{i rho} v_C with
      v_C = a + i b.  Splitting that eigen-relation into real and imaginary
      parts (M is real) gives

          M a = a cos(rho) - b sin(rho)
          M b = a sin(rho) + b cos(rho)

      Apply M to the MINUS form and collect with the angle-sum identities:

          M[ a cos(th) - b sin(th) ] = a cos(th + rho) - b sin(th + rho)

      which is (*).  The PLUS form gives theta - rho instead, i.e. it is the
      seed for the OPPOSITE residual convention.

    Why the wrong sign got in: this seed cites McCarthy & Howell Eq. (6), and
    invariance_jacobian's own docstring records that M&H write
    DG = Phi_tilde (R kron I) -- the opposite operator order, a different
    residual convention.  Their Eq. (6) is right in their convention; copying
    it into this one is the mixing error.

    Confirmed numerically on an L2 halo at rho/2pi = 0.282: halving epsilon
    from 3e-4 down to 3e-6, the MINUS residual falls quadratically (ratio
    11.1 per 3.33x step, i.e. exactly eps^2) while the PLUS residual falls
    only linearly (ratio 3.33).  See notebooks/qpit_solver.ipynb.

    The base periodic orbit state is added here; the published equation omits
    it in the PDF text, but without it the curve is not near the torus.

    CHOOSING EPSILON.  The quadratic coefficient is large -- about 1.9e3 for
    the L2 halo tested, inherited from the hyperbolic multiplier (~1.2e3) in
    the STM -- so the linear/quadratic crossover sits near eps ~ 1e-3 and the
    eps^2 signature is only visible BELOW that.  Useful window is roughly
    3e-5 to 3e-4: the residual floor from rtol=atol=1e-12 propagation is
    ~1e-10, and eps = 1e-4 gives a seed residual ~2e-5, comfortably above it.

    Parameters
    ----------
    x_halo   : (6,) initial state on the underlying periodic orbit
    v_center : (6,) complex eigenvector of the monodromy matrix associated
               with the unit-modulus center eigenvalue
    epsilon  : small amplitude; start around 1e-4 non-dimensional and grow
               by continuation, not by guessing large

    Returns (N, 6) real.
    """
    check_odd(N)
    th = node_angles(N)
    re = np.real(v_center)[None, :]
    im = np.imag(v_center)[None, :]
    pert = np.cos(th)[:, None] * re - np.sin(th)[:, None] * im
    return x_halo[None, :] + epsilon * pert


def seed_rho(lambda_center: complex) -> float:
    """rho^0 = atan2(Im lambda_C, Re lambda_C), in radians.

    McCarthy & Howell Eq. (7): the seed rotation angle is the angular
    position of the center eigenvalue on the unit circle -- which is exactly
    the rotation number already tabulated in your Floquet sweeps, once
    converted out of normalised units.
    """
    return float(np.arctan2(np.imag(lambda_center), np.real(lambda_center)))


def canonical_eigenvector_phase(v: np.ndarray) -> np.ndarray:
    """Fix the arbitrary complex phase of a center eigenvector.

    numpy.linalg.eig returns eigenvectors up to an arbitrary factor e^{i*phi},
    and the factor is not reproducible across calls.  That phase rotates the
    starting point on the invariant curve -- harmless for correctness, since
    it is exactly the latitudinal symmetry, but it means an unpinned seed is
    not reproducible.  For dataset generation that is a genuine problem: the
    same halo orbit would yield differently-phased curves on different runs
    and therefore different feature vectors.

    Convention here: rotate so the largest-magnitude component is real and
    positive.  Largest-magnitude rather than first, because the first
    component can be near zero and the phase then ill-conditioned.
    """
    j = int(np.argmax(np.abs(v)))
    return v * np.exp(-1j * np.angle(v[j]))


def seed_from_halo(orbit: dict, mu: float, N: int, epsilon: float,
                   canonical_phase: bool = True):
    """Build a GMOS seed from an orbits.py halo family record.

    Consumes the dicts produced by orbits.compute_halo_family, which carry
    'state0' (a list) and 'period'.  Returns (U0, T0, rho0) ready for
    invariance_residual.

    rho0 comes from the center eigenvalue directly, in RADIANS.  Note that
    orbits.sweep_rotation_numbers reports 'rho' NORMALISED (theta/2pi) and
    'theta' unwrapped along the family; neither is what goes in here.  The
    rotation operator is 2*pi-periodic in rho, so an unwrapped theta would
    also work, but taking the eigenvalue argument avoids the question.

    Raises if the orbit has no clean center pair -- i.e. it is fully
    hyperbolic, or sits at the bifurcation where the center pair collides
    with the trivial fuzz.  Those orbits have no torus to seed.
    """
    state0 = np.asarray(orbit['state0'], dtype=float)
    T0 = float(orbit['period'])
    stab = _halo_stability(state0, T0, mu)
    if stab['center'] is None:
        raise ValueError(
            "orbit has no clean center pair (fully hyperbolic, or at the "
            "bifurcation) -- no quasi-periodic torus exists here")
    v_c = stab['center']['vec']
    if canonical_phase:
        v_c = canonical_eigenvector_phase(v_c)
    return seed_curve(state0, v_c, N, epsilon), T0, seed_rho(stab['center']['mu'])


# =========================================================================
# SECTION 5 -- INVARIANCE RESIDUAL AND JACOBIAN
# =========================================================================

def propagate_curve(U0: np.ndarray, T: float, mu: float,
                    with_stm: bool = False):
    """Propagate every node for the stroboscopic time T.

    This is the expensive step: N independent 6-dim propagations, or N
    independent 42-dim propagations when STMs are needed.  The nodes do not
    interact, so this loop is the natural place to parallelise on Titan.

    Returns U1 of shape (N, 6), and if with_stm also a list of N (6,6) STMs.
    """
    _require_adapters()
    N = U0.shape[0]
    U1 = np.empty_like(U0)
    if not with_stm:
        for j in range(N):
            U1[j] = propagate(U0[j], T, mu)
        return U1
    stms = []
    for j in range(N):
        xf, Phi = propagate_stm(U0[j], T, mu)
        U1[j] = xf
        stms.append(Phi)
    return U1, stms


def invariance_residual(U0: np.ndarray, T: float, rho: float,
                        mu: float) -> np.ndarray:
    """F = R(-rho) @ U1 - U0, flattened to (6N,).

    Zero exactly when the discretised curve is invariant under the
    stroboscopic map up to a rotation by rho.
    """
    N = U0.shape[0]
    U1 = propagate_curve(U0, T, mu, with_stm=False)
    Rm = rotation_operator(N, -rho)
    return flatten_curve(apply_node_operator(Rm, U1) - U0)


def invariance_jacobian(U0: np.ndarray, T: float, rho: float, mu: float):
    """Blocks of dF/d[U0, T, rho].  Returns (J_U0, J_T, J_rho, F).

    J_U0 : (6N, 6N)  = (R(-rho) kron I6) @ blockdiag(Phi_j) - I
    J_T  : (6N,)     = (R(-rho) kron I6) @ f(U1)      since dU1/dT = f(U1)
    J_rho: (6N,)     = (dR/drho(-rho) kron I6) @ U1, with a sign from the
                       chain rule because the operator argument is -rho

    ORDERING -- VERIFIED.  Our residual F = R U1 - U0 differentiates to
    (R kron I6) @ Phi_tilde, and that order is confirmed against finite
    differences (tests/test_qpit_integration.py).  McCarthy & Howell write
    their stability differential as DG = Phi_tilde (R kron I), the opposite
    order; since these matrices do not commute, that reflects a different
    residual convention on their side, not an error on either.

    FD CAUTION.  Near a low-amplitude seed the residual is O(1e-6), so a
    central difference with h below about 1e-6 is dominated by subtractive
    cancellation and will appear to disagree.  Error growing as 1/h is
    roundoff, not a wrong Jacobian.  Use h ~ 1e-5.
    """
    _require_adapters()
    N = U0.shape[0]
    U1, stms = propagate_curve(U0, T, mu, with_stm=True)

    Rm = rotation_operator(N, -rho)
    dRm = -d_rotation_d_rho(N, -rho)   # chain rule: d/drho R(-rho)

    R_kron = np.kron(Rm, np.eye(6))                       # (6N, 6N)
    Phi_tilde = np.zeros((6 * N, 6 * N))
    for j, Phi in enumerate(stms):
        Phi_tilde[6 * j:6 * j + 6, 6 * j:6 * j + 6] = Phi

    J_U0 = R_kron @ Phi_tilde - np.eye(6 * N)

    f_U1 = np.array([vector_field(U1[j], mu) for j in range(N)])
    J_T = R_kron @ flatten_curve(f_U1)

    J_rho = np.kron(dRm, np.eye(6)) @ flatten_curve(U1)

    F = flatten_curve(apply_node_operator(Rm, U1) - U0)
    return J_U0, J_T, J_rho, F


# =========================================================================
# SECTION 6 -- PHASE CONSTRAINTS
# =========================================================================
# Two continuous symmetries leave the residual unchanged, so the Jacobian
# has two flat directions that must be pinned by hand:
#
#   latitudinal  -- v(theta + c) traces the same curve, relabelled
#   longitudinal -- slicing the torus at theta_0 = s instead of 0 gives a
#                   different curve on the SAME torus
#
# Each constraint says the solution carries no component along the
# corresponding tangent of a reference curve U_ref.

def latitudinal_tangent(U_ref: np.ndarray) -> np.ndarray:
    """dU/dtheta_1 by spectral differentiation.  Shape (N, 6)."""
    N = U_ref.shape[0]
    return spectral_derivative_operator(N) @ U_ref


def longitudinal_tangent(U_ref: np.ndarray, T: float, rho: float,
                         mu: float) -> np.ndarray:
    """dU/dtheta_0 from the PDE invariance equation.  Shape (N, 6).

    Rearranging  omega_0 du/dtheta_0 + omega_1 du/dtheta_1 = f(u)
    with omega_0 = 2*pi/T and omega_1 = rho/T gives

        du/dtheta_0 = ( T * f(u) - rho * du/dtheta_1 ) / (2*pi)

    So the longitudinal tangent costs one vector-field evaluation per node
    plus the latitudinal tangent already computed -- this is the step-2 PDE
    earning its keep after the derivation moved to the map form.
    """
    _require_adapters()
    N = U_ref.shape[0]
    f_ref = np.array([vector_field(U_ref[j], mu) for j in range(N)])
    return (T * f_ref - rho * latitudinal_tangent(U_ref)) / (2.0 * np.pi)


def phase_constraints(U: np.ndarray, U_ref: np.ndarray, T: float,
                      rho: float, mu: float):
    """Scalars (F_theta0, F_theta1), each an inner product against a tangent.

    McCarthy & Howell (2021) Eqs. (34)-(35).  Both reference a PREVIOUSLY
    CONVERGED curve U_ref.  For the very first torus there is no previous
    member; passing the seed curve as U_ref is the natural adaptation, but
    that is an inference and not the published prescription -- this is the
    open question in the Olikara email.

    THE DIFFERENCE MATTERS (Week 8 fix).  The constraint is
    <U - U_ref, tan> = 0, not <U, tan> = 0.  For the latitudinal tangent the
    two coincide identically, because <U_ref, D_theta U_ref> = N sum_k
    i k |C_k|^2 = 0 by antisymmetry of the symmetric wavenumbers against a
    real curve.  For the longitudinal tangent they do NOT coincide in
    general, since <U_ref, f(U_ref)> = sum_j (r_j . v_j + v_j . a_j) has no
    reason to vanish.

    It does happen to vanish at the seed, and that is a trap rather than a
    reprieve: a halo state0 from halo_corrector sits at a perpendicular x-z
    crossing, where y = vx = vz = 0 makes r.v = 0 and ay = 0, hence
    x0 . f(x0) = 0 exactly.  So the wrong form is invisible on iteration one
    and switches on as soon as Newton moves off the symmetric crossing, and
    again the moment U_ref is a previously converged member instead of U
    itself.  Measured 2.5e-16 at the seed, which is why Week 7's numbers did
    not expose it.
    """
    tan0 = longitudinal_tangent(U_ref, T, rho, mu)
    tan1 = latitudinal_tangent(U_ref)
    dU = U - U_ref
    return (float(np.sum(dU * tan0)), float(np.sum(dU * tan1)))


# =========================================================================
# SECTION 6B -- PARAMETER CONSTRAINTS (amplitude closure)
# =========================================================================
# The two phase constraints kill the two SYMMETRY directions.  They do not
# kill the two PHYSICAL directions: the solution set through a genuine torus
# is 4-dimensional (2 phases x 2 parameters -- which underlying periodic
# orbit, and amplitude within its centre manifold), so the invariance block
# has rank 6N - 2 in the 6N + 2 unknowns [U0, T, rho].
#
# Consequence if you stop at the phase constraints: amplitude is unpinned,
# and v == x_halo (the constant curve) is an exact solution of the invariance
# equation for ANY rho.  Newton is then free to bleed epsilon to zero and
# "converge" to the periodic orbit you started from.
#
# Closure used here for the FIRST torus: 2 phase constraints + hold the mean
# Jacobi constant.  rho and T both stay FREE.
#
# DO NOT ALSO PIN rho (Week 8, measured).  rho at the seed is the LINEAR
# rotation number, read off the monodromy eigenvalue at zero amplitude.  The
# true torus at finite amplitude has rho shifted by O(eps^2), so pinning rho
# at rho0 while also pinning C demands a torus that does not exist and leaves
# an irreducible O(eps^3) residual:
#
#     eps      |F|inf with rho pinned     |F|inf with rho free   rho - rho0
#     1e-4          6.3e-12                    6.8e-13            2.43e-7
#     3e-4          1.7e-10                    5.6e-13            2.19e-6
#     1e-3          6.3e-09                    7.4e-13            2.43e-5
#
# The shift is clean eps^2 (24.3 eps^2 for the orbit tested) and the stalled
# residual is clean eps^3, N-independent -- so it is not Fourier truncation
# and not integrator error, it is an inconsistent constraint set.
#
# Also do NOT pin rho and T together: the eps = 0 halo sits at exactly
# (T0, rho0), so pinning both aims Newton straight at the degenerate answer.
#
# With rho free the system carries a 1-D nullspace -- 6N + 2 unknowns against
# rank (6N - 2) + 3 = 6N + 1.  That is not an oversight; it is exactly the
# nullspace McCarthy & Howell leave for pseudo-arclength, and it means the
# torus is isolated only up to sliding along the constant-C family.  Least
# squares takes minimum-norm steps and therefore stays put, which is enough
# for a first torus.  Closing that last direction is continuation work.

def jacobi_constant(x, mu):
    """Jacobi constant of a single state.  Delegates to cr3bp_3d."""
    return float(_jacobi_constant_3d(np.asarray(x, float), mu))


def jacobi_gradient(x, mu):
    """dC/dx at one state, shape (6,).

    With omega = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2 and C = 2 omega - |v|^2,

        dC/d(pos) = 2 * [omega_x, omega_y, omega_z]
        dC/d(vel) = -2 * [vx, vy, vz]

    and omega_x, omega_y, omega_z are exactly the non-Coriolis parts of the
    accelerations in cr3bp_rhs_3d, which is what makes the self-test below
    convention-proof.

    SELF-TEST (tests/test_qpit_jacobi.py): dC/dx . f(x) == 0 to machine
    precision at a random state, using YOUR vector field.  That identity
    holds only if this gradient and cr3bp_rhs_3d agree on mu placement, on
    the sign of the Coriolis block, and on the omega convention, so one
    assertion covers every convention question at once.  It is insensitive
    to the additive constant in omega, which is fine -- only differences of
    C are ever used here.
    """
    x = np.asarray(x, float)
    px, py, pz, vx, vy, vz = x
    r1 = np.sqrt((px + mu)**2 + py**2 + pz**2)
    r2 = np.sqrt((px - 1 + mu)**2 + py**2 + pz**2)
    r1_3, r2_3 = r1**3, r2**3
    ox = px - (1 - mu) * (px + mu) / r1_3 - mu * (px - 1 + mu) / r2_3
    oy = py - (1 - mu) * py / r1_3 - mu * py / r2_3
    oz = -(1 - mu) * pz / r1_3 - mu * pz / r2_3
    return np.array([2 * ox, 2 * oy, 2 * oz, -2 * vx, -2 * vy, -2 * vz])


def mean_jacobi(U, mu):
    """Mean Jacobi constant over the nodes of a curve.

    A converged torus is an invariant set of an autonomous system, so every
    node carries the SAME C and the mean is that common value.  Off the
    solution the nodes disagree, and the mean is a better-conditioned
    amplitude measure than any single node.
    """
    return float(np.mean([jacobi_constant(U[j], mu) for j in range(U.shape[0])]))


def mean_jacobi_row(U, mu):
    """Gradient of mean_jacobi with respect to the flattened curve, (6N,)."""
    N = U.shape[0]
    return flatten_curve(np.array([jacobi_gradient(U[j], mu)
                                   for j in range(N)])) / N


def curve_amplitude(U):
    """Max distance of any node from the curve's mean POSITION, a scalar.

    Used for step limiting, for collapse detection, and for reporting -- note
    that epsilon is not itself a physical amplitude, because the eigenvector
    is normalised across all six mixed position/velocity components.
    """
    return float(np.max(np.linalg.norm(U[:, :3] - U[:, :3].mean(axis=0),
                                       axis=1)))


# =========================================================================
# SECTION 6C -- OFF-GRID EVALUATION
# =========================================================================

def evaluate_curve(U, theta):
    """Fourier-interpolate a discretised curve at arbitrary angles.

    U is (N, 6) at the node angles; theta is a scalar or (M,) array.
    Returns (M, 6) real.  Exact for band-limited curves -- same spectral
    machinery as the rotation operator, so no interpolation error is
    introduced that the residual would then have to fight.
    """
    N = U.shape[0]
    theta = np.atleast_1d(np.asarray(theta, float))
    C = dft_matrix(N) @ U                       # (N, 6) complex, symmetric k
    E = np.exp(1j * theta[:, None] * wavenumbers(N)[None, :])   # (M, N)
    return np.real(E @ C)


def offgrid_residual(U0, T, rho, mu, offsets=None):
    """Invariance residual at points BETWEEN the nodes.  Returns max |.|.

    Newton driving F to zero only says the curve is invariant AT the nodes.
    A curve with too few nodes can satisfy that exactly and still not lie on
    a torus.  The honest check evaluates (*) off-grid:

        max_theta || phi_T( v(theta) ) - v(theta + rho) ||_inf

    with v obtained by Fourier interpolation.  Default sample points are the
    node midpoints, the worst case for a band-limited interpolant.

    If this is orders of magnitude above the on-grid residual, raise N.
    """
    _require_adapters()
    N = U0.shape[0]
    if offsets is None:
        offsets = node_angles(N) + np.pi / N
    lhs = np.array([propagate(x, T, mu) for x in evaluate_curve(U0, offsets)])
    rhs = evaluate_curve(U0, offsets + rho)
    return float(np.abs(lhs - rhs).max())


# =========================================================================
# SECTION 7 -- CORRECTION STEP
# =========================================================================

RCOND_DEFAULT = 1e-8
CONSTRAINT_ROWS = 3          # 2 phase + 1 Jacobi, when rho is held fixed


def _constraint_row(row_U, rhs_val, n_extra):
    """Normalise one constraint row to unit 2-norm, scaling its rhs to match.

    Least squares balances rows by magnitude.  The invariance rows carry STM
    entries of order 1e3 for a halo this hyperbolic, so an unnormalised
    constraint row would be weighted almost arbitrarily against them.  Unit
    norm is a defensible default because each constraint is nearly aligned
    with a direction the invariance block barely sees (singular value ~1e-5),
    so unit weight already dominates there without distorting the
    well-conditioned directions.
    """
    row = np.concatenate([row_U, np.zeros(n_extra)])
    nrm = np.linalg.norm(row)
    if nrm == 0.0:
        raise ValueError("degenerate constraint row (zero norm)")
    return row / nrm, rhs_val / nrm


def newton_step(U0, T, rho, mu, U_ref=None, C_target=None, free_rho=True,
                rcond=RCOND_DEFAULT):
    """ONE least-squares correction step.  Returns (U0_new, T_new, rho_new, info).

    SIGNATURE CHANGED IN WEEK 8.  The Week 7 version returned |F| as a bare
    float and imposed only the two phase constraints; it was under-determined
    by two.  See SECTION 6B.

    Free variables : [U0 (6N), T, rho]   (rho dropped when free_rho=False)
    Rows           : 6N invariance (rank 6N - 2)
                     + 2 phase constraints  (kill the symmetry directions)
                     + 1 Jacobi constraint  (pins amplitude)

    free_rho=True is the default and is the correct setting -- see SECTION 6B
    for the measurements.  free_rho=False is kept only so the over-constrained
    case can be reproduced on demand; it stalls at O(eps^3).

    lstsq rather than a square solve is retained deliberately.  The rank
    argument above is structural, and the two redundant rows it implies are
    absorbed harmlessly by minimum-norm least squares; a square solve would
    make a wrong rank count fatal rather than merely wasteful.

    RCOND.  1e-8 relative against sigma_max ~ 2.2e3 puts the cutoff at ~2e-5,
    which is exactly where the two deficient singular values sat before the
    parameter constraints were added -- that is why the Week 7 empirical
    value worked, and independent evidence that the deficiency is 2.  With
    the constraints in place those directions are pinned and the cutoff
    should matter much less; sweep it and check, do not assume.  What no
    rcond can fix is the conditioning inherited from the hyperbolic
    multiplier (~1.2e3 here, so cond(J) ~ 1e8): that is a property of single
    shooting over a full period, not a bug.

    info keys: 'res_inf', 'res_2', 'step_norm', 'phase0', 'phase1', 'dC',
               'rank', 'sv' (singular values of the assembled system).
    """
    if U_ref is None:
        U_ref = U0
    N = U0.shape[0]
    n_extra = 2 if free_rho else 1          # columns beyond U0: T, and maybe rho

    J_U0, J_T, J_rho, F = invariance_jacobian(U0, T, rho, mu)
    cols = [J_U0, J_T[:, None]] + ([J_rho[:, None]] if free_rho else [])
    rows, rhs = [np.hstack(cols)], [F]

    # --- phase constraints: <U - U_ref, tan> = 0.  With the self-referencing
    #     default the value is identically zero, so these rows constrain only
    #     the STEP -- they forbid drift along the two symmetries.
    tan0 = longitudinal_tangent(U_ref, T, rho, mu)
    tan1 = latitudinal_tangent(U_ref)
    dU = U0 - U_ref
    phase_vals = []
    for tan in (tan0, tan1):
        val = float(np.sum(dU * tan))
        phase_vals.append(val)
        row, r = _constraint_row(flatten_curve(tan), val, n_extra)
        rows.append(row[None, :]); rhs.append(np.array([r]))

    # --- amplitude constraint: mean Jacobi held at C_target.
    if C_target is None:
        C_target = mean_jacobi(U0, mu)
    dC = mean_jacobi(U0, mu) - C_target
    row, r = _constraint_row(mean_jacobi_row(U0, mu), dC, n_extra)
    rows.append(row[None, :]); rhs.append(np.array([r]))

    J = np.vstack(rows)
    rvec = np.concatenate(rhs)
    dx, _res, rank, sv = np.linalg.lstsq(J, -rvec, rcond=rcond)

    U0_new = U0 + unflatten_curve(dx[:6 * N])
    T_new = T + dx[6 * N]
    rho_new = rho + dx[6 * N + 1] if free_rho else rho
    info = dict(res_inf=float(np.abs(F).max()), res_2=float(np.linalg.norm(F)),
                step_norm=float(np.linalg.norm(dx)), phase0=phase_vals[0],
                phase1=phase_vals[1], dC=dC, rank=int(rank), sv=sv)
    return U0_new, T_new, rho_new, info


def converge_torus(U0, T, rho, mu, U_ref=None, C_target=None, free_rho=True,
                   tol=1e-10, itmax=30, rcond=RCOND_DEFAULT,
                   step_limit_frac=0.5, collapse_frac=0.1, verbose=True):
    """Drive newton_step to a converged invariant curve.

    Convergence is declared on ||F||_inf, not the 2-norm.  The 2-norm grows
    like sqrt(6N) purely from discretisation, so a tolerance stated in it
    silently tightens as N rises; the infinity norm is the same number at
    every N.  (Week 7 reported 2-norms -- they are not comparable to these.)

    STEP LIMITING is relative, not absolute.  The step is capped so no node
    moves further than step_limit_frac times the curve's own amplitude, which
    is scale-free and therefore survives the epsilon sweep without retuning.

    Returns a dict:
      'U0','T','rho'  : the iterate at exit (converged or not)
      'converged'     : bool
      'code'          : 'converged' | 'max_iter' | 'collapsed' | 'diverged'
                        | 'nonfinite'
      'history'       : list of per-iteration info dicts, each with the
                        newton_step keys plus 'amplitude'
      'C'             : mean Jacobi at exit

    C_target -- READ THIS BEFORE SWEEPING EPSILON.  When left at None it is
    taken from the SEED: C_target = mean_jacobi(U_seed).  The seed is
    x_halo + eps * Re[e^{i theta} v_C], whose mean Jacobi differs from the
    halo's by O(eps^2).  So a sweep over eps with C_target=None converges every
    torus AT A DIFFERENT ENERGY, and any quantity compared across those rows
    mixes the amplitude dependence with the energy dependence.

    That is exactly what happened to the Week 8 amplitude sweep.  Its
    'rho - rho0' column rises as eps^2 and reads as a positive
    amplitude-frequency coefficient.  It is not one.  Holding energy fixed the
    true coefficient is NEGATIVE, kappa = -11.08 in rho/2pi per A^2, confirmed
    three ways: directly from a constant-C continuation, from the identity
    kappa = c * drho_halo/dC against the Phase 2 halo family, and by rerunning
    the eps sweep with C_target pinned.  The sweep column was measuring
    drho/dC * dC/deps^2, which carries the opposite sign.

    Nothing about the solver was wrong and the default is not changed here --
    every Week 8 number stays reproducible.  What changes is that 'C_target' is
    now returned, so a sweep that assumes fixed energy can see at a glance that
    it is not getting it.  To pin the energy, pass the underlying orbit's value:

        C_halo = qpit.jacobi_constant(orbit['state0'], mu)
        converge_torus(..., C_target=C_halo)

    FAILURE CODES map onto the Week 8 taxonomy:
      'collapsed'  -- amplitude fell below collapse_frac of its initial value.
                      The constant curve v == x_halo solves the invariance
                      equation for any rho, so it is a genuine attractor if
                      amplitude is not pinned.  Seeing this means the Jacobi
                      constraint is not doing its job.
      'diverged'   -- residual rose an order of magnitude above its best.
                      Suspect the seed sign or an epsilon outside the basin.
      'max_iter' with the residual flat around 1e-8..1e-10 is the integrator
                      tolerance floor, not a formulation problem.
    """
    _require_adapters()
    U0 = np.array(U0, dtype=float, copy=True)
    C_seed = mean_jacobi(U0, mu)          # reported so a sweep can see it move
    if C_target is None:
        C_target = C_seed
    amp0 = curve_amplitude(U0)
    best = np.inf
    history = []
    code = 'max_iter'

    for it in range(itmax):
        U0_new, T_new, rho_new, info = newton_step(
            U0, T, rho, mu, U_ref=U_ref, C_target=C_target,
            free_rho=free_rho, rcond=rcond)
        amp = curve_amplitude(U0)
        info['amplitude'] = amp
        history.append(info)
        if verbose:
            print(f"  it{it:2d}: |F|inf={info['res_inf']:.4e}  "
                  f"|dx|={info['step_norm']:.3e}  A={amp:.4e}  "
                  f"T={T:.9f}  dC={info['dC']:+.2e}  rank={info['rank']}")

        if info['res_inf'] < tol:
            code = 'converged'
            break
        if not np.isfinite(info['res_inf']) or not np.all(np.isfinite(U0_new)):
            code = 'nonfinite'
            break
        if info['res_inf'] > 10.0 * best:
            code = 'diverged'
            break
        best = min(best, info['res_inf'])

        # --- relative step limiting
        dU = U0_new - U0
        move = np.max(np.linalg.norm(dU[:, :3], axis=1))
        cap = step_limit_frac * max(amp, 1e-14)
        if move > cap:
            scale = cap / move
            U0_new = U0 + scale * dU
            T_new = T + scale * (T_new - T)
            rho_new = rho + scale * (rho_new - rho)

        U0, T, rho = U0_new, T_new, rho_new
        if curve_amplitude(U0) < collapse_frac * amp0:
            code = 'collapsed'
            break

    return dict(U0=U0, T=T, rho=rho, converged=(code == 'converged'),
                code=code, history=history, C=mean_jacobi(U0, mu),
                amplitude=curve_amplitude(U0), n_iter=len(history),
                C_target=float(C_target), C_seed=float(C_seed))


# =========================================================================
# SECTION 8 -- STABILITY
# =========================================================================

def stability_matrix(U0: np.ndarray, T: float, rho: float,
                     mu: float) -> np.ndarray:
    """DG, the (6N, 6N) differential of the invariance condition.

    McCarthy & Howell Eq. (9).  Note this is the same object as the first
    Jacobian block before the identity is subtracted, so once a torus has
    converged the stability information costs nothing extra -- the STMs were
    already propagated for the corrector.
    """
    _require_adapters()
    N = U0.shape[0]
    _, stms = propagate_curve(U0, T, mu, with_stm=True)
    Phi_tilde = np.zeros((6 * N, 6 * N))
    for j, Phi in enumerate(stms):
        Phi_tilde[6 * j:6 * j + 6, 6 * j:6 * j + 6] = Phi
    return np.kron(rotation_operator(N, -rho), np.eye(6)) @ Phi_tilde


def stability_index(DG: np.ndarray) -> float:
    """nu = 0.5 * (R_u + 1/R_u), where R_u is the unstable-circle radius.

    Eigenvalues of DG lie on concentric circles about the origin
    (lambda = Lambda_r * e^{-i k rho}); radius > 1 is the unstable mode.
    nu == 1 means stable, nu > 1 unstable with hyperbolic manifolds.
    McCarthy & Howell Eq. (12).

    Taking R_u as the maximum eigenvalue modulus is the simple reading and
    is correct when a single unstable circle dominates.  Verify against a
    plotted eigenstructure before relying on it near bifurcations.
    """
    R_u = float(np.abs(np.linalg.eigvals(DG)).max())   # one eigensolve, not two
    return float(0.5 * (R_u + 1.0 / R_u))


# =========================================================================
# SECTION 9 -- CONTINUATION  (Week 9)
# =========================================================================
# Section 6B left the first torus isolated only up to a 1-D nullspace: with
# rho and T free and mean Jacobi pinned, the assembled system is
# (6N + 3) x (6N + 2) of rank 6N + 1.  newton_step reports exactly that
# (rank=187 at N=31), and least squares takes minimum-norm steps, so it sits
# still.  Continuation is the act of choosing to move along that direction
# instead, with one extra row fixing HOW FAR.
#
# MEASURED STRUCTURE at the converged torus (N=31, A=17.2 km):
#   invariance block  (6N) x (6N+2) = 186 x 188
#   zero singular values                      2
#   nullity = (188 - 186) + 2                 4   <- the 4-D solution set
#   latitudinal tangent in that nullspace     overlap 1.000000000000
#   longitudinal tangent in that nullspace    overlap 1.000000000000
#   directions left after projecting both out 2   <- the physical family
#
# So: 2 symmetries + 2 physical parameters, confirmed rather than assumed.
# One phase constraint kills each symmetry; one parameter pin kills one
# physical direction; the arclength row kills the last one.
#
# WHICH PARAMETER TO PIN.  Three consistent choices, and they are different
# families, not different parameterisations of one family:
#
#   pin='rho'  constant frequency ratio.  omega_0 = 2pi/T and omega_1 =
#              rho/T, so omega_1/omega_0 = rho/2pi and pinning rho pins the
#              ratio exactly.  The family never approaches a resonance
#              because it never moves in rho.  DEFAULT, per McCarthy &
#              Howell -- and this is the family that reaches large amplitude.
#   pin='C'    constant energy.  rho drifts with amplitude (measured
#              +26 eps^2 rad on this orbit), so this family walks toward
#              whatever rational rho is nearest and terminates there.  Kept
#              because that drift is the mechanism the resonance study needs.
#   pin='T'    constant mapping time.  Same termination problem as 'C'.
#
# NOT A CONTRADICTION WITH SECTION 6B.  That section says do not pin rho.
# It means: do not pin rho AT THE LINEAR SEED VALUE WHILE ALSO PINNING C.
# rho0 is read off the monodromy at zero amplitude and the true torus sits
# O(eps^2) away, so demanding both is demanding a torus that does not exist,
# and the residual stalls at O(eps^3).  Pinning rho ALONE, at the CONVERGED
# value of an existing family member, with C free to move, is a different
# and entirely consistent request: it asks for the torus of that rotation
# number at whatever energy hosts it.  torus_tangent refuses to pin rho at a
# value that did not come from a converged solve is not something code can
# check -- so it is stated here instead, and the notebook checks it by
# confirming the first family member reproduces its own seed.


def pack_X(U0, T, rho):
    """[U0 flattened, T, rho] -> (6N + 2,).  The continuation state vector."""
    return np.concatenate([flatten_curve(U0), [float(T), float(rho)]])


def unpack_X(X):
    """Inverse of pack_X.  Returns (U0 (N,6), T, rho)."""
    return unflatten_curve(X[:-2]), float(X[-2]), float(X[-1])


# ---------------------------------------------------------------- resolution

def fourier_tail(U):
    """||C_k|| at the highest retained |k|, relative to the FUNDAMENTAL (k=+/-1).

    The cheap per-step resolution monitor.  offgrid_residual is the honest
    check but costs N propagations; this costs one DFT, so it runs every
    continuation step and offgrid_residual runs only on accepted members.

    A converged curve should show this at the integrator noise floor
    (2.85e-10 at N=31 on the first torus).  Once it climbs above ~1e-8 the
    curve is no longer band-limited at this N and the invariance residual is
    measuring the truncation, not the torus.

    NORMALISED AGAINST THE FUNDAMENTAL (k = +/-1), NOT k = 0.  The k = 0
    coefficient is the centre of the curve -- order 1 for an orbit near the
    Moon -- while the whole shape of a 17 km torus lives at 1e-5.  Dividing
    by k = 0 therefore reports a tail about 1e4 times healthier than it is,
    and a refinement trigger set on it would never fire.
    """
    N = U.shape[0]
    C = dft_matrix(N) @ U
    mag = np.linalg.norm(C, axis=1)
    k = np.abs(wavenumbers(N))
    return float(mag[k == k.max()].max() / mag[k == 1].max())


def resample_curve(U, N_new):
    """Re-discretise a curve onto N_new nodes by zero-padding in Fourier space.

    Exact for band-limited curves in both directions: padding adds modes that
    were zero, truncating removes modes that are at the noise floor.  No
    interpolation error is introduced, which is the whole reason the spectral
    representation was chosen in Section 2.

    Both N and N_new must be odd (convention 2).  Refining mid-family changes
    the dimension of X, so any tangent held across the change is invalid and
    must be recomputed -- continue_torus_family does that.
    """
    check_odd(N_new)
    N = U.shape[0]
    if N_new == N:
        return U.copy()
    C = dft_matrix(N) @ U                      # (N, 6) complex, symmetric k
    k_old, k_new = wavenumbers(N), wavenumbers(N_new)
    C_new = np.zeros((N_new, 6), dtype=complex)
    keep = np.isin(k_old, k_new)
    C_new[np.searchsorted(k_new, k_old[keep])] = C[keep]
    return np.real(idft_matrix(N_new) @ C_new)


# ------------------------------------------------------------------ assembly

def _pin_row(pin, U0, T, rho, mu, target, n_extra):
    """One parameter-pin row and its residual value.  Returns (row, value).

    'C'   holds the mean Jacobi constant -- a dense row over the curve.
    'rho' and 'T' are single unit entries in the parameter columns, already
    unit-norm, so they bypass _constraint_row's rescaling.
    """
    N = U0.shape[0]
    if pin == 'C':
        return _constraint_row(mean_jacobi_row(U0, mu),
                               mean_jacobi(U0, mu) - target, n_extra)
    row = np.zeros(6 * N + n_extra)
    if pin == 'T':
        row[6 * N] = 1.0
        return row, T - target
    if pin == 'rho':
        if n_extra < 2:
            raise ValueError("pin='rho' requires rho to be a free variable")
        row[6 * N + 1] = 1.0
        return row, rho - target
    raise ValueError(f"unknown pin {pin!r}; expected 'C', 'rho' or 'T'")


def assemble_system(U0, T, rho, mu, U_ref=None, pin='C', pin_target=None,
                    tangent=None, X_prev=None, ds=None, free_rho=True):
    """The full least-squares system.  Returns (J, rvec, info).

    Rows, in order:
        6N  invariance                     rank 6N - 2
         1  longitudinal phase             kills the flow-shift symmetry
         1  latitudinal phase              kills the curve-relabel symmetry
         1  parameter pin                  kills one physical direction
         1  arclength   (only if tangent)  kills the last one

    Columns: [U0 (6N) | T | rho], the last dropped when free_rho=False.

    Without the arclength row this is the Week 8 system and carries a 1-D
    nullspace -- which is precisely the object torus_tangent extracts.  With
    it, rank rises to 6N + 2 and lstsq should report full column rank; that
    jump (187 -> 188 at N=31) is the cheapest confirmation that the arclength
    row is doing its job and is asserted in the notebook.

    EQUIVALENCE.  With pin='C' and tangent=None this reproduces newton_step's
    matrix and right-hand side exactly, bit for bit; tests assert it.
    newton_step is deliberately left untouched so every Week 8 number stays
    reproducible from the same code path that produced it.
    """
    if U_ref is None:
        U_ref = U0
    N = U0.shape[0]
    n_extra = 2 if free_rho else 1

    J_U0, J_T, J_rho, F = invariance_jacobian(U0, T, rho, mu)
    cols = [J_U0, J_T[:, None]] + ([J_rho[:, None]] if free_rho else [])
    rows, rhs = [np.hstack(cols)], [F]

    tan0 = longitudinal_tangent(U_ref, T, rho, mu)
    tan1 = latitudinal_tangent(U_ref)
    dU = U0 - U_ref
    phase_vals = []
    for tan in (tan0, tan1):
        val = float(np.sum(dU * tan))
        phase_vals.append(val)
        row, r = _constraint_row(flatten_curve(tan), val, n_extra)
        rows.append(row[None, :]); rhs.append(np.array([r]))

    if pin_target is None:
        pin_target = dict(C=mean_jacobi(U0, mu), rho=rho, T=T)[pin]
    row, r = _pin_row(pin, U0, T, rho, mu, pin_target, n_extra)
    rows.append(row[None, :]); rhs.append(np.array([r]))

    arclen = None
    if tangent is not None:
        X = pack_X(U0, T, rho) if free_rho else np.concatenate(
            [flatten_curve(U0), [T]])
        arclen = float(np.dot(X - X_prev, tangent)) - float(ds)
        rows.append(tangent[None, :]); rhs.append(np.array([arclen]))

    info = dict(res_inf=float(np.abs(F).max()), res_2=float(np.linalg.norm(F)),
                phase0=phase_vals[0], phase1=phase_vals[1],
                pin_err=float(r), arclen_err=arclen)
    return np.vstack(rows), np.concatenate(rhs), info


# ------------------------------------------------------------------- tangent

def torus_tangent(U0, T, rho, mu, pin='rho', pin_target=None,
                  prev_tangent=None, free_rho=True, gap_warn=1e3):
    """Unit tangent to the 1-parameter family.  Returns (t, diag).

    The tangent is the nullspace of the system WITHOUT the arclength row --
    (6N+3) x (6N+2), rank 6N+1, nullity exactly 1 -- taken as the last right
    singular vector.

    COMPUTE IT FROM THE CONSTRAINED SYSTEM, NOT FROM THE INVARIANCE BLOCK.
    The invariance block alone has nullity 4 (measured above), and two of
    those four are the phase symmetries.  A "tangent" drawn from that space
    would generically be a mixture, and the predictor would step along a
    relabelling of the same curve -- amplitude unchanged, residual small,
    a family that goes nowhere while looking healthy.  Including the phase
    rows forces t orthogonal to both symmetry tangents, which is also what
    makes the arclength row independent of the phase rows at the next step.

    ORIENTATION.  Sign is continued from prev_tangent when given; otherwise
    it is chosen so the curve amplitude increases, since the small-amplitude
    end of every family here is the periodic orbit itself.

    diag: 'sv_null' (should be ~0), 'sv_next' (should not be), 'gap',
          't_rho', 't_T', 'dA_ds'.  A gap below gap_warn means the nullspace
          is not cleanly 1-D -- a bifurcation, or a pin that is degenerate
          with the family direction -- and the tangent is not trustworthy.
    """
    J, _, _ = assemble_system(U0, T, rho, mu, pin=pin, pin_target=pin_target,
                              free_rho=free_rho)
    sv, Vt = np.linalg.svd(J)[1:]
    t = Vt[-1].copy()
    gap = float(sv[-2] / sv[-1]) if sv[-1] > 0 else np.inf

    # orientation: finite-difference the amplitude along +t, in units that
    # move each node by ~1% of the curve's own amplitude.
    A0 = curve_amplitude(U0)
    tU = unflatten_curve(t[:U0.size])
    move = np.max(np.linalg.norm(tU[:, :3], axis=1))
    h = 0.01 * A0 / move if move > 0 else 0.0
    dA_ds = (curve_amplitude(U0 + h * tU) - A0) / h if h > 0 else 0.0
    if prev_tangent is not None:
        if np.dot(t, prev_tangent) < 0:
            t, dA_ds = -t, -dA_ds
    elif dA_ds < 0:
        t, dA_ds = -t, -dA_ds

    diag = dict(sv_null=float(sv[-1]), sv_next=float(sv[-2]), gap=gap,
                t_T=float(t[U0.size]),
                t_rho=float(t[U0.size + 1]) if free_rho else 0.0,
                dA_ds=float(dA_ds))
    if gap < gap_warn:
        diag['warning'] = (f"nullspace gap {gap:.2e} below {gap_warn:.0e}: "
                           "tangent may be contaminated (bifurcation?)")
    return t, diag


# ------------------------------------------------------------------ corrector

def correct_along_family(U0, T, rho, mu, U_ref, pin, pin_target,
                         tangent=None, X_prev=None, ds=None, free_rho=True,
                         tol=1e-11, itmax=12, rcond=RCOND_DEFAULT,
                         step_limit_frac=0.5, verbose=False):
    """Newton on the arclength-closed system.  Returns a result dict.

    Differs from converge_torus in exactly two ways, both required:
      * the constraint set is chosen by `pin` rather than hardwired to C;
      * U_ref is the PREVIOUS CONVERGED MEMBER, not the iterate itself.

    The second is what the phase constraints were designed for.  With
    U_ref = U0 the constraints are identically zero and only restrain the
    step; with U_ref a genuine previous member they carry a real value and
    actively pull the new curve back to the same phase origin, which is what
    keeps the family from drifting around the torus as it advances.

    No collapse guard is needed here: the arclength row forbids standing
    still, so the constant curve v == x_halo is no longer reachable.
    """
    U0 = np.array(U0, dtype=float, copy=True)
    history = []
    code = 'max_iter'
    for it in range(itmax):
        J, rvec, info = assemble_system(
            U0, T, rho, mu, U_ref=U_ref, pin=pin, pin_target=pin_target,
            tangent=tangent, X_prev=X_prev, ds=ds, free_rho=free_rho)
        dx, _r, rank, sv = np.linalg.lstsq(J, -rvec, rcond=rcond)
        info.update(rank=int(rank), step_norm=float(np.linalg.norm(dx)),
                    amplitude=curve_amplitude(U0))
        history.append(info)
        if verbose:
            print(f"    it{it:2d}: |F|inf={info['res_inf']:.3e} "
                  f"|dx|={info['step_norm']:.2e} rank={info['rank']}")
        if info['res_inf'] < tol:
            code = 'converged'
            break
        if not np.isfinite(info['res_inf']):
            code = 'nonfinite'
            break
        if it and info['res_inf'] > 10.0 * min(h['res_inf'] for h in history[:-1]):
            code = 'diverged'
            break

        dU = unflatten_curve(dx[:U0.size])
        move = np.max(np.linalg.norm(dU[:, :3], axis=1))
        cap = step_limit_frac * max(curve_amplitude(U0), 1e-14)
        scale = min(1.0, cap / move) if move > 0 else 1.0
        U0 = U0 + scale * dU
        T = T + scale * dx[U0.size]
        if free_rho:
            rho = rho + scale * dx[U0.size + 1]

    return dict(U0=U0, T=T, rho=rho, converged=(code == 'converged'),
                code=code, history=history, n_iter=len(history),
                C=mean_jacobi(U0, mu), amplitude=curve_amplitude(U0),
                res_inf=history[-1]['res_inf'])


# --------------------------------------------------------------- family loop

def continue_torus_family(U0, T, rho, mu, pin='rho', n_steps=40,
                          ds=None, ds_frac=0.10, ds_min_frac=1e-3,
                          ds_max_frac=1.0, it_target=3, tail_tol=1e-8,
                          N_max=81, tol=1e-11, itmax=12, offgrid_every=1,
                          verbose=True):
    """Pseudo-arclength continuation of a converged torus into a family.

    START FROM A CONVERGED TORUS.  The pin target is read off the state you
    pass in, so passing a seed instead of a solution pins the wrong value --
    for pin='rho' that is exactly the Section 6B mistake, one level up.

    STEP SIZE is expressed as a fraction of the curve's own amplitude:
    ds = ds_frac * A * sqrt(N) makes each node move by about ds_frac * A,
    which is scale-free and so needs no retuning as the family grows by three
    orders of magnitude.  It adapts on the Newton iteration count -- grow by
    1.5 when a step converges in <= it_target, halve and retry when it fails.

    RESOLUTION is monitored by fourier_tail every step.  Above tail_tol the
    curve has outgrown its node count, so N is raised by 10 (odd -> odd), the
    member is re-converged at the new N, and the tangent is recomputed --
    it must be, since X changed dimension.

    Returns dict with 'members' (list of per-torus records), 'pin',
    'pin_target', 'terminated' (why the loop stopped).  Each member record
    carries U0, T, rho, C, amplitude, N, res_inf, offgrid, tail, n_iter, ds.
    """
    _require_adapters()
    U0 = np.array(U0, dtype=float, copy=True)
    pin_target = dict(C=mean_jacobi(U0, mu), rho=rho, T=T)[pin]

    def record(U, T_, r_, extra):
        rec = dict(U0=U.copy(), T=T_, rho=r_, C=mean_jacobi(U, mu),
                   amplitude=curve_amplitude(U), N=U.shape[0],
                   tail=fourier_tail(U))
        rec.update(extra)
        return rec

    t, tdiag = torus_tangent(U0, T, rho, mu, pin=pin, pin_target=pin_target)
    members = [record(U0, T, rho, dict(
        res_inf=float(np.abs(invariance_residual(U0, T, rho, mu)).max()),
        offgrid=offgrid_residual(U0, T, rho, mu), n_iter=0, ds=0.0,
        tangent_gap=tdiag['gap']))]
    if verbose:
        m = members[0]
        print(f"pin={pin} at {pin_target:.12f}   tangent gap {tdiag['gap']:.2e}"
              f"   dA/ds {tdiag['dA_ds']:+.3e}")
        print(f"{'step':>4} {'N':>3} {'A [km]':>10} {'C':>14} {'rho/2pi':>11} "
              f"{'|F|inf':>10} {'tail':>9} {'its':>4} {'ds':>9}")
        print(f"{0:4d} {m['N']:3d} {m['amplitude']*384400:10.2f} {m['C']:14.9f} "
              f"{m['rho']/(2*np.pi):11.8f} {m['res_inf']:10.2e} "
              f"{m['tail']:9.1e} {0:4d} {0.0:9.2e}")

    N = U0.shape[0]
    ds = ds_frac * curve_amplitude(U0) * np.sqrt(N) if ds is None else ds
    terminated = 'n_steps'

    for step in range(1, n_steps + 1):
        prev = members[-1]
        ds_floor = ds_min_frac * prev['amplitude'] * np.sqrt(prev['N'])
        ds_ceil = ds_max_frac * prev['amplitude'] * np.sqrt(prev['N'])
        ds = min(ds, ds_ceil)
        accepted = None

        while ds >= ds_floor:
            X_prev = pack_X(prev['U0'], prev['T'], prev['rho'])
            Xp = X_prev + ds * t
            Up, Tp, rp = unpack_X(Xp)
            out = correct_along_family(
                Up, Tp, rp, mu, U_ref=prev['U0'], pin=pin,
                pin_target=pin_target, tangent=t, X_prev=X_prev, ds=ds,
                tol=tol, itmax=itmax, verbose=False)
            if out['converged']:
                accepted = out
                break
            ds *= 0.5

        if accepted is None:
            terminated = 'ds_underflow'
            break

        U0, T, rho = accepted['U0'], accepted['T'], accepted['rho']
        tail = fourier_tail(U0)

        if tail > tail_tol and U0.shape[0] < N_max:
            N_new = U0.shape[0] + 10
            U0 = resample_curve(U0, N_new)
            out2 = correct_along_family(U0, T, rho, mu, U_ref=U0, pin=pin,
                                        pin_target=pin_target, tol=tol,
                                        itmax=itmax)
            if not out2['converged']:
                terminated = f'refine_failed_at_N={N_new}'
                break
            U0, T, rho = out2['U0'], out2['T'], out2['rho']
            accepted['n_iter'] += out2['n_iter']
            t = None                        # dimension changed; tangent void

        og = (offgrid_residual(U0, T, rho, mu)
              if offgrid_every and step % offgrid_every == 0 else np.nan)
        members.append(record(U0, T, rho, dict(
            res_inf=accepted['res_inf'], offgrid=og,
            n_iter=accepted['n_iter'], ds=ds)))

        t, tdiag = torus_tangent(U0, T, rho, mu, pin=pin,
                                 pin_target=pin_target, prev_tangent=t)
        members[-1]['tangent_gap'] = tdiag['gap']

        if verbose:
            m = members[-1]
            print(f"{step:4d} {m['N']:3d} {m['amplitude']*384400:10.2f} "
                  f"{m['C']:14.9f} {m['rho']/(2*np.pi):11.8f} "
                  f"{m['res_inf']:10.2e} {m['tail']:9.1e} "
                  f"{m['n_iter']:4d} {ds:9.2e}")

        if accepted['n_iter'] <= it_target:
            ds *= 1.5

    return dict(members=members, pin=pin, pin_target=pin_target,
                terminated=terminated, mu=mu)


# =========================================================================
# SECTION 10 -- TORUS INVARIANT BUNDLES  (Week 9, manifolds)
# =========================================================================

def torus_bundle(U0, T, rho, mu, which='unstable', n_power=12, DG=None):
    """Real unstable (or stable) direction attached to each node.  (N, 6).

    THE BUNDLE EQUATION.  Writing DG = (R(-rho) kron I) Phi_tilde and reading
    an eigenvector w as a curve of vectors w(theta_j), DG w = lambda w says

        Dphi_T( u(theta) ) w(theta) = lambda w(theta + rho)

    -- the direction is transported by the STM but lands one rotation further
    round the curve, not back on itself.  That is why this is a bundle
    problem and not a per-node eigenproblem, and why the eigenvalues come in
    circles: if w solves it then e^{i k theta} w solves it with
    lambda e^{-i k rho}, so each real bundle contributes N eigenvalues of
    equal modulus.

    WHY POWER ITERATION AND NOT eig(DG).  The circle structure means the
    dominant eigenvalue is N-fold degenerate in modulus, and numpy hands back
    complex eigenvectors whose per-node phase is arbitrary -- reassembling a
    real bundle from one of them means undoing a phase that differs node to
    node.  Iterating DG on a real curve avoids the question entirely: it
    stays real throughout and converges to the dominant real bundle at a rate
    set by the modulus gap, which here is R_u : 1 = 146 : 1, so 12 iterations
    leave 1e-26.  For 'stable', iterate the inverse map.

    Returns (V, R) with V of shape (N, 6), unit rows, and R the measured
    growth factor per application -- which should reproduce the radius behind
    stability_index, giving an independent check on both.
    """
    _require_adapters()
    N = U0.shape[0]
    if DG is None:
        DG = stability_matrix(U0, T, rho, mu)
    M = DG if which == 'unstable' else np.linalg.inv(DG)
    rng = np.random.default_rng(0)           # fixed: the bundle is unique
    v = rng.standard_normal(6 * N)
    v /= np.linalg.norm(v)
    R = np.nan
    for _ in range(n_power):
        w = M @ v
        R = float(np.linalg.norm(w))
        v = w / R
    V = unflatten_curve(v)
    V /= np.linalg.norm(V, axis=1)[:, None]
    return V, (R if which == 'unstable' else 1.0 / R)


def torus_manifold(U0, T, rho, mu, V, d=1e-6, n_periods=4.0, side=+1,
                   n_out=400, which='unstable'):
    """Globalise one branch of a torus manifold.  Returns a list of N arrays.

    Each node is displaced by side * d along its bundle direction and
    propagated -- forward for the unstable branch, backward for the stable
    one.  Same construction as the periodic-orbit manifolds in Phase 2, with
    the eigenvector of a single monodromy matrix replaced by the bundle.

    d is a displacement in nondimensional units; 1e-6 is about 0.4 km, which
    sits above the 1e-10 residual floor and well inside the linear regime for
    a torus of this amplitude.  Check that halving d leaves the globalised
    surface unchanged before trusting a picture of it.
    """
    sgn = 1.0 if which == 'unstable' else -1.0
    tf = sgn * n_periods * T
    ts = np.linspace(0.0, tf, n_out)
    strands = []
    for j in range(U0.shape[0]):
        x0 = U0[j] + side * d * V[j]
        sol = _solve_ivp(_cr3bp_rhs_3d, [0.0, tf], x0, args=(mu,),
                         t_eval=ts, **_INTEGRATOR)
        strands.append(sol.y.T)
    return strands
