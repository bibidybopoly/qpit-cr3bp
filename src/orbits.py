"""Periodic-orbit machinery for the CR3BP: differential correctors,
pseudo-arclength continuation, Floquet analysis and manifold computation.

IMPORTING THIS MODULE HAS NO SIDE EFFECTS.  It used to: lines 1-85 were the
exploratory L1-Lyapunov derivation, run at import time, printing ~30 lines and
integrating an orbit before the caller got control.  Notebooks worked around it
(`halo_families.ipynb` wrapped the import in `redirect_stdout(os.devnull)`;
`qpit_solver.ipynb` annotated it "expect ~30 lines of noise"), and no test file
imported this module at all -- which is why the largest module in src/ had zero
coverage.  The derivation is preserved verbatim in `_l1_lyapunov_derivation()`
at the bottom and still runs under `python src/orbits.py`.

The `sys.path.insert(0, '../src')` that used to sit at line 3 is also gone: it
resolved against the working directory, so it only did anything when the
process happened to start one level below src/.
"""

import numpy as np
from scipy.integrate import solve_ivp

from cr3bp import (cr3bp_jacobian, cr3bp_rhs, find_lagrange_points,
                   integrate_with_stm, MU_EARTH_MOON)


def y_axis_crossing(t, state, mu):
    """Event function: returns y.  solve_ivp detects the x-axis crossing."""
    return state[1]


y_axis_crossing.terminal = True
y_axis_crossing.direction = -1

def correct_lyapunov_orbit(x0, vy0_guess, mu, tol=1e-11, max_iter=20,
                            min_crossing_time=0.5, verbose=True):
    """Differential corrector for a planar Lyapunov orbit via single shooting.

    Now with a guard against degenerate near-zero-time crossings.
    """
    vy0 = vy0_guess

    for iteration in range(max_iter):
        state0 = [x0, 0.0, 0.0, vy0]

        def crossing_event(t, aug, mu):
            return aug[1]
        crossing_event.terminal = True
        crossing_event.direction = -1

        sol = integrate_with_stm(state0, (0, 10), mu, events=crossing_event)

        if len(sol.t_events[0]) == 0:
            raise RuntimeError("No x-axis crossing found.")

        t_cross = sol.t_events[0][0]

        # GUARD: reject degenerate crossings that fire too early
        if t_cross < min_crossing_time:
            raise RuntimeError(
                f"Degenerate crossing at t={t_cross:.3e} (< {min_crossing_time}). "
                f"Guess vy0={vy0:.6f} likely outside convergence basin.")

        aug_cross = sol.y_events[0][0]
        state_cross = aug_cross[:4]
        Phi = aug_cross[4:].reshape(4, 4)
        vx_cross = state_cross[2]

        if verbose:
            print(f"  iter {iteration}: vy0={vy0:.12f}, residual vx={vx_cross:.3e}")

        if abs(vx_cross) < tol:
            if verbose:
                print(f"  Converged: vx residual {vx_cross:.2e}")
            return vy0, t_cross, state_cross

        deriv = cr3bp_rhs(0, state_cross, mu)
        ydot, vxdot = deriv[1], deriv[2]
        sensitivity = Phi[2, 3] - (vxdot / ydot) * Phi[1, 3]
        delta_vy0 = -vx_cross / sensitivity
        vy0 = vy0 + delta_vy0

    raise RuntimeError(f"Did not converge in {max_iter} iterations.")

def periodicity_constraint(X, mu):
    """The periodicity constraint F and its Jacobian dF/dX for a symmetric
    Lyapunov orbit.

    Free variables X = [x0, vy0, T_half].
    Integrates from (x0, 0, 0, vy0) for time T_half (no event detection).

    Returns
    -------
    F : ndarray, shape (2,)
        [y(T_half), vx(T_half)] — both should be zero for a periodic orbit.
    dF : ndarray, shape (2, 3)
        Jacobian of F with respect to [x0, vy0, T_half].
    final_state : ndarray, shape (4,)
        The full state at T_half (useful for diagnostics).
    """
    x0, vy0, T_half = X
    state0 = [x0, 0.0, 0.0, vy0]

    # Integrate state + STM for exactly T_half (no events — fixed time)
    sol = integrate_with_stm(state0, (0, T_half), mu, t_eval=[T_half])
    aug_final = sol.y[:, -1]
    final_state = aug_final[:4]
    Phi = aug_final[4:].reshape(4, 4)   # STM at T_half

    # F = [y, vx] at T_half  (state indices: x=0, y=1, vx=2, vy=3)
    F = np.array([final_state[1], final_state[2]])

    # --- Jacobian dF/dX ---
    # F depends on x0, vy0 through the STM (how final state responds to initial state)
    # and on T_half through the equations of motion (how state changes with more time).

    # Derivatives of [y, vx] w.r.t. [x0, vy0]:
    # x0 is initial component 0, vy0 is initial component 3.
    # Phi[i, j] = d(final state i) / d(initial state j)
    dF_dx0  = np.array([Phi[1, 0], Phi[2, 0]])   # d[y,vx]/d(x0)
    dF_dvy0 = np.array([Phi[1, 3], Phi[2, 3]])   # d[y,vx]/d(vy0)

    # Derivative w.r.t. T_half: how do y and vx change if we integrate longer?
    # That's just their time derivatives at the final state = the equations of motion.
    deriv = cr3bp_rhs(0, final_state, mu)         # [vx, vy, ax, ay] at final state
    dF_dT = np.array([deriv[1], deriv[2]])        # d[y,vx]/dt = [vy, ax] at final

    # Assemble the 2x3 Jacobian: columns are d/dx0, d/dvy0, d/dT_half
    dF = np.column_stack([dF_dx0, dF_dvy0, dF_dT])

    return F, dF, final_state

def family_tangent(dF, previous_tangent=None):
    """Compute the tangent to the orbit family at a point.

    The tangent is the null space of the 2x3 constraint Jacobian dF —
    the direction along which F stays zero (i.e. you stay on the family).

    Parameters
    ----------
    dF : ndarray, shape (2, 3)
        The constraint Jacobian at the current orbit.
    previous_tangent : ndarray, shape (3,), optional
        The tangent from the previous step. Used to keep the direction
        consistent (the null space is defined only up to sign, so we
        pick the sign that agrees with where we were heading).

    Returns
    -------
    tangent : ndarray, shape (3,)
        Unit tangent vector to the family, sign-consistent with previous.
    """
    # SVD: dF = U @ diag(S) @ Vh.  The null space is the row of Vh
    # corresponding to the smallest singular value (last row).
    U, S, Vh = np.linalg.svd(dF)
    tangent = Vh[-1, :]   # last right-singular vector = null space direction

    # Normalize to unit length (so steps are measured in true arclength)
    tangent = tangent / np.linalg.norm(tangent)

    # Sign consistency: the null space direction could point "forward" or
    # "backward" along the family. Keep it pointing the same way as last step.
    if previous_tangent is not None:
        if np.dot(tangent, previous_tangent) < 0:
            tangent = -tangent

    return tangent

def predict_step(X, tangent, ds):
    """Predict the next orbit by stepping along the tangent (finite Taylor step).

    Parameters
    ----------
    X : ndarray, shape (3,)
        Current orbit [x0, vy0, T_half].
    tangent : ndarray, shape (3,)
        Unit tangent to the family.
    ds : float
        Arclength step size.

    Returns
    -------
    X_predicted : ndarray, shape (3,)
        Predicted next orbit (near the family, slightly off it).
    """
    return X + ds * tangent

def pseudo_arclength_correct(X_predicted, X_previous, tangent, ds, mu,
                              tol=1e-11, max_iter=20, verbose=False,
                              constraint_fn=periodicity_constraint):
    """Correct a predicted orbit back onto the family using Newton's method
    with the pseudo-arclength constraint.

    Solves the square 3x3 system:
      F(X) = 0                              (periodicity: be on the family)
      (X - X_previous) . tangent - ds = 0   (arclength: be ds along the tangent)

    Parameters
    ----------
    X_predicted : ndarray, shape (3,)
        The predicted orbit (from predict_step) — starting guess for Newton.
    X_previous : ndarray, shape (3,)
        The previous converged orbit (anchor for the arclength constraint).
    tangent : ndarray, shape (3,)
        Unit tangent used for the prediction and the arclength constraint.
    ds : float
        Arclength step size.
    mu : float
        Mass parameter.

    Returns
    -------
    X : ndarray, shape (3,)
        The corrected orbit, on the family at arclength ds from X_previous.
    """
    X = X_predicted.copy()

    for iteration in range(max_iter):
        # Periodicity constraint and its Jacobian (Lyapunov 2x3, or halo 3x4)
        F, dF, _ = constraint_fn(X, mu)

        # Arclength constraint: (X - X_previous) . tangent - ds
        arc = np.dot(X - X_previous, tangent) - ds

        # Assemble the augmented residual G and its square Jacobian dG.
        # Concatenate ALL constraint components with the arclength row, so this
        # works whether F has 2 entries (Lyapunov) or 3 (halo).
        G = np.concatenate([np.atleast_1d(F), [arc]])
        dG = np.vstack([dF, tangent])   # (k x n) constraint on top, tangent as bottom row

        # Newton step: solve dG @ delta = -G
        delta = np.linalg.solve(dG, -G)
        X = X + delta

        residual = np.max(np.abs(G))
        if verbose:
            print(f"    Newton iter {iteration}: |G| = {residual:.3e}")

        if residual < tol:
            return X

    raise RuntimeError(f"Pseudo-arclength correction did not converge "
                       f"(|G| = {residual:.2e} after {max_iter} iters)")

def compute_family_pseudo_arclength(X_start, ds, n_orbits, mu,
                                     initial_direction=None, verbose=True):
    """Compute a Lyapunov orbit family via pseudo-arclength continuation.

    Parameters
    ----------
    X_start : ndarray, shape (3,)
        A known starting orbit [x0, vy0, T_half] (already on the family).
    ds : float
        Arclength step size.
    n_orbits : int
        Number of orbits to compute.
    initial_direction : ndarray, shape (3,), optional
        Hint for which way to march initially (the tangent's sign is
        otherwise arbitrary). If None, uses the raw SVD direction.
    verbose : bool

    Returns
    -------
    family : list of dict
        Each orbit bundled with x0, vy0, period, jacobi, state0, X.
    """
    from cr3bp import jacobi_constant

    family = []
    X = X_start.copy()
    previous_tangent = initial_direction

    for i in range(n_orbits):
        # Evaluate the constraint and its Jacobian at the current orbit
        F, dF, _ = periodicity_constraint(X, mu)

        # Compute the tangent (sign-consistent with the previous step)
        tangent = family_tangent(dF, previous_tangent)

        # Bundle this orbit's data together (the stale-variable lesson)
        state0 = [X[0], 0.0, 0.0, X[1]]
        orbit = {
            'x0': X[0], 'vy0': X[1], 'T_half': X[2],
            'period': 2 * X[2],
            'jacobi': jacobi_constant(state0, mu),
            'state0': state0,
            'X': X.copy(),
        }
        family.append(orbit)

        if verbose and i % 10 == 0:
            print(f"  Orbit {i}: x0={X[0]:.5f}, vy0={X[1]:.5f}, "
                  f"C={orbit['jacobi']:.5f}, T={orbit['period']:.5f}")

        # Predict the next orbit, then correct it back onto the family
        try:
            X_pred = predict_step(X, tangent, ds)
            X_new = pseudo_arclength_correct(X_pred, X, tangent, ds, mu)
        except RuntimeError as e:
            if verbose:
                print(f"  Stopped at orbit {i}: {e}")
            break

        # Advance
        X = X_new
        previous_tangent = tangent

    return family

def monodromy_and_multipliers(state0, period, mu):
    """Compute the monodromy matrix and Floquet multipliers for a periodic orbit.

    The monodromy matrix M = Phi(T, 0) is the STM integrated over one full
    period. Its eigenvalues are the Floquet multipliers.

    Parameters
    ----------
    state0 : array_like, shape (4,)
        Initial state on the periodic orbit.
    period : float
        The full orbit period T.
    mu : float
        Mass parameter.

    Returns
    -------
    M : ndarray, shape (4, 4)
        The monodromy matrix.
    multipliers : ndarray, shape (4,)
        The Floquet multipliers (eigenvalues of M).
    """
    # Integrate state + STM over one full period
    sol = integrate_with_stm(state0, (0, period), mu, t_eval=[period])
    aug_final = sol.y[:, -1]
    M = aug_final[4:].reshape(4, 4)

    multipliers = np.linalg.eigvals(M)
    return M, multipliers

def compute_manifold(state0, period, mu, stable=False, n_points=40,
                      epsilon=1e-6, integration_time=None, n_strand_points=500):
    """Compute the stable or unstable manifold of a periodic orbit.

    Perturbs along the (transported) stable/unstable eigenvector at n_points
    around the orbit and integrates each perturbation to trace a manifold strand.

    Parameters
    ----------
    state0 : array_like, shape (4,)
        Initial state on the periodic orbit.
    period : float
        Orbit period.
    mu : float
        Mass parameter.
    stable : bool
        If True, compute the stable manifold (integrate backward).
        If False, the unstable manifold (integrate forward).
    n_points : int
        Number of base points around the orbit to seed strands from.
    epsilon : float
        Perturbation size along the eigenvector.
    integration_time : float or None
        How long to integrate each strand. Defaults to ~period if None.
    n_strand_points : int
        Output resolution per strand.

    Returns
    -------
    strands : list of ndarray
        Each is shape (n_strand_points, 4): one manifold strand trajectory.
    """
    if integration_time is None:
        integration_time = period

    # 1. Get the monodromy matrix and the stable/unstable eigenvector at t=0
    M, eigvals = monodromy_and_multipliers(state0, period, mu)
    eigvals_full, eigvecs = np.linalg.eig(M)
    if stable:
        idx = np.argmin(np.abs(eigvals_full))   # |lambda| < 1
    else:
        idx = np.argmax(np.abs(eigvals_full))   # |lambda| > 1
    v0 = np.real(eigvecs[:, idx])
    v0 = v0 / np.linalg.norm(v0)

    # 2. Integrate the orbit WITH the STM, sampling at n_points around it
    t_samples = np.linspace(0, period, n_points, endpoint=False)
    sol = integrate_with_stm(state0, (0, period), mu, t_eval=t_samples)

    strands = []
    for k in range(n_points):
        # State and STM at this base point
        aug = sol.y[:, k]
        state_k = aug[:4]
        Phi_k = aug[4:].reshape(4, 4)

        # Transport the eigenvector to this point: v_k = Phi_k @ v0
        v_k = Phi_k @ v0
        v_k = v_k / np.linalg.norm(v_k)

        # Perturb off the orbit along the transported direction
        perturbed = state_k + epsilon * v_k

        # Integrate the strand: forward for unstable, backward for stable
        if stable:
            t_span = (0, -integration_time)
            t_eval = np.linspace(0, -integration_time, n_strand_points)
        else:
            t_span = (0, integration_time)
            t_eval = np.linspace(0, integration_time, n_strand_points)

        strand_sol = solve_ivp(cr3bp_rhs, t_span, perturbed, args=(mu,),
                               method='DOP853', rtol=1e-11, atol=1e-11,
                               t_eval=t_eval)
        strands.append(strand_sol.y[:4, :].T)   # shape (n_strand_points, 4)

    return strands

# ============================================================
#  Halo families — same PAC engine, 3D constraint
# ============================================================

def halo_periodicity_constraint(X, mu):
    """Perpendicular x-z-plane crossing constraint for a symmetric halo.

    The 3D analog of periodicity_constraint. Free variables
    X = [x0, z0, vy0, T_half]; integrates from (x0, 0, z0, 0, vy0, 0) for
    exactly T_half (fixed time, no event).

    Returns
    -------
    F : ndarray, shape (3,)
        [y, vx, vz] at T_half — all zero at a perpendicular crossing.
    dF : ndarray, shape (3, 4)
        Jacobian of F w.r.t. [x0, z0, vy0, T_half].
    final_state : ndarray, shape (6,)
    """
    from cr3bp_3d import cr3bp_rhs_3d, integrate_with_stm_3d
    x0, z0, vy0, T_half = X
    state0 = [x0, 0.0, z0, 0.0, vy0, 0.0]

    sol = integrate_with_stm_3d(state0, (0, T_half), mu, t_eval=[T_half])
    aug_final = sol.y[:, -1]
    final_state = aug_final[:6]
    Phi = aug_final[6:].reshape(6, 6)          # STM at T_half

    # 3D state indices: x=0, y=1, z=2, vx=3, vy=4, vz=5
    F = np.array([final_state[1], final_state[3], final_state[5]])   # [y, vx, vz]

    # dF/d(free vars) through the STM. Free vars map to initial-state columns:
    #   x0 -> col 0,  z0 -> col 2,  vy0 -> col 4
    dF_dx0  = np.array([Phi[1, 0], Phi[3, 0], Phi[5, 0]])
    dF_dz0  = np.array([Phi[1, 2], Phi[3, 2], Phi[5, 2]])
    dF_dvy0 = np.array([Phi[1, 4], Phi[3, 4], Phi[5, 4]])

    # dF/dT_half = time derivatives of [y, vx, vz] at the final state (the EOM)
    deriv = cr3bp_rhs_3d(0, final_state, mu)   # [vx, vy, vz, ax, ay, az]
    dF_dT = np.array([deriv[1], deriv[3], deriv[5]])   # d[y,vx,vz]/dt

    dF = np.column_stack([dF_dx0, dF_dz0, dF_dvy0, dF_dT])   # 3x4
    return F, dF, final_state


def first_orbit(point, mu, Az=0.05, northern=True):
    """Richardson seed -> corrector -> one on-family halo [x0, z0, vy0, T_half].

    PROMOTED FROM THE NOTEBOOKS.  This lived in `halo_families.ipynb` and
    `floquet_analysis.ipynb` as two pasted copies which had already drifted --
    one wrote `T / 2.0`, the other `T / 2` -- and a third copy was about to be
    made in `notebooks/time_solve.py`.  Same fix `metrics.dead_zone` records for
    nb11 section 9: one implementation, three importers.

    `northern` is flipped at L2 so the seed has +z at BOTH points.  That is not
    a detail to re-derive at each call site; it is the reason a shared copy
    matters.

    cr3bp_3d is imported inside the function, matching `compute_halo_family`,
    so importing this module stays free of the 3D stack.
    """
    from cr3bp_3d import richardson_halo_seed, halo_corrector
    seed_flag = northern if point == 'L1' else (not northern)
    seed, _, _ = richardson_halo_seed(Az, mu, point=point, northern=seed_flag)
    s0, T, ok = halo_corrector(seed, mu)
    if not ok:
        raise RuntimeError(f"{point} halo corrector failed to converge "
                           f"from a Richardson seed at Az={Az}")
    return np.array([s0[0], s0[2], s0[4], T / 2.0])


def amplitude_direction(X, mu):
    """Pseudo-arclength tangent pointed toward growing |z0|, i.e. deeper into
    the halo family rather than back toward the planar bifurcation."""
    _, dF, _ = halo_periodicity_constraint(X, mu)
    t = family_tangent(dF)
    return t if np.sign(t[1]) == np.sign(X[1]) else -t


def halo_family(point, mu, Az=0.05, northern=True, ds=0.01, n_orbits=130):
    """Seed, correct, and continue a halo family in one call.

    Defaults are the ones the notebooks used, so a call with no keywords
    reproduces the families behind the Week 15 timing and the Floquet sweep.
    """
    X0 = first_orbit(point, mu, Az=Az, northern=northern)
    return compute_halo_family(X0, ds=ds, n_orbits=n_orbits, mu=mu,
                               initial_direction=amplitude_direction(X0, mu),
                               verbose=False)


def compute_halo_family(X_start, ds, n_orbits, mu,
                        initial_direction=None, verbose=True):
    """Halo family via pseudo-arclength continuation.

    Reuses the SAME engine as the Lyapunov families — family_tangent,
    predict_step, pseudo_arclength_correct — because the continuation
    machinery is dimension-blind. Only the constraint (halo_periodicity_
    constraint) and the state unpacking are problem-specific.

    Free variables X = [x0, z0, vy0, T_half].

    Returns
    -------
    family : list of dict, each with x0, z0, vy0, T_half, period, jacobi,
             state0 (6-vector), and X.
    """
    from cr3bp_3d import jacobi_constant_3d

    family = []
    X = np.asarray(X_start, dtype=float).copy()
    previous_tangent = initial_direction

    for i in range(n_orbits):
        F, dF, _ = halo_periodicity_constraint(X, mu)
        tangent = family_tangent(dF, previous_tangent)

        state0 = [X[0], 0.0, X[1], 0.0, X[2], 0.0]
        orbit = {
            'x0': X[0], 'z0': X[1], 'vy0': X[2], 'T_half': X[3],
            'period': 2 * X[3],
            'jacobi': jacobi_constant_3d(state0, mu),
            'state0': state0,
            'X': X.copy(),
        }
        family.append(orbit)

        if verbose and i % 10 == 0:
            print(f"  Orbit {i}: x0={X[0]:.5f}, z0={X[1]:.5f}, vy0={X[2]:.5f}, "
                  f"C={orbit['jacobi']:.5f}, T={orbit['period']:.5f}")

        try:
            X_pred = predict_step(X, tangent, ds)
            X_new = pseudo_arclength_correct(
                X_pred, X, tangent, ds, mu,
                constraint_fn=halo_periodicity_constraint)
        except RuntimeError as e:
            if verbose:
                print(f"  Stopped at orbit {i}: {e}")
            break

        X = X_new
        previous_tangent = tangent

    return family


# ============================================================
#  Floquet sweep: rotation number along a halo family
# ============================================================

def sweep_rotation_numbers(family, mu, anchor=None, overlap_min=0.9,
                           circle_tol=1e-4, verbose=False):
    """Track the center pair's rotation number across a halo family.

    Consumes a family object (list of orbit dicts from compute_halo_family) and
    calls the stateless halo_stability per orbit; the CROSS-ORBIT memory lives
    here. Strategy, exactly as derived:

      1. Anchor on the cleanest single-center orbit -- center on the circle,
         well clear of +/-1 (theta near 90 deg), and the hyperbolic pair still
         off the circle so the 3-way split is unambiguous. Not the large-
         amplitude end (that's bifurcation soup); the clean middle.
      2. Walk outward in BOTH directions. At each step, match the carried center
         eigenvector against every candidate eigenvector (both members of every
         pair -- so the match survives the pair crossing -1, where the Im>=0
         member flips to its conjugate). Pick maximum overlap |<v_prev,v_cand>|.
      3. Accumulate theta continuously as arg(mu_i / mu_prev) -- this unwraps
         through 180 deg automatically, no branch juggling.
      4. Stop a direction when the matched pair leaves the circle (center has
         gone hyperbolic -> torus ends) or the overlap collapses (lost track).

    Returns dict of arrays over the tracked orbits (sorted by family index):
      'index', 'C' (Jacobi), 'theta' (unwrapped, rad), 'rho' = theta/2pi,
      'anchor' (the chosen anchor index).
    """
    from cr3bp_3d import halo_stability

    R = [halo_stability(o['state0'], o['period'], mu, circle_tol=circle_tol)
         for o in family]

    if anchor is None:
        # Anchor in the FIRST clean block -- before the hyperbolic pair ever
        # joins the circle -- so tracking walks outward from clean territory
        # instead of starting inside the two-elliptic-pair region.
        mess = [i for i, r in enumerate(R) if r['hyp_on_circle']]
        hi = mess[0] if mess else len(R)
        clean = [i for i in range(hi)
                 if R[i]['center'] is not None and not R[i]['hyp_on_circle']]
        if not clean:   # fall back to any clean orbit anywhere
            clean = [i for i, r in enumerate(R)
                     if r['center'] is not None and not r['hyp_on_circle']]
        if not clean:
            raise RuntimeError("no clean single-center anchor orbit found")
        anchor = min(clean, key=lambda i: abs(R[i]['theta'] - np.pi/2))
    if verbose:
        print(f"  anchor = orbit {anchor}  (theta = {np.degrees(R[anchor]['theta']):.1f} deg)")

    def candidates(r, trivial_tol=1e-3):
        # Every eigenvector is a candidate EXCEPT the trivial pair, which sits
        # pinned at +1. Excluding it stops the tracker from ever latching onto
        # the zero-rotation direction. The center pair, even at theta ~ 3 deg,
        # sits |mu-1| ~ 0.05 >> trivial_tol, so it's never wrongly excluded --
        # only once theta < ~0.06 deg, i.e. the torus has degenerated anyway.
        c = []
        for p in r['pairs']:
            if abs(p['mu'] - 1.0) < trivial_tol:
                continue
            c.append((p['mu'], p['vec']))
            c.append((np.conj(p['mu']), np.conj(p['vec'])))
        return c

    theta = {anchor: float(np.angle(R[anchor]['center']['mu']))}

    def walk(seq, mu0, v0, th0):
        mu_prev, v_prev, th_prev = mu0, v0, th0
        for i in seq:
            mu_i, v_i = max(candidates(R[i]),
                            key=lambda c: abs(np.vdot(v_prev, c[1])))
            if abs(np.vdot(v_prev, v_i)) < overlap_min:
                if verbose: print(f"  stop at {i}: overlap collapsed")
                break
            if abs(abs(mu_i) - 1.0) > circle_tol:
                if verbose: print(f"  stop at {i}: center left the circle (|mu|={abs(mu_i):.3f})")
                break
            th_i = th_prev + float(np.angle(mu_i / mu_prev))
            theta[i] = th_i
            mu_prev, v_prev, th_prev = mu_i, v_i, th_i

    a = R[anchor]['center']
    walk(range(anchor - 1, -1, -1), a['mu'], a['vec'], theta[anchor])
    walk(range(anchor + 1, len(family)), a['mu'], a['vec'], theta[anchor])

    idx = np.array(sorted(theta))
    th  = np.array([theta[i] for i in idx])
    C   = np.array([family[i]['jacobi'] for i in idx])
    return dict(index=idx, C=C, theta=th, rho=th / (2*np.pi), anchor=anchor)


def _l1_lyapunov_derivation():
    """The original module-level exploratory script, verbatim.

    Derives a linearised initial guess for a small L1 Lyapunov orbit from
    the eigenstructure of the Jacobian at L1, then integrates it to the
    first x-axis crossing to show the residual the corrector has to remove.
    Kept because it documents where the seed in the notebooks came from.
    """
    mu = MU_EARTH_MOON
    lps = find_lagrange_points(mu)

    # The L1 point as a full state (on x-axis, at rest)
    L1_state = [lps['L1'][0], 0.0, 0.0, 0.0]

    # Jacobian at L1
    A_L1 = cr3bp_jacobian(L1_state, mu)
    print("Jacobian at L1:")
    print(A_L1)
    print()

    # Its eigenvalues
    eigvals = np.linalg.eigvals(A_L1)
    print("Eigenvalues at L1:")
    for ev in eigvals:
        print(f"  {ev:.6f}")

    # Get eigenvalues AND eigenvectors
    eigvals, eigvecs = np.linalg.eig(A_L1)

    # Find the eigenvalue with POSITIVE imaginary part (the +i*omega one)
    imag_idx = np.argmax(eigvals.imag)   # index of the most-positive-imaginary eigenvalue
    omega = eigvals[imag_idx].imag
    print(f"Oscillation frequency omega = {omega:.6f}")
    print(f"Linearized orbit period T = {2*np.pi/omega:.6f}")
    print()

    # The corresponding (complex) eigenvector
    v = eigvecs[:, imag_idx]
    print("Complex eigenvector for +i*omega:")
    print(f"  x-component:  {v[0]:.6f}")
    print(f"  y-component:  {v[1]:.6f}")
    print(f"  vx-component: {v[2]:.6f}")
    print(f"  vy-component: {v[3]:.6f}")

    # Extract the real parts that define the symmetric initial condition
    x_comp = v[0].real      # -0.105758
    vy_comp = v[3].real     #  0.885435

    # The ratio of vy0 to x-offset for the linearized Lyapunov orbit
    ratio = vy_comp / x_comp
    print(f"vy0 / delta_x ratio = {ratio:.6f}")

    # Choose a small orbit: offset x from L1 by a small amount.
    # Sign convention: let's offset to the LEFT of L1 (toward Earth), delta_x < 0,
    # which makes vy0 positive. Small magnitude = small orbit = closer to linear.
    delta_x = -0.001
    x0 = lps['L1'][0] + delta_x
    vy0 = ratio * delta_x

    print(f"\nLinearized initial guess for a small L1 Lyapunov orbit:")
    print(f"  x0  = {x0:.8f}   (L1 offset by {delta_x})")
    print(f"  y0  = 0.0")
    print(f"  vx0 = 0.0")
    print(f"  vy0 = {vy0:.8f}")

    # Sanity check: integrate this guess to the x-axis crossing and see the residual.
    # If the linearization is good, the residual vx should be SMALL (unlike the 0.63 before).
    state0 = [x0, 0.0, 0.0, vy0]
    sol = solve_ivp(cr3bp_rhs, (0, 10), state0, args=(mu,),
                    method='DOP853', rtol=1e-12, atol=1e-12,
                    events=y_axis_crossing)

    if len(sol.t_events[0]) > 0:
        state_cross = sol.y_events[0][0]
        t_cross = sol.t_events[0][0]
        print(f"\nHalf-period crossing at t = {t_cross:.6f} (linear prediction: {np.pi/omega:.6f})")
        print(f"  x at crossing  = {state_cross[0]:.8f}")
        print(f"  vx at crossing = {state_cross[2]:.8e}   <- residual to correct")
    else:
        print("\nNo crossing found — check the guess.")


if __name__ == '__main__':
    _l1_lyapunov_derivation()
