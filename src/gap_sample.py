"""Sampling for the Week 10 gap experiment.

Three arms, one energy, one indicator:

  TREATMENT         points between adjacent converged family members
  POSITIVE CONTROL  points on the converged members themselves
  NEGATIVE CONTROL  points displaced along the torus's own unstable bundle

The treatment result is meaningless except bracketed by the two controls:
a test that always answers 'quasi-periodic' would reproduce the predicted
treatment result exactly, and only the negative control can rule that out.

UNIT OF REPLICATION IS ALPHA, NOT THETA.  All theta on one interpolated
curve lie on (or off) the same underlying object, so they are not
independent trials.  Sampling n_theta points from one alpha and calling it
n_theta observations is pseudo-replication and would understate the
confidence interval by sqrt(n_theta).  Confidence intervals are computed
over alpha; the within-alpha agreement rate is reported separately as a
diagnostic (a genuinely invariant curve should return a unanimous label).
"""

from __future__ import annotations

import numpy as np

from qpit import (evaluate_curve, jacobi_constant, mean_jacobi,
                  node_angles, torus_bundle)


# =========================================================================
# SECTION 1 -- ENERGY REPROJECTION
# =========================================================================

def reproject_energy(x, mu, C_target):
    """Move a state onto the C_target level set by rescaling |v|.

    C = 2*Omega(pos) - |v|^2 and Omega depends on POSITION ONLY, so holding
    position fixed gives 2*Omega = C(x) + |v|^2 and the corrected speed is

        |v_new| = sqrt( C(x) + |v|^2 - C_target ).

    Exact to roundoff, closed form, direction preserving, and it reads
    Omega back out of jacobi_constant rather than recomputing it -- so it
    inherits that function's mu placement and Omega convention instead of
    duplicating them.  Same convention-proofing as the jacobi_gradient
    self-test.

    Returns (x_new, rel_change) or (None, nan) in the forbidden region.
    The linear interpolant misses the level set by O(delta^2), so
    rel_change should be SMALL; if it is not, the gap is too wide for
    linear interpolation and alpha resolution must increase.  Recorded per
    sample so this is checkable rather than assumed.
    """
    x = np.asarray(x, float).copy()
    v = x[3:]
    v2 = float(v @ v)
    if v2 < 1e-300:
        return None, np.nan
    v2_new = jacobi_constant(x, mu) + v2 - float(C_target)
    if v2_new <= 0.0:
        return None, np.nan                       # forbidden region
    s = np.sqrt(v2_new / v2)
    x[3:] = v * s
    return x, float(abs(s - 1.0))


# =========================================================================
# SECTION 2 -- PHASE ALIGNMENT
# =========================================================================

def phase_offset(U_a, U_b, n_scan=721):
    """Shift sigma minimising ||u_a(theta) - u_b(theta + sigma)||.

    continue_torus_family pins each member's phase against U_ref =
    prev['U0'], so ADJACENT members should already agree and sigma ~ 0.
    A nonzero sigma between neighbours indicates a bug, not a convention.
    Because the pinning is relative it can drift cumulatively across many
    members, so this is measured rather than assumed.

    Returns (sigma, mismatch_at_sigma, mismatch_at_zero).
    """
    th = node_angles(U_a.shape[0])
    A = evaluate_curve(U_a, th)
    sig = np.linspace(-np.pi, np.pi, n_scan)
    d = np.array([np.linalg.norm(A - evaluate_curve(U_b, th + s)) for s in sig])
    k = int(np.argmin(d))
    return float(sig[k]), float(d[k]), float(d[n_scan // 2])


# =========================================================================
# SECTION 3 -- THE THREE ARMS
# =========================================================================

def sample_gap(rec_a, rec_b, mu, n_alpha, n_theta, rng, sigma=0.0,
               alpha_lo=0.15, alpha_hi=0.85):
    """TREATMENT.  Points between two adjacent converged members.

    alpha is drawn strictly inside (0, 1) -- endpoints are the members
    themselves and belong to the positive control, and points very near
    them would inflate the treatment pass rate for a trivial reason.

    C_target is the MEAN of the two members' converged C.  Interpolating
    two states of a level set leaves it by O(delta^2); reprojection puts
    the sample back exactly.  Sampling a gap at an energy that drifts with
    alpha is the Week 9 contamination in new clothes.
    """
    U_a, U_b = rec_a['U0'], rec_b['U0']
    C_target = 0.5 * (rec_a['C'] + rec_b['C'])
    T_ref = 0.5 * (rec_a['T'] + rec_b['T'])
    rows = []
    for i in range(n_alpha):
        a = rng.uniform(alpha_lo, alpha_hi)
        th = rng.uniform(0.0, 2 * np.pi, n_theta)
        Xa = evaluate_curve(U_a, th)
        Xb = evaluate_curve(U_b, th + sigma)
        for j in range(n_theta):
            x, rel = reproject_energy((1 - a) * Xa[j] + a * Xb[j], mu, C_target)
            if x is None:
                continue
            rows.append(dict(x0=x, arm='gap', alpha_id=i, alpha=a,
                             theta=float(th[j]), T=T_ref, C_target=C_target,
                             reproj_rel=rel))
    return rows


def sample_on_torus(rec, mu, n_alpha, n_theta, rng):
    """POSITIVE CONTROL.  Points on a converged member.

    alpha_id indexes independent draws so the same clustered statistics
    apply as in the treatment arm; theta values within one alpha_id share
    the curve exactly as gap points share an interpolant.
    """
    U, C = rec['U0'], rec['C']
    rows = []
    for i in range(n_alpha):
        th = rng.uniform(0.0, 2 * np.pi, n_theta)
        X = evaluate_curve(U, th)
        for j in range(n_theta):
            rows.append(dict(x0=X[j], arm='pos', alpha_id=i, alpha=np.nan,
                             theta=float(th[j]), T=rec['T'], C_target=C,
                             reproj_rel=0.0))
    return rows


def sample_off_manifold(rec, mu, n_alpha, n_theta, rng, d=1e-6, DG=None):
    """NEGATIVE CONTROL.  Displaced along the torus's own unstable bundle.

    torus_bundle returns V (N, 6) with unit rows and the measured growth
    factor R per stroboscopic period.  A displacement d grows to O(1) in
    about log(1/d)/log(R) periods -- roughly 2.8 at d=1e-6 and R=146 --
    so any window long enough for NAFF is orders of magnitude longer than
    the escape time.  This control should fail at essentially 100%, and
    the escape time is PREDICTED rather than observed after the fact.

    The bundle is defined at the nodes, so displacement is applied at node
    angles and theta is drawn from the nodes rather than continuously.
    Reprojected to the member's own C so energy carries no signal between
    the arms.
    """
    U, T, rho, C = rec['U0'], rec['T'], rec['rho'], rec['C']
    V, R = torus_bundle(U, T, rho, mu, which='unstable', DG=DG)
    N = U.shape[0]
    rows = []
    for i in range(n_alpha):
        for _ in range(n_theta):
            j = int(rng.integers(N))
            side = 1.0 if rng.random() < 0.5 else -1.0
            x, rel = reproject_energy(U[j] + side * d * V[j], mu, C)
            if x is None:
                continue
            rows.append(dict(x0=x, arm='neg', alpha_id=i, alpha=np.nan,
                             theta=float(node_angles(N)[j]), T=T,
                             C_target=C, reproj_rel=rel, bundle_R=float(R)))
    return rows


# =========================================================================
# SECTION 4 -- RESONANCE PROXIMITY
# =========================================================================

def resonance_distance(rho, q_max=5):
    """min |rho/2pi - p/q| over q <= q_max.  Recorded on every sample.

    Week 12 needs this and it costs nothing now; regenerating a dataset to
    add a column that was computable at generation time is avoidable work.
    """
    r = float(rho) / (2 * np.pi)
    r = r - np.floor(r)
    best, arg = np.inf, None
    for q in range(1, q_max + 1):
        for p in range(0, q + 1):
            d = abs(r - p / q)
            if d < best:
                best, arg = d, (p, q)
    return float(best), arg
