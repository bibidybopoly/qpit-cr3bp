"""Distance from a state to the torus family surface -- the Phase 4 label.

THE OBJECT.  At fixed Jacobi constant the family sweeps a continuous set
    S = { u_alpha(theta) : alpha in [0,1] per gap, theta in [0,2pi) }
and the label is d(x) = min over S of ||x - s||.  Distance to the SURFACE,
not to the nearest member of a finite list: a point midway between members
spaced g apart is ~g/2 from both while its true nearest torus is at
essentially zero distance, so enumeration overestimates by g/2 and would
need thousands of members.  Interpolating instead puts the error at
c*g^2 (densify.interp_error_km), which is 10 cm at g = 12.3 km.

WHY NOT NEWTON PER SAMPLE.  probe_curve is faithful only inside its linear
regime -- 34.2 m on L2 halo #29, measured -- and costs ~2.4 s per sample.
This is geometric: no basin to fall out of, no censoring ceiling, unlimited
dynamic range, and thousands of points per second.

THE ALPHA MINIMISATION IS CLOSED FORM.  The interpolant is linear in alpha,
so with r(theta) = x - u_a(theta) and d(theta) = u_b(theta) - u_a(theta),

    alpha*(theta) = (r.d)/||d||^2   clipped to [0,1]
    min_alpha ||x - u_alpha(theta)||^2 = ||r||^2 - (r.d)^2/||d||^2

leaving only a 1-D minimisation over theta.  Clipping is what makes taking
the min over gaps correct: alpha* outside [0,1] means the true minimum sits
in a neighbouring gap, and that gap reports it.

SIX DIMENSIONS.  The network is handed a full state, so a position-only
label would assign identical targets to two states at the same point with
different velocities -- one of which may be on a torus and the other not.
Nondimensional CR3BP units already fix the scaling.
"""

from __future__ import annotations

import numpy as np

from qpit import evaluate_curve

LU = 384400.0


def _gap_distance(X, U_a, U_b, thetas):
    """min over alpha in [0,1] of ||X - u_alpha(theta)||, for each theta.

    Returns (D, alpha) both shaped (P, n_theta).
    """
    A = evaluate_curve(U_a, thetas)                   # (n_theta, 6)
    B = evaluate_curve(U_b, thetas)
    D_vec = B - A                                     # (n_theta, 6)
    dd = np.einsum('tj,tj->t', D_vec, D_vec)          # (n_theta,)
    dd = np.maximum(dd, 1e-300)

    R = X[:, None, :] - A[None, :, :]                 # (P, n_theta, 6)
    rr = np.einsum('ptj,ptj->pt', R, R)
    rd = np.einsum('ptj,tj->pt', R, D_vec)

    alpha = np.clip(rd / dd[None, :], 0.0, 1.0)
    # ||r - alpha*d||^2, valid for clipped alpha too
    D2 = rr - 2.0 * alpha * rd + alpha ** 2 * dd[None, :]
    return np.sqrt(np.maximum(D2, 0.0)), alpha


def surface_distance(X, members, n_grid=512, n_refine=48, chunk=512,
                     amp_margin_km=None):
    """Distance from each state in X to the family surface.

    members must be sorted by amplitude (densify.densify guarantees this).

    amp_margin_km prunes gaps whose amplitude range is far from the query
    point's own scale.  Pruning is a speed device only and is OFF by
    default, because a wrong margin silently returns a too-large distance
    rather than an error -- the failure mode is a plausible wrong label.

    Returns dict(d, alpha, theta, gap, member_lo).
    """
    X = np.atleast_2d(np.asarray(X, float))
    P = X.shape[0]
    n_gap = len(members) - 1
    if n_gap < 1:
        raise ValueError("need at least two members to form a surface")

    th = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    best_d = np.full(P, np.inf)
    best = dict(alpha=np.zeros(P), theta=np.zeros(P),
                gap=np.zeros(P, dtype=int))

    for i in range(n_gap):
        U_a, U_b = members[i]['U0'], members[i + 1]['U0']
        for p0 in range(0, P, chunk):
            Xc = X[p0:p0 + chunk]
            D, al = _gap_distance(Xc, U_a, U_b, th)
            k = np.argmin(D, axis=1)
            rows = np.arange(Xc.shape[0])
            dmin = D[rows, k]
            upd = dmin < best_d[p0:p0 + chunk]
            idx = np.nonzero(upd)[0] + p0
            if idx.size:
                best_d[idx] = dmin[upd]
                best['alpha'][idx] = al[rows, k][upd]
                best['theta'][idx] = th[k][upd]
                best['gap'][idx] = i

    # --- refine theta by golden section in the winning gap AND ITS NEIGHBOURS
    #
    # Refining only the winner is wrong.  A point sitting near a member sits
    # at a gap BOUNDARY, where one side reports it at a clipped alpha (0 or
    # 1) and the other reports it interior.  At coarse resolution the clipped
    # side can win, and refinement inside that gap can never recover because
    # the clipped endpoint is genuinely the nearest point OF THAT GAP.
    # Observed: an exactly-on-surface point labelled 84 m instead of 0.
    # Adjacent gaps are the only possible confusion, so +/-1 suffices.
    dth = 2 * np.pi / n_grid
    invphi = (np.sqrt(5.0) - 1.0) / 2.0

    def _refine(sel, i):
        U_a, U_b = members[i]['U0'], members[i + 1]['U0']
        Xs = X[sel]
        def f(t):
            D, al = _gap_distance(Xs, U_a, U_b, t)
            return np.diagonal(D).copy(), np.diagonal(al).copy()
        th0 = best['theta'][sel]
        a, b = th0 - dth, th0 + dth
        c, e = b - invphi * (b - a), a + invphi * (b - a)
        fc, _ = f(c); fe, _ = f(e)
        for _ in range(n_refine):
            m = fc < fe
            b, a = np.where(m, e, b), np.where(m, a, c)
            c, e = b - invphi * (b - a), a + invphi * (b - a)
            fc, _ = f(c); fe, _ = f(e)
        t_star = np.where(fc < fe, c, e)
        d_star, a_star = f(t_star)
        return d_star, a_star, t_star

    home = best['gap'].copy()
    for off in (0, -1, 1):
        for i in np.unique(np.clip(home + off, 0, n_gap - 1)):
            sel = np.nonzero(np.clip(home + off, 0, n_gap - 1) == i)[0]
            if sel.size == 0:
                continue
            d_s, a_s, t_s = _refine(sel, int(i))
            upd = d_s < best_d[sel]
            if upd.any():
                idx = sel[upd]
                best_d[idx] = d_s[upd]
                best['alpha'][idx] = a_s[upd]
                best['theta'][idx] = np.mod(t_s[upd], 2 * np.pi)
                best['gap'][idx] = i

    return dict(d=best_d, alpha=best['alpha'], theta=best['theta'],
                gap=best['gap'])


def label_states(X, members, floor_km=1e-4, **kw):
    """Regression label: log10 distance in km, floored at the surface error.

    The floor is the interpolation bound, not machine epsilon.  Below it the
    label is not resolved by the surface and reporting a smaller number
    would be reporting the family's discretisation as physics.
    """
    out = surface_distance(X, members, **kw)
    d_km = out['d'] * LU
    out['d_km'] = d_km
    out['log_d_km'] = np.log10(np.maximum(d_km, floor_km))
    out['at_floor'] = d_km < floor_km
    return out
