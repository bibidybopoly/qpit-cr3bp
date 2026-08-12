"""Finite-time torus probe: does an invariant curve pass through here?

WHY NOT A TRAJECTORY DIAGNOSTIC.  These tori are strongly partially
hyperbolic -- R_u = 146.32 on L2 halo #29 -- so integrator error reaches
O(1) after log(1e12)/log(146.32) = 5.5 stroboscopic periods, ~4 of them
clean.  Every forward-time indicator (NAFF drift, Das-Yorke weighted
Birkhoff, FLI, MEGNO) needs one to two orders of magnitude more than that:
weighted Birkhoff measured >95% Lyapunov-verified classification only from
~32 iterates, and is UNDEFINED at the 4 available.  These tori are
computable but not observable.

converge_torus never integrates longer than ONE stroboscopic period, so the
shadowing bound does not bind on it.  That makes the invariance equation
itself the only available probe.

WHAT THE MINIMUM-NORM STEP BUYS.  With C pinned, two phase constraints, and
rho and T free, the assembled system carries a deliberate 1-D nullspace --
the family direction (qpit.py Section 6B).  Least squares takes
minimum-norm steps and therefore does NOT slide along the family.  For
continuation that is an obstacle, closed with an arclength row.  Here it is
the measurement: Newton moves only as far as invariance demands, so the
distance it moves is a genuine distance to the nearest invariant torus
rather than an artifact of solver drift.

U_ref is set to the CANDIDATE, not to a family member, so the phase
constraints pin the curve against the candidate's own phase and it cannot
slide in theta either.  Passing a family member instead would let the
solver rotate the curve and inflate the measured distance for a reason that
has nothing to do with whether a torus is there.

The score is CONTINUOUS (a distance), not binary.  A binary label is a
thresholded readout of it, and the threshold is reportable as a curve --
accuracy versus tolerance -- rather than a number that has to be defended.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

import qpit
from qpit import (converge_torus, curve_amplitude, evaluate_curve, torus_bundle,
                  invariance_residual, mean_jacobi, node_angles)


# =========================================================================
# SECTION 1 -- CURVE DISTANCE
# =========================================================================

def curve_distance(U_a, U_b, normalise=True):
    """Node-wise distance between two curves, POSITION components only.

    Positions only for the same reason curve_amplitude uses them: the six
    components mix position and velocity units, so a 6-D norm weights the
    answer by an arbitrary choice of length and time scale.  Normalised by
    the candidate's amplitude so the score is comparable across a family
    spanning 17 to 890 km.

    Returns (max_node_distance, rms_node_distance), normalised or raw.
    """
    d = np.linalg.norm(U_a[:, :3] - U_b[:, :3], axis=1)
    s = curve_amplitude(U_a) if normalise else 1.0
    s = max(s, 1e-300)
    return float(d.max() / s), float(np.sqrt((d ** 2).mean()) / s)


def curve_local_distance(X, U_ref, n_grid=1024, n_refine=64, chunk=4096):
    """Minimum distance from each state to a curve treated as a SET.

    d(x) = min over theta' of ||x - u_ref(theta')||, i.e. the foot of the
    perpendicular, NOT the parameter-matched distance curve_distance returns.

    WHY THIS AND NOT NODE-WISE.  Newton adjusts rho and T, so the converged
    curve's parameterisation can slide relative to the candidate's even when
    the phase constraints pin the candidate.  Node-wise distance then reports
    pure reparameterisation as motion.  It is always an UPPER BOUND on this
    quantity -- the minimum over theta' can only beat the theta' = theta
    choice -- so results already computed with it remain valid, just loose.

    SIX DIMENSIONS, NOT THREE.  curve_distance uses positions because a 6-D
    norm mixes units, which is right for a diagnostic and wrong for a
    regression target: the network is handed a full state, so a position-only
    label assigns identical targets to two states at the same point with
    different velocities, one of which may be on a torus and the other not.
    Nondimensional CR3BP units already fix the scaling (LU = 384400 km, TU
    such that mean motion is 1), and converge_torus's own convergence test is
    a mixed 6-D infinity norm, so the codebase already treats these as
    commensurate.  d3 and dv are returned alongside for sensitivity checks.

    REFINEMENT IS NOT OPTIONAL.  For x ON the curve, d(theta) is V-shaped --
    linear in theta near the minimum, not quadratic -- so a grid of 1024 caps
    resolution at |u'| * 2*pi/1024 ~ 1.2 km on a 200 km curve, against a
    target range starting near 1e-3 km.  The grid brackets; golden section
    finds the foot.  Golden section rather than Newton because it needs only
    unimodality on the bracket and handles the V-shaped and smooth cases
    identically.

    Returns dict with d6, d3, dv, theta_foot, theta_grid_gap.
    """
    X = np.atleast_2d(np.asarray(X, float))
    th6, d6 = _min_over_theta(X, U_ref, slice(0, 6), n_grid, n_refine, chunk)
    th3, d3 = _min_over_theta(X, U_ref, slice(0, 3), n_grid, n_refine, chunk)
    dv = np.linalg.norm((X - evaluate_curve(U_ref, th6))[:, 3:], axis=1)
    return dict(d6=d6, d3=d3, dv=dv,
                theta_foot=np.mod(th6, 2 * np.pi),
                theta_foot_pos=np.mod(th3, 2 * np.pi),
                theta_grid_gap=float(2 * np.pi / n_grid))


def _min_over_theta(X, U_ref, sl, n_grid, n_refine, chunk):
    """min over theta of ||X[:, sl] - u_ref(theta)[sl]||, and the minimiser.

    d3 must be its OWN minimisation, not the position part of the 6-D foot.
    Those differ -- a velocity mismatch pulls the 6-D minimiser to a different
    angle -- and only the independent minimum is a position-only distance.
    Reporting the component at the 6-D foot under the name d3 would make a
    Week 11 sensitivity check between metrics compare the wrong things.
    """
    P = X.shape[0]
    Xs = X[:, sl]

    th_g = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    G = evaluate_curve(U_ref, th_g)[:, sl]
    g2 = (G ** 2).sum(1)
    k = np.empty(P, dtype=int)
    for i0 in range(0, P, chunk):                    # avoids a (P, n_grid, m)
        Xc = Xs[i0:i0 + chunk]
        D2 = (Xc ** 2).sum(1)[:, None] - 2.0 * (Xc @ G.T) + g2[None, :]
        k[i0:i0 + chunk] = np.argmin(D2, axis=1)
    dth = 2 * np.pi / n_grid

    def dist(th):
        return np.linalg.norm(Xs - evaluate_curve(U_ref, th)[:, sl], axis=1)

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = th_g[k] - dth, th_g[k] + dth
    c, e = b - invphi * (b - a), a + invphi * (b - a)
    fc, fe = dist(c), dist(e)
    for _ in range(n_refine):
        m = fc < fe
        b, a = np.where(m, e, b), np.where(m, a, c)
        c, e = b - invphi * (b - a), a + invphi * (b - a)
        fc, fe = dist(c), dist(e)
    th_star = np.where(fc < fe, c, e)
    return th_star, dist(th_star)


def regression_target(d, floor=1e-12, censor=np.inf):
    """log10 distance, floored at solver resolution and censored above.

    LOG, NOT LINEAR.  Distances span roughly 1e-10 to 1e1; a linear target
    would be determined almost entirely by the tail and would carry no
    gradient in the regime that matters.

    FLOOR.  Below solver resolution the value is noise, not measurement, so
    it is clipped rather than reported as an ever-smaller number.

    CENSOR.  Above the probe's linear regime, dist records how far Newton
    travelled rather than how far the nearest torus is.  Those are lower
    bounds; the flag keeps them out of the fit while preserving them in the
    dataset.  Measure the censor level -- do not assume it (see the bundle
    control in nb10_gap_experiment).
    """
    d = np.asarray(d, float)
    return np.log10(np.maximum(d, floor)), (d > censor)


def sample_probe(p, thetas, n_grid=1024, floor=1e-12, censor=np.inf,
                 calibration=1.0, **kw):
    """Per-point regression targets from a completed probe.

    The candidate curve evaluated at `thetas` gives the states; each state's
    distance to the CONVERGED curve gives its own target.  This is the point
    of the exercise: one Newton solve yields len(thetas) distinct labels
    instead of len(thetas) copies of the curve's dist_max.

    theta_foot minus theta lets a caller see the reparameterisation directly
    -- a systematic offset is exactly what makes node-wise distance overstate.

    CALIBRATION.  The distance is measured to the torus NEWTON FOUND, which is
    not quite the nearest one: min-norm Newton slides a little along the
    family's 1-D nullspace, so it lands past the true nearest torus and the
    measured distance overstates.  On L2 halo #29 the overshoot factor is
    1.1433, constant to five figures across four decades of displacement and a
    16x amplitude range, against a true nearest-torus distance that recovers an
    imposed displacement to 0.07%.  Pass `calibration` (the `plateau` field
    from calibrate_censor) to divide it out.  Default 1.0 leaves the raw
    measurement, so the correction is always visible rather than baked in.

    The factor is measured for UNSTABLE-BUNDLE displacements.  Whether it
    transfers to centre-manifold displacements (gap interpolants) is assumed,
    not verified -- there is no known displacement to calibrate against there.
    """
    if not p.get('converged') or 'U_cand' not in p or p.get('result') is None:
        return None
    thetas = np.atleast_1d(np.asarray(thetas, float))
    X = evaluate_curve(p['U_cand'], thetas)
    out = curve_local_distance(X, p['result']['U0'], n_grid=n_grid, **kw)
    out['states'] = X
    out['theta'] = thetas
    out['dtheta'] = np.mod(out['theta_foot'] - thetas + np.pi, 2*np.pi) - np.pi
    out['calibration'] = float(calibration)
    out['d6_cal'] = out['d6'] / calibration
    out['log_d6'], out['censored'] = regression_target(out['d6_cal'], floor, censor)
    return out


# =========================================================================
# SECTION 2 -- THE PROBE
# =========================================================================

def probe_curve(U_cand, T_cand, rho_cand, mu, C_target=None,
                tol=1e-12, itmax=15, offgrid=True):
    """Ask whether an invariant curve passes through a candidate curve.

    Returns a dict of CONTINUOUS scores plus the raw converged object.
    Never returns a label -- see label_from_probe.

    C_target defaults to the candidate's own mean Jacobi.  Pinning it is
    what keeps the probe from answering a question about a different energy;
    C_seed is returned alongside so a sweep can see drift, the same
    countermeasure added after the Week 9 contamination bug.
    """
    U_cand = np.asarray(U_cand, float)
    if C_target is None:
        C_target = mean_jacobi(U_cand, mu)

    with contextlib.redirect_stdout(io.StringIO()):
        res = converge_torus(U_cand, T_cand, rho_cand, mu,
                             U_ref=U_cand,          # phase pinned to candidate
                             C_target=C_target,
                             free_rho=True, tol=tol, itmax=itmax)

    out = dict(U_cand=U_cand,                  # needed by sample_probe
               converged=bool(res['converged']), code=res['code'],
               n_iter=res['n_iter'],
               C_target=float(C_target), C_seed=float(res['C_seed']),
               C_final=float(res['C']),
               amp_cand=float(curve_amplitude(U_cand)),
               amp_final=float(res['amplitude']),
               T_cand=float(T_cand), T_final=float(res['T']),
               rho_cand=float(rho_cand), rho_final=float(res['rho']),
               dist_max=np.nan, dist_rms=np.nan, damp_rel=np.nan,
               res_inf=np.nan, offgrid=np.nan, result=res)

    dmax, drms = curve_distance(U_cand, res['U0'])
    out['dist_max'], out['dist_rms'] = dmax, drms
    out['damp_rel'] = float(abs(res['amplitude'] - out['amp_cand'])
                            / max(out['amp_cand'], 1e-300))

    if res['converged']:
        out['res_inf'] = float(np.abs(
            invariance_residual(res['U0'], res['T'], res['rho'], mu)).max())
        if offgrid:
            out['offgrid'] = float(qpit.offgrid_residual(
                res['U0'], res['T'], res['rho'], mu))
    return out


def label_from_probe(p, thr):
    """Thresholded readout of the continuous distance score.

    Conjunctive.  damp_rel guards the case where Newton converged to a
    genuine torus of a very different amplitude -- that is a torus, but not
    one passing through the candidate, and counting it positive would make
    the probe answer 'does a torus exist at this energy', which is trivially
    yes everywhere the family exists.
    """
    if not p['converged']:
        return False
    return bool(p['dist_max'] <= thr['dist_max']
                and p['damp_rel'] <= thr['damp_rel']
                and p['res_inf'] <= thr['res_inf'])


# =========================================================================
# SECTION 3 -- CALIBRATION
# =========================================================================

def calibrate_probe(members, mu, quantile=95.0, margin=10.0,
                    perturb=1e-6, seed=0, floor=1e-9, **kw):
    """POSITIVE CONTROL.  Probe curves known to be tori, slightly displaced.

    PERTURB IS NOT OPTIONAL.  Re-probing a member from its own converged U0
    is degenerate: Newton exits after one iteration with dist_max identically
    0.0, and a floor of exactly zero makes every threshold zero, so nothing
    but the members themselves can ever be labelled positive.  Measured on
    L2 halo #29 -- the first run of this function returned 0.000e+00 for both
    control members and produced an unusable threshold.

    Displacing each member by a small relative amount first makes the control
    an honest analogue of the treatment, where the candidate is a nearby
    non-solution (an interpolant) rather than an exact one.  The floor then
    measures what it should: how close to an invariant curve the probe can
    resolve, given solver tolerance and the pinned energy.

    'floor' is a hard lower bound on the distance threshold, so a control
    that still comes back degenerate cannot silently yield a threshold no
    candidate can meet.
    """
    rng = np.random.default_rng(seed)
    ps = []
    for m in members:
        U = np.asarray(m['U0'], float).copy()
        if perturb:
            U[:, :3] += (perturb * curve_amplitude(U)
                         * rng.standard_normal((U.shape[0], 3)))
        p = probe_curve(U, m['T'], m['rho'], mu, C_target=m['C'], **kw)
        p['perturb'] = float(perturb)
        ps.append(p)

    ok = [p for p in ps if p['converged']]
    if not ok:
        raise ValueError("positive control did not converge; probe unusable")

    def q(key):
        return float(np.percentile([p[key] for p in ok], quantile) * margin)

    return dict(dist_max=max(q('dist_max'), floor),
                damp_rel=max(q('damp_rel'), floor),
                res_inf=max(q('res_inf'), 1e-11),
                n_control=len(ok), n_failed=len(ps) - len(ok),
                quantile=quantile, margin=margin, perturb=perturb, probes=ps)


# =========================================================================
# SECTION 4 -- GAP SWEEP
# =========================================================================

def probe_gap(rec_a, rec_b, mu, alphas, sigma=0.0, reproject=True, **kw):
    """TREATMENT.  Probe interpolants between two adjacent members.

    Linear interpolation of two converged curves is the only construction
    that yields a FULL candidate curve at gap amplitude: a gap point has no
    fixed point to linearise about, so seed_from_halo's route is unavailable
    there, and a stroboscopic-orbit fit would need >= N iterates against the
    ~4 the shadowing bound allows.

    T and rho are interpolated too, and both are free in the solve, so a
    poor guess costs iterations rather than correctness.

    Adjacent members of a pin='C' family already share an energy, so the
    reprojection correction should be near zero; it is applied anyway and
    the magnitude recorded, because a large correction means the pair is not
    actually adjacent.
    """
    from gap_sample import reproject_energy

    U_a, U_b = rec_a['U0'], rec_b['U0']
    th = node_angles(U_a.shape[0])
    A = evaluate_curve(U_a, th)
    B = evaluate_curve(U_b, th + sigma)
    C_target = 0.5 * (rec_a['C'] + rec_b['C'])

    rows = []
    for a in np.atleast_1d(alphas):
        U = (1.0 - a) * A + a * B
        rel = 0.0
        if reproject:
            V = np.empty_like(U)
            bad = False
            for j in range(U.shape[0]):
                xj, rj = reproject_energy(U[j], mu, C_target)
                if xj is None:
                    bad = True
                    break
                V[j] = xj
                rel = max(rel, rj)
            if bad:
                rows.append(dict(alpha=float(a), converged=False,
                                 code='forbidden_region', reproj_rel=np.nan))
                continue
            U = V
        p = probe_curve(U, (1 - a) * rec_a['T'] + a * rec_b['T'],
                        (1 - a) * rec_a['rho'] + a * rec_b['rho'],
                        mu, C_target=C_target, **kw)
        p['alpha'] = float(a)
        p['reproj_rel'] = float(rel)
        rows.append(p)
    return rows

def calibrate_censor(member, mu, displacements=None, tol=0.05, DG=None, **kw):
    """MEASURE where the probe stops measuring distance.

    Displace a converged member along its own unstable bundle by a known d.
    torus_bundle returns unit rows, so every node moves by exactly d in 6-D
    and the original member -- still a torus -- sits at distance d.  A probe
    that is measuring distance returns d.  The RATIO d_returned/d is therefore
    an absolute calibration, not merely a linearity check: it should be 1.

    Beyond some displacement Newton leaves the basin of the nearby torus and
    converges to a different one, at which point dist records how far Newton
    travelled, not how far the nearest torus is.  Those samples are lower
    bounds and must be censored out of a regression fit.

    Note a position-only metric cannot pass this test even in the faithful
    regime: the bundle vector is a unit 6-vector whose position part has norm
    below 1, so d3/d saturates at ||V_pos|| (~0.417 on L2 halo #29) rather
    than 1.  That is why the regression target is d6.

    Returns dict with the sweep and `censor`, the largest displacement whose
    ratio stays within `tol` of the small-d plateau, halved for safety.

    MEASURED ON L2 HALO #29 (members at 31.5, 202.5, 514.9 km):
        plateau  = 1.1433   (Newton overshoot; see sample_probe)
        break    between d = 1.78e-7 and 3.16e-7, ratio jumping 1.14 -> 160
        censor   = 8.891e-08 nondim = 34.2 m, IDENTICAL across all three
    The censor level is amplitude-independent, which is what makes a single
    value defensible for the whole family.  It is also four orders of
    magnitude below the 1e-3 placeholder it replaces.
    """
    if displacements is None:
        displacements = np.logspace(-9, -5, 17)
    U, T, rho, C = member['U0'], member['T'], member['rho'], member['C']
    V, R_u = torus_bundle(U, T, rho, mu, which='unstable', DG=DG)

    rows = []
    for dd in np.atleast_1d(displacements):
        p = probe_curve(U + dd * V, T, rho, mu, C_target=C, offgrid=False, **kw)
        r = dict(displacement=float(dd), converged=p['converged'],
                 dist_max=p['dist_max'], d6=np.nan, d3=np.nan)
        if p['converged']:
            loc = curve_local_distance(p['U_cand'], p['result']['U0'])
            r['d6'], r['d3'] = float(loc['d6'].max()), float(loc['d3'].max())
        rows.append(r)

    dd = np.array([r['displacement'] for r in rows])
    d6 = np.array([r['d6'] for r in rows])
    ok = np.array([r['converged'] for r in rows]) & np.isfinite(d6)
    ratio = np.where(ok, d6 / dd, np.nan)

    # plateau from the three smallest converged displacements
    base = np.nanmedian(ratio[ok][:3]) if ok.sum() >= 3 else np.nan
    good = ok & (np.abs(ratio - base) <= tol * base)
    last = dd[good].max() if good.any() else np.nan

    return dict(displacement=dd, d6=d6, d3=np.array([r['d3'] for r in rows]),
                dist_max=np.array([r['dist_max'] for r in rows]),
                converged=ok, ratio=ratio, plateau=float(base),
                last_faithful=float(last), censor=float(last / 2.0),
                R_u=float(R_u), tol=tol, V_pos_norm=float(
                    np.linalg.norm(V[:, :3], axis=1).mean()))
