"""Sampling for the Phase 4 labelled dataset.

Compute stopped being the constraint once labelling became geometric
(~320 states/s vs ~0.4 with Newton per sample), so the only thing that
matters now is COVERAGE.  Three distributions are chosen separately;
conflating them is how this goes wrong.

DISTANCE.  The target is log10 d, so distance is drawn log-uniform and the
state is CONSTRUCTED to realise it -- not the other way round.  Sampling
states uniformly in phase space would put almost everything at large
distance and leave the near-surface region, the one that matters for
warm-starting a solver, with no training support.  Drawing d first also
makes every sample a free regression test: the label comes back and should
equal what was drawn.

DIRECTION.  A uniformly random 6-D direction has a large tangential
component, so stepping by d lands CLOSER than d, and the shortfall varies
over the surface -- a bias that is both wrong and non-uniform.  Projecting
out both local tangents (theta along the curve, alpha across the family)
makes the realised distance equal the drawn one to ~2%.  A declared
fraction of unprojected directions is kept for coverage, with the realised
label recorded either way.

BASE POINT.  Amplitude is drawn uniformly over the family's span and then
converted to (gap, alpha), rather than drawing a gap index uniformly.
Uniform-in-index would inherit the densification's own spacing history --
the same class of contamination as the Week 9 energy bug.  theta is drawn
uniformly because that is the invariant measure of the quasi-periodic flow
on each torus.

GROUPING.  Samples from the same neighbourhood are not independent.
group_id bins (family, energy, gap, alpha, theta) so Week 11 can split on
it; splitting on rows would put near-duplicates on both sides of the
train/test boundary.

FLOOR.  Measured, not assumed: re-deriving a torus by a different route
lands 2-155 mm from the surface, so the surface is not determined below
~0.15 m however finely the family is densified.  That is the sigma_3
~ 2.0e-4 result -- the curves themselves are the limit, not the spacing.
"""

from __future__ import annotations

import numpy as np

from qpit import evaluate_curve, torus_bundle
from torus_field import surface_distance

LU = 384400.0
FLOOR_KM = 1.55e-4          # 0.155 m, measured reproducibility floor


# =========================================================================
# SECTION 1 -- SURFACE GEOMETRY
# =========================================================================

def amplitude_to_cell(members, A_target):
    """Map an amplitude to (gap index, alpha) by linear inversion.

    Drawing amplitude and inverting -- rather than drawing a gap index --
    is what keeps the densification's spacing out of the sample measure.
    """
    A = np.array([m['amplitude'] for m in members])
    i = np.clip(np.searchsorted(A, A_target) - 1, 0, len(A) - 2)
    a = (A_target - A[i]) / np.maximum(A[i + 1] - A[i], 1e-300)
    return i, np.clip(a, 0.0, 1.0)


def surface_point(members, i, alpha, theta):
    """u_alpha(theta) on the interpolated surface, plus both tangents.

    t_theta from a central difference on evaluate_curve; t_alpha is EXACT
    (u_b - u_a) because the interpolant is linear in alpha.
    """
    U_a, U_b = members[i]['U0'], members[i + 1]['U0']
    A_, B_ = evaluate_curve(U_a, theta), evaluate_curve(U_b, theta)
    x = (1.0 - alpha)[:, None] * A_ + alpha[:, None] * B_

    h = 1e-6
    Ap, Bp = evaluate_curve(U_a, theta + h), evaluate_curve(U_b, theta + h)
    Am, Bm = evaluate_curve(U_a, theta - h), evaluate_curve(U_b, theta - h)
    t_th = ((1 - alpha)[:, None] * (Ap - Am)
            + alpha[:, None] * (Bp - Bm)) / (2 * h)
    t_al = B_ - A_
    return x, t_th, t_al


def _normalise(V):
    return V / np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-300)


def normal_directions(t_th, t_al, rng):
    """Random unit directions with both surface tangents projected out."""
    V = rng.standard_normal(t_th.shape)
    for T in (t_th, t_al):
        Tn = _normalise(T)
        V = V - np.sum(V * Tn, axis=1)[:, None] * Tn
    return _normalise(V)


# =========================================================================
# SECTION 2 -- SAMPLERS
# =========================================================================

def sample_offsets(members, n, rng, d_lo_km=FLOOR_KM, d_hi_km=100.0,
                   frac_random_dir=0.25):
    """Backbone sampler: log-uniform distance, mostly normal directions.

    Returns states and the metadata needed to label and group them.  The
    DRAWN distance is returned alongside so a caller can compare it with the
    measured label -- for normal directions they should agree to ~2%, and a
    systematic gap means the tangent projection is wrong.
    """
    A = np.array([m['amplitude'] for m in members])
    A_t = rng.uniform(A[0], A[-1], n)
    i, alpha = amplitude_to_cell(members, A_t)
    theta = rng.uniform(0.0, 2 * np.pi, n)

    X = np.empty((n, 6))
    d_drawn = 10.0 ** rng.uniform(np.log10(d_lo_km), np.log10(d_hi_km), n) / LU
    is_rand = rng.random(n) < frac_random_dir

    for gi in np.unique(i):
        s = np.nonzero(i == gi)[0]
        x0, t_th, t_al = surface_point(members, int(gi), alpha[s], theta[s])
        V = normal_directions(t_th, t_al, rng)
        Vr = _normalise(rng.standard_normal(V.shape))
        V = np.where(is_rand[s][:, None], Vr, V)
        X[s] = x0 + d_drawn[s][:, None] * V

    return dict(states=X, gap=i, alpha=alpha, theta=theta,
                amplitude=A_t, d_drawn_km=d_drawn * LU,
                random_dir=is_rand, source=np.array(['offset'] * n))


def sample_on_surface(members, n, rng):
    """Anchor the near-surface end: states with d = 0 by construction."""
    A = np.array([m['amplitude'] for m in members])
    A_t = rng.uniform(A[0], A[-1], n)
    i, alpha = amplitude_to_cell(members, A_t)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    X = np.empty((n, 6))
    for gi in np.unique(i):
        s = np.nonzero(i == gi)[0]
        X[s] = surface_point(members, int(gi), alpha[s], theta[s])[0]
    return dict(states=X, gap=i, alpha=alpha, theta=theta, amplitude=A_t,
                d_drawn_km=np.zeros(n), random_dir=np.zeros(n, bool),
                source=np.array(['surface'] * n))


def sample_bundle(members, mu, n, rng, d_lo_km=FLOOR_KM, d_hi_km=100.0,
                  n_members=8):
    """Off centre-manifold coverage, along each torus's unstable bundle.

    The bundle is a per-node object, so states are drawn at node angles
    rather than continuously.  Restricted to n_members tori because
    torus_bundle needs a stability_matrix per torus, which is the expensive
    part; the states themselves are cheap.
    """
    idx = np.unique(np.linspace(0, len(members) - 1, n_members).astype(int))
    per = max(n // len(idx), 1)
    Xs, gaps, ths, amps, dd = [], [], [], [], []
    for mi in idx:
        m = members[int(mi)]
        V, _ = torus_bundle(m['U0'], m['T'], m['rho'], mu, which='unstable')
        N = V.shape[0]
        j = rng.integers(N, size=per)
        d = 10.0 ** rng.uniform(np.log10(d_lo_km), np.log10(d_hi_km), per) / LU
        sign = np.where(rng.random(per) < 0.5, 1.0, -1.0)
        Xs.append(m['U0'][j] + (sign * d)[:, None] * V[j])
        gaps.append(np.full(per, min(int(mi), len(members) - 2)))
        ths.append(2 * np.pi * j / N)
        amps.append(np.full(per, m['amplitude']))
        dd.append(d * LU)
    n_out = per * len(idx)
    return dict(states=np.vstack(Xs), gap=np.concatenate(gaps),
                alpha=np.zeros(n_out), theta=np.concatenate(ths),
                amplitude=np.concatenate(amps), d_drawn_km=np.concatenate(dd),
                random_dir=np.zeros(n_out, bool),
                source=np.array(['bundle'] * n_out))


# =========================================================================
# SECTION 3 -- ASSEMBLY
# =========================================================================

def group_ids(fam_id, energy_id, gap, alpha, theta, n_alpha=4, n_theta=8):
    """Bin the base-point cell. THE key for Week 11's train/test split."""
    a = np.clip((alpha * n_alpha).astype(int), 0, n_alpha - 1)
    t = np.clip((theta / (2 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    return (((fam_id * 1000 + energy_id) * 10000 + gap) * n_alpha + a) * n_theta + t


def label_and_pack(parts, members, fam_id, energy_id, family_name, C,
                   floor_km=FLOOR_KM, **kw):
    """Label every sampled state against the surface and assemble columns."""
    cat = lambda k: np.concatenate([p[k] for p in parts])
    X = np.vstack([p['states'] for p in parts])
    lab = surface_distance(X, members, **kw)
    d_km = lab['d'] * LU
    gap, alpha, theta = cat('gap'), cat('alpha'), cat('theta')
    return dict(
        states=X, d_km=d_km,
        log_d_km=np.log10(np.maximum(d_km, floor_km)),
        at_floor=d_km < floor_km,
        d_drawn_km=cat('d_drawn_km'), source=cat('source'),
        random_dir=cat('random_dir'), amplitude_km=cat('amplitude') * LU,
        gap=gap, alpha=alpha, theta=theta,
        alpha_found=lab['alpha'], theta_found=lab['theta'],
        group_id=group_ids(fam_id, energy_id, gap, alpha, theta),
        family_id=np.full(X.shape[0], fam_id),
        family=np.array([family_name] * X.shape[0]),
        energy_id=np.full(X.shape[0], energy_id),
        C=np.full(X.shape[0], C),
        jacobi=np.full(X.shape[0], C))
