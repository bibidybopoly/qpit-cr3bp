"""Mutual-information diagnostics for the Week 13 dead-zone test.

The question: is the ~125 m dead zone an information limit or a capacity limit?
This module estimates I(features; log10 d) inside a near-field window and a
span-matched far-field control, and converts the estimate into an upper bound on
achievable R^2.

NOTHING IN THIS MODULE FITS ANYTHING. Every quantity is a functional of the
frozen dataset. If a future edit introduces a fit, the Week 13 result stops
being admissible as evidence about the model.

Theory the code depends on
--------------------------

(1) MI is exactly invariant under smooth invertible per-variable
    reparameterisation, so I(X; d) == I(X; log10 d) identically. This is why MI
    is the admissible instrument after Week 12: no choice of target transform
    can move it, which is exactly the failure mode that R^2 had.

    The KSG *estimator* is not invariant. Every estimate below is therefore
    taken on copula-transformed (rank) coordinates. A per-coordinate rank map is
    monotone, so the estimand is unchanged, and as a bonus the marginals become
    uniform by construction -- identical in both windows -- which removes the
    log-span mismatch between near and far field from the comparison.

(2) Shannon lower bound, which converts MI into something comparable to R^2.
    With Y = log10 d,

        I(X;Y) = h(Y) - h(Y|X)
        h(Y|X) <= 0.5 * log(2*pi*e*mmse)

    the second line because the Gaussian maximises differential entropy at fixed
    variance, and log is concave so Jensen carries the conditional average
    inside. Rearranging,

        mmse >= N(Y) * exp(-2I),        N(Y) := exp(2 h(Y)) / (2*pi*e)
        R^2  <= 1 - (N(Y)/Var(Y)) * exp(-2I)

    N(Y) is the entropy power; N(Y)/Var(Y) <= 1 with equality iff Y is Gaussian.

    The inequality runs in the useful direction. A small MI yields a genuine
    ceiling on R^2, so the "informational limit" verdict rests on a bound rather
    than on absence of evidence. A large MI yields a high ceiling, which on its
    own proves nothing -- an upper bound being loose is not evidence.

    h(Y) and Var(Y) are computed on RAW log10 d inside the window, because that
    is the space M1's R^2 is reported in. The MI in the exponent is estimated on
    copula coordinates. Mixing the two spaces is correct precisely because of
    (1), and would be a bug for any non-invariant statistic.

(3) The far-field window is a POSITIVE CONTROL FOR ESTIMATOR POWER, not a
    reference value. "MI is small in the near field" is only credible if the
    same estimator, at the same n, the same dimension and the same marginal
    shape, returns something large elsewhere. Pre-registration P13.1: if the
    far-field ceiling lands below M1's achieved far-field R^2 on the same rows,
    the estimator is under-reading MI and the whole test is void.

Conventions / things to check against the rest of the repo
----------------------------------------------------------
- This module deliberately imports nothing from src/constants.py. FLOOR_KM and
  the feature matrix are passed in by the caller, so there is no second copy of
  a constant to drift. DEAD_ZONE_KM_W12 below is a Week 12 *measurement*, not a
  physical constant, which is why it is not in constants.py.
- Requires scipy >= 1.6 for cKDTree.query_ball_point(return_length=True).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

__all__ = [
    "DEAD_ZONE_KM_W12",
    "MIN_N_JOINT",
    "copula_transform",
    "ksg_mi",
    "kl_entropy",
    "entropy_power_ceiling",
    "make_windows",
    "match_window_sizes",
    "thin_by_group",
    "per_feature_mi",
    "permutation_null",
    "bootstrap_mi",
]

# Week 12 measured dead-zone boundary in km (124.62 m), from results_wk12.json.
# Empirical, not physical -- hence here rather than in constants.py.
#
# NOT THE ONLY COPY, and the other one is deliberately a different number:
# `ablation.NEAR_FIELD_KM = 0.125` is the WEEK 11 measurement of the same
# boundary, and Week 12's ablation tables are scored against the number that
# was current when they were pre-registered.  They agree to 0.3%, which is
# well inside the 0.21%-wide band of boundaries that select identical rows for
# the far window and so cannot move a Week 13 or Week 15 count.  Keeping both
# is right; quietly unifying them would re-score Week 12 against a boundary it
# did not use.  This comment exists because "One copy" used to sit here and
# was wrong.
DEAD_ZONE_KM_W12 = 0.12462

# Below this many rows a joint KSG estimate at dim > ~5 is not credible and the
# week reports Tier-1 (per-feature) MI only. Pre-registered guard.
MIN_N_JOINT = 500

# Copula coordinates are k/(n+1), so the minimum spacing between distinct values
# is 1/(n+1) >= 1e-5 for any n we will see. Jitter seven orders below that
# breaks bootstrap-induced ties without perturbing geometry.
DEFAULT_JITTER = 1e-12


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _as_2d(a: np.ndarray) -> np.ndarray:
    """(n,) -> (n,1), (n,d) unchanged.

    np.atleast_2d is NOT usable here: on a 1-D array of length n it returns
    shape (1, n), i.e. one n-dimensional point, which silently turns every
    estimate below into nonsense.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got ndim={a.ndim}")
    return a


def _knn_radius(pts: np.ndarray, k: int) -> np.ndarray:
    """Chebyshev distance from each point to its k-th nearest *other* point.

    Raises on duplicate rows. The check is on column 1 -- the nearest OTHER
    point -- not on the returned k-th radius, and the difference is the whole
    reason this guard lives here.

    With k=5 and one exact twin per point, the k-th radius is comfortably
    positive while the tie-free assumption underpinning the KSG and KL bias
    cancellation is already violated: the estimate comes back finite,
    plausible, and wrong. Checking only the k-th radius would let that through.
    Bootstrap resampling with replacement produces exactly this case on almost
    every draw, which is why bootstrap_mi jitters by default.
    """
    n = pts.shape[0]
    if not (1 <= k < n):
        raise ValueError(f"need 1 <= k < n; got k={k}, n={n}")
    tree = cKDTree(pts)
    # k+1 neighbours because the first hit is the point itself at distance 0.
    dists, _ = tree.query(pts, k=k + 1, p=np.inf, workers=-1)
    if np.any(dists[:, 1] <= 0.0):
        raise ValueError(
            "duplicate points in the k-NN input: KSG/KL assume no ties. Pass "
            "jitter>0 (bootstrap resamples always need it) or de-duplicate. "
            "Note copula-transformed input cannot produce ties, so this firing "
            "means the caller passed raw coordinates or a resample."
        )
    return dists[:, k]


def copula_transform(X: np.ndarray) -> np.ndarray:
    """Rank-transform each column to (0,1). MI is invariant under this map.

    Uses k/(n+1) rather than k/n so no point sits exactly on the boundary.
    Ties are broken by first-occurrence order (argsort of argsort), which is
    deterministic; with continuous features ties should not occur at all, and if
    they do the resulting duplicate coordinates will be caught by the eps <= 0
    check in the estimators rather than passing silently.
    """
    X = _as_2d(X)
    n = X.shape[0]
    ranks = np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0, kind="stable")
    return (ranks + 1.0) / (n + 1.0)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def ksg_mi(x, y, k: int = 5, jitter: float = 0.0, rng=None) -> float:
    """KSG algorithm-1 estimate of I(X;Y) in NATS.

    Kraskov, Stoegbauer & Grassberger, Phys. Rev. E 69, 066138 (2004), Eq. 8:

        I_hat = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >

    where eps_i is the distance from point i to its k-th nearest neighbour in
    the JOINT space under the max-norm, and n_x(i), n_y(i) count neighbours
    lying STRICTLY inside eps_i in each marginal.

    The estimator is NOT clamped at zero. On independent data it returns small
    values of either sign, and clamping would pile the permutation null up at 0
    and destroy the null band that P13 uses as its arbiter. If a caller wants a
    non-negative number it must clamp downstream, after the null is built.
    """
    x = _as_2d(x)
    y = _as_2d(y)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"x has {n} rows, y has {y.shape[0]}")

    if jitter > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        x = x + rng.uniform(-jitter, jitter, size=x.shape)
        y = y + rng.uniform(-jitter, jitter, size=y.shape)

    # _knn_radius raises on duplicate joint points; see its docstring for why
    # the check is on the nearest neighbour rather than on eps itself.
    eps = _knn_radius(np.hstack([x, y]), k)

    # query_ball_point uses <=. nextafter(eps, 0) is the largest float strictly
    # below eps, so <= against it reproduces the strict < that Eq. 8 requires.
    # Points sitting exactly at eps are excluded, which is the whole point of
    # the KSG bias cancellation.
    r = np.nextafter(eps, 0.0)

    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # -1 removes the point itself, which is inside every ball of radius >= 0.
    # workers=-1 is pure parallelism: identical results, ~4-8x wall clock on a
    # multicore box. The Chebyshev query cost grows steeply with feature
    # dimension, so this is what keeps the pre-registered replicate counts
    # affordable at dim > 12.
    nx = np.asarray(tree_x.query_ball_point(
        x, r, p=np.inf, return_length=True, workers=-1)) - 1
    ny = np.asarray(tree_y.query_ball_point(
        y, r, p=np.inf, return_length=True, workers=-1)) - 1

    return float(
        digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )


def kl_entropy(y, k: int = 5) -> float:
    """Kozachenko-Leonenko differential entropy in NATS, max-norm.

        h_hat = -psi(k) + psi(N) + log(c_d) + (d/N) * sum_i log(eps_i)

    with c_d = 2^d the volume of the unit max-norm ball in d dimensions and
    eps_i the distance to point i's k-th nearest neighbour.

    Sanity check the constant against Y ~ U[0,1] (d=1, true h=0): the k-th
    neighbour distance is typically eps ~ k/(2N), and to leading order
    -log k + log N + log 2 + log(k/(2N)) = 0.
    """
    y = _as_2d(y)
    n, d = y.shape
    eps = _knn_radius(y, k)   # raises on duplicate points
    return float(
        -digamma(k) + digamma(n) + d * np.log(2.0) + d * np.mean(np.log(eps))
    )


def entropy_power_ceiling(mi_nats: float, y_raw, k: int = 5) -> dict:
    """R^2 ceiling from the Shannon lower bound. See module docstring (2).

    y_raw MUST be in the space the model's R^2 is reported in -- log10 d_km --
    NOT copula coordinates. Passing copula coordinates here is a silent error
    that produces a plausible-looking number, so it is the first thing to check
    if a ceiling looks strange.

    entropy_power_over_var is reported rather than clipped. Theory puts it at
    <= 1; a value above 1 means the KL entropy estimate is off, not that the
    data are super-Gaussian, and it should void the ceiling rather than be
    silently corrected.
    """
    y_raw = np.asarray(y_raw, dtype=float).ravel()
    var = float(np.var(y_raw, ddof=1))
    if var <= 0.0:
        raise ValueError("zero variance in y_raw; ceiling undefined")

    h = kl_entropy(y_raw, k=k)
    entropy_power = float(np.exp(2.0 * h) / (2.0 * np.pi * np.e))
    ratio = entropy_power / var
    return {
        "mi_nats": float(mi_nats),
        "mi_bits": float(mi_nats / np.log(2.0)),
        "h_nats": h,
        "var": var,
        "entropy_power": entropy_power,
        "entropy_power_over_var": ratio,
        "r2_ceiling": float(1.0 - ratio * np.exp(-2.0 * mi_nats)),
        "ratio_exceeds_theory": bool(ratio > 1.0),
    }


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def make_windows(d_km, measured, floor_km: float,
                 boundary_km: float = DEAD_ZONE_KM_W12) -> dict:
    """Near-field and span-matched far-field masks over measured rows.

    Near field is [floor_km, boundary_km]. Its span is
    log10(boundary_km / floor_km), which for the Week 12 numbers is
    log10(0.12462 / 1.55e-4) = 2.905 decades -- exactly half of the dataset's
    5.810. The matched far window is [boundary_km, boundary_km * F] with the
    same multiplicative factor F = boundary_km / floor_km, so both windows span
    the same number of decades by construction.

    The realised far span is reported separately because the data may simply
    stop before boundary_km * F. Copula transforming equalises the marginals
    regardless, so a shortfall does not corrupt the MI comparison -- but it does
    change h(Y) and Var(Y), hence the ceiling, so it must be visible.
    """
    d_km = np.asarray(d_km, dtype=float).ravel()
    measured = np.asarray(measured, dtype=bool).ravel()
    if d_km.shape != measured.shape:
        raise ValueError("d_km and measured must have the same length")
    if not (0.0 < floor_km < boundary_km):
        raise ValueError("require 0 < floor_km < boundary_km")

    factor = boundary_km / floor_km
    far_hi_intended = boundary_km * factor

    # The lower bound is enforced, not just documented. Rows below the
    # surface-determination floor have no meaningful distance to the true
    # surface, and the caller's log target almost certainly clamps them --
    # log10(max(d, floor)) maps every one of them onto a single value, an atom
    # that lands in the near window and silently corrupts both h(Y) and every
    # k-NN estimate built on it. They are excluded and COUNTED, never clamped.
    below = measured & (d_km < floor_km)
    near = measured & (d_km >= floor_km) & (d_km <= boundary_km)
    far = measured & (d_km > boundary_km) & (d_km <= far_hi_intended)

    near_decades = float(np.log10(factor))
    if far.sum() > 0:
        far_decades = float(np.log10(d_km[far].max() / boundary_km))
    else:
        far_decades = 0.0

    return {
        "near": near,
        "far": far,
        "n_near": int(near.sum()),
        "n_far": int(far.sum()),
        "n_below_floor": int(below.sum()),
        "boundary_km": float(boundary_km),
        "floor_km": float(floor_km),
        "far_hi_intended_km": float(far_hi_intended),
        "near_decades": near_decades,
        "far_decades_realised": far_decades,
        # 0.9 is a reporting threshold, not a gate. It exists so a shortfall is
        # never discovered after the ceiling has been quoted.
        "span_shortfall": bool(far_decades < 0.9 * near_decades),
        "joint_credible_near": bool(near.sum() >= MIN_N_JOINT),
        "joint_credible_far": bool(far.sum() >= MIN_N_JOINT),
    }


def match_window_sizes(near_idx, far_idx, rng) -> tuple:
    """Subsample the larger window to the size of the smaller, without
    replacement. Random subsampling preserves the marginal shape and the
    dependence structure; it only reduces n. Both windows must share n because
    every k-NN estimator's bias is an explicit function of n."""
    near_idx = np.asarray(near_idx)
    far_idx = np.asarray(far_idx)
    m = min(near_idx.size, far_idx.size)
    if near_idx.size > m:
        near_idx = np.sort(rng.choice(near_idx, size=m, replace=False))
    if far_idx.size > m:
        far_idx = np.sort(rng.choice(far_idx, size=m, replace=False))
    return near_idx, far_idx


def thin_by_group(group_id, rng) -> np.ndarray:
    """One row per group_id, drawn uniformly within each group.

    KSG assumes i.i.d. rows. With 6,564 groups over 20,400 rows (median size 2)
    the dependence is weak, but weak is not measured -- the thinned estimate is
    what turns that assumption into a reported number.
    """
    group_id = np.asarray(group_id).ravel()
    perm = rng.permutation(group_id.size)
    _, first = np.unique(group_id[perm], return_index=True)
    return np.sort(perm[first])


# ---------------------------------------------------------------------------
# tiers, nulls, spread
# ---------------------------------------------------------------------------

def per_feature_mi(X, y, k: int = 5, feature_names=None) -> dict:
    """Tier 1: I(X_j; Y) for each feature separately, in bits.

    1-D against 1-D, so KSG is on solid ground at any n we will see. This tier
    stays credible when the joint estimate does not, and it is the fallback the
    pre-registration falls to when n < MIN_N_JOINT.
    """
    X = _as_2d(X)
    y = _as_2d(y)
    names = list(feature_names) if feature_names is not None else [
        f"f{j}" for j in range(X.shape[1])
    ]
    if len(names) != X.shape[1]:
        raise ValueError(f"{len(names)} names for {X.shape[1]} features")
    ln2 = np.log(2.0)
    return {
        name: ksg_mi(X[:, [j]], y, k=k) / ln2
        for j, name in enumerate(names)
    }


def permutation_null(X, y, k: int = 5, n_perm: int = 200, rng=None) -> np.ndarray:
    """Null distribution of the KSG estimate under independence, in NATS.

    Y is shuffled within the window, which preserves both marginals and n
    exactly and destroys only the dependence. The estimator's finite-n bias is
    therefore reproduced, which is the entire point: "MI is near zero" means
    "inside this band", not "close to the number zero".
    """
    rng = np.random.default_rng() if rng is None else rng
    X = _as_2d(X)
    y = _as_2d(y)
    n = X.shape[0]
    out = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        out[b] = ksg_mi(X, y[rng.permutation(n)], k=k)
    return out


def bootstrap_mi(X, y, k: int = 5, n_boot: int = 200, rng=None,
                 jitter: float = DEFAULT_JITTER) -> np.ndarray:
    """Bootstrap spread of the KSG estimate, in NATS.

    Resampling with replacement produces exact duplicates, which give zero k-NN
    radii and break the estimator outright. jitter > 0 is therefore mandatory
    here, not optional -- it is the reason DEFAULT_JITTER exists.

    The copula transform is applied ONCE by the caller, before bootstrapping.
    Re-ranking each resample would fold the marginal estimation into the spread
    and is not what the pre-registered 95th percentile refers to.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = _as_2d(X)
    y = _as_2d(y)
    n = X.shape[0]
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = ksg_mi(X[idx], y[idx], k=k, jitter=jitter, rng=rng)
    return out
