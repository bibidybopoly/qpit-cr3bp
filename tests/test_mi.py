"""Tests for src/mi.py.

Standalone: no pytest. Run with `python tests/test_mi.py`.

Design note. Every estimator here has a closed-form oracle, so the tests check
NUMBERS, not "did it run". That matters for mutation testing: a suite that only
asserts shapes and non-negativity survives almost every arithmetic mutant. The
tests that specifically exist to kill mutants are marked KILLS in their
docstring, naming the mutation they are there to catch.

Tolerances are set against the fixed seeds below. They are roughly 3x the
observed seed-to-seed spread -- tight enough that a wrong constant fails, loose
enough not to flake. If you change a seed you must re-check the tolerance.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import numpy as np  # noqa: E402

import mi  # noqa: E402

SEED = 20260827
LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# assertion helpers
# ---------------------------------------------------------------------------

def _close(got, want, tol, msg):
    if not abs(got - want) <= tol:
        raise AssertionError(f"{msg}: got {got:.6f}, want {want:.6f} (tol {tol})")


def _true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _gauss_pair(rho, n, rng):
    return rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)


# ---------------------------------------------------------------------------
# kl_entropy -- closed-form oracles
# ---------------------------------------------------------------------------

def test_kl_entropy_uniform_1d():
    """h(U[0,1]) = 0. KILLS: wrong c_d constant (log 2 -> log 1 or log 3)."""
    rng = np.random.default_rng(SEED)
    y = rng.uniform(0.0, 1.0, 20000)
    _close(mi.kl_entropy(y, k=5), 0.0, 0.03, "h(U[0,1])")


def test_kl_entropy_gaussian_1d():
    """h(N(0,1)) = 0.5*ln(2*pi*e) = 1.41894."""
    rng = np.random.default_rng(SEED + 1)
    y = rng.normal(0.0, 1.0, 20000)
    _close(mi.kl_entropy(y, k=5), 0.5 * np.log(2 * np.pi * np.e), 0.04, "h(N(0,1))")


def test_kl_entropy_uniform_2d():
    """h(U[0,1]^2) = 0.

    KILLS: dropping either `d *` multiplier. Both the log(c_d) term and the
    mean-log-radius term carry a factor d, and the 1-D tests above cannot see
    either of them because d = 1 there. This is the gap inspection misses.
    """
    rng = np.random.default_rng(SEED + 2)
    y = rng.uniform(0.0, 1.0, (20000, 2))
    _close(mi.kl_entropy(y, k=5), 0.0, 0.10, "h(U[0,1]^2)")


def test_kl_entropy_scaling_law():
    """h(aY) = h(Y) + log|a| exactly.

    KILLS: a wrong coefficient on mean(log eps), independently of the c_d
    constant -- the additive c_d term cancels in the difference, so only the
    radius term can move this.
    """
    rng = np.random.default_rng(SEED + 3)
    y = rng.normal(size=8000)
    a = 7.0
    got = mi.kl_entropy(a * y, k=5) - mi.kl_entropy(y, k=5)
    _close(got, np.log(a), 1e-9, "h(aY) - h(Y)")


def test_kl_entropy_rejects_duplicates():
    """Duplicate points give a zero k-NN radius and must raise, not return -inf."""
    y = np.zeros(50)
    try:
        mi.kl_entropy(y, k=3)
    except ValueError:
        return
    raise AssertionError("kl_entropy accepted degenerate all-identical input")


# ---------------------------------------------------------------------------
# ksg_mi -- closed-form oracles
# ---------------------------------------------------------------------------

def test_ksg_gaussian_closed_form():
    """I = -0.5*ln(1-rho^2) for a bivariate Gaussian."""
    rng = np.random.default_rng(SEED + 10)
    for rho, tol in ((0.5, 0.05), (0.8, 0.06), (0.95, 0.08)):
        z = _gauss_pair(rho, 6000, rng)
        want = -0.5 * np.log(1.0 - rho**2)
        _close(mi.ksg_mi(z[:, [0]], z[:, [1]], k=5), want, tol, f"I at rho={rho}")


def test_ksg_copula_invariance():
    """I(X;Y) = I(rank X; rank Y). The module's whole design rests on this."""
    rng = np.random.default_rng(SEED + 11)
    z = _gauss_pair(0.8, 6000, rng)
    raw = mi.ksg_mi(z[:, [0]], z[:, [1]], k=5)
    cop = mi.copula_transform(z)
    trans = mi.ksg_mi(cop[:, [0]], cop[:, [1]], k=5)
    _close(trans, raw, 0.06, "copula invariance")


def test_ksg_symmetry():
    """I(X;Y) = I(Y;X). The joint max-norm radius is permutation-invariant and
    the formula is symmetric in n_x, n_y, so this should hold to float noise."""
    rng = np.random.default_rng(SEED + 12)
    z = _gauss_pair(0.7, 3000, rng)
    x, y = z[:, [0]], z[:, [1]]
    _close(mi.ksg_mi(x, y, k=5), mi.ksg_mi(y, x, k=5), 1e-9, "I symmetry")


def test_ksg_accepts_1d_input():
    """A 1-D array of length n must mean n points, not one n-dimensional point.

    KILLS: replacing the _as_2d helper with np.atleast_2d, which reshapes (n,)
    to (1, n) and silently destroys every estimate downstream.
    """
    rng = np.random.default_rng(SEED + 13)
    z = _gauss_pair(0.8, 3000, rng)
    flat = mi.ksg_mi(z[:, 0], z[:, 1], k=5)
    col = mi.ksg_mi(z[:, [0]], z[:, [1]], k=5)
    _close(flat, col, 1e-9, "1-D vs column input")


def test_ksg_counts_are_strict_not_inclusive():
    """Marginal neighbour counts must use STRICT inequality against eps.

    KILLS: `r = eps` instead of `r = np.nextafter(eps, 0.0)`.

    On continuous data this mutant biases the estimate by only ~0.009 nats at
    n=6000, which is below the estimator's own seed-to-seed spread -- no
    tolerance on the Gaussian oracle can separate them. An integer lattice makes
    every distance an integer, so exact ties AT the k-NN radius become the rule
    rather than a measure-zero accident, and the two versions separate by 2.16
    nats. The input carries no randomness, so the expected value below is
    deterministic; the tolerance only absorbs scipy tie-breaking differences.
    """
    m = 30
    x = (np.arange(m * m) // m).astype(float)
    y = (np.arange(m * m) % m).astype(float)
    got = mi.ksg_mi(x, y, k=3)
    _true(got > 0.5, f"lattice estimate {got:.4f}: counts are inclusive, not strict")
    # Deterministic to float precision, so this pins the whole formula rather
    # than just the strictness -- including psi(N), whose off-by-one shifts the
    # value by 1/(n-1) = 1.1e-3 here. Re-baseline if scipy changes its k-NN tie
    # handling; the property assertion above is the version-robust guard.
    _close(got, 0.955750, 1e-5, "lattice estimate")


def test_ksg_independent_inside_null():
    """Independent data must sit inside its own permutation null."""
    rng = np.random.default_rng(SEED + 14)
    x = mi.copula_transform(rng.normal(size=(1200, 4)))
    y = mi.copula_transform(rng.normal(size=1200))
    point = mi.ksg_mi(x, y, k=5)
    null = mi.permutation_null(x, y, k=5, n_perm=60, rng=rng)
    _true(point <= np.percentile(null, 95), f"independent point {point} above null")


def test_ksg_is_not_clamped_at_zero():
    """KSG must be able to return negative values.

    KILLS: `return max(mi, 0.0)`. Clamping looks harmless and is the single most
    damaging mutant in this module -- it would pile the permutation null up at
    zero, collapse the null band, and make P13's arbiter unfalsifiable. Nothing
    else in this suite detects it.
    """
    rng = np.random.default_rng(SEED + 15)
    x = mi.copula_transform(rng.normal(size=(600, 6)))
    y = mi.copula_transform(rng.normal(size=600))
    null = mi.permutation_null(x, y, k=5, n_perm=80, rng=rng)
    _true((null < 0).any(), "no negative values in the independence null")


def test_ksg_detects_dependence():
    """y = f(X0, X3) + noise must land far above the null in 10-D."""
    rng = np.random.default_rng(SEED + 16)
    x = rng.normal(size=(1200, 10))
    y = x[:, 0] + 0.5 * x[:, 3] + 0.3 * rng.normal(size=1200)
    xc, yc = mi.copula_transform(x), mi.copula_transform(y)
    point = mi.ksg_mi(xc, yc, k=5)
    null = mi.permutation_null(xc, yc, k=5, n_perm=40, rng=rng)
    _true(point > np.percentile(null, 95), f"dependence missed: {point}")


def test_ksg_rejects_duplicates_without_jitter():
    """Duplicated rows give a zero joint radius and must raise."""
    rng = np.random.default_rng(SEED + 17)
    z = _gauss_pair(0.6, 200, rng)
    z = np.vstack([z, z])  # every point now has an exact twin
    try:
        mi.ksg_mi(z[:, [0]], z[:, [1]], k=5, jitter=0.0)
    except ValueError:
        return
    raise AssertionError("ksg_mi accepted duplicated rows with jitter=0")


def test_ksg_jitter_rescues_duplicates():
    """The same duplicated input must work once jitter is on."""
    rng = np.random.default_rng(SEED + 18)
    z = _gauss_pair(0.6, 200, rng)
    z = np.vstack([z, z])
    val = mi.ksg_mi(z[:, [0]], z[:, [1]], k=5, jitter=1e-10, rng=rng)
    _true(np.isfinite(val), "jittered estimate not finite")


def test_ksg_length_mismatch_raises():
    rng = np.random.default_rng(SEED + 19)
    try:
        mi.ksg_mi(rng.normal(size=100), rng.normal(size=99), k=5)
    except ValueError:
        return
    raise AssertionError("ksg_mi accepted mismatched lengths")


# ---------------------------------------------------------------------------
# entropy_power_ceiling
# ---------------------------------------------------------------------------

def test_ceiling_recovers_gaussian_r2():
    """For a jointly Gaussian pair the ceiling must return rho^2: N/Var = 1 and
    1 - exp(-2I) = 1 - (1-rho^2). This is the P13.3 arbiter's oracle."""
    rng = np.random.default_rng(SEED + 20)
    for rho in (0.6, 0.85):
        z = _gauss_pair(rho, 8000, rng)
        c = mi.copula_transform(z)
        m = mi.ksg_mi(c[:, [0]], c[:, [1]], k=5)
        out = mi.entropy_power_ceiling(m, z[:, 1], k=5)
        _close(out["r2_ceiling"], rho**2, 0.06, f"ceiling at rho={rho}")


def test_ceiling_entropy_power_ratio_gaussian():
    """N/Var = 1 exactly for a Gaussian."""
    rng = np.random.default_rng(SEED + 21)
    out = mi.entropy_power_ceiling(0.0, rng.normal(size=20000), k=5)
    _close(out["entropy_power_over_var"], 1.0, 0.06, "N/Var Gaussian")


def test_ceiling_entropy_power_ratio_uniform():
    """N/Var = 12/(2*pi*e) = 0.70261 for U[0,1].

    A sharp non-unit oracle. KILLS: a wrong 2*pi*e constant, which the Gaussian
    case cannot see because there the error would have to cancel against h.
    """
    rng = np.random.default_rng(SEED + 22)
    out = mi.entropy_power_ceiling(0.0, rng.uniform(0.0, 1.0, 20000), k=5)
    _close(out["entropy_power_over_var"], 12.0 / (2 * np.pi * np.e), 0.05, "N/Var U")


def test_ceiling_zero_mi_gives_zero_for_gaussian():
    """With I = 0 and Gaussian y, the ceiling collapses to 1 - 1 = 0.
    KILLS: a sign error or missing factor 2 in exp(-2I) at the I=0 point."""
    rng = np.random.default_rng(SEED + 23)
    out = mi.entropy_power_ceiling(0.0, rng.normal(size=20000), k=5)
    _close(out["r2_ceiling"], 0.0, 0.06, "ceiling at I=0")


def test_ceiling_monotone_in_mi():
    rng = np.random.default_rng(SEED + 24)
    y = rng.normal(size=4000)
    vals = [mi.entropy_power_ceiling(m, y, k=5)["r2_ceiling"]
            for m in (0.0, 0.25, 0.5, 1.0, 2.0)]
    _true(all(b > a for a, b in zip(vals, vals[1:])), f"not monotone: {vals}")


def test_ceiling_variance_uses_ddof_one():
    """Pins ddof=1. KILLS: a silent switch to the population variance."""
    rng = np.random.default_rng(SEED + 25)
    y = rng.normal(size=300)
    out = mi.entropy_power_ceiling(0.1, y, k=5)
    _close(out["var"], float(np.var(y, ddof=1)), 1e-12, "reported var")


def test_ceiling_bits_conversion():
    out = mi.entropy_power_ceiling(1.0, np.random.default_rng(SEED + 26).normal(size=500))
    _close(out["mi_bits"], 1.0 / LN2, 1e-12, "nats to bits")


def test_ceiling_zero_variance_raises():
    try:
        mi.entropy_power_ceiling(0.1, np.full(100, 3.0), k=5)
    except ValueError:
        return
    raise AssertionError("ceiling accepted zero-variance y")


# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------

def _window_fixture():
    """d spanning the real dataset's range; every row measured unless stated."""
    floor, boundary = 1.55e-4, mi.DEAD_ZONE_KM_W12
    d = np.concatenate([
        np.geomspace(floor, boundary, 400),          # near field
        np.geomspace(boundary * 1.01, 100.0, 400),   # far field
    ])
    return d, np.ones(d.size, dtype=bool), floor, boundary


def test_windows_near_span_is_2905_decades():
    """log10(0.12462 / 1.55e-4) = 2.9053. Pins the real geometry of the split."""
    d, meas, floor, bnd = _window_fixture()
    w = mi.make_windows(d, meas, floor, bnd)
    _close(w["near_decades"], 2.9053, 1e-3, "near span")


def test_windows_far_upper_limit_is_boundary_squared_over_floor():
    """Matched span means the far window ends at boundary * (boundary/floor)."""
    d, meas, floor, bnd = _window_fixture()
    w = mi.make_windows(d, meas, floor, bnd)
    _close(w["far_hi_intended_km"], bnd * bnd / floor, 1e-9, "far upper limit")


def test_windows_boundary_row_goes_to_near():
    """A row at exactly the boundary belongs to near. KILLS: <= flipped to <."""
    floor, bnd = 1.55e-4, mi.DEAD_ZONE_KM_W12
    d = np.array([bnd])
    w = mi.make_windows(d, np.ones(1, bool), floor, bnd)
    _true(w["n_near"] == 1 and w["n_far"] == 0, "boundary row misassigned")


def test_windows_exclude_and_count_sub_floor_rows():
    """Rows below the determination floor belong to neither window, and the
    count is reported rather than silently dropped.

    KILLS: dropping the `d_km >= floor_km` term. Those rows would land in the
    near window, where the caller's log10(max(d, floor)) clamp collapses them
    onto one value -- an atom that makes h(Y) undefined and makes every KSG
    estimate depend on archive row order.
    """
    floor, bnd = 1.55e-4, mi.DEAD_ZONE_KM_W12
    d = np.concatenate([np.full(7, floor / 10.0),          # below the floor
                        np.geomspace(floor, bnd, 40)])     # in the near window
    w = mi.make_windows(d, np.ones(d.size, bool), floor, bnd)
    _true(w["n_below_floor"] == 7, f"n_below_floor {w['n_below_floor']}, want 7")
    _true(w["n_near"] == 40, f"n_near {w['n_near']}, want 40")
    _true(not w["near"][:7].any(), "sub-floor rows leaked into the near window")
    _true(not w["far"][:7].any(), "sub-floor rows leaked into the far window")


def test_windows_floor_row_goes_to_near():
    """A row at exactly the floor is IN. KILLS: >= flipped to > on the lower
    bound, which would silently shave the lowest decade of the near window."""
    floor, bnd = 1.55e-4, mi.DEAD_ZONE_KM_W12
    w = mi.make_windows(np.array([floor]), np.ones(1, bool), floor, bnd)
    _true(w["n_near"] == 1 and w["n_below_floor"] == 0, "floor row misassigned")


def test_windows_exclude_unmeasured():
    """Censored rows must appear in neither window."""
    d, meas, floor, bnd = _window_fixture()
    meas = meas.copy()
    meas[:100] = False
    w = mi.make_windows(d, meas, floor, bnd)
    _true(not w["near"][:100].any(), "censored rows leaked into near window")
    _true(w["n_near"] == 300, f"expected 300 near rows, got {w['n_near']}")


def test_windows_span_shortfall_flag():
    """Data that stops early must set the flag rather than pass silently."""
    floor, bnd = 1.55e-4, mi.DEAD_ZONE_KM_W12
    d = np.geomspace(floor, bnd * 2.0, 200)  # far side spans only 0.30 decades
    w = mi.make_windows(d, np.ones(d.size, bool), floor, bnd)
    _true(w["span_shortfall"], "shortfall not flagged")


def test_windows_min_n_guard_boundary():
    """n = MIN_N_JOINT-1 -> False, n = MIN_N_JOINT -> True. KILLS: >= to >."""
    floor, bnd = 1.55e-4, mi.DEAD_ZONE_KM_W12
    for n, want in ((mi.MIN_N_JOINT - 1, False), (mi.MIN_N_JOINT, True)):
        d = np.geomspace(floor, bnd, n)
        w = mi.make_windows(d, np.ones(n, bool), floor, bnd)
        _true(w["joint_credible_near"] is want, f"guard wrong at n={n}")


def test_windows_bad_floor_raises():
    try:
        mi.make_windows(np.array([1.0]), np.ones(1, bool), 0.5, 0.2)
    except ValueError:
        return
    raise AssertionError("make_windows accepted floor >= boundary")


# ---------------------------------------------------------------------------
# match_window_sizes / thin_by_group
# ---------------------------------------------------------------------------

def test_match_sizes_equalises_and_keeps_subset():
    """Both orderings are exercised deliberately: the near-larger and far-larger
    branches subsample independently, so a fixture that only ever makes `far`
    the bigger window leaves the `near` branch untested and its mutants alive."""
    rng = np.random.default_rng(SEED + 30)
    for near, far, small in (
        (np.arange(300), np.arange(1000, 2200), "near"),
        (np.arange(1200), np.arange(5000, 5300), "far"),
    ):
        n2, f2 = mi.match_window_sizes(near, far, rng)
        _true(n2.size == f2.size == 300, f"[{small}] sizes {n2.size}, {f2.size}")
        _true(set(n2) <= set(near) and set(f2) <= set(far),
              f"[{small}] invented indices")
        # KILLS: replace=False -> True, on either branch. Duplicated rows tie in
        # the copula space and blow up _knn_radius downstream.
        _true(len(set(n2)) == n2.size and len(set(f2)) == f2.size,
              f"[{small}] subsample drawn with replacement")
        untouched = n2 if small == "near" else f2
        original = near if small == "near" else far
        _true(np.array_equal(untouched, original),
              f"[{small}] smaller window was resampled")


def test_thin_by_group_one_per_group():
    rng = np.random.default_rng(SEED + 31)
    g = np.repeat(np.arange(500), 3)
    idx = mi.thin_by_group(g, rng)
    _true(idx.size == 500, f"expected 500 rows, got {idx.size}")
    _true(len(set(g[idx])) == 500, "duplicate groups in thinned set")


def test_thin_by_group_is_random_not_first():
    """Different rngs must give different picks.
    KILLS: dropping the permutation and always taking the first row per group,
    which would correlate the thinned sample with dataset row order."""
    g = np.repeat(np.arange(400), 4)
    a = mi.thin_by_group(g, np.random.default_rng(1))
    b = mi.thin_by_group(g, np.random.default_rng(2))
    _true(not np.array_equal(a, b), "thinning is deterministic in row order")


# ---------------------------------------------------------------------------
# per_feature_mi / permutation_null / bootstrap_mi
# ---------------------------------------------------------------------------

def test_per_feature_returns_bits_and_finds_signal():
    """Tier 1 must be in bits, and must rank the one informative feature first."""
    rng = np.random.default_rng(SEED + 40)
    x = rng.normal(size=(1500, 5))
    y = x[:, 2] + 0.3 * rng.normal(size=1500)
    xc, yc = mi.copula_transform(x), mi.copula_transform(y)
    out = mi.per_feature_mi(xc, yc, k=5, feature_names=list("abcde"))
    _true(max(out, key=out.get) == "c", f"wrong top feature: {out}")
    _close(out["c"], mi.ksg_mi(xc[:, [2]], yc, k=5) / LN2, 1e-9, "bits conversion")


def test_per_feature_name_mismatch_raises():
    rng = np.random.default_rng(SEED + 41)
    try:
        mi.per_feature_mi(rng.normal(size=(50, 3)), rng.normal(size=50),
                          feature_names=["a", "b"])
    except ValueError:
        return
    raise AssertionError("per_feature_mi accepted a short name list")


def test_permutation_null_destroys_dependence():
    """The null built from dependent data must match the null built from
    independent data. KILLS: shuffling X instead of Y, or shuffling neither."""
    rng = np.random.default_rng(SEED + 42)
    x = rng.normal(size=(800, 3))
    y_dep = x[:, 0] + 0.3 * rng.normal(size=800)
    xc = mi.copula_transform(x)
    dep_null = mi.permutation_null(xc, mi.copula_transform(y_dep), k=5,
                                   n_perm=40, rng=np.random.default_rng(7))
    ind_null = mi.permutation_null(xc, mi.copula_transform(rng.normal(size=800)),
                                   k=5, n_perm=40, rng=np.random.default_rng(7))
    _close(dep_null.mean(), ind_null.mean(), 0.03, "shuffled null still dependent")
    _true(len(dep_null) == 40, "wrong number of permutations returned")


def test_bootstrap_is_reproducible_and_survives_ties():
    """Resampling with replacement always duplicates rows; the default jitter
    must absorb that. KILLS: setting DEFAULT_JITTER to 0."""
    rng = np.random.default_rng(SEED + 43)
    z = _gauss_pair(0.7, 700, rng)
    c = mi.copula_transform(z)
    a = mi.bootstrap_mi(c[:, [0]], c[:, [1]], k=5, n_boot=25,
                        rng=np.random.default_rng(99))
    b = mi.bootstrap_mi(c[:, [0]], c[:, [1]], k=5, n_boot=25,
                        rng=np.random.default_rng(99))
    _true(np.allclose(a, b), "bootstrap not reproducible under a fixed seed")
    _true(np.isfinite(a).all(), "non-finite bootstrap replicate")


# ---------------------------------------------------------------------------
# copula_transform
# ---------------------------------------------------------------------------

def test_copula_values_and_range():
    """Exactly (rank+1)/(n+1), strictly inside (0,1)."""
    x = np.array([[30.0], [10.0], [20.0]])
    c = mi.copula_transform(x)
    _true(np.allclose(c.ravel(), [3 / 4, 1 / 4, 2 / 4]), f"wrong values: {c.ravel()}")
    _true((c > 0).all() and (c < 1).all(), "copula values touched the boundary")


def test_copula_is_monotone_and_columnwise():
    rng = np.random.default_rng(SEED + 50)
    x = rng.normal(size=(200, 3))
    c = mi.copula_transform(x)
    for j in range(3):
        _true(np.array_equal(np.argsort(x[:, j]), np.argsort(c[:, j])),
              f"column {j} order not preserved")
    x2 = x.copy()
    x2[:, 0] = rng.normal(size=200)
    _true(np.allclose(mi.copula_transform(x2)[:, 1:], c[:, 1:]),
          "columns are not transformed independently")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _collect():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


def main():
    tests = _collect()
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 -- runner reports, does not raise
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
