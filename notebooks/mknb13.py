#!/usr/bin/env python3
"""Builds notebooks/wk13_mi.ipynb -- the Week 13 mutual-information test.

Usage:
    python mknb13.py            # write if absent, no-op if identical, fail if diverged
    python mknb13.py --force    # overwrite regardless

Divergence is measured on CELL SOURCE ONLY. Executing the notebook writes
outputs and execution counts back into the .ipynb, and that must not read as a
hand-edit -- otherwise the builder would cry divergence after every single run
and you would learn to pass --force reflexively, which defeats the guard.

Scope note. This builder covers the MI test only: P13.1, P13.2, P13.3, plus the
calibration gate. P13.0 (the eikonal / benchmark selector) needs the frozen M1
weights and autodiff, and gets its own builder later in the week.
"""

import argparse
import json
import os
import sys

NB_PATH = os.path.join("notebooks", "wk13_mi.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [

md(r"""
# Week 13 — Mutual information: is the 125 m dead zone informational or capacity?

**Pre-registered 2026-08-27, before any estimate was computed. Do not edit the
conditions below in response to what the numbers turn out to be.**

Week 12 assumed the dead zone was a capacity limit (bounded network gradient
against unbounded ∇log d), applied the direct remedy, and made things worse at
every one of the 32 fits. This notebook asks the prior question instead: is the
information there at all?

**Why MI and not something cheaper.** MI is exactly invariant under smooth
invertible reparameterisation of either variable, so `I(X; d) = I(X; log10 d)`
identically. It is the one quantity the asinh episode could not have corrupted,
which is precisely the property R² lacked.

**Conditions.**

| ID | Statement | Arbiter |
|----|-----------|---------|
| **P13.1** | *Validity gate.* `ceiling_far ≥ achieved_far_R2` on the same rows | Failure ⇒ **test VOID**, no near-field number reported |
| **P13.2** | *Direction.* `I_near < I_far` | Point estimates, matched n and dimension |
| **P13.3** | *Verdict.* Ceiling from the **point estimate**, corrected by locally inverting the calibration map (erring high) | `corrected_ceiling_near ≤ achieved_near_R2 + MARGIN` ⇒ **informational**; else information is present and the architecture question stays open |

`MARGIN = 0.10`.

---

### Amendment, 2026-08-28 — logged before any real number was computed

The calibration gate fired, exactly as it was written to. Three changes follow
from it. All were decided from synthetic-data behaviour only; no window
statistic had been computed.

1. **The bootstrap is removed from the arbiter.** It is invalid for k-NN
   estimators: resampling with replacement duplicates ~26% of rows, the 1e-12
   jitter separates each twin by far less than the ~1e-4 spacing of genuine
   neighbours, every duplicate becomes its own nearest neighbour, and the radii
   collapse. Measured effect is a *fixed* +0.19 nat offset at every true MI —
   an offset, not a spread. Significance now comes from the permutation null,
   which creates no duplicates and is unaffected. An uncertainty band needs
   subsampling without replacement and is parked to Week 17–18 with the other
   CIs.

2. **The point estimate under-reads the ceiling, and the under-read is local.**
   It runs −0.003 at true R² 0.05 and −0.101 at 0.75, so a single global
   worst-case correction would apply the error from an operating point we are
   not at, leaving +0.0007 of usable slack. The calibration map is inverted
   locally instead, from replicated points, and the notebook refuses to
   extrapolate outside the calibrated range. `z = +1.645` for P13.3 so the
   corrected ceiling errs high (a too-low ceiling is what would falsely declare
   the informational verdict); `z = −1.645` for P13.1 so the validity gate
   cannot be passed by an inflated one.

3. **Calibration now runs at the real configuration.** The first version used
   dim 9 and a plain joint KSG. The analysis uses **6** continuous dimensions
   under a 6-cell conditioning, because `C` turned out to have three distinct
   values and is categorical, not continuous.

**A verdict of "information present" is not a bad outcome and does not reopen
Week 16.** With 14 and 16 cut, it costs one sentence in Limitations: the
information is there, and we did not establish whether a different architecture
could reach it.

**Guards.** `n < MIN_N_JOINT` ⇒ Tier-1 (per-feature) only. `k ∈ {3,5,10}` for
stability. Group-thinned estimate reported alongside the full one, because 6,564
groups over 20,400 rows makes the rows nearly independent but "nearly" is not a
measurement.

**A limit this test does not resolve.** The handoff frames the question as a
binary. It is a trichotomy: informational, capacity, and *data*. A high MI with
poor model performance is consistent with either capacity or data starvation,
and separating those needs a learning curve, which is not Week 13 work. Say so
in the writeup rather than over-claiming.

**Nothing here fits anything.** Every quantity is a functional of the frozen
dataset and the frozen Week 12 predictions. If a future edit introduces a fit,
this stops being admissible as evidence about the model.
"""),

code(r"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join("..", "src"))
import mi
import constants

# ---- run configuration -----------------------------------------------------
SEED     = 20260827
K_MAIN   = 5
K_SWEEP  = (3, 5, 10)
N_PERM   = 200          # permutation null replicates
MARGIN   = 0.10         # P13.3
CAL_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70)  # true R^2 calibration grid,
                        # dense at the low end where the near field will sit
CAL_REPS = 5            # replicates per grid point; the map gets inverted, so
                        # one draw per point is not enough
CAL_GROUP_FRAC = 0.30   # share of explained variance the categorical carries
                        # in the synthetic calibration cases
# No bootstrap: invalid for k-NN estimators. See the amendment in cell 1.

DATA   = os.path.join("..", "data", "qpit_dataset_w10.npz")
PRED12    = os.path.join("..", "data", "wk12", "predictions_wk12.npz")
RESULTS12 = os.path.join("..", "data", "wk12", "results_wk12.json")
OUT       = os.path.join("..", "data", "wk13", "results_wk13_mi.json")

rng = np.random.default_rng(SEED)
print("dead-zone boundary (Week 12) =", mi.DEAD_ZONE_KM_W12, "km")
"""),

md(r"""
## Adapter — the only cell coupled to repo APIs

Every assumption about column names, function signatures and file layout lives
here and nowhere else. If anything below is wrong, it is wrong *in this cell*,
and the asserts print the real contents rather than failing somewhere downstream
with a plausible-looking number.
"""),

code(r"""
# ---- names verified against the real archive 2026-08-28. Fix HERE only. ---
KEY_STATE  = "states"     # (n, 6) phase-space states  -- plural, not "state"
KEY_D_KM   = "d_km"       # distance to nearest QPIT surface, km. THIS is M1's
                          # target; d_drawn_km is the drawn offset used to
                          # validate the labeller and must not be swapped in.
KEY_CENSOR = "at_floor"   # True where d was not measured
KEY_GROUP  = "group_id"   # sampling group, for thinning
KEY_SOURCE = "source"     # sampler label; used only for the orientation check
KEY_FLOOR  = "floor_km"   # the floor THIS dataset was labelled with

npz = np.load(DATA, allow_pickle=True)
missing = [k for k in (KEY_STATE, KEY_D_KM, KEY_CENSOR, KEY_GROUP,
                       KEY_SOURCE, KEY_FLOOR) if k not in npz.files]
assert not missing, f"missing {missing}; archive has {list(npz.files)}"

state    = np.asarray(npz[KEY_STATE], dtype=float)
d_km     = np.asarray(npz[KEY_D_KM], dtype=float).ravel()
censored = np.asarray(npz[KEY_CENSOR]).ravel().astype(bool)
group_id = np.asarray(npz[KEY_GROUP]).ravel()
measured = ~censored
_src = np.asarray(npz[KEY_SOURCE]).ravel()
source = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in _src])

# The floor comes from the ARCHIVE, not from constants.py: it is the value this
# frozen dataset was actually labelled with, which is the provenance-correct
# number. constants.FLOOR_KM is then cross-checked against it. This is not
# shadowing -- it is the drift check. A mismatch would silently move every
# window boundary in this notebook.
FLOOR_KM = float(npz[KEY_FLOOR])
_c = getattr(constants, "FLOOR_KM", None)
if _c is None:
    print("!! constants.FLOOR_KM not found -- check the attribute name")
else:
    assert np.isclose(FLOOR_KM, _c, rtol=1e-12), (
        f"archive floor {FLOOR_KM!r} != constants.FLOOR_KM {_c!r}: "
        "constants.py has drifted from the frozen data")
print(f"FLOOR_KM = {FLOOR_KM:.6g} km (from archive; agrees with constants.py)")

# ---- censoring orientation check ------------------------------------------
# If at_floor were inverted, `measured` would select exactly the wrong rows and
# every number downstream would be garbage that still looks entirely plausible
# -- the silent-success pattern this project has now hit four times. Week 12
# measured 12.26% censored, ~96% of them from the `surface` sampler. Both are
# reproduced here. The hard assert is on the fraction, which is name-
# independent; the sampler breakdown is printed for diagnosis.
frac_cens = float(censored.mean())
print(f"censored {100*frac_cens:.2f}%  (Week 12 measured 12.26%)")
if censored.any():
    lbl, cnt = np.unique(source[censored], return_counts=True)
    print("  censored rows by sampler: " +
          ", ".join(f"{a}={100*b/cnt.sum():.1f}%" for a, b in zip(lbl, cnt)))
assert 0.10 < frac_cens < 0.15, (
    f"censored fraction {frac_cens:.4f} is not Week 12's 12.26% -- at_floor "
    "may be inverted, or this is not the frozen w10 dataset")

# log_d_km is absent from the archive by design and is derived, never stored.
log_d_km = np.log10(np.maximum(d_km, FLOOR_KM))

# ---- features: the model's own view, not raw states -----------------------
import inspect
from features import build_features
print("build_features", inspect.signature(build_features))
_doc = (build_features.__doc__ or "").strip().splitlines()
print("  doc:", " ".join(l.strip() for l in _doc[:3]) if _doc else "(none)")

jacobi    = np.asarray(npz["jacobi"], dtype=float).ravel()
family_id = np.asarray(npz["family_id"]).ravel()
energy_id = np.asarray(npz["energy_id"]).ravel()

# family_levels is the one argument the archive does not name directly. Show
# every candidate rather than guessing silently.
for k in ("family", "family_id", "family_tags", "family_sizes"):
    a = np.asarray(npz[k]).ravel()
    print(f"  {k:14s} shape {str(a.shape):10s} dtype {str(a.dtype):10s} "
          f"head {a[:3]}")

# features.py checks `set(np.unique(fid)) - set(levels)`, i.e. SET MEMBERSHIP,
# and names columns f'family_{lv}'. So family_levels holds the VALUES appearing
# in family_id, not tags that family_id indexes into.
#
# My earlier check asserted family_id.max() < len(family_tags). It passed, and
# it was wrong: it verified an indexing relationship against a function that
# does membership. Passing for the wrong reason is the failure worth naming,
# because nothing about that assert looked suspicious.
#
# Column ORDER does not matter anywhere in this notebook: joint MI is invariant
# to permuting features, and P13.1 reads M1's stored predictions rather than
# re-running the model, so M1's own encoder ordering never has to be reproduced.
family_levels = np.unique(family_id)
print(f"  family_levels = {list(family_levels)} ({family_levels.size} families)")
if family_levels.size < 3:
    print(f"  note: only {family_levels.size} families here. That is a Week 15 "
          "feasibility fact, not something to act on now.")

_fx = build_features(state, jacobi, family_id, family_levels)
if isinstance(_fx, tuple):
    X, feat_names = _fx[0], list(_fx[1])
else:
    X, feat_names = _fx, [f"f{j}" for j in range(np.asarray(_fx).shape[1])]
X = np.asarray(X, dtype=float)

assert X.shape[0] == d_km.size == group_id.size, "row count mismatch"
assert state.ndim == 2, f"states is {state.shape}, expected (n, 6)"
print(f"rows {X.shape[0]}, states {state.shape}, features {X.shape[1]}, "
      f"measured {measured.sum()} ({100*measured.mean():.2f}%), "
      f"groups {len(np.unique(group_id))}")
print("feature names:", feat_names)

# ---- are any feature columns effectively discrete? ------------------------
# KSG assumes continuous coordinates. copula_transform breaks ties by ROW
# ORDER (argsort of argsort), so a categorical or one-hot column would get an
# arbitrary within-tie ranking and the MI estimate would depend on the order
# rows happen to sit in the archive. That is a correctness problem, not a
# cosmetic one, and it has to be caught here rather than explained later.
_nuniq = np.array([np.unique(X[:, j]).size for j in range(X.shape[1])])
_disc = [(feat_names[j], int(_nuniq[j])) for j in np.argsort(_nuniq)[:5]]
print("fewest distinct values: " + ", ".join(f"{a}={b}" for a, b in _disc))
_disc_found = {feat_names[j] for j in np.flatnonzero(_nuniq < 0.01 * X.shape[0])}
_disc_expected = {"C", "family_0", "family_1"}
if _disc_found:
    print(f"\neffectively discrete feature columns: {sorted(_disc_found)}")
if _disc_found == _disc_expected:
    print("  -> expected, and handled: these never enter a KSG call. The "
          "definitions cell conditions on G = (family_id, energy_id) and uses "
          "the exact chain rule I([X,G];Y) = I(G;Y) + I(X;Y|G), which is "
          "better than either dropping them or randomising tie-breaking.")
elif _disc_found:
    raise RuntimeError(
        f"discrete columns {sorted(_disc_found)} differ from the expected set "
        f"{sorted(_disc_expected)}. The conditioning in the definitions cell "
        "covers C and family only. A NEW discrete column would go into KSG "
        "with ties broken by archive row order, so the estimate would depend "
        "on storage order. Extend the conditioning before running further.")

# ---- probe the Week 12 predictions archive HERE, so both adapters get fixed
# in one pass instead of failing eight cells apart.
import glob
_cands = [PRED12] + sorted(glob.glob(
    os.path.join("..", "**", "predictions_wk12.npz"), recursive=True))
_found = next((p for p in _cands if os.path.exists(p)), None)
if _found is None:
    print("\n!! predictions_wk12.npz not found -- set PRED12 by hand in cell 1")
else:
    PRED12 = _found
    print("\npredictions archive:", PRED12)
    print("  keys:", list(np.load(PRED12, allow_pickle=True).files))
    print("  -> M1 is delta-independent, so P13.1 reads boundary__M1 and "
          "verifies it against the other three delta keys. The archive stores "
          "no row index; the held-out mask is regenerated from src/splits.py "
          "with the parameters in results_wk12.json and verified by "
          "reproducing M1's overall R2.")

# Present but deliberately unused this week: family / family_id / energy_id /
# C / jacobi (Week 15 cross-family transfer), gap / target_gap_km /
# family_max_gap_km (Risk 19), score_C / label_B (the classification arm).
# Week 15's split variables existing in the archive is good news for its
# feasibility; it is not a reason to start it.
"""),

code(r"""
w = mi.make_windows(d_km, measured, FLOOR_KM, mi.DEAD_ZONE_KM_W12)
for key in ("n_near", "n_far", "n_below_floor", "near_decades", "far_decades_realised",
            "far_hi_intended_km", "span_shortfall",
            "joint_credible_near", "joint_credible_far"):
    print(f"  {key:24s} {w[key]}")

if w["span_shortfall"]:
    print("\n!! far window falls short of the matched span. The copula transform "
          "still equalises the marginals, so the MI comparison holds, but h(Y) "
          "and Var(Y) differ between windows and the ceilings are not directly "
          "comparable. Report this.")
if not (w["joint_credible_near"] and w["joint_credible_far"]):
    print(f"\n!! a window has < {mi.MIN_N_JOINT} rows: joint KSG is not credible. "
          "Pre-registration falls back to Tier-1 per-feature MI only.")
"""),

code(r"""
# ---- target ties ----------------------------------------------------------
# Ties in the target break two things at once, silently: differential entropy
# is undefined for a distribution with atoms, and copula_transform ranks tied
# values by archive row order, so every KSG estimate would depend on how the
# rows happen to be stored.
#
# Two regimes have to be told apart, and they need opposite treatments.
#
#   CHANCE FLOAT COLLISIONS -- multiplicity 2, a handful of pairs. For ~9000
#   values over ~3 decades in float32 the expected number of collisions is of
#   order one, so this is the null expectation, not a defect. Dropping one row
#   of each pair costs nothing and invents nothing; jittering would invent
#   precision the archive does not have.
#
#   STRUCTURAL ATOMS -- a sampler placing many rows at one drawn offset, or a
#   clamp collapsing a range onto a point. These change what the target IS and
#   must never be deduplicated away quietly. The clamp case is already handled:
#   log10(max(d, FLOOR_KM)) would map every sub-floor row onto one value, and
#   make_windows now excludes and counts those instead.
#
# The threshold below comes from the collision model, not from what makes the
# data pass: chance collisions cannot produce multiplicity above 2, and cannot
# produce dozens of tied values at this n.
print(f"measured rows below the determination floor (excluded): "
      f"{w['n_below_floor']}")

in_windows = w["near"] | w["far"]          # sub-floor rows are already out
idx_w = np.flatnonzero(in_windows)
vals, cnt = np.unique(log_d_km[idx_w], return_counts=True)
tied_vals, max_mult = vals[cnt > 1], int(cnt.max())
n_tied = int(cnt[cnt > 1].sum())
print(f"in-window rows {idx_w.size}, distinct targets {vals.size}, "
      f"tied rows {n_tied}, max multiplicity {max_mult}")

if n_tied:
    if max_mult > 2 or tied_vals.size > 20:
        for j in np.argsort(-cnt)[:5]:
            if cnt[j] < 2:
                continue
            at = idx_w[log_d_km[idx_w] == vals[j]]
            lv, lc = np.unique(source[at], return_counts=True)
            print(f"  d = {10 ** vals[j]:.6g} km  x{cnt[j]}  "
                  f"groups={np.unique(group_id[at]).size}  "
                  f"samplers={dict(zip(lv.tolist(), lc.tolist()))}")
        raise RuntimeError(
            "ties are structural, not chance collisions: multiplicity above 2 "
            "or too many tied values. The target has genuine atoms. Decide how "
            "to handle them and log the decision before going further -- "
            "deduplicating would change what the analysis population is.")

    # Chance collisions. Keep one row per distinct target, chosen at RANDOM --
    # keeping the first would correlate the surviving row with archive order.
    keep = np.ones(d_km.size, dtype=bool)
    for vv in tied_vals:
        rows = idx_w[log_d_km[idx_w] == vv]
        drop = rng.permutation(rows)[1:]
        keep[drop] = False
        print(f"  d = {10 ** vv:.6g} km x{rows.size} across "
              f"{np.unique(group_id[rows]).size} group(s), samplers "
              f"{np.unique(source[rows]).tolist()} -> dropped {drop.size}")
    measured = measured & keep
    print(f"deduplicated: {int((~keep).sum())} row(s) dropped "
          f"({100 * (~keep).sum() / idx_w.size:.3f}% of in-window rows)")

    # Windows must be rebuilt on the deduplicated population, or every count
    # downstream would refer to the pre-dedup mask.
    w = mi.make_windows(d_km, measured, FLOOR_KM, mi.DEAD_ZONE_KM_W12)
    print(f"rebuilt windows: n_near {w['n_near']}, n_far {w['n_far']}, "
          f"n_below_floor {w['n_below_floor']}")

for lbl in ("near", "far"):
    yy = log_d_km[w[lbl]]
    assert np.unique(yy).size == yy.size, f"ties remain in {lbl} after dedup"
    print(f"  {lbl}: no ties, {yy.size} distinct values")
"""),

md(r"""
## Definitions — no real numbers computed here

Feature blocks, the conditioning variable, and the estimator pipeline. The
calibration cell below depends on these, so they must be defined first.
"""),

code(r"""
# ---- feature blocks -------------------------------------------------------
# features.py returns [x y z vx vy vz, C, one-hot(family)]. TWO of those blocks
# are categorical, not one: C has three distinct values (three energy levels),
# and family has two. copula_transform breaks ties by row order, so feeding
# either to KSG would make the estimate depend on the archive's row ordering.
# Neither ever enters a KSG call.
CONT_NAMES = ["x", "y", "z", "vx", "vy", "vz"]
CONT_COLS = [feat_names.index(nm) for nm in CONT_NAMES]
DISC_COLS = [j for j in range(X.shape[1]) if j not in CONT_COLS]
print(f"continuous ({len(CONT_COLS)}): {CONT_NAMES}")
print(f"categorical ({len(DISC_COLS)}): {[feat_names[j] for j in DISC_COLS]}")

# The conditioning variable is the (family, energy) pair -- 2 x 3 = 6 cells,
# which is exactly the six family_tags. This is only EXACT if C is constant
# within energy_id and the map energy_id <-> C is one-to-one; otherwise
# conditioning on G would not capture C and the chain rule below would be an
# approximation rather than an identity.
for e in np.unique(energy_id):
    assert np.unique(np.asarray(npz["C"], float).ravel()[energy_id == e]).size == 1, \
        f"C is not constant within energy_id={e}; conditioning on G is not exact"
assert np.unique(np.asarray(npz["C"], float).ravel()).size == np.unique(energy_id).size, \
    "energy_id and C are not one-to-one; conditioning on G would lose C"

G_all = np.asarray([f"{a}_{b}" for a, b in zip(family_id, energy_id)])
cells, cell_sizes = np.unique(G_all, return_counts=True)
print(f"conditioning cells ({cells.size}): " +
      ", ".join(f"{c}:{n}" for c, n in zip(cells, cell_sizes)))
X_cont = X[:, CONT_COLS]


def mi_decomposed(Xc, G, y_raw, rng, k=K_MAIN):
    '''I([X_cont, G]; Y) = I(G;Y) + I(X_cont; Y | G), in nats.

    Exact chain rule, no approximation, and no categorical coordinate ever
    enters a KSG call:

        I(G;Y)   = h(Y) - sum_g p_g h(Y|G=g)        1-D KL estimator
        I(X;Y|G) = sum_g p_g I(X_cont; Y | G=g)     plain continuous KSG

    The first term is invariant to the space Y is measured in: both entropies
    pick up the same Jacobian and it cancels in the difference, so computing it
    on log10 d rather than d is the same number, not an approximation.

    Strictly more informative than dropping the categorical columns: it
    separates how much is "which family and energy am I at" from how much is
    "where in phase space am I".'''
    i_g = mi.kl_entropy(y_raw, k=k)
    i_cond = 0.0
    per_cell = {}
    for g in np.unique(G):
        m = (G == g)
        p = float(m.mean())
        yg = y_raw[m]
        i_g -= p * mi.kl_entropy(yg, k=k)
        # Estimate whenever it is feasible at all, and report cell sizes
        # separately. A hard MIN_N_JOINT gate here returns NaN that propagates
        # silently into every downstream comparison. It also conflates two
        # different questions: whether a MAGNITUDE is quotable (needs large n)
        # and whether a point estimate sits inside its own permutation null
        # (internally valid at any n, because the null is built the same way on
        # the same rows). window_analysis flags the first; this does the second.
        v = (mi.ksg_mi(mi.copula_transform(Xc[m]), mi.copula_transform(yg), k=k)
             if m.sum() > k + 1 else np.nan)
        per_cell[str(g)] = {"n": int(m.sum()), "p": p, "mi_nats": float(v)}
        i_cond += p * v
    return {"I_G_Y": float(i_g), "I_cond": float(i_cond),
            "I_total": float(i_g + i_cond), "per_cell": per_cell}


def window_analysis(idx, label, rng, k=K_MAIN, k_sweep=K_SWEEP, n_perm=N_PERM):
    '''Full pipeline for one window, given row indices into the frozen arrays.

    The ceiling takes h(Y) and Var(Y) from RAW log10 d, the space M1 R2 lives
    in, while the MI comes from copula coordinates. Mixing the two is correct
    only because MI is reparameterisation-invariant.

    No bootstrap: see the amendment in cell 1.'''
    Xc, G, y = X_cont[idx], G_all[idx], log_d_km[idx]
    n = idx.size
    cell_n = np.array([(G == g).sum() for g in np.unique(G)])
    out = {"label": label, "n": int(n), "dim_cont": int(Xc.shape[1]),
           "n_cells": int(np.unique(G).size), "min_cell_n": int(cell_n.min()),
           "magnitude_credible": bool(cell_n.min() >= mi.MIN_N_JOINT)}
    if not out["magnitude_credible"]:
        print(f"  [{label}] smallest cell has {cell_n.min()} rows "
              f"(< MIN_N_JOINT={mi.MIN_N_JOINT}): the null comparison is still "
              "valid, the MAGNITUDE is not quotable.")
    base = mi_decomposed(Xc, G, y, rng, k=k)
    out.update(base)
    out["mi_bits"] = base["I_total"] / np.log(2.0)
    out["by_k"] = {str(kk): mi_decomposed(Xc, G, y, rng, k=kk)["I_total"]
                   for kk in k_sweep}
    null = np.array([mi_decomposed(Xc, G, y[rng.permutation(n)], rng, k=k)["I_total"]
                     for _ in range(n_perm)])
    out["null_mean"] = float(null.mean())
    out["null_p95"] = float(np.percentile(null, 95))
    out["above_null"] = bool(base["I_total"] > out["null_p95"])
    out["ceiling"] = mi.entropy_power_ceiling(base["I_total"], y, k=k)
    return out


# ---- runtime estimate at the REAL configuration ---------------------------
# One mi_decomposed call is n_cells KSG calls at ~n/n_cells rows in 6
# dimensions -- not one call at n rows in 9. Measure that unit, then count.
n_match = min(w["n_near"], w["n_far"])
n_cells = int(cells.size)
_m = n_match // n_cells
_r = np.random.default_rng(0)
_Xs = mi.copula_transform(_r.normal(size=(_m, len(CONT_COLS))))
_ys = mi.copula_transform(_r.normal(size=_m))
_t = time.time(); mi.ksg_mi(_Xs, _ys, k=K_MAIN); per_ksg = time.time() - _t

# 2 windows x (1 base + |K_SWEEP| + N_PERM), + far_test (2 + 60),
# + 2 thinned (2 + 60), + |CAL_GRID| calibration cases.
n_dec = 2 * (1 + len(K_SWEEP) + N_PERM) + 62 + 2 * 62 + len(CAL_GRID)
n_ksg = n_dec * n_cells + 2 * len(CONT_COLS)     # + Tier-1 1-D calls
print(f"n per window {n_match}, {n_cells} cells -> ~{_m} rows/cell, "
      f"dim {len(CONT_COLS)}, {per_ksg:.3f} s per KSG call")
print(f"~{n_dec} decomposed calls = ~{n_ksg} KSG calls  ->  "
      f"~{n_ksg * per_ksg / 60:.0f} min single-core")
if n_ksg * per_ksg > 3 * 3600:
    print("!! over three hours. Cut N_PERM to 100 (a noisier null p95, not a "
          "biased one), or run on Titan -- the null loop is embarrassingly "
          "parallel. Do NOT cap n: the cells are already only ~n/6 rows each.")
"""),

md(r"""
## Calibration gate — run and read this before any real number

Measures how far the ceiling sits from a *known* R² at the realised window
size, the real continuous dimension, and the real cell structure, using the
same pipeline the arbiter uses.

Synthetic cases are jointly Gaussian, where the ceiling formula is exact, so
any gap is pure estimator slack. **This bounds estimator slack, not model
mismatch** — the real features are neither Gaussian nor linearly related to the
target, and this cell says nothing about that. That residual is what MARGIN is
for.

The numbers that matter are the **interior leave-one-out error** and the
**calibrated range**. The map is inverted to correct the observed ceiling, so
it has to be both smooth enough to interpolate and wide enough to bracket
wherever the real ceiling lands. If the LOO error exceeds half of MARGIN the
map cannot carry the verdict; if the real ceiling falls outside the calibrated
range the cell raises rather than extrapolating. Neither failure is fixed by
widening MARGIN.
"""),

code(r"""
def _cal_case(n, dim, r2, sizes, f_group, rng):
    '''Synthetic case with a KNOWN total R^2, matching the analysis structure:
    a categorical with the observed cell proportions plus `dim` continuous
    features, with f_group of the explained variance carried by the
    categorical. g and s are each standardised and independent, so Var(y) = 1
    by construction and the best predictor from (X, G) achieves exactly r2.'''
    p = np.asarray(sizes, float); p = p / p.sum()
    G = rng.choice(len(p), size=n, p=p)
    mu = rng.normal(size=len(p))
    g = mu[G]
    g = (g - g.mean()) / g.std() if g.std() > 0 else g
    Xg = rng.normal(size=(n, dim)); wt = rng.normal(size=dim)
    s = Xg @ wt; s = (s - s.mean()) / s.std()
    y = (np.sqrt(f_group * r2) * g + np.sqrt((1.0 - f_group) * r2) * s
         + np.sqrt(1.0 - r2) * rng.normal(size=n))
    return Xg, G.astype(str), y


# Replicates matter most where we expect to operate: the ceiling SD is LARGEST
# at low true R^2 (~0.016) and smallest at high (~0.003), and the near field is
# a low-R^2 regime. One draw per grid point is not enough to invert the map.
cal = []
for r2_true in CAL_GRID:
    cs = []
    for rep in range(CAL_REPS):
        r = np.random.default_rng(SEED + 977 * rep + int(round(r2_true * 1000)))
        Xg, Gg, yg = _cal_case(n_match, len(CONT_COLS), r2_true, cell_sizes,
                               CAL_GROUP_FRAC, r)
        I = mi_decomposed(Xg, Gg, yg, r, k=K_MAIN)["I_total"]
        cs.append(mi.entropy_power_ceiling(I, yg, k=K_MAIN)["r2_ceiling"])
    cs = np.asarray(cs)
    cal.append({"r2_true": float(r2_true), "ceiling_mean": float(cs.mean()),
                "ceiling_sd": float(cs.std(ddof=1)),
                "slack": float(cs.mean() - r2_true)})
    print(f"  R2_true {r2_true:.2f}   ceiling {cs.mean():+.4f} +/- {cs.std(ddof=1):.4f}"
          f"   slack {cs.mean() - r2_true:+.4f}")

cal_true = np.array([z["r2_true"] for z in cal])
cal_obs = np.array([z["ceiling_mean"] for z in cal])
cal_sd = np.array([z["ceiling_sd"] for z in cal])
assert np.all(np.diff(cal_obs) > 0), (
    "the calibration map is not monotone in the observed ceiling; it cannot be "
    "inverted. Raise CAL_REPS or widen the grid spacing.")


def corrected_ceiling(obs, z=0.0, label=""):
    '''Invert the calibration map: observed ceiling -> bias-corrected ceiling.

    The under-read is strongly dependent on where you are on the curve (-0.003
    at true R2 0.05, -0.101 at 0.75), so a single global worst-case correction
    applies the error from an operating point we are not at. This inverts
    locally instead.

    REFUSES TO EXTRAPOLATE. Outside the calibrated range, clamping to an
    endpoint would silently invent a number; the leave-one-out check below
    measures endpoint extrapolation error at 0.10-0.15 against 0.003-0.011 in
    the interior.

    z shifts the observation by z local SDs before inverting, so the caller
    picks which way to err:
      z = +1.645  errs HIGH -- used by P13.3, because a too-low ceiling is what
                  would falsely declare the informational verdict we assert.
      z = -1.645  errs LOW  -- used by P13.1, so the validity gate cannot be
                  passed by an inflated ceiling.
    '''
    if not (cal_obs.min() <= obs <= cal_obs.max()):
        raise ValueError(
            f"observed ceiling {obs:.4f} {label} is outside the calibrated "
            f"range [{cal_obs.min():.4f}, {cal_obs.max():.4f}]. Extend CAL_GRID "
            "to bracket it and re-run the calibration. Do not extrapolate.")
    o = obs + z * float(np.interp(obs, cal_obs, cal_sd))
    o = float(np.clip(o, cal_obs.min(), cal_obs.max()))
    return float(np.interp(o, cal_obs, cal_true))


# Leave-one-out on the INTERIOR points only. LOO at a grid endpoint forces the
# interpolator to clamp, which measures the boundary, not the method.
loo = []
for i in range(1, len(cal_true) - 1):
    keep = [j for j in range(len(cal_true)) if j != i]
    loo.append(float(np.interp(cal_obs[i], cal_obs[keep], cal_true[keep]) - cal_true[i]))
print("interior leave-one-out error: " + "  ".join(f"{e:+.4f}" for e in loo))
worst_loo = max(abs(e) for e in loo)
print(f"worst interior LOO {worst_loo:.4f}   MARGIN {MARGIN:.2f}")
if worst_loo > 0.5 * MARGIN:
    print("!! the inverted map is not accurate enough to carry the verdict. "
          "Densify CAL_GRID or raise CAL_REPS. Do NOT widen MARGIN.")
print(f"cell sizes used: {sorted(set(cell_sizes.tolist()))} -- the synthetic "
      "cases draw from exactly these proportions, so cell structure is MATCHED, "
      "not approximated.")
print("MARGIN covers what is NOT matched: the synthetic target is Gaussian and "
      "the dependence sits in a 1-D linear projection. The first of those is "
      "not a small caveat -- the entropy-power term N/Var is ~1 for a Gaussian "
      "target and ~0.70 for the near-log-uniform real one, and that single "
      "mismatch is what makes the ceiling arm inadmissible downstream.")
"""),

md(r"""
## Everything below touches real data

The pre-registration is closed. Numbers from here on get reported whichever way
they come out.
"""),

code(r"""
near_i = np.flatnonzero(w["near"])
far_i = np.flatnonzero(w["far"])
near_i, far_i = mi.match_window_sizes(near_i, far_i, rng)
print(f"matched to n = {near_i.size} per window")

res_near = window_analysis(near_i, "near", rng)
res_far = window_analysis(far_i, "far", rng)

for r in (res_near, res_far):
    print(f"[{r['label']}] n={r['n']}  I_total {r['mi_bits']:.4f} bits "
          f"({r['I_total']:.4f} nats)")
    print(f"    I(G;Y) {r['I_G_Y']:.4f}   I(X;Y|G) {r['I_cond']:.4f} nats")
    print("    per cell: " + "  ".join(
        f"{c}(n={d['n']}):{d['mi_nats']:.3f}" for c, d in r["per_cell"].items()))
    print(f"    null p95 {r['null_p95']:.4f} -> above null: {r['above_null']}")
    print(f"    ceiling {r['ceiling']['r2_ceiling']:.4f}  "
          f"(N/Var {r['ceiling']['entropy_power_over_var']:.4f})")
    print("    k sweep: " + "  ".join(f"k={a}:{b:.4f}" for a, b in r["by_k"].items()))
    if r["ceiling"]["ratio_exceeds_theory"]:
        print("    !! N/Var > 1 is impossible in theory; the KL entropy estimate "
              "is off and this ceiling is void.")
    if any(np.isnan(d["mi_nats"]) for d in r["per_cell"].values()):
        print(f"    !! a cell has < {mi.MIN_N_JOINT} rows; its conditional term "
              "is NaN and I_total is unusable. Report it, do not drop the cell.")
"""),

code(r"""
# ---- Tier 1: which features carry the information -------------------------
# Continuous features via KSG; the categorical block via the same KL difference
# used above, because it has no business inside a k-NN estimator.
def tier1(idx):
    Xc, G, y = X_cont[idx], G_all[idx], log_d_km[idx]
    Xk, yk = mi.copula_transform(Xc), mi.copula_transform(y)
    d = {CONT_NAMES[j]: mi.ksg_mi(Xk[:, [j]], yk, k=K_MAIN) / np.log(2.0)
         for j in range(len(CONT_COLS))}
    h = mi.kl_entropy(y, k=K_MAIN)
    d["family x energy"] = float(
        (h - sum((G == g).mean() * mi.kl_entropy(y[G == g], k=K_MAIN)
                 for g in np.unique(G))) / np.log(2.0))
    return d

t1_near, t1_far = tier1(near_i), tier1(far_i)
print(f"{'feature':>18} {'near (bits)':>12} {'far (bits)':>12}")
for f in sorted(t1_far, key=lambda z: -t1_far[z]):
    print(f"{f:>18} {t1_near[f]:12.4f} {t1_far[f]:12.4f}")
print("Tier 1 stays credible when the joint estimate does not. If the joint "
      "numbers sit at the null while a feature here carries real 1-D signal, "
      "the joint estimator is failing -- that is not an informational verdict.")
"""),

code(r"""
# ---- Tier 1 permutation null ----------------------------------------------
# The table above has no reference scale, so "near-field features sit near
# zero" is not yet a claim. The joint null does not transfer: it is the null
# for a 6-dimensional estimate under a 6-cell conditioning, not for a 1-D
# marginal.
#
# MULTIPLICITY. Six continuous features times two windows is twelve
# comparisons; at a per-test 95th percentile you expect roughly one spurious
# hit, which is exactly how a 0.0099-bit reading turns into a finding. The
# threshold used below is the MAX-STATISTIC null: per permutation take the
# largest MI across the six features, and read the 95th percentile of that
# maximum. This controls the family-wise error rate exactly and assumes
# nothing about independence between features -- they are not independent,
# since all six are scored against one shuffled y.
#
# The continuous and categorical rows need separate nulls: the continuous ones
# are KSG on copula coordinates, the categorical one is a KL entropy
# difference on RAW log10 d.
def tier1_null(idx, n_perm=N_PERM, k=K_MAIN):
    Xc, G, y = X_cont[idx], G_all[idx], log_d_km[idx]
    n, ln2 = idx.size, np.log(2.0)
    Xk = mi.copula_transform(Xc)
    yk = mi.copula_transform(y)
    h_all = mi.kl_entropy(y, k=k)      # invariant under permutation of y
    cells_here = np.unique(G)
    per_feat = np.empty((n_perm, len(CONT_NAMES)))
    cat = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n)
        ykp, yp = yk[p], y[p]
        for j in range(len(CONT_NAMES)):
            per_feat[b, j] = mi.ksg_mi(Xk[:, [j]], ykp, k=k) / ln2
        cat[b] = (h_all - sum((G == g).mean() * mi.kl_entropy(yp[G == g], k=k)
                              for g in cells_here)) / ln2
    return per_feat, cat


t1_null = {}
for lbl, idx, obs in (("near", near_i, t1_near), ("far", far_i, t1_far)):
    pf, cat = tier1_null(idx)
    fam = float(np.percentile(pf.max(axis=1), 95))      # family-wise
    per = float(np.percentile(pf, 95))                  # per-test, for contrast
    cat95 = float(np.percentile(cat, 95))
    t1_null[lbl] = {"family_wise_95": fam, "per_test_95": per,
                    "categorical_95": cat95}
    print(f"[{lbl}] family-wise threshold {fam:.4f} bits "
          f"(per-test would be {per:.4f} -- the gap is the multiplicity cost)")
    print(f"      categorical threshold {cat95:.4f} bits")
    for nm in CONT_NAMES:
        flag = "SIGNIFICANT" if obs[nm] > fam else "-"
        print(f"   {nm:>16} {obs[nm]:+.4f}   {flag}")
    nm = "family x energy"
    print(f"   {nm:>16} {obs[nm]:+.4f}   "
          f"{'SIGNIFICANT' if obs[nm] > cat95 else '-'}")

# Sanity check on the estimates rather than on the data. Data processing gives
# I([X,G];Y) >= I(X_j;Y) as a theorem, so a marginal above the joint means the
# two estimates disagree by more than their own noise. Reported, not asserted:
# near the null both are consistent with zero and a small violation is
# expected, but it bounds how much any single Tier-1 reading can carry.
for lbl, obs, res in (("near", t1_near, res_near), ("far", t1_far, res_far)):
    worst = max(obs[nm] for nm in CONT_NAMES) - res["mi_bits"]
    print(f"[{lbl}] largest marginal minus joint: {worst:+.4f} bits "
          f"(negative respects data processing; positive means the estimates "
          f"disagree by that much, against a joint null p95 of "
          f"{res['null_p95'] / np.log(2.0):.4f} bits)")
"""),

code(r"""
# ---- M1's achieved R2, and the P13.1 validity gate ------------------------
res12 = json.load(open(RESULTS12))
print("results_wk12.json keys:", list(res12.keys()))
print("  recorded week11_m1_r2:", res12.get("week11_m1_r2"))
print("  gate:", res12.get("gate"), " verdict:", res12.get("verdict"))

p12 = np.load(PRED12, allow_pickle=True)
# M1 carries no delta dependence, so all four delta keys should be identical.
# Verify rather than assume -- if they differ, the archive is not what we think.
yhat_M1 = np.asarray(p12["boundary__M1"], dtype=float)
for kk in ("output_floor__M1", "boundary_over3__M1", "boundary_x3__M1"):
    assert np.allclose(np.asarray(p12[kk], float), yhat_M1), \
        f"{kk} differs from boundary__M1 -- M1 is not delta-independent"
n_seeds, n_test = yhat_M1.shape
print(f"M1 predictions: {n_seeds} seeds x {n_test} held-out rows")
print("  CHECK res12['scorecard'] for budget_ltd / dead_seeds on M1 before "
      "quoting anything below.")

# The archive stores no row index. Recover the held-out set, then VERIFY the
# recovery by reproducing Week 12's recorded overall R2 -- if the rows were
# wrong, that number would not come back.
# results_wk12.json records the split PARAMETERS, not the indices, so the mask
# is regenerated from src/splits.py with exactly those parameters. Note the
# split is by GROUP, so the row fraction will not be exactly frac_test.
import splits
_sp = res12["split"]
print(f"  regenerating grouped_split with {_sp}")
_masks = splits.grouped_split(group_id, frac_test=_sp["frac_test"],
                              frac_val=_sp["frac_val"], seed=_sp["seed"])
test_idx = np.flatnonzero(_masks["test"])
assert test_idx.size == n_test, (
    f"regenerated split gives {test_idx.size} held-out rows but the archive "
    f"has {n_test} prediction columns -- parameters or splitter have changed")


def achieved_r2(rows):
    '''Mean of PER-SEED R^2, not the R^2 of the seed-averaged prediction.
    Ensembling would inflate the number, which makes the informational verdict
    easier to declare -- the wrong direction. This also matches how Week 12
    reported 0.6288 +/- 0.0214.'''
    sel = np.isin(test_idx, rows)
    if sel.sum() < 2:
        return float("nan"), float("nan"), int(sel.sum())
    yt = log_d_km[test_idx[sel]]
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2s = [1.0 - float(np.sum((yt - yhat_M1[s][sel]) ** 2)) / ss
           for s in range(n_seeds)]
    return float(np.mean(r2s)), float(np.std(r2s)), int(sel.sum())


r2_chk, sd_chk, n_chk = achieved_r2(np.flatnonzero(measured))
print(f"reconstructed overall M1 R2 on measured held-out rows: "
      f"{r2_chk:.4f} +/- {sd_chk:.4f} (n={n_chk}); Week 12 recorded 0.6288")
assert abs(r2_chk - 0.6288) < 0.02, (
    "split reconstruction is WRONG: the overall R2 does not reproduce Week 12. "
    "Every number below would be aligned to the wrong rows.")

r2_far, sd_far, n_ft = achieved_r2(np.flatnonzero(w["far"]))
r2_near, sd_near, n_nt = achieved_r2(np.flatnonzero(w["near"]))
print(f"achieved R2  far {r2_far:+.4f}+/-{sd_far:.4f} (n={n_ft})   "
      f"near {r2_near:+.4f}+/-{sd_near:.4f} (n={n_nt})")

# ---- ceiling admissibility -------------------------------------------------
# The Shannon bound R2 <= 1 - (N/Var)*exp(-2I) has an INFORMATION-FREE FLOOR:
# at I = 0 it returns 1 - N/Var, which is zero only for a Gaussian target.
# log_d_km here is near log-uniform -- 12/(2*pi*e) = 0.7026 is the uniform
# value for N/Var -- so the floor sits near 0.30. Below that the bound is
# rigorous but uninformative: it cannot fall further no matter how little
# information the features carry.
#
# The calibration did not surface this. It matched n, dimension and cell
# structure but used a GAUSSIAN target, where the floor is ~0, so the offset
# never appeared and the inverted map does not account for it. That is a gap
# in the calibration design, not a property of the data, and it means the
# calibration correction is not applicable to a window whose target marginal
# is this far from Gaussian.
for r in (res_near, res_far):
    fl = 1.0 - r["ceiling"]["entropy_power_over_var"]
    r["ceiling_floor"] = float(fl)
    r["ceiling_excess"] = float(r["ceiling"]["r2_ceiling"] - fl)
    print(f"[{r['label']:8s}] ceiling {r['ceiling']['r2_ceiling']:.4f}   "
          f"info-free floor {fl:.4f}   excess {r['ceiling_excess']:+.4f}")

# The ceiling can only decide P13.3 if its own floor lies BELOW the decision
# threshold. Otherwise it is structurally incapable of returning the
# informational verdict and the arm is void. This criterion comes from the
# bound itself, not from what makes the data pass.
CEIL_ADMISSIBLE = bool(res_near["ceiling_floor"] <= r2_near + MARGIN)
print(f"\nceiling arm admissible for P13.3: {CEIL_ADMISSIBLE}  "
      f"(floor {res_near['ceiling_floor']:.4f} vs threshold "
      f"{r2_near + MARGIN:.4f})")
if not CEIL_ADMISSIBLE:
    print("  -> P13.3's ceiling comparison is VOID. The bound's floor alone "
          "exceeds the threshold, so it could not have returned "
          "'informational' whatever the MI was. Report it as void; do not "
          "report the number it would have produced.")

# ---- P13.1 validity gate --------------------------------------------------
# The ceiling-based form of this gate is VOID, for the same reason leg B is.
# The bound's information-free floor 1 - N/Var is ~0.27 in the far window, so
# the far ceiling is inflated by that amount and the gate would be passed
# almost automatically -- easier, not harder, which is the wrong direction for
# a safety catch. Recorded as an instrument failure, not as a result.
#
# A same-rows version is also not available: 6 conditioning cells over the
# held-out far rows leaves far too few per cell for a quotable magnitude.
far_test = np.intersect1d(np.flatnonzero(w["far"]), test_idx)
print(f"held-out far rows {far_test.size} over {cells.size} cells = "
      f"~{far_test.size // max(cells.size, 1)} per cell, against "
      f"MIN_N_JOINT={mi.MIN_N_JOINT}: no same-rows ceiling is quotable.")
print(f"far-window ceiling info-free floor: {res_far['ceiling_floor']:.4f}")

# Reverting to the handoff's own formulation, which the ceiling was only ever
# an implementation of: run the same estimator on far-field rows as a control,
# because "MI is small" means nothing without one. The control is the far
# window in the main analysis -- matched n, matched dimension, matched cells.
P13_1 = bool(res_far["above_null"])
print(f"P13.1  far-field control: I_far {res_far['I_total']:.4f} nats vs its "
      f"own null p95 {res_far['null_p95']:.4f}  ->  "
      f"{'PASS' if P13_1 else 'FAIL - TEST VOID'}")

# Gaussian-equivalent R2 sidesteps the entropy-power floor by assuming N/Var=1.
# It is an INTERPRETATION under a stated assumption, not a rigorous bound for
# a non-Gaussian target -- but it is the standard way MI is read in R2 terms
# and it is quotable, which the ceiling is not.
def gauss_equiv_r2(nats):
    return float(1.0 - np.exp(-2.0 * max(nats, 0.0)))

print(f"\nGaussian-equivalent R2 (assumes joint Gaussianity; stated, not proved):")
print(f"  near, bounded by its null p95 {res_near['null_p95']:.4f} nats  ->  "
      f"<= {gauss_equiv_r2(res_near['null_p95']):.4f}")
print(f"  far,  at its point estimate {res_far['I_total']:.4f} nats       ->  "
      f"~  {gauss_equiv_r2(res_far['I_total']):.4f}")
"""),

code(r"""
# ---- group-thinned robustness ---------------------------------------------
# KSG assumes i.i.d. rows. One row per group removes the dependence entirely at
# the cost of n; a large shift means the full estimates are inflated by
# within-group repetition.
thin = mi.thin_by_group(group_id, rng)
thin_near = np.intersect1d(near_i, thin)
thin_far = np.intersect1d(far_i, thin)
thin_near, thin_far = mi.match_window_sizes(thin_near, thin_far, rng)
print(f"thinned to n = {thin_near.size} per window "
      f"({100 * thin_near.size / max(near_i.size, 1):.0f}% retained)")

thin_res = {}
if thin_near.size > 0:
    for lbl, idx in (("near_thinned", thin_near), ("far_thinned", thin_far)):
        thin_res[lbl] = window_analysis(idx, lbl, rng, k_sweep=(K_MAIN,), n_perm=60)
        print(f"  {lbl:14s} {thin_res[lbl]['mi_bits']:.4f} bits   "
              f"ceiling {thin_res[lbl]['ceiling']['r2_ceiling']:.4f}")
    print("  Thinning removes within-group dependence entirely. It matters "
          "most for the FAR window, where the positive finding is and where "
          "dependence would inflate; for the near window it cuts the harmless "
          "way, since dependence inflates MI and the near field is at zero.")
else:
    print("!! thinning leaves no rows. NOT RUNNABLE, not a null result.")
"""),

code(r"""
# ---- verdicts -------------------------------------------------------------
ceil_near = res_near["ceiling"]["r2_ceiling"]
P13_2 = bool(res_near["I_total"] < res_far["I_total"])

# P13.3 rests on two independent legs. Leg A is the permutation null, which was
# pre-registered as "the strongest form of the informational verdict" and does
# not use the Shannon bound at all. Leg B is the ceiling comparison, which is
# void whenever the bound's information-free floor exceeds the threshold.
LEG_A = bool((not res_near["above_null"]) and res_far["above_null"])
if CEIL_ADMISSIBLE:
    adj_near = corrected_ceiling(ceil_near, z=+1.645, label="(near)")
    LEG_B = bool(adj_near <= r2_near + MARGIN)
else:
    adj_near, LEG_B = None, None
P13_3_informational = LEG_A if not CEIL_ADMISSIBLE else bool(LEG_A and LEG_B)

print(f"P13.1  validity gate ........ {'PASS' if P13_1 else 'FAIL -> VOID'}")
print(f"P13.2  I_near < I_far ....... {'CONFIRMED' if P13_2 else 'FALSIFIED'} "
      f"({res_near['mi_bits']:.4f} vs {res_far['mi_bits']:.4f} bits)")
if P13_1:
    print(f"P13.3  leg A (permutation null): near inside its own null = "
          f"{not res_near['above_null']}, far above its null = "
          f"{res_far['above_null']}  ->  {'MET' if LEG_A else 'NOT MET'}")
    if CEIL_ADMISSIBLE:
        print(f"       leg B (ceiling): {ceil_near:.4f} -> corrected "
              f"{adj_near:.4f} vs achieved {r2_near:.4f} + {MARGIN:.2f} "
              f"-> {'MET' if LEG_B else 'NOT MET'}")
    else:
        print(f"       leg B (ceiling): VOID -- the bound's information-free "
              f"floor {res_near['ceiling_floor']:.4f} exceeds the threshold "
              f"{r2_near + MARGIN:.4f}, so it could not have returned the "
              f"informational verdict whatever the MI was.")
    print("       VERDICT: " + (
        "INFORMATIONAL - the features do not contain the information. The dead "
        "zone is a characterised limitation of this feature representation, "
        "not of the architecture, and the lever is better-determined surfaces. "
        "This is the strong form of the central claim."
        if P13_3_informational else
        "INFORMATION PRESENT - the features carry more about d than M1 "
        "extracts. Weeks 14 and 16 are cut, so this is not reopened; it costs "
        "one sentence in Limitations, that we did not establish whether a "
        "different architecture could reach it. It does NOT separate capacity "
        "from data, which would need a learning curve."))
    if not res_near["above_null"]:
        print("       (near-field MI sits inside its own permutation null - "
              "the strongest form of the informational verdict)")
else:
    print("P13.3  not evaluated: P13.1 failed, test is void.")
"""),

code(r"""
payload = {
    "week": 13, "seed": SEED, "margin": MARGIN, "k_main": K_MAIN,
    "n_perm": N_PERM,
    "amendment": ("2026-08-28: bootstrap removed from the arbiter (invalid for "
                  "k-NN estimators, fixed +0.19 nat offset); ceiling read from "
                  "the point estimate with the calibrated under-read added "
                  "back; calibration moved to the real 6-dim / 6-cell "
                  "configuration. Decided from synthetic data before any "
                  "window statistic was computed."),
    "features": {"continuous": CONT_NAMES,
                 "categorical": [feat_names[j] for j in DISC_COLS],
                 "conditioning": "G = (family_id, energy_id), 6 cells",
                 "decomposition": "I([X,G];Y) = I(G;Y) + I(X;Y|G)"},
    "windows": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in w.items() if k not in ("near", "far")},
    "calibration": cal,
    "calibration_map": {"true": cal_true.tolist(), "obs": cal_obs.tolist(),
                        "sd": cal_sd.tolist(), "interior_loo": loo},
    "tier1": {"near": t1_near, "far": t1_far, "null": t1_null},
    "near": res_near, "far": res_far,
    "gaussian_equiv_r2": {"near_bound": gauss_equiv_r2(res_near["null_p95"]),
                          "far_point": gauss_equiv_r2(res_far["I_total"])},
    "thinned": thin_res,
    "achieved_r2": {"near": r2_near, "near_sd": sd_near, "n_near_test": n_nt,
                    "far": r2_far, "far_sd": sd_far, "n_far_test": n_ft,
                    "overall_check": r2_chk},
    "ceiling_admissible": CEIL_ADMISSIBLE,
    "verdicts": {"P13_1_valid": P13_1,
                 "P13_1_form": "far-field permutation control; ceiling form void", "P13_2_direction": P13_2,
                 "P13_3_leg_A_null": LEG_A, "P13_3_leg_B_ceiling": LEG_B,
                 "P13_3_informational": (P13_3_informational if P13_1 else None)},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(payload, fh, indent=2)
print("wrote", OUT)
"""),

]


def build():
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def sources(nb):
    """Cell type + source only. Outputs and execution counts are what running
    the notebook legitimately changes, and must not read as divergence."""
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if the notebook has diverged")
    args = ap.parse_args()

    nb = build()
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)

    if os.path.exists(NB_PATH) and not args.force:
        with open(NB_PATH) as fh:
            existing = json.load(fh)
        new_src, old_src = sources(nb), sources(existing)
        if new_src == old_src:
            print(f"{NB_PATH}: identical, no write")
            return 0
        print(f"{NB_PATH}: DIVERGED from mknb13.py")
        print(f"  builder has {len(new_src)} cells, file has {len(old_src)}")
        for i, (a, b) in enumerate(zip(old_src, new_src)):
            if a == b:
                continue
            print(f"  first differing cell: {i} ({a[0]})")
            # Report the first differing LINE, not the first line of the cell --
            # those are usually identical and the message is then useless.
            al, bl = a[1].splitlines(), b[1].splitlines()
            for j in range(max(len(al), len(bl))):
                x = al[j] if j < len(al) else "<absent>"
                y = bl[j] if j < len(bl) else "<absent>"
                if x != y:
                    print(f"    line {j}")
                    print(f"      file:    {x}")
                    print(f"      builder: {y}")
                    break
            break
        print("  Hand-edits belong back in mknb13.py. --force to overwrite.")
        return 1

    with open(NB_PATH, "w") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
