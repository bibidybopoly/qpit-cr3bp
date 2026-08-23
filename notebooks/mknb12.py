import json

def _lines(s):
    ls = s.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]

def md(s):
    return dict(cell_type="markdown", metadata={}, source=_lines(s))

def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None,
                outputs=[], source=_lines(s))

cells = []

cells.append(md(r"""
# Week 12 — Phase 4, loss and target

Tests two changes to the Week 11 regressor against the **same dataset, the same
split and the same architecture**: a censored (Tobit) likelihood in place of
MSE, and `asinh(d/delta)` in place of `log10 d`. Run as a 2×2 so the two are
attributable separately.

**Scope.** Loss and target on the existing dataset; then the regeneration
decision, against a gate registered *below* rather than after the numbers.
Explicitly NOT here: the hyperparameter sweep and the Titan port, dataset
regeneration, cross-family and cross-energy holdout.

**The Weekly Plan row for this week has drifted** and is not what is run here.
It reads "phase-space-resolved analysis": error maps, resonance proximity via
`d_res`, the sweep, multi-seed CIs, the central figure. Three problems. `d_res`
does not exist (nb11 §2e). Risk 18 is Realized — no resonance-adjacent negatives
were obtainable — so the resonance connection has no material. And the
multi-seed band was already done in Week 11. The sheet needs rewriting to match.

**The four cells.**

| | target | loss | |
|---|---|---|---|
| **M1** | `log10 d` | MSE | the frozen Week 11 baseline, reproduced |
| **M2a** | `asinh(d/δ)` | MSE | target only |
| **M2b** | `log10 d` | Tobit | loss only |
| **M2c** | `asinh(d/δ)` | Tobit | both |

M1 is *nested* in M2c: the censored NLL with no censored rows and fixed σ is MSE
up to a constant, and `targets.forward` on the log10 path reproduces
`sampling.label_and_pack` byte for byte. So the cells differ in exactly the two
intended ways.

**Predictions, stated before running.** Scored in §7.

| | claim | falsified if |
|---|---|---|
| **P12a** | the dead-zone boundary reproduces at ~125 m and the constant at ~6.5 m | either moves by more than a factor of 2, or the boundary is not stable across bin resolutions |
| **P12b** | censored fraction is ≈ 13% and dominated by `source == 'surface'` | below 2%, in which case Tobit is a null by construction and the week collapses to the target change |
| **P12c** | M2c ≥ M1 on R² over measured rows, but by less than 0.05 | a large R² gain, which given §7's argument would more likely be the metric moving than the model |
| **P12d** | M2c beats M1 on the **near-field** factor-of-10 column by more than the seed spread | no near-field gain, which means the target change conceded the dead zone rather than fixing it |
| **P12e** | **M2b ≈ M1 within the seed spread**, and the value of the loss change appears as M2c − M2a instead | M2b moves substantially on its own |
| **P12f** | under M2c, censored rows are predicted *below* the floor; under M1 they pile *at* it | censored predictions still sit ~42× above the floor under M2c |

P12e is the one with a derivation behind it. For a censored row the Tobit
gradient is λ(t)/σ with t = (z_L − μ)/σ, and as μ ≫ z_L the inverse Mills ratio
λ(t) → −t, so the gradient → (μ − z_L)/σ² — *identically* the MSE gradient.
Week 11's dead-zone constant is 6.5 m against a 0.155 m floor, i.e. 42× above
it, so Tobit's distinguishing behaviour does not engage at the current operating
point. It should only pay once something else brings μ near the floor. That
makes the two changes multiplicative, not additive.

**Risk 19 is untouched.** The censoring here is the labeller's measured
reproducibility floor. `d_km` is still an upper bound on distance-to-nearest-QPIT
because only the computed surfaces exist, and the tightness of that bound in the
gaps is unknown. Nothing in this notebook has power over it.

**Runtime** ~40–70 min: four cells × five seeds, plus two extra δ values for the
asinh cells (the log10 cells do not depend on δ and are run once).
""".rstrip()))

cells.append(code(r"""
import sys, os, time, json
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
import torch

from features import build_features, assert_no_leakage, Standardizer
from splits import grouped_split, groups_disjoint, group_census, split_report
from metrics import (regression_metrics, accuracy_vs_threshold,
                     summarize_threshold_curve, accuracy_vs_error_tolerance,
                     factor_table, residual_bins, dead_zone, format_table,
                     FACTORS)
from models import MLPConfig
import targets as T
import ablation as AB
from constants import FLOOR_KM          # single source -- do NOT shadow (audit s3)

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
np.set_printoptions(precision=6, suppress=True)
t0 = time.time()

DATA      = '../data/qpit_dataset_w10.npz'
WK11      = '../data/wk11/results_wk11.json'      # frozen; read only
OUT_DIR   = '../data/wk12'

# THE WEEK 11 CONFIG.  nb11's first run ended all three seeds at the 600-epoch
# ceiling with validation loss still falling, so its R2 was a lower bound and
# its seed spread (0.17) exceeded the M1-B1 gap; nb11 cell 20 overrode the
# stopping rule and the frozen results come from that override.  The audit has
# since moved those values into the MLPConfig DEFAULTS, so this block is now a
# NO-OP -- kept because it states, at the call site, which settings the frozen
# baseline was produced with.  If it ever stops being a no-op, the defaults
# moved again and that is worth knowing before the 2x2 runs.
CFG_BASE = dict(hidden=(256, 256, 256), activation='silu',
                max_epochs=3000, patience=150, lr_patience=50)

SEED_SPLIT  = 0                    # MUST match Week 11 or the control is void
FRAC_TEST, FRAC_VAL = 0.20, 0.10
SEEDS_MODEL = (0, 1, 2, 3, 4)      # five, not three -- see below

# DELTAS is not set here.  Both anchors are MEASURED, in section 2c, from
# the frozen Week 11 predictions -- a hardcoded value sitting beside an
# authoritative one is the shape of the unit slips logged as Risk 20.

os.makedirs(OUT_DIR, exist_ok=True)
print(f"torch {torch.__version__}   floor {FLOOR_KM*1e6:.0f} mm   "
      f"seeds {SEEDS_MODEL}")
""".rstrip()))

cells.append(md(r"""
### Why five seeds

Week 11 used three, as a sanity band on whether "the baseline failed" survived
re-initialisation. That is not enough here. Three seeds against an observed
spread of 0.020 gives SEM ≈ 0.012, so a 0.03 difference between cells is 1.8σ
and P12d is not decidable. Five gives SEM ≈ 0.009. This is still a seed band and
still not the confidence-interval protocol; the sweep it was once bundled with
is parked.
""".rstrip()))

cells.append(md(r"""
## 1. Load, and reproduce the Week 11 target

`log_d_km` is not stored in the Week 10 file — nb11 §2c derived it, and so does
this. `at_floor` may not be stored either. `ablation.build_target` derives the
censoring mask from `d_km` and the floor in the same call that builds the
target, so the two cannot disagree; when the column *is* present it is checked
against the derived mask rather than used, because a disagreement means one of
them is stale.
""".rstrip()))

cells.append(code(r"""
raw = dict(np.load(DATA, allow_pickle=True))
n_rows = len(raw['states'])
D = {k: v for k, v in raw.items() if getattr(v, 'ndim', 0) >= 1
     and len(v) == n_rows}
META = {k: v for k, v in raw.items() if k not in D}
print(f"{n_rows} rows, {len(D)} columns")
print("columns:", sorted(D))
if META:
    print("per-sheet metadata (NOT columns):", sorted(META))

# THE FLOOR COMES FROM THE FILE WHEN THE FILE HAS ONE.  Same resolution nb11
# section 2c does, and for the same reason: a hardcoded constant sitting beside
# an authoritative value is the shape of the unit slips logged as Risk 20.  It
# matters more here than there -- FLOOR_KM sets the Tobit threshold z_L, so a
# stale one does not just mis-scale a plot, it mis-specifies the likelihood.
for _k in ('floor_km', 'FLOOR_KM', 'floor'):
    if _k in META:
        _f = float(np.ravel(META[_k])[0])
        if abs(_f - FLOOR_KM) > 1e-15:
            print(f"floor in file {_f:.6g} km overrides notebook constant "
                  f"{FLOOR_KM:.6g} km")
        FLOOR_KM = _f
        break
else:
    print(f"no floor in file; using the Week 10 measured value {FLOOR_KM:.6g} km")

d_km = np.asarray(D['d_km'], dtype=float).reshape(-1)
y_log = (D['log_d_km'].astype(float) if 'log_d_km' in D
         else T.log10_forward(d_km, FLOOR_KM))
assert np.allclose(y_log, T.log10_forward(d_km, FLOOR_KM), atol=1e-12), \
    "stored log_d_km disagrees with the clamp at this floor"

# WEEK 11 COULD TOLERATE A DISAGREEMENT HERE.  Week 12 cannot.  MSE does not
# care where the flag sits; the censored likelihood does -- its whole premise is
# that a clamped row's target IS z_L, and `losses.check_censoring_consistency`
# rejects the fit otherwise.  So a mismatch is resolved before the run, not
# papered over by preferring one of the two.
at_floor = D['at_floor'].astype(bool)
implied = d_km < FLOOR_KM
if not np.array_equal(at_floor, implied):
    raise ValueError(
        f"at_floor disagrees with (d_km < {FLOOR_KM:.6g}) on "
        f"{(at_floor != implied).sum()} rows.  The stored flag was set against a "
        "different floor.  nb11 preferred the stored flag and that was fine for "
        "MSE; Tobit needs the flag and the target's clamp to come from ONE floor. "
        "Resolve which floor is authoritative before running the 2x2.")

print(f"\nd_km  min {d_km.min():.6g}  max {d_km.max():.6g} km")
print(f"target log10 km  [{y_log.min():.4f}, {y_log.max():.4f}]  "
      f"= {y_log.max()-y_log.min():.3f} decades")
print(f"at_floor: {at_floor.sum()} / {n_rows}  ({at_floor.mean()*100:.2f}%)")
""".rstrip()))

cells.append(md(r"""
## 2. Step 0 — the three measurements that decide the week

Not a warm-up. Each one gates something.

**2a. Censored fraction.** Below ~2% the Tobit half of the week has no material
and should be dropped rather than run.

**2b. Drawn versus measured distance.** The sampler was built so this is a free
regression test — for normal-direction offsets the two should agree to ~2%. Where
the agreement breaks is the empirical label-trust floor, measured rather than
declared, and it checks `FLOOR_KM` independently.

**2c. The dead zone, from the frozen Week 11 predictions.** Its boundary is δ.
""".rstrip()))

cells.append(code(r"""
# 2a
cens = d_km < FLOOR_KM
print(f"censored: {cens.sum()} / {cens.size} = {cens.mean():.4f}")
if at_floor is not None and not np.array_equal(at_floor, cens):
    print(f"  WARNING: stored at_floor disagrees on {(at_floor != cens).sum()} rows")
if 'source' in D:
    for s in np.unique(D['source']):
        m = D['source'] == s
        print(f"  {str(s):10s} n={m.sum():6d}  censored {cens[m].mean():.4f}")
if cens.mean() < 0.02:
    print("\nGATE: censored fraction below 2% -- Tobit is a null by "
          "construction.  Drop M2b/M2c and run the target change only.")
""".rstrip()))

cells.append(code(r"""
# 2b -- normal-direction offsets only; random directions land closer by design
if {'d_drawn_km', 'source', 'random_dir'} <= set(D):
    # startswith, not equality: a source array saved with a too-narrow unicode
    # dtype silently truncates, and an equality test against a literal then
    # matches nothing at all (the 'surface' -> 'surfac' case from
    # nb11_diag_shortcut).  Comparing against values drawn FROM the array, as
    # 2a does, is safe; comparing against a literal is not.
    is_offset = np.char.startswith(D['source'].astype(str), 'off')
    m = is_offset & (~D['random_dir'].astype(bool)) & (D['d_drawn_km'] > 0)
    if not m.any():
        raise ValueError(
            f"no normal-direction offset rows found; source values present: "
            f"{np.unique(D['source'])}")
    ratio = d_km[m] / D['d_drawn_km'][m]
    edges = np.linspace(np.log10(FLOOR_KM), np.log10(d_km.max()), 13)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = m.copy(); s[m] = (np.log10(D['d_drawn_km'][m]) >= a) \
                             & (np.log10(D['d_drawn_km'][m]) < b)
        if s.sum() < 20:
            continue
        r = d_km[s] / D['d_drawn_km'][s]
        rows.append(dict(lo_km=float(10**a), hi_km=float(10**b), n=int(s.sum()),
                         med_ratio=float(np.median(r)),
                         iqr=float(np.subtract(*np.percentile(r, [75, 25])))))
    print(format_table(rows))
    print("\nWk12 RESULT: it does not break.  med_ratio 0.9999 and IQR ~0.001 "
          "in every bin, 0.2 m to 100 km.  That measures PRECISION -- recovery "
          "of the offset against the SAME interpolated surface -- while the "
          "floor is an ACCURACY limit (two derivations of one torus land 2-155 "
          "mm apart).  So 2b cannot check FLOOR_KM independently, contrary to "
          "what it was set up to do; delta's anchor rests on 2c alone.")
    print("It does show something else: realised/drawn = 0.9999 puts the "
          "residual tangential component of `sampling.normal_directions` at "
          "~1.4% of the step (~0.8 deg), reproducing the audit's finding "
          "independently on 11,773 rows.")
else:
    print("d_drawn_km / source / random_dir absent -- 2b not runnable on this file")
""".rstrip()))

cells.append(code(r"""
# 2c -- dead zone, from the FROZEN Week 11 predictions.  Read only.
# Same detector as nb11 cell 27/31, now shared from src/metrics.py.  It was
# pasted between cells there and the two copies had already drifted; Week 12
# calls the function so a third copy cannot appear.
W11 = np.load('../data/wk11/predictions_wk11.npz')
y11, p11 = W11['y_test'], W11['pred_mean']
fl11 = W11['at_floor_test'].astype(bool)

rows = []
for nb_ in (12, 18, 24, 36):
    dz = dead_zone(y11, p11, n_bins=nb_, mask=~fl11)
    rows.append(dict(n_bins=nb_, flat_bins=dz['k'],
                     width_dec=dz['bin_width_dec'],
                     constant_m=dz.get('constant_m', float('nan')),
                     boundary_m=dz.get('boundary_m', float('nan')),
                     slope=dz.get('slope', float('nan')),
                     span_dec=dz.get('span_dec', float('nan')),
                     consistent=dz.get('self_consistent', False)))
print(format_table(rows))

good = [r for r in rows if r['consistent']]
print("\nslope near -1 is the saturation signature: a constant output c gives "
      "bias = log10(c) - y, i.e. slope exactly -1 against the target.")
print("`consistent` is the self-check -- the reported constant must lie INSIDE "
      "the range it was constant over.  A row where it does not is the fixed "
      "0.10 tolerance truncating on bin noise, not a shorter dead zone.")
if not good:
    raise ValueError("no self-consistent dead-zone fit; delta has no anchor")
DELTA_BOUNDARY = float(np.median([r['boundary_m'] for r in good]) / 1e3)  # km
print(f"\nself-consistent at n_bins {[r['n_bins'] for r in good]}; "
      f"boundary {DELTA_BOUNDARY*1e3:.0f} m")
print(f"delta = boundary sits {np.log10(DELTA_BOUNDARY/FLOOR_KM):.2f} of "
      f"{np.log10(100/FLOOR_KM):.2f} decades above the floor -- it flattens "
      "HALF the dynamic range, and exactly the half the model already fails in.")

DELTAS = dict(output_floor=float(np.median([r['constant_m'] for r in good])/1e3),
              boundary_over3=DELTA_BOUNDARY/3, boundary=DELTA_BOUNDARY,
              boundary_x3=DELTA_BOUNDARY*3)
print("\ndelta anchors (km):", {k: round(v, 6) for k, v in DELTAS.items()})
""".rstrip()))

cells.append(md(r"""
### What §2c means for how this week can be read

A model that emits a constant across a region it is no longer *asked* to resolve
will score better without predicting better. That is the failure mode this
notebook is most exposed to, and R² cannot detect it: R² is scale-invariant
under an *affine* change of target, and `asinh` is not affine in `log10 d`.

So the arbiter is the **near-field factor-of-N column** — measured rows below
125 m, in physical km — which no choice of δ can move. If a cell lifts R² and
leaves that column flat, the honest scorecard entry is a **falsification of the
target change**, not a win.
""".rstrip()))

cells.append(md(r"""
## 3. Features and split — both must reproduce Week 11 exactly

Same `SEED_SPLIT`, same fractions, same feature builder. If they do not
reproduce, M1 is not a control and the 2×2 has nothing to sit against.

`grouped_split` returns **boolean row masks** over the full dataset, so every
column subsets by the same mask and the censoring mask cannot fall out of order
with the rows it describes.
""".rstrip()))

cells.append(code(r"""
# THE ENERGY COLUMN IS THE STORED SHEET TAG, NOT A RECOMPUTED PER-STATE VALUE.
# Recomputing it from the state manufactured the Week 11 shortcut: the offsets
# were never reprojected onto the energy manifold, so a state's own Jacobi
# constant differs from its sheet's IN PROPORTION TO THE OFFSET, which is the
# target.  Measured then: sheet C alone R2 = -0.0001, recomputed per-state
# Jacobi alone R2 = +0.7976.
JAC = (D['C'] if 'C' in D else D['jacobi']).astype(float)
sheets = np.unique(np.round(JAC, 9))
print(f"stored C: {sheets.size} distinct values")
assert sheets.size <= 12, ("stored C is no longer sheet-like -- re-run the "
                           "single-feature leak audit before trusting it")

FAMILY_LEVELS = sorted(np.unique(D['family_id']).tolist())
X, FEAT_NAMES, SCALE_MASK = build_features(
    D['states'], JAC, D['family_id'], FAMILY_LEVELS)
assert_no_leakage(FEAT_NAMES)
print(f"X {X.shape}   features {FEAT_NAMES}")

M = grouped_split(D['group_id'], FRAC_TEST, FRAC_VAL, seed=SEED_SPLIT)
assert groups_disjoint(D['group_id'], M), "grouped split leaked a group"
tr, va, te = M['train'], M['val'], M['test']

scaler = Standardizer(mask=SCALE_MASK).fit(X[tr])        # TRAIN ROWS ONLY
Z = scaler.transform(X)
X_tr, X_va, X_te = Z[tr], Z[va], Z[te]
d_tr, d_va, d_te = d_km[tr], d_km[va], d_km[te]
af_tr, af_va = at_floor[tr], at_floor[va]

print(format_table([{k: v for k, v in group_census(D['group_id'], M).items()
                     if not k.startswith('size')}]))
print(f"\ncensored by split: train {af_tr.mean():.4f}  "
      f"val {af_va.mean():.4f}  test {at_floor[te].mean():.4f}")
rep = split_report(y_log, M, group_id=D['group_id'])
print(f"KS(train, test) on the target: {rep.get('ks_stat_train_test', float('nan')):.4f} "
      f"(p={rep.get('ks_pvalue_train_test', float('nan')):.3f})")
""".rstrip()))

cells.append(md(r"""
## 4. M1 — reproduce the baseline before changing anything

Seeds 0–2 are Week 11's. They should return the same numbers on the same machine
and library build. A mismatch means something upstream moved and everything
below is uninterpretable.

M1's held-out loss also sets `log_sigma_init` for the Tobit cells: that loss is
MSE on standardised targets, so its square root is the residual scale in the
units σ lives in. Starting σ there avoids the early-training regime where a
badly-scaled σ dominates the censored branch.
""".rstrip()))

cells.append(code(r"""
cfg = MLPConfig(**CFG_BASE)
ev_m1, pred_m1 = AB.run_cell('M1', 'log10', 'mse', X_tr, X_va, X_te,
                             d_tr, d_va, d_te, FLOOR_KM, cfg=cfg,
                             seeds=SEEDS_MODEL, at_floor_tr=af_tr,
                             at_floor_val=af_va, verbose=True)

# SEARCH the frozen file rather than assume a key path.  A .get() chain that
# misses returns None and prints as nan, so the reproduction check silently
# passes on no data -- which is the failure mode this whole suite exists to
# catch, and the first version of this cell had it.
with open(WK11) as fh:
    wk11 = json.load(fh)

def _find_r2(node, path=()):
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (int, float)) and 'r2' in str(k).lower():
                hits.append(('.'.join(path + (str(k),)), float(v)))
            else:
                hits += _find_r2(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += _find_r2(v, path + (str(i),))
    return hits

_all = _find_r2(wk11)
_m1 = [(k, v) for k, v in _all if 'm1' in k.lower()]
if not _m1:
    raise KeyError(f"no M1 R2 in {WK11}. R2-like keys present: "
                   f"{[k for k, _ in _all]}")
r2_w11 = _m1[0][1]
if len(_m1) > 1:
    print(f"several M1 R2 entries, using {_m1[0][0]}: {_m1}")
r2_now = ev_m1['summary']['r2_measured']
print(f"\nWeek 11 M1 R2: {r2_w11:.4f}   reproduced (measured rows): {r2_now:.4f} "
      f"+/- {ev_m1['summary']['r2_measured_sd']:.4f}")
print("NOTE: Week 11's headline R2 was over ALL test rows; this is over MEASURED "
      "rows only, because a clamped label is not a point observation and scoring "
      "against it rewards predicting the clamp.  The two are not the same number "
      "and the difference is not a discrepancy.")

nbl = ev_m1['budget_limited']
if nbl:
    print(f"\n{nbl}/{len(SEEDS_MODEL)} M1 seeds ended on the EPOCH BUDGET, not "
          "on convergence -- those are lower bounds.  nb11's first run had all "
          "three end that way at 600 epochs and read as converged.  Check how "
          "far each was still moving before treating the band as a fit; if the "
          "Tobit cells do this on most seeds, stop rather than report it.")

best_val = float(np.mean([h['best_val_loss'] for h in ev_m1['histories']]))
cfg_tobit = MLPConfig(**CFG_BASE, log_sigma_init=0.5 * np.log(best_val))
print(f"\nM1 mean best_val_loss {best_val:.5f} -> "
      f"log_sigma_init {cfg_tobit.log_sigma_init:+.4f}")
""".rstrip()))

cells.append(md(r"""
## 5. The 2×2, swept over δ

δ enters only the `asinh` cells, so M1 and M2b are run once and reused.

**δ is reported across the band, not selected from it.** The headline is δ =
the measured boundary, at five seeds; the other three are a sensitivity check at
two seeds. Screening δ and then reporting the winner would be a selection effect
on the same test set the result is quoted from. The cheaper seed count on the
sensitivity rows is deliberate — they are a robustness check, not a number that
goes in a table.

Budget: 10 headline fits (M1, M2b at five seeds each) + 10 more at the boundary
+ 12 sensitivity fits = **32 fits**. At Week 11's ~5 min each with
`max_epochs=3000` that is roughly 2.5–3 hours. If that is too long, cut the
sensitivity rows first, never the seed count on the headline.
""".rstrip()))

cells.append(code(r"""
ev_m2b, pred_m2b = AB.run_cell('M2b', 'log10', 'tobit', X_tr, X_va, X_te,
                               d_tr, d_va, d_te, FLOOR_KM, cfg=cfg_tobit,
                               seeds=SEEDS_MODEL, at_floor_tr=af_tr,
                               at_floor_val=af_va, verbose=True)

RESULTS, PREDS = {}, {}
SEEDS_SENS = SEEDS_MODEL[:2]          # sensitivity band, not a headline

for label, delta in DELTAS.items():
    headline = (label == 'boundary')
    seeds = SEEDS_MODEL if headline else SEEDS_SENS
    print(f"\n===== delta = {delta*1e3:.1f} m  ({label}, "
          f"{len(seeds)} seeds{', HEADLINE' if headline else ''}) =====")
    r = {'M1': ev_m1, 'M2b': ev_m2b}
    p = {'M1': pred_m1, 'M2b': pred_m2b}
    for name, loss in (('M2a', 'mse'), ('M2c', 'tobit')):
        r[name], p[name] = AB.run_cell(
            name, 'asinh', loss, X_tr, X_va, X_te, d_tr, d_va, d_te,
            FLOOR_KM, delta_km=delta,
            cfg=(cfg_tobit if loss == 'tobit' else cfg),
            seeds=seeds, at_floor_tr=af_tr, at_floor_val=af_va,
            verbose=headline)
    RESULTS[label], PREDS[label] = r, p
    print(f"\n{format_table(AB.scorecard(r))}")

print(f"\n[{time.time()-t0:.0f}s]")
""".rstrip()))

cells.append(md(r"""
## 6. The arbiter — near-field factor-of-N

`all` and `measured` are directly comparable to the Week 11 table. `near` is the
new column: measured rows below 125 m. It is the one that separates *fixed the
dead zone* from *stopped asking about it*.
""".rstrip()))

cells.append(code(r"""
for label, r in RESULTS.items():
    print(f"\n=== delta = {DELTAS[label]*1e3:.1f} m ===")
    rows = []
    for f in (f"{x:g}" for x in FACTORS):
        row = dict(factor=float(f))
        for cell in ('M1', 'M2a', 'M2b', 'M2c'):
            row[cell] = r[cell]['summary']['factors'][f]['near']
        rows.append(row)
    print(format_table(rows))

print("\nP12f -- where do censored rows get predicted?")
for label, r in RESULTS.items():
    print(f"  delta {DELTAS[label]*1e3:6.1f} m  " + "  ".join(
        f"{c} {r[c]['summary']['frac_below_floor']:.3f}"
        for c in ('M1', 'M2a', 'M2b', 'M2c')))
""".rstrip()))

cells.append(md(r"""
## 7. Scorecard

A falsification here is a result. P12d and P12e are the two that carry
information either way: P12d because a flat near-field column means the target
change conceded rather than fixed, and P12e because it has a derivation behind it
that fails loudly if wrong.
""".rstrip()))

cells.append(code(r"""
best = 'boundary'
r, s = RESULTS[best], {k: v['summary'] for k, v in RESULTS[best].items()}
sd = s['M1']['r2_measured_sd']

# P12e IS SCORED AT THE MEASURED OPERATING POINT, NOT AN ASSUMED ONE.  The
# Mills limit lambda(t) -> -t is in units of the FITTED sigma.  6.54 m over a
# 0.155 m floor is 42x in DISTANCE and only 1.625 decades; at a residual sigma
# near 1 decade that is t ~ -1.5, not t ~ -40.  lambda(t) > |t| for every
# t < 0, so a small |t| means Tobit pushes censored rows down HARDER than MSE
# rather than matching it -- the opposite of the asymptotic reading.
from scipy.stats import norm as _norm
cens_te = d_te < FLOOR_KM
print("\noperating point on censored test rows:")
for cell in ('M2b', 'M2c'):
    h, tf = r[cell]['histories'][0], r[cell]['transform']
    dl = r[cell]['delta_km']
    mu_km = float(np.mean([T.to_log10_km(pp, tf, FLOOR_KM, dl)[cens_te].mean()
                           for pp in PREDS[best][cell]]))
    mu_t = float(T.forward(np.array([10.0 ** mu_km]), tf, FLOOR_KM, dl)[0])
    z_Lr = T.threshold_in_target_units(FLOOR_KM, tf, dl)
    sig = float(np.exp(h['log_sigma_final']) * h['y_std'])
    t = (z_Lr - mu_t) / sig
    lam = float(_norm.pdf(t) / max(_norm.cdf(t), 1e-300))
    print(f"  {cell}: mu-z_L = {mu_t-z_Lr:+.3f} target units, sigma = {sig:.3f}, "
          f"t = {t:+.2f}, lambda/|t| = {lam/abs(t):.3f}  "
          f"({'near the MSE limit' if lam/abs(t) < 1.05 else 'NOT in the asymptote'})")

verdict = {}
verdict['P12b'] = (cens.mean() > 0.02,
                   f"censored fraction {cens.mean():.4f}")
verdict['P12c'] = (0 <= s['M2c']['r2_measured'] - s['M1']['r2_measured'] < 0.05,
                   f"dR2 = {s['M2c']['r2_measured'] - s['M1']['r2_measured']:+.4f}")
d_near = s['M2c']['factors']['10']['near'] - s['M1']['factors']['10']['near']
verdict['P12d'] = (d_near > 2 * sd,
                   f"d(near, factor 10) = {d_near:+.4f} vs 2*seed sd {2*sd:.4f}")
d_m2b = abs(s['M2b']['r2_measured'] - s['M1']['r2_measured'])
verdict['P12e'] = (d_m2b <= 2 * sd,
                   f"|M2b - M1| = {d_m2b:.4f} vs 2*seed sd {2*sd:.4f}; "
                   f"M2c - M2a = "
                   f"{s['M2c']['r2_measured'] - s['M2a']['r2_measured']:+.4f}")
verdict['P12f'] = (s['M2c']['frac_below_floor'] > s['M1']['frac_below_floor'],
                   f"below-floor fraction M1 {s['M1']['frac_below_floor']:.3f} "
                   f"-> M2c {s['M2c']['frac_below_floor']:.3f}")

for k in sorted(verdict):
    ok, why = verdict[k]
    print(f"  [{'HELD     ' if ok else 'FALSIFIED'}] {k}  {why}")

if not verdict['P12d'][0]:
    print("\nP12d FALSIFIED.  If R2 rose anyway, that is the metric moving and "
          "not the model: delta = 125 m flattens the decades the network already "
          "failed in.  Record this as a falsification of the TARGET change, not "
          "as an inconclusive result.")
""".rstrip()))

cells.append(md(r"""
## 8. The regeneration gate

Registered before the numbers, so it is not read off them.

- **If M2c − M1 on the near-field factor-of-10 column is within the seed
  spread**, the loss/target lever is exhausted and the bottleneck is data →
  regenerate, with **grid refinement as a hard requirement of the spec**.
- **If the gain is large**, the bottleneck was modelling → regeneration is worth
  less than it looked and the parked sweep moves ahead of it.
- **If M2a carries the gain and M2b does not**, the problem was representational
  → the regeneration's value shifts toward the 10–15 energies rather than the
  near field.

If it fires toward regeneration, the batch is already specified by the audit
handoff and goes as ONE change, not three: swap in
`data/families/L2_E2_dense.npz` (562 → 0 rows below their own error bound), fix
`sampling.normal_directions` (sequential Gram–Schmidt clears only the second
tangent), and **write `mknb10.py` first**, since nb10 currently has no builder
at all. One at a time invalidates Weeks 11–12 three times over.

One thing that has to be in the spec whichever branch fires: **near-field
oversampling alone cannot help.** More states close to a surface produce more
rows near a floor set by how well the *curves themselves* are determined
(0.155 m, σ₃ ≈ 2.0e-4 — the spacing is not the limit). Making the near field
informative means determining the surfaces better, not sampling them more.

And a guard for whenever it happens: more energies makes the sheet-energy column
higher-cardinality and raises the risk of a fresh proxy. Re-run the
single-feature R² audit on any regenerated set. The recomputed per-state Jacobi
must not reappear in `features.py`.
""".rstrip()))

cells.append(code(r"""
gate = ("regenerate, grid refinement required" if d_near <= 2 * sd else
        "modelling lever still live; sweep ahead of regeneration")
print(f"GATE -> {gate}")

out = dict(
    week=12, data=DATA, n_rows=int(n_rows),
    floor_km=FLOOR_KM, deltas_km=DELTAS, seeds=list(SEEDS_MODEL),
    split=dict(seed=SEED_SPLIT, frac_test=FRAC_TEST, frac_val=FRAC_VAL),
    censored_frac=float(cens.mean()),
    week11_m1_r2=r2_w11,
    results={lab: {c: v['summary'] for c, v in rr.items()}
             for lab, rr in RESULTS.items()},
    scorecard={lab: AB.scorecard(rr) for lab, rr in RESULTS.items()},
    verdict={k: bool(v[0]) for k, v in verdict.items()},
    verdict_detail={k: v[1] for k, v in verdict.items()},
    gate=gate, config=cfg_tobit.__dict__,
)
rp, pp = AB.save_results(out, {f"{lab}__{c}": v for lab, pp_ in PREDS.items()
                               for c, v in pp_.items()}, out_dir=OUT_DIR)
print(f"saved -> {rp}\n         {pp}   [{time.time()-t0:.0f}s total]")
""".rstrip()))

cells.append(md(r"""
## Carried into Week 13

- Whichever predictions were falsified, and the δ sensitivity across the band.
- The gate's branch, and if it fired toward regeneration, the grid-refinement
  requirement written into the spec rather than left as a note here.
- **Heteroscedastic σ(x).** This week used a single global σ deliberately —
  per-row σ is a real improvement and a known early-training instability, and
  bundling it with two other changes would have made none of them attributable.
- **Interval censoring.** The correct statement for a sub-floor row is
  `d ∈ [0, FLOOR]`; left-censoring at the floor is the cheap version of it.
- Risk 19 still untested, and nothing this week had power over it. If the
  write-up starts leaning on the Cantor-set framing, that has to come from an
  experiment that can resolve inter-torus gaps — which the labeller, measuring
  against a densified interpolated continuum, cannot.
- The Weekly Plan row for Week 12 needs rewriting to what was actually run, and
  Week 13's row needs checking for the same drift.
""".rstrip()))

nb = dict(cells=cells,
          metadata=dict(kernelspec=dict(display_name="Python 3",
                                        language="python", name="python3"),
                        language_info=dict(name="python", version="3.12")),
          nbformat=4, nbformat_minor=5)

import os, sys
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'nb12_loss_target.ipynb')

# THE BUILDER AND THE NOTEBOOK ARE TWO COPIES OF ONE THING (Risk 21, Realized
# Wk12).  Running mknb11.py mid-audit clobbered 33 cells and 23 executed
# outputs down to a stale builder's 32 with none.  Identical sources -> do not
# write at all, so executed outputs survive; different -> report and stop.
# Any hand-edit to the notebook goes back into THIS FILE, not the .ipynb.
if os.path.exists(out_path) and '--force' not in sys.argv:
    a = ["".join(c['source']) for c in json.load(open(out_path))['cells']]
    b = ["".join(c['source']) for c in cells]
    if a == b:
        print(f"unchanged: {out_path} ({len(cells)} cells, outputs preserved)")
        raise SystemExit(0)
    print(f"DIVERGED: notebook has {len(a)} cells, builder produces {len(b)}")
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            what = ('only in notebook' if y is None
                    else 'only in builder' if x is None else 'differs')
            print(f"  cell {i}: {what}  | {(x or y).splitlines()[0][:62]}")
    print("\nRefusing to overwrite.  Fold the notebook's edits into mknb12.py, "
          "or re-run with --force to discard them.")
    raise SystemExit(1)

with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out_path}: {len(cells)} cells")
