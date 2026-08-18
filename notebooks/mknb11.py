import json
import os

def _lines(s):
    ls = s.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]

def md(s):
    return dict(cell_type="markdown", metadata={}, source=_lines(s))

def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None,
                outputs=[], source=_lines(s))


def write_notebook(nb, out_path):
    """Write `nb` to `out_path`, but NEVER silently clobber a diverged notebook.

    This builder used to write unconditionally.  On 2026-08-16 that discarded a
    notebook carrying five cells of hand-applied corrections -- including the
    epoch-budget fix the retraction turned on -- together with every executed
    output, because the builder had drifted behind the notebook and nobody
    diffed before running it.

    Rules:
      * nothing on disk        -> write.
      * cell sources identical -> DO NOT write.  A rewrite would be a no-op on
                                  the source and would strip executed outputs,
                                  which is a silent loss of work.
      * cell sources differ    -> refuse, print the divergence, exit 1.  Pass
                                  --force to overwrite anyway.
    """
    import difflib
    import sys

    def sources(cs):
        return [''.join(c['source']) if isinstance(c['source'], list)
                else c['source'] for c in cs]

    name = os.path.basename(out_path)
    force = '--force' in sys.argv

    if os.path.exists(out_path) and not force:
        with open(out_path, encoding='utf-8') as fh:
            disk = json.load(fh)
        old, new = sources(disk['cells']), sources(nb['cells'])
        n_out = sum(len(c.get('outputs', [])) for c in disk['cells']
                    if c['cell_type'] == 'code')

        if old == new:
            print(f"{name}: already up to date ({len(new)} cells, "
                  f"{n_out} executed outputs preserved) -- not rewriting")
            return

        print(f"REFUSING TO OVERWRITE {name}")
        print(f"  on disk : {len(old)} cells, {n_out} executed outputs")
        print(f"  builder : {len(new)} cells")
        print("  The notebook on disk has content this builder would destroy.\n")
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, old, new).get_opcodes():
            if tag == 'equal':
                continue
            print(f"  {tag.upper()}  disk[{i1}:{i2}] -> builder[{j1}:{j2}]")
            for k in range(i1, i2):
                print(f"      - disk    cell {k}: "
                      f"{old[k].strip().splitlines()[0][:84]}")
            for k in range(j1, j2):
                print(f"      + builder cell {k}: "
                      f"{new[k].strip().splitlines()[0][:84]}")
        print(f"\n  Port the notebook's version into this builder, or re-run "
              f"with --force\n  to discard it.  It is in git:  "
              f"git checkout -- notebooks/{name}")
        raise SystemExit(1)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out_path}: {len(nb['cells'])} cells")


cells = []

cells.append(md(r"""
# Week 11 — Phase 4, first regressor

Trains the baseline MLP on `log10` distance-to-nearest-QPIT, against the
Week 10 labelled dataset, and produces the week's deliverable: **baseline
metrics plus an accuracy-vs-tolerance curve**.

**Scope.** Weekly Plan row 13. Feature representation; shortcut baseline;
grouped train/test split; MLP regressor; classification as a thresholded
readout. Nothing else. Explicitly NOT here: the hyperparameter sweep and the
Titan port, the phase-space error map, the resonance-proximity connection and
the central figure (all Week 12); speedup benchmarking (Week 13); cross-family
and cross-energy holdout (Week 15).

**Decisions taken into this notebook.**

1. *Floor samples are kept*, and every test metric is reported split by
   `at_floor`. They are left-censored observations — the true distance is below
   what the surface determines — so a censored (Tobit) loss is the principled
   treatment, but that is a Week 12 change if the split reporting shows it
   matters.
2. *Three seeds*, as a sanity band on whether "the baseline failed" survives
   re-initialisation. The confidence-interval protocol is Week 12.
3. *Learning curve in*, over group fractions. It is the bias–variance
   checkpoint made concrete for the dad session.
4. *B2 under both splits.* The grouped-vs-random gap is the empirical
   correlation length of the dataset and the evidence that grouping is
   necessary rather than decorative.

**Predictions, stated before running.** Scored in section 10.

| | claim | falsified if |
|---|---|---|
| **P1** | the shortcut baseline B1 scores near R² ≈ 0 — location relative to L1/L2 plus energy does not determine distance-to-torus | B1 R² > 0.5, which means the sampling left a location shortcut in and the dataset needs fixing |
| **P2** | 1-NN scores markedly higher under a row-random split than under the grouped split | the gap is ≈ 0, meaning either the group key is finer than the correlation length or the split is not doing what it claims |
| **P3** | ∇(log d) ~ 1/d normal to the surface, so the target's Lipschitz constant is unbounded as d → 0 while a smooth-activation MLP's gradient is bounded. Residuals are strongly heteroscedastic with a **systematic positive bias near the floor** — the network over-predicts distance for near-surface points. Overall R² is **high**, above 0.9 | uniform residual scale across the target range, or no near-field bias |

P3 contradicts Risk 19, deliberately. Risk 19 expects mediocre R² from
Cantor-set non-smoothness, but the labeller measures distance to a densified
*interpolated continuum* at ~12.3 km spacing and cannot resolve the gaps — and
the Week 10 gap experiment (80/80, exponent 2.003) says the interpolant is
populated by real tori anyway. So the target as generated should be
piecewise-smooth with cut-locus kinks, not Cantor-rough. If P3 holds, the
paper's framing improves: *"R² is high, but the near field — precisely the
regime a warm start needs — is where it degrades, and here is the mechanism"*
is a finding, where *"R² is mediocre and that is physics"* is a defence.

**Runtime** ~15–30 min on a laptop: seven MLP fits (3 seeds + 4 learning-curve
folds) plus the baselines.
""".rstrip()))

cells.append(code(r"""
import sys, os, time, json, warnings
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
import torch

from features import (jacobi_constant, build_features, assert_no_leakage,
                      Standardizer, MU_EARTH_MOON, LU_KM)
from splits import (grouped_split, random_row_split, groups_disjoint,
                    group_census, split_report, nearest_point_overlap,
                    subsample_groups)
from metrics import (regression_metrics, accuracy_vs_threshold,
                     summarize_threshold_curve, accuracy_vs_error_tolerance,
                     residual_bins, format_table)
from baselines import (collinear_points, shortcut_features, MeanPredictor,
                       KNNRegressor, fit_flexible)
from models import MLPConfig, train_mlp, predict, train_multi_seed, count_parameters

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
np.set_printoptions(precision=6, suppress=True)
t0 = time.time()

# ---- SET THIS to the Week 10 output. A single path, or a list to concatenate
# if the per-(family, energy) checkpoints were never merged.
DATA = '../data/qpit_dataset_w10.npz'

SEED_SPLIT  = 0
SEEDS_MODEL = (0, 1, 2)
FRAC_TEST, FRAC_VAL = 0.20, 0.10
FLOOR_KM = 1.55e-4            # measured in Week 10: 155 mm
os.makedirs('../data/wk11', exist_ok=True)
print(f"torch {torch.__version__}   floor {FLOOR_KM*1e6:.0f} mm")
""".rstrip()))

cells.append(md(r"""
## 1. Load

The Week 10 generator checkpointed per (family, energy), so `DATA` may be
several files. They are concatenated on the row axis; `group_id` already
encodes family and energy, so groups stay distinct across files without any
renumbering.
""".rstrip()))

cells.append(code(r"""
import glob
from collections import Counter

def load_dataset(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        # Say what IS there.  An error that reports only what is absent makes
        # the reader go hunting for a filename the machine already knows.
        found = []
        for d in sorted({os.path.dirname(p) or '.' for p in paths}):
            found += sorted(glob.glob(os.path.join(d, '*.npz')))
        listing = '\n  '.join(found) if found else '(none in that directory)'
        raise FileNotFoundError(
            f"not found: {missing}\n"
            f".npz files actually present:\n  {listing}\n"
            f"Set DATA in the config cell. If the Week 10 per-(family, energy) "
            f"checkpoints were never merged, pass the list of them instead of "
            f"one path -- load_dataset concatenates.")

    parts = [dict(np.load(p, allow_pickle=True)) for p in paths]
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    dropped = set(parts[0]) - keys
    if dropped:
        print(f"columns absent from some files, dropped: {sorted(dropped)}")

    # WHAT COUNTS AS A COLUMN: leading dimension == the row count, anchored on
    # `states`.  An earlier version tested for 0-d scalars instead and tried to
    # concatenate `energies` (3), `family_sizes` (6), `family_tags` (6) and
    # `family_max_gap_km` (6) as though they were columns.  Metadata in this
    # dataset is 1-D per-sheet arrays, not scalars.
    anchors = [len(p['states']) if 'states' in p else
               max(Counter(len(v) for v in p.values() if v.ndim).items(),
                   key=lambda kv: kv[1])[0]
               for p in parts]

    cols, meta = {}, {}
    for k in sorted(keys):
        vals = [p[k] for p in parts]
        if all(v.ndim >= 1 and len(v) == a for v, a in zip(vals, anchors)):
            cols[k] = np.concatenate(vals)
        else:
            if any(not np.array_equal(v, vals[0]) for v in vals[1:]):
                print(f"  NOTE: metadata '{k}' differs between files; kept the first")
            meta[k] = vals[0]

    counts = {k: len(v) for k, v in cols.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"columns disagree on row count: {counts}")
    return cols, meta

D, META = load_dataset(DATA)
n = len(D['states'])
print(f"{n} rows, {len(D)} columns")
print(f"columns: {sorted(D)}")
for k in sorted(META):
    v = META[k]
    print(f"  meta {k:22s} shape {v.shape} {np.array2string(v, max_line_width=100)[:90]}")
for k in ('family_id', 'energy_id'):
    if k in D:
        u, c = np.unique(D[k], return_counts=True)
        print(f"{k}: {dict(zip(u.tolist(), c.tolist()))}")
""".rstrip()))

cells.append(md(r"""
### 1b. What is actually in the file

Written against the schema the Week 10 generator produced, not against the one
inferred from `label_and_pack` -- they differ. Two columns could be the
regression target (`d_km`, the geometric surface distance, and `score_C`, which
in Week 10 was the probe's invariance residual `dist_max`), and they are not the
same quantity. This prints enough to decide which is which rather than guessing,
because guessing wrong would invalidate every number downstream.
""".rstrip()))

cells.append(code(r"""
print(f"{'column':16s} {'dtype':10s} {'shape':14s} min / median / max")
for k in sorted(D):
    v = D[k]
    if v.dtype.kind in 'fiu' and v.ndim == 1:
        print(f"{k:16s} {str(v.dtype):10s} {str(v.shape):14s} "
              f"{v.min():.6g} / {np.median(v):.6g} / {v.max():.6g}")
    elif v.dtype.kind == 'b':
        print(f"{k:16s} {str(v.dtype):10s} {str(v.shape):14s} True: {v.sum()}")
    else:
        u = np.unique(v) if v.ndim == 1 else np.array(['-'])
        print(f"{k:16s} {str(v.dtype):10s} {str(v.shape):14s} "
              f"{u[:6] if u.size <= 12 else str(u.size) + ' distinct'}")

# Is score_C the LOG of d_km, or a different quantity?  Compare it against
# log10(max(d_km, floor)) directly.  An earlier version took log10 OF score_C,
# which is already in log units, and reported a meaningless 0.87 correlation.
if {'d_km', 'score_C'} <= set(D):
    a, b = D['d_km'].astype(float), D['score_C'].astype(float)
    lg = np.log10(np.maximum(a, FLOOR_KM))
    print(f"\nscore_C vs log10(max(d_km, {FLOOR_KM:g})): "
          f"allclose {np.allclose(b, lg, atol=1e-9)}, "
          f"max |diff| {np.abs(b - lg).max():.3e}")
    print(f"  score_C range {b.min():.5f} .. {b.max():.5f}   "
          f"10**min = {10**b.min():.3e} km")
    if not np.allclose(b, lg, atol=1e-6):
        print("  NOT the same column -- resolve which is the target before "
              "going further")

if 'label_B' in D:
    print(f"\nlabel_B: {int(np.sum(D['label_B']))} positive of {len(D['label_B'])}")
""".rstrip()))

cells.append(md(r"""
## 2. Audit

Five checks, all cheap, all before any model touches the data.

**2a is the one that can change the feature set.** `sampling.label_and_pack`
writes `C=np.full(n, C)` — the *family's* energy broadcast to every row, not
each state's own Jacobi constant. Off-surface samples were stepped along a
normal direction, and unless they were reprojected onto the energy manifold
their true C differs from that column. If it does, `C` is a six-valued sheet
tag rather than a state feature, and handing it to the shortcut baseline
alongside a radial coordinate could let B1 succeed for a bookkeeping reason —
condemning a dataset that does not deserve it.

**2b is a free regression test on the labeller.** For offset samples drawn
along a projected normal, the *drawn* distance and the *measured* one should
agree to ~2% by construction. A fifth independent validation, at no cost.

**2e** checks whether `d_res` is actually in the dataset. Week 12 assumes it.
Finding out now costs a minute; finding out in Week 12 costs a day.
""".rstrip()))

cells.append(code(r"""
# 2a. WHICH energy column is safe to use as a feature?
#
# Not "is the stored C per-state or a sheet tag" -- that was the wrong question
# and answering it by recomputing C from the state manufactured a shortcut.
# The offsets were never reprojected onto the energy manifold, so a state's own
# Jacobi constant differs from its sheet's IN PROPORTION TO THE OFFSET, which
# is the target.  The sheet tag is the clean column; the per-state value is the
# leak.  Measured in nb11_diag_shortcut: sheet C alone R2 = -0.0001,
# recomputed per-state Jacobi alone R2 = +0.7976.
C_stored = (D['C'] if 'C' in D else D['jacobi']).astype(float)
C_state = jacobi_constant(D['states'], MU_EARTH_MOON)
sheets = np.unique(np.round(C_stored, 9))
print(f"stored C: {sheets.size} distinct values -> {np.array2string(sheets[:8], precision=9)}")

SHEET_LIKE = sheets.size <= 12
if SHEET_LIKE:
    JAC = C_stored
    print("-> stored C is the SHEET energy. Using it. This is the clean column.")
else:
    JAC = C_stored
    print("-> stored C varies continuously. It may already be reprojected, or it "
          "may\n   itself carry the offset. Check the leak size below before "
          "trusting it.")

# size of the leak, so it is a measured quantity in the record rather than an
# assertion carried over from a diagnostic script
dC = np.abs(C_state - C_stored)
print(f"\n|C_state - C_sheet|: median {np.median(dC):.3e}  max {dC.max():.3e}")
_m = (dC > 0) & np.isfinite(dC)
if _m.sum() > 50:
    _y = np.log10(np.maximum(D['d_km'].astype(float), FLOOR_KM))
    print(f"corr(log10|C_state - C_sheet|, target) = "
          f"{np.corrcoef(np.log10(dC[_m]), _y[_m])[0,1]:.4f}")
    print("A high correlation is why the per-state Jacobi is NOT a feature: with "
          "only\nthree energy sheets it is a proxy for the offset. That is a "
          "coverage artefact,\nnot dynamics -- real QPIT families fill a "
          "continuous energy range.")

_off = np.abs(np.median(C_state - C_stored) - MU_EARTH_MOON * (1 - MU_EARTH_MOON) / 2)
if _off < 1e-6:
    print("\nNOTE: the median offset equals mu(1-mu)/2 -- a CONVENTION mismatch "
          "with\ncr3bp_3d, not a physical displacement. Reconcile before reading "
          "anything\nelse into it.")
""".rstrip()))

cells.append(code(r"""
# 2b. drawn vs measured distance -- regression test on the labeller
if {'d_drawn_km', 'source', 'random_dir'} <= set(D):
    m = ((D['source'] == 'offset') & (~D['random_dir'].astype(bool))
         & (~D['at_floor'].astype(bool)))
    rel = np.abs(D['d_km'][m] - D['d_drawn_km'][m]) / np.maximum(D['d_drawn_km'][m], 1e-30)
    print(f"projected-normal offsets: {m.sum()} samples")
    print(f"  relative error  median {np.median(rel)*100:.3f}%   "
          f"p95 {np.percentile(rel, 95)*100:.3f}%   max {rel.max()*100:.3f}%")
    print(f"  {'PASS' if np.median(rel) < 0.02 else 'FAIL'} against the 2% construction target")
else:
    print("d_drawn_km / source / random_dir absent -- labeller check skipped")
""".rstrip()))

cells.append(code(r"""
# 2c. RESOLVE THE TARGET, then census the censoring.
# log_d_km is not stored in this dataset, so it is derived here -- and the
# floor it is derived with comes from the FILE when the file has one, because a
# hardcoded constant sitting beside an authoritative value is the exact shape of
# the unit slips logged as Risk 20.
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

if 'log_d_km' in D:
    y_all = D['log_d_km'].astype(float)
    TARGET_COL = 'log_d_km (stored)'
else:
    y_all = np.log10(np.maximum(D['d_km'].astype(float), FLOOR_KM))
    TARGET_COL = 'log10(max(d_km, floor)) [derived]'
print(f"target: {TARGET_COL}")

at_floor = D['at_floor'].astype(bool)
_implied = D['d_km'].astype(float) < FLOOR_KM
if not np.array_equal(at_floor, _implied):
    print(f"  NOTE: at_floor disagrees with (d_km < floor) on "
          f"{(at_floor != _implied).sum()} rows -- the stored flag was set "
          f"against a different floor. Using the STORED flag.")
print(f"target log10 d[km]: {y_all.min():.3f} to {y_all.max():.3f}  "
      f"(={10**y_all.min()*1e3:.2f} m to {10**y_all.max():.1f} km)")
print(f"at_floor: {at_floor.sum()} / {n}  ({at_floor.mean()*100:.2f}%)")
print(f"floor value log10({FLOOR_KM}) = {np.log10(FLOOR_KM):.3f}; "
      f"rows sitting exactly there: {(np.abs(y_all - np.log10(FLOOR_KM)) < 1e-12).sum()}")

# 2d. group census
print()
for k, v in group_census(D['group_id']).items():
    print(f"  {k:14s} {v}")
""".rstrip()))

cells.append(code(r"""
# 2e. is d_res present? Week 12 needs it.
if 'd_res' in D:
    print(f"d_res present: {np.nanmin(D['d_res']):.4f} to {np.nanmax(D['d_res']):.4f}")
else:
    print("d_res ABSENT from the dataset.\n"
          "  Week 12 ('connect high-error regions to resonance proximity') "
          "assumes this column.\n"
          "  It is derivable from alpha_found -> member rho -> "
          "gap_sample.resonance_distance.\n"
          "  NOT built here: that is Week 12 scope. Flagged so it is not a "
          "surprise then.")

# 2f. cross-check the libration points against the project's own module
L = collinear_points(MU_EARTH_MOON)
print(f"\nL1 {L['L1']:.9f}   L2 {L['L2']:.9f}   L3 {L['L3']:.9f}")
try:
    import orbits
    for name in ('lagrange_points', 'libration_points', 'collinear_points'):
        if hasattr(orbits, name):
            print(f"  orbits.{name} ->", getattr(orbits, name)(MU_EARTH_MOON))
            break
    else:
        print("  (no comparable function found in orbits; values stand on the "
              "root check in tests/test_baselines.py)")
except Exception as e:
    print(f"  (orbits not importable here: {e})")
""".rstrip()))

cells.append(md(r"""
## 3. Features

Allowlist, not denylist. The dataset carries the label (`d_km`, `log_d_km`),
the quantity it was drawn from (`d_drawn_km`), and the surface coordinates the
state was constructed from and recovered at (`gap`, `alpha`, `theta`,
`amplitude_km`, `alpha_found`, `theta_found`). Any of those reaching the model
produces a spectacular and meaningless result, and a denylist that misses one
fails silently. `assert_no_leakage` fails loudly instead.

Energy enters as **continuous C**, not a one-hot over `energy_id`: a one-hot
cannot extrapolate to an unseen energy, and Week 15's transfer study needs
exactly that. Family enters as a **one-hot**, because L1 and L2 are different
objects and integer-coding them would assert an ordering that does not exist.
""".rstrip()))

cells.append(code(r"""
FAMILY_LEVELS = sorted(np.unique(D['family_id']).tolist())
X, FEAT_NAMES, SCALE_MASK = build_features(
    D['states'], JAC, D['family_id'], FAMILY_LEVELS)
assert_no_leakage(FEAT_NAMES)
print(f"X {X.shape}   features {FEAT_NAMES}")
print(f"standardised: {[n for n, s in zip(FEAT_NAMES, SCALE_MASK) if s]}")
print(f"left alone:   {[n for n, s in zip(FEAT_NAMES, SCALE_MASK) if not s]}")

y = y_all.copy()
""".rstrip()))

cells.append(md(r"""
## 4. Splits, and what they leave open

`group_id` bins (family, energy, gap, alpha, theta) — a patch of the family
surface. Samples inside one bin share a base point and a direction, so they are
near-duplicates. Splitting on rows would put them on both sides of the boundary
and the test score would measure memorisation.

What the grouped split asks is: *does the field interpolate to unseen patches of
sheets it has already seen.* Holding out a whole energy or a whole family is
strictly harder and is **Week 15**, not this week. Worth being explicit, since
the two get conflated and a reader will assume the harder claim.

**The leak grouping cannot close.** For a large offset the nearest surface point
is not the base point the state was built from, so a test sample's nearest torus
can sit inside a training patch even though its own group is held out. That is
not fixable by regrouping — the label is defined by the *found* point — so it
gets measured and reported beside the test score.
""".rstrip()))

cells.append(code(r"""
M = grouped_split(D['group_id'], FRAC_TEST, FRAC_VAL, seed=SEED_SPLIT)
assert groups_disjoint(D['group_id'], M), "grouped split leaked a group"

strata = None
if 'energy_id' in D and 'family_id' in D:
    strata = np.array([f"f{a}e{b}" for a, b in zip(D['family_id'], D['energy_id'])])
rep = split_report(y, M, group_id=D['group_id'], strata=strata)
for k in sorted(rep):
    v = rep[k]
    print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")
""".rstrip()))

cells.append(code(r"""
# residual contamination: does a test sample's NEAREST SURFACE POINT sit in a
# training patch? bins the recovered coordinates directly, so it does not
# matter whether alpha_found is global or within-gap -- only that both sides
# use the same convention.
if {'alpha_found', 'theta_found'} <= set(D):
    coords = np.column_stack([D['alpha_found'], D['theta_found']])
    keys = np.column_stack([D['family_id'], D.get('energy_id', np.zeros(n, int))])
    rng_ = (coords.min(axis=0), coords.max(axis=0))
    for nb in (16, 64, 256):
        r = nearest_point_overlap(coords[M['train']], coords[M['test']], n_bins=nb,
                                  keys_train=keys[M['train']], keys_test=keys[M['test']],
                                  ranges=rng_)
        print(f"  n_bins={nb:4d}  overlap {r['overlap_fraction']*100:6.2f}%  "
              f"train bins occupied {r['n_bins_occupied_train']}")
    print("\nThis BOUNDS how much of the test score could be near-neighbour "
          "recall.\nIt is a number to quote beside the headline metric, not a "
          "reason to redesign the split.")
else:
    print("alpha_found / theta_found absent -- overlap diagnostic skipped")
""".rstrip()))

cells.append(code(r"""
scaler = Standardizer(mask=SCALE_MASK).fit(X[M['train']])   # TRAIN ROWS ONLY
Z = scaler.transform(X)
print("train feature means after scaling:", np.round(Z[M['train']].mean(axis=0), 6))
print("train feature stds  after scaling:", np.round(Z[M['train']].std(axis=0), 6))
print("test  feature means after scaling:", np.round(Z[M['test']].mean(axis=0), 6),
      "\n  (not zero, and should not be -- the scaler never saw these rows)")
""".rstrip()))

cells.append(md(r"""
## 5. Baselines

**B0** mean predictor — defines the zero of R².

**B1** the shortcut probe: `(|r − L1|, |r − L2|, C)` and nothing else. The
Weekly Plan asks this to do *poorly*. But a weak learner fails on every
dataset, so a weak learner's failure proves nothing about the data — it is
fitted with gradient-boosted trees and k-NN, both of which can represent
essentially any function of three variables at this sample count, so that
failure is evidence of **absent information** rather than absent capacity.
Position only; adding velocity would make it a different and much stronger
baseline whose failure would say less.

**B2** 1-NN in standardised state space, under **both** splits. The gap between
them is the empirical correlation length of the dataset and the evidence that
grouping is necessary.
""".rstrip()))

cells.append(code(r"""
results = {}
tr, va, te = M['train'], M['val'], M['test']

# B0
b0 = MeanPredictor().fit(None, y[tr])
p_b0 = b0.predict(X[te])
results['B0 mean'] = regression_metrics(y[te], p_b0)

# B1
Xs, s_names = shortcut_features(D['states'], JAC)
sc_s = Standardizer().fit(Xs[tr]); Zs = sc_s.transform(Xs)
p_b1, info_b1 = fit_flexible(Zs[tr], y[tr], Zs[va], y[va], Zs[te], seed=0)
results['B1 shortcut'] = regression_metrics(y[te], p_b1)
print(f"B1 features {s_names}; selected {info_b1['selected']}; "
      f"val RMSE by learner: " +
      ", ".join(f"{k} {v:.3f}" for k, v in info_b1['candidates'].items()))
if not info_b1['sklearn_available']:
    print("  WARNING: sklearn missing, so no tree model was tried. B1's "
          "failure is then weaker evidence -- install sklearn and rerun.")

# B2, both splits
for tag, mm in (('grouped', M),
                ('random-row', random_row_split(n, FRAC_TEST, FRAC_VAL, seed=SEED_SPLIT))):
    sc_k = Standardizer(mask=SCALE_MASK).fit(X[mm['train']])
    Zk = sc_k.transform(X)
    p = KNNRegressor(k=1).fit(Zk[mm['train']], y[mm['train']]).predict(Zk[mm['test']])
    results[f'B2 1-NN ({tag})'] = regression_metrics(y[mm['test']], p)

rows = [dict(model=k, **{c: v[c] for c in ('n', 'r2', 'rmse', 'mae', 'bias')})
        for k, v in results.items()]
print()
print(format_table(rows, ['model', 'n', 'r2', 'rmse', 'mae', 'bias']))
""".rstrip()))

cells.append(md(r"""
## 6. M1 — the MLP

Standardised features in, `log10` km out. Smooth activation (SiLU) on purpose:
P3 says the network cannot follow the target into the near field because
∇(log d) ~ 1/d while the network's gradient is bounded by the product of its
weight norms. A ReLU network is also Lipschitz, but its kinks invite the
objection that any near-field failure is an activation artefact.

Early stopping is on the **group-disjoint** validation set. Stopping on a
row-random validation set would select the epoch that best memorises training
patches, which is exactly the quantity this week is trying not to measure.

Three seeds — a sanity band on whether the result survives re-initialisation,
not a confidence interval. If `hit_max_epochs` is True for any seed, the budget
ended the fit rather than convergence and the number below is a lower bound.
""".rstrip()))

cells.append(code(r"""
# BUDGET, NOT TUNING. The first run ended all three seeds at the epoch ceiling
# (best epochs 589/598/595 of 600) with validation loss still falling, so its
# R2 was a lower bound and its seed spread (0.17) exceeded the M1-B1 gap (0.13).
# Architecture, lr and batch size are unchanged; only the stopping rule moves.
cfg = MLPConfig(hidden=(256, 256, 256), activation='silu',
                max_epochs=3000, patience=150, lr_patience=50)
print("expect ~30-45 min for this cell plus the learning curve")
t1 = time.time()
P, H = train_multi_seed(Z[tr], y[tr], Z[va], y[va], Z[te], cfg,
                        seeds=SEEDS_MODEL, verbose=True)
print(f"\n{len(SEEDS_MODEL)} fits in {time.time()-t1:.0f}s")

per_seed = [regression_metrics(y[te], p) for p in P]
for s, h, m in zip(SEEDS_MODEL, H, per_seed):
    flag = "  <-- BUDGET-LIMITED" if h['hit_max_epochs'] else ""
    print(f"  seed {s}: R2 {m['r2']:+.4f}  RMSE {m['rmse']:.4f}  "
          f"epochs {h['epochs_run']} (best {h['best_epoch']}){flag}")

r2s = np.array([m['r2'] for m in per_seed])
print(f"\nR2 across seeds: {r2s.mean():+.4f} +/- {r2s.std():.4f} "
      f"(spread {r2s.max()-r2s.min():.4f})")

p_m1 = P.mean(axis=0)                      # seed-averaged predictions
results['M1 MLP (seed mean)'] = regression_metrics(y[te], p_m1)
print("seed-averaged:", {k: round(v, 4) for k, v in results['M1 MLP (seed mean)'].items()})
""".rstrip()))

cells.append(code(r"""
# censoring split -- decision 1. Are floor samples a different regime?
fl = at_floor[te]
print(f"test rows at floor: {fl.sum()} / {fl.size} ({fl.mean()*100:.2f}%)")
if fl.any() and (~fl).any():
    for tag, mm in (('at floor (left-censored)', fl), ('measured', ~fl)):
        m = regression_metrics(y[te][mm], p_m1[mm])
        print(f"  {tag:26s} n {m['n']:6d}  RMSE {m['rmse']:.4f}  "
              f"MAE {m['mae']:.4f}  bias {m['bias']:+.4f}")
    print("\nIf the censored rows carry a much larger and one-signed bias, a "
          "Tobit\nloss is worth the ten lines -- in WEEK 12. Recorded, not "
          "acted on.")
else:
    print("one side empty; censoring split not informative")
""".rstrip()))

cells.append(md(r"""
## 7. Learning curve

Trained on 10 / 25 / 50 / 100% of the **training groups**, held-out set fixed.
Subsampling groups rather than rows keeps each fold's effective sample size
honest: dropping rows inside a retained group removes near-duplicates, which is
not the same as having less data.

Reading it: a gap between train and test error that persists as data grows is
variance and more data helps; curves that meet at a high error are bias and
more data will not. This is the concrete artefact for the bias–variance dad
session.
""".rstrip()))

cells.append(code(r"""
from dataclasses import replace

FRACS = (0.10, 0.25, 0.50, 1.00)
lc = []
for f in FRACS:
    sub = subsample_groups(D['group_id'], tr, f, seed=SEED_SPLIT)
    # An epoch is not a fixed amount of work.  At batch 256 the 10% fold ran
    # five steps per epoch and never left the mean predictor (train RMSE 1.796
    # vs B0's 1.851) -- that was a failed fit being read as a data-size result.
    n_sub = int(sub.sum())
    cfg_f = replace(cfg, batch_size=int(min(cfg.batch_size, max(32, n_sub // 25))))
    mdl, h = train_mlp(Z[sub], y[sub], Z[va], y[va], cfg_f, seed=SEEDS_MODEL[0])
    m_te = regression_metrics(y[te], predict(mdl, Z[te], h))
    m_tr = regression_metrics(y[sub], predict(mdl, Z[sub], h))
    lc.append(dict(frac=f, n_groups=int(len(np.unique(D['group_id'][sub]))),
                   n_rows=int(sub.sum()), train_rmse=m_tr['rmse'],
                   test_rmse=m_te['rmse'], test_r2=m_te['r2'],
                   batch=cfg_f.batch_size, budget_limited=h['hit_max_epochs']))
    print(f"  {f:5.0%}  rows {sub.sum():6d}  batch {cfg_f.batch_size:4d}  "
          f"train RMSE {m_tr['rmse']:.4f}  "
          f"test RMSE {m_te['rmse']:.4f}  test R2 {m_te['r2']:+.4f}")

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot([r['n_rows'] for r in lc], [r['train_rmse'] for r in lc], 'o-', label='train')
ax.plot([r['n_rows'] for r in lc], [r['test_rmse'] for r in lc], 's-', label='test')
ax.set_xscale('log'); ax.set_xlabel('training rows (groups subsampled)')
ax.set_ylabel('RMSE  [log10 km]'); ax.set_title('Learning curve')
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.show()
""".rstrip()))

cells.append(md(r"""
## 8. The two curves

**Left — the deliverable.** Research Question row 9 specifies classification
recovered by thresholding the regression: true class (d < τ), predicted class
(d̂ < τ), agreement swept over τ. It **degenerates to 1 at both ends** — once τ
leaves the data range every sample is one class and any predictor is right — so
the mean or max over τ is meaningless. The honest scalar is the **minimum**,
and balanced accuracy and MCC are shown alongside because the class prior at
each τ is set by the log-uniform draw rather than by anything physical.

**Right — the one the warm-start story wants.** Fraction with |ŷ − y| ≤ tol, in
log₁₀ km. An RMSE of 0.2 in log₁₀ is a typical factor-of-1.6 error in the
distance handed to a solver, which is legible in a way R² is not.
""".rstrip()))

cells.append(code(r"""
curve = accuracy_vs_threshold(y[te], p_m1, n_tau=80)
summary = summarize_threshold_curve(curve)
etol = accuracy_vs_error_tolerance(y[te], p_m1, tols=np.linspace(0, 2.0, 81))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
a = axes[0]
a.plot(curve['tau'], curve['accuracy'], label='accuracy')
a.plot(curve['tau'], curve['balanced_accuracy'], label='balanced accuracy')
a.plot(curve['tau'], curve['mcc'], label='MCC', ls='--')
a.axvline(summary.get('tau_at_min', np.nan), color='0.5', lw=1, ls=':')
a.axhline(0.5, color='0.8', lw=1)
a.set_xlabel(r'threshold $\tau$  [log10 km]'); a.set_ylabel('score')
a.set_title('Classification as a thresholded readout')
a.grid(alpha=.3); a.legend(fontsize=8)

b = axes[1]
b.plot(etol['tol'], etol['fraction_within'])
for t in (0.1, 0.3, 1.0):
    f = float(np.interp(t, etol['tol'], etol['fraction_within']))
    b.plot([t], [f], 'o', ms=5)
    b.annotate(f'{f:.2f} within x{10**t:.1f}', (t, f), textcoords='offset points',
               xytext=(6, -10), fontsize=8)
b.set_xlabel('error tolerance  [log10 km]'); b.set_ylabel('fraction within')
b.set_title('Accuracy vs error tolerance')
b.grid(alpha=.3)
plt.tight_layout(); plt.show()

print("threshold-curve summary (restricted to informative tau):")
for k, v in summary.items():
    print(f"  {k:26s} {v}")
""".rstrip()))

cells.append(md(r"""
## 9. Residuals against the target — the P3 check

Binned on the **target's own magnitude**, not on phase-space location. The
phase-space error map is Week 12; this is only enough to score P3.
""".rstrip()))

cells.append(code(r"""
# MEASURED ROWS ONLY. 13% of the test set is pinned at the floor, all of it in
# the smallest-d bins, and left-censoring produces a large positive bias there
# mechanically. Scoring P3 on all rows confounds the two.
rb = residual_bins(y[te][~fl], p_m1[~fl], n_bins=12)
print("measured rows only (at_floor excluded):")
print(format_table(rb, ['bin_lo', 'bin_hi', 'n', 'bias', 'rmse', 'mae']))

ctr = np.array([(r['bin_lo'] + r['bin_hi']) / 2 for r in rb])
bias_b = np.array([r['bias'] for r in rb])
rmse_b = np.array([r['rmse'] for r in rb])
sd_b = np.sqrt(np.maximum(rmse_b ** 2 - bias_b ** 2, 0.0))   # RMSE^2 = bias^2 + var

# IS THERE A DEAD ZONE?  Under mean reversion pred = mu + lam*(y-mu), so
# bias + centre = mu + (lam-1)*... is NOT flat unless lam = 0.  A flat
# (bias + centre) therefore means the model emits a CONSTANT and ignores the
# input over those bins -- a stronger failure than smoothing, and the thing to
# measure.  RMSE alone cannot see it, which is why residual SD is reported.
implied = bias_b + ctr
k = 1
while k < len(implied) and np.ptp(implied[:k + 1]) < 0.10:
    k += 1
print(f"\nimplied constant output per bin (bias + centre):")
print("  " + "  ".join(f"{v:+.3f}" for v in implied))
if k >= 3:
    slope = np.polyfit(ctr[:k], bias_b[:k], 1)[0]
    c0 = float(implied[:k].mean())
    print(f"\nDEAD ZONE: first {k} bins are flat to {np.ptp(implied[:k]):.3f}")
    print(f"  bias-vs-centre slope {slope:+.3f} (-1 means the output ignores the input)")
    print(f"  constant output {c0:+.3f} log10 km = {10**c0*1e3:.1f} m")
    print(f"  boundary {rb[k-1]['bin_hi']:+.3f} log10 km = "
          f"{10**rb[k-1]['bin_hi']*1e3:.0f} m -- below this the regressor does "
          f"not respond to the state")
else:
    print("\nno flat prefix: the near field is tracked, not abandoned")

near, far = [r for r in rb if r['n'] > 0][:3], [r for r in rb if r['n'] > 0][-3:]
nb = float(np.mean([r['bias'] for r in near])); fb = float(np.mean([r['bias'] for r in far]))
nr = float(np.mean([r['rmse'] for r in near])); fr = float(np.mean([r['rmse'] for r in far]))
nsd = float(np.sqrt(max(nr ** 2 - nb ** 2, 0))); fsd = float(np.sqrt(max(fr ** 2 - fb ** 2, 0)))
print(f"\nnear field: bias {nb:+.4f}  RMSE {nr:.4f}  residual SD {nsd:.4f}")
print(f"far  field: bias {fb:+.4f}  RMSE {fr:.4f}  residual SD {fsd:.4f}")
print("\nP3a is scored on residual SD, NOT RMSE. RMSE^2 = bias^2 + variance, so "
      "an\nRMSE comparison counts mean reversion as scatter and reports 'held' "
      "when the\nnear field is in fact the LESS scattered end.")

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
a = axes[0]
a.scatter(y[te][~fl], p_m1[~fl] - y[te][~fl], s=3, alpha=.25, lw=0, label='measured')
a.scatter(y[te][fl], p_m1[fl] - y[te][fl], s=3, alpha=.25, lw=0, color='tab:orange',
          label='at floor')
a.plot(ctr, bias_b, 'o-', color='tab:red', label='bin bias (measured)')
a.axhline(0, color='k', lw=1)
a.set_xlabel('true log10 d [km]'); a.set_ylabel('residual  (pred - true)')
a.set_title('Residual vs target'); a.grid(alpha=.3); a.legend(fontsize=8)

b = axes[1]
b.plot(ctr, rmse_b, 'o-', label='RMSE')
b.plot(ctr, sd_b, 's-', label='residual SD')
b.plot(ctr, np.abs(bias_b), '^--', label='|bias|')
b.set_xlabel('true log10 d [km]'); b.set_ylabel('log10 km')
b.set_title('Error decomposition'); b.grid(alpha=.3); b.legend(fontsize=8)
plt.tight_layout(); plt.show()
""".rstrip()))

cells.append(md(r"""
## 10. Scorecard

Predictions were stated in the header, before anything ran. Filled in here
against what actually happened, falsifications included.
""".rstrip()))

cells.append(code(r"""
r2_b1 = results['B1 shortcut']['r2']
r2_g = results['B2 1-NN (grouped)']['r2']
r2_r = results['B2 1-NN (random-row)']['r2']
r2_m1 = results['M1 MLP (seed mean)']['r2']

verdict = {
    'P1 shortcut baseline fails (R2 < 0.5)':
        (r2_b1 < 0.5, f"B1 R2 = {r2_b1:+.4f}"),
    'P2 1-NN gap random > grouped by >0.1':
        (r2_r - r2_g > 0.1, f"random {r2_r:+.4f} vs grouped {r2_g:+.4f}, "
                            f"gap {r2_r - r2_g:+.4f}"),
    'P3a near-field residual SD exceeds far-field':
        (nsd > fsd, f"near SD {nsd:.4f} vs far SD {fsd:.4f} "
                    f"(RMSE {nr:.4f} vs {fr:.4f} -- bias-inflated)"),
    'P3b near-field bias is positive (measured rows only)':
        (nb > 0, f"near-field bias {nb:+.4f}, censored rows excluded"),
    'P3c overall R2 > 0.9 (contradicts Risk 19)':
        (r2_m1 > 0.9, f"M1 R2 = {r2_m1:+.4f}"),
}
for k, (ok, detail) in verdict.items():
    print(f"  [{'HELD' if ok else 'FALSIFIED':9s}] {k}\n              {detail}")

print("\nA falsification here is a result, not a failure.\n")
print("READ P3c CAREFULLY. A falsified P3c means the >0.9 prediction was wrong.")
print("It is NOT evidence for Risk 19's mechanism. Risk 19 attributes mediocre")
print("R2 to Cantor-set non-smoothness, but the labeller measures distance to a")
print("DENSIFIED INTERPOLATED CONTINUUM at ~12.3 km spacing and cannot resolve")
print("the gaps -- that structure is absent from the target as generated. So")
print("Risk 19 stays untested; this experiment has no power over it.\n")
print("P3a and P3b holding is the substantive half: the predicted MECHANISM")
print("(bounded network gradients cannot track grad(log d) ~ 1/d) is visible in")
print("the near-field bias, even though the magnitude prediction failed.\n")
print("If B2 scores near or below zero under BOTH splits, P2 is falsified but")
print("UNINFORMATIVE -- a baseline at zero cannot resolve a gap. The evidence")
print("about row dependence then comes from the group census, not from B2.")
""".rstrip()))

cells.append(code(r"""
rows = [dict(model=k, **{c: v[c] for c in ('n', 'r2', 'rmse', 'mae', 'bias')})
        for k, v in results.items()]
print(format_table(rows, ['model', 'n', 'r2', 'rmse', 'mae', 'bias']))

out = dict(
    week=11, generated=time.strftime('%Y-%m-%d %H:%M'),
    data=DATA, n_rows=int(n), features=FEAT_NAMES,
    energy_feature='stored sheet C' if SHEET_LIKE else 'stored C (continuous)',
    n_energy_sheets=int(sheets.size),
    energy_leak_corr=float(np.corrcoef(np.log10(dC[_m]), _y[_m])[0, 1])
    if _m.sum() > 50 else None,
    split=dict(seed=SEED_SPLIT, frac_test=FRAC_TEST, frac_val=FRAC_VAL),
    split_report={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in rep.items()},
    results={k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    per_seed_r2=[float(m['r2']) for m in per_seed],
    threshold_summary={k: (list(v) if isinstance(v, tuple) else float(v))
                       for k, v in summary.items()},
    learning_curve=lc,
    residual_bins=rb,
    verdict={k: bool(v[0]) for k, v in verdict.items()},
    config=cfg.__dict__,
)
with open('../data/wk11/results_wk11.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
np.savez_compressed('../data/wk11/predictions_wk11.npz',
                    test_mask=te, y_test=y[te], pred_seeds=P, pred_mean=p_m1,
                    pred_b1=p_b1, at_floor_test=fl,
                    tau=curve['tau'], acc=curve['accuracy'],
                    bal_acc=curve['balanced_accuracy'], mcc=curve['mcc'],
                    tol=etol['tol'], frac_within=etol['fraction_within'])
print(f"\nsaved -> ../data/wk11/  [{time.time()-t0:.0f}s total]")
""".rstrip()))

cells.append(code(r"""
import sys
import numpy as np
sys.path.insert(0, '../src')
from metrics import residual_bins

z = np.load('../data/wk11/predictions_wk11.npz')
y, p, fl = z['y_test'], z['pred_mean'], z['at_floor_test']

print("fraction of test rows whose predicted distance is within a factor of N")
print(f"{'factor':>7s} {'log10 tol':>10s} {'all':>8s} {'measured':>9s}")
for factor in (1.3, 2, 3, 5, 10, 30, 100):
    t = np.log10(factor)
    print(f"{factor:7.1f} {t:10.3f} {np.mean(np.abs(p - y) <= t):8.3f} "
          f"{np.mean(np.abs(p[~fl] - y[~fl]) <= t):9.3f}")

print("\ndead-zone boundary vs bin resolution (measured rows only)")
for nb in (12, 18, 24, 36):
    rb = residual_bins(y[~fl], p[~fl], n_bins=nb)
    ctr = np.array([(r['bin_lo'] + r['bin_hi']) / 2 for r in rb])
    bias = np.array([r['bias'] for r in rb])
    imp = bias + ctr
    k = 1
    while k < len(imp) and np.ptp(imp[:k + 1]) < 0.10:
        k += 1
    if k >= 3:
        print(f"  n_bins={nb:3d}  flat over {k:2d} bins  "
              f"constant {10**imp[:k].mean()*1e3:6.1f} m  "
              f"boundary {10**rb[k-1]['bin_hi']*1e3:7.0f} m  "
              f"slope {np.polyfit(ctr[:k], bias[:k], 1)[0]:+.3f}  "
              f"width {rb[0]['bin_hi']-rb[0]['bin_lo']:.3f} dec")
    else:
        print(f"  n_bins={nb:3d}  no flat prefix")
"""))

cells.append(md(r"""
## Carried into Week 12

- `d_res`, if section 2e reported it absent — Week 12's resonance-proximity
  analysis assumes the column exists.
- The censoring split from section 6. If the floor rows carry a large one-signed
  bias, a Tobit loss is the principled fix and it is a Week 12 change.
- Whichever predictions were falsified.
- Risk 18 is marked **Realized** in the tracker: no resonance-adjacent negatives
  were obtainable. Week 12's "connect high-error regions to resonance proximity"
  was written before that. Worth raising with Dr. Lang before Week 12 starts
  rather than discovering it mid-week.

**Still open from Week 10, not Week 11 work:** the project title decision
(Research Question row 13) needs sign-off *before* the STS abstract is written,
and the two STS essays. This week's STS task assumes those drafts exist.
""".rstrip()))

nb = dict(cells=cells,
          metadata=dict(kernelspec=dict(display_name="Python 3",
                                        language="python", name="python3"),
                        language_info=dict(name="python", version="3.11")),
          nbformat=4, nbformat_minor=5)

# relative to THIS FILE, not the working directory -- otherwise the script only
# works when run from the repo root and fails from inside notebooks/
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'nb11_first_regressor.ipynb')
write_notebook(nb, out_path)
