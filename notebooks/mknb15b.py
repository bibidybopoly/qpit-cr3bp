"""Builder for nb15_fits.ipynb -- Week 15's eight fits and their readouts.

Separate from nb15_gates.ipynb on purpose.  The gates can still change the
week's design; these cells assume they passed.  Run nb15_gates first.

Divergence is measured on CELL SOURCE ONLY.  Any hand-edit goes back here.
"""

import json
import os
import sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nb15_fits.ipynb')

CELLS = [

("markdown", """# Week 15 -- transfer fits

Pre-registration: `W15_PREREG.md`. Eight fits, five seeds each, one feature set
throughout. Run `nb15_gates.ipynb` first; G15.1 and G15.2 gate this notebook.

Predictions are written to disk before anything is scored, so scoring can be
re-run without refitting and a scoring change cannot quietly refit.
"""),

("code", """# --- ADAPTER CELL -------------------------------------------------------
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join('..', 'src'))

from splits import leave_one_level_out, grouped_split, subsample_groups, group_census
from features import (build_features, Standardizer, assert_feature_set,
                      assert_sheet_energy, FEATURE_SET_W15)
from models import train_mlp, predict, MLPConfig
from metrics import regression_metrics, retention, retention_difference
from mi import make_windows, DEAD_ZONE_KM_W12
import constants

z = np.load('../data/qpit_dataset_w10.npz', allow_pickle=True)
gid, fid, eid = z['group_id'], z['family_id'], z['energy_id']
states, C, d_km, at_floor = z['states'], z['C'], z['d_km'], z['at_floor']
FLOOR_KM = float(z['floor_km'])
assert abs(FLOOR_KM - constants.FLOOR_KM) < 1e-12

# THE ENERGY COLUMN.  CORRECTED 2026-09-11: the archive does NOT carry a
# per-state Jacobi.  `sampling.label_and_pack` writes `jacobi=np.full(n, C)`,
# so the archive's `jacobi` and `C` columns are byte-identical (verified on
# data/qpit_dataset_w10.npz).  The dangerous column is the RECOMPUTED
# per-state Jacobi -- `features.jacobi_constant(states)` -- which is not in
# the archive at all.  It scores R2 = +0.7976 alone against -0.0001 for the
# sheet column, because the offsets were never reprojected, so a state's own
# energy differs from its sheet's by an amount proportional to the target.
# Feeding it manufactures the shortcut and it would look like a triumph.
# `assert_sheet_energy` still earns its place: it is what makes the column
# handed to build_features the sheet energy rather than something recomputed.
assert_sheet_energy(C, eid)
print('sheet energy confirmed: C is constant within energy_id')

y = np.log10(np.maximum(d_km, FLOOR_KM))
W = make_windows(d_km, ~at_floor, floor_km=FLOOR_KM,
                 boundary_km=DEAD_ZONE_KM_W12)
FAR, NEAR = W['far'], W['near']
assert W['n_far'] == 8908, f"far window drifted: {W['n_far']} != 8908"
print('rows', len(y), ' far', W['n_far'], ' near', W['n_near'])
"""),

("markdown", "## Feature set\n\nOne feature set for all eight fits. The family "
             "one-hot is dropped: with a family held out its column is "
             "constant-zero across training, so its weights receive zero "
             "gradient and stay at initialisation while the survivor trains as "
             "a bias, and both flip at test time. The energy folds drop it too, "
             "so a family fold and an energy fold are the same model and their "
             "retention ratios are comparable."),

("code", """X_all, names, scale_mask = build_features(
    states, C, fid, [0, 1], include_family=False)
assert_feature_set(names, FEATURE_SET_W15)
print('features:', names, X_all.shape)
"""),

("markdown", "## The eight fits\n\nOne control per axis serves every fold on "
             "that axis: its training pool excludes every test set on the axis, "
             "and all folds on an axis have the same nominal n. Control pools "
             "come to 16,320 rows either way, since 2 x 2,040 and 3 x 1,360 are "
             "both 4,080."),

("code", """SEEDS = [0, 1, 2, 3, 4]
# NOT MLPConfig().  The defaults are max_epochs=600, min_epochs=60,
# patience=80.  Research log line 1188: M1 moved 0.539 -> 0.632 +/- 0.020 only
# after the epoch budget went to 3000/150/50, and "the budget-limited 600-epoch
# run reporting a converged R2" is entry 2 on the silent-success list.  Running
# the bare default here would reproduce that failure across all eight fits and
# every one would return a plausible number.
#
# CORRECTED 2026-09-11.  "3000/150/50" is (max_epochs, patience, lr_patience),
# as written in mknb11.py:613 and mknb12.py:114.  This line previously read
#     MLPConfig(max_epochs=3000, min_epochs=150, patience=50)
# which put 150 into min_epochs and 50 into patience, leaving lr_patience at
# its default 25.  So the archived Week 15 run used patience=50 (a third of
# the frozen configuration) and lr_patience=25 (half).  That is visible in
# data/wk15/history_wk15.json: epochs_run - best_epoch is exactly 51 on all
# forty fits, i.e. patience + 1 at patience=50; at 150 it would be 151.
# Every fold and its matched-n control ran the same schedule, so the WITHIN-
# Week-15 comparisons are unaffected; the cross-week comparison against the
# Week 11/12 archive is not.  See docs/W15_RESULTS.md section 1.
# max_epochs 3000 -> 8000, 2026-09-11.  With patience corrected to 150 the
# budget became the binding constraint instead: 13 of the 40 fits in the
# 3000-epoch run reported hit_max_epochs, concentrated in T1 (4/5 seeds) and
# T2 (4/5), with epochs_run - best_epoch as low as 1 -- still improving when
# the budget ended.  By this project's own rule those are LOWER BOUNDS, not
# fits, and P15.2 is computed entirely from T1 and T2.  The fits that did
# converge needed at most best_epoch 2509, so 8000 is roughly 3x headroom over
# the largest converged run and early stopping should end most fits well
# before it.  If hit_max_epochs still fires at 8000, that is itself the
# finding -- do not raise it again without saying so in the log.
CFG = MLPConfig(max_epochs=8000, patience=150, lr_patience=50)

# G15.2 IS A GATE, SO ASSERT IT RATHER THAN RECORDING THAT IT PASSED.  The
# interpolation/extrapolation reading of P15.3 is void unless energy_id is
# monotone in C; the comment used to carry that as a claim.  It holds on this
# archive (3.0294, 3.0647, 3.1306) and costs one line to keep holding.
_C_by_eid = [float(np.unique(C[eid == e])[0]) for e in (0, 1, 2)]
assert _C_by_eid == sorted(_C_by_eid), (
    f"G15.2 FAILS: energy_id is not monotone in C ({_C_by_eid}); E_a/E_b/E_c "
    "below would mislabel which fold is interpolation")
E_A, E_B, E_C = 0, 1, 2    # energy_id, monotone in C -- asserted above

FOLDS = {}
for tag, key, lv in [('T1', fid, 1), ('T2', fid, 0),
                     ('T3', eid, E_A), ('T4', eid, E_B), ('T5', eid, E_C)]:
    m, meta = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                  frac_val=0.1, seed=0)
    FOLDS[tag] = dict(masks=m, meta=meta, key=key, level=lv,
                      axis='family' if tag in ('T1', 'T2') else 'energy')
    print(f"{tag}: train={meta['train_rows']} val={meta['val_rows']} "
          f"test={meta['test_rows']} excluded={meta['excluded_rows']}")

# Controls: mixed-composition training at the transfer run's n, scored on the
# transfer run's exact test rows.  Without this, a family fold trains on 36%
# less data than B0 and any degradation is confounded with sample size.
CONTROLS = {}
for axis, tags in (('family', ['T1', 'T2']), ('energy', ['T3', 'T4', 'T5'])):
    held_out_test = np.zeros(len(gid), bool)
    for t in tags:
        held_out_test |= FOLDS[t]['masks']['test']
    pool = ~held_out_test
    target_n = int(np.mean([FOLDS[t]['meta']['train_rows'] +
                            FOLDS[t]['meta']['val_rows'] for t in tags]))
    frac = target_n / int(pool.sum())
    sub = subsample_groups(gid, pool, frac, seed=0)
    inner = grouped_split(gid[sub], frac_test=0.0, frac_val=0.1, seed=0)
    idx = np.flatnonzero(sub)
    tr = np.zeros(len(gid), bool); tr[idx[inner['train']]] = True
    va = np.zeros(len(gid), bool); va[idx[inner['val']]] = True
    CONTROLS[axis] = dict(train=tr, val=va, serves=tags)
    got = int(tr.sum()) + int(va.sum())
    print(f"C_{axis}: target n={target_n} achieved={got} "
          f"({abs(got-target_n)/target_n:.2%} off; G15.3 allows 3%)")
    assert abs(got - target_n) / target_n <= 0.03, 'G15.3 matched-n tolerance'

B0 = grouped_split(gid, frac_test=0.2, frac_val=0.1, seed=0)
print(f"B0: train={int(B0['train'].sum())} val={int(B0['val'].sum())} "
      f"test={int(B0['test'].sum())}")
"""),

("markdown", "## Fit\n\nEight fits x five seeds. Predictions over all 20,400 "
             "rows are stored so any fold's test rows can be scored later "
             "without refitting. `is_mean_predictor` and `val_r2` are recorded "
             "per seed: Week 12's M2b seed 2 collapsed and was averaged into a "
             "five-seed band as though it were a fit."),

("code", """def fit_one(train_mask, val_mask, seed):
    sc = Standardizer(scale_mask).fit(X_all[train_mask])
    model, hist = train_mlp(sc.transform(X_all[train_mask]), y[train_mask],
                            sc.transform(X_all[val_mask]), y[val_mask],
                            cfg=CFG, loss='mse', seed=seed)
    return predict(model, sc.transform(X_all), hist), hist


RUNS = {'B0': (B0['train'], B0['val'])}
for axis, c in CONTROLS.items():
    RUNS[f'C_{axis}'] = (c['train'], c['val'])
for tag, f in FOLDS.items():
    RUNS[tag] = (f['masks']['train'], f['masks']['val'])

PRED, HIST = {}, {}
for name, (tr, va) in RUNS.items():
    t0 = time.time()
    ps, hs = [], []
    for s in SEEDS:
        p, h = fit_one(tr, va, s)
        ps.append(p); hs.append(h)
        # Both ends of the budget, because both are silent.  A mean predictor
        # reports early_stopped=True as though converged; a budget-limited run
        # reports a converged R2 it never reached.
        if h['is_mean_predictor']:
            print(f"  !! {name} seed {s} is a MEAN PREDICTOR "
                  f"(val_r2={h['val_r2']:.4f}) -- do not average it in silently")
        if h['hit_max_epochs']:
            print(f"  !! {name} seed {s} HIT MAX EPOCHS at {h['epochs_run']} "
                  f"-- the fit is budget-limited, not converged")
    PRED[name] = np.stack(ps)
    # `config` IS RECORDED.  It was left out of this allowlist, and that is
    # why a mis-specified CFG survived a whole week: the archive recorded what
    # the fits DID and not what they were ASKED to do, so the only trace of
    # patience=50 was an epochs_run - best_epoch gap of 51 that nobody had a
    # reason to look at.  train_mlp already puts asdict(cfg) in the history;
    # it costs a few hundred bytes to keep it.
    HIST[name] = [{k: h[k] for k in ('val_r2', 'is_mean_predictor',
                                     'hit_max_epochs', 'early_stopped',
                                     'epochs_run', 'best_epoch',
                                     'best_val_loss', 'config',
                                     'seed')} for h in hs]
    print(f"{name}: {len(SEEDS)} seeds in {time.time()-t0:.0f}s  "
          f"val_r2={np.mean([h['val_r2'] for h in hs]):.4f}")

bad = {n: [i for i, h in enumerate(HIST[n])
           if h['is_mean_predictor'] or h['hit_max_epochs']]
       for n in HIST}
bad = {n: v for n, v in bad.items() if v}
if bad:
    print()
    print('SEEDS THAT ARE NOT CLEAN FITS:', bad)
    print('Decide what happens to these BEFORE scoring.  Week 12 averaged a '
          'collapsed seed into a five-seed band as though it were a fit.')

os.makedirs('../data/wk15', exist_ok=True)
np.savez_compressed('../data/wk15/predictions_wk15.npz', **PRED)
with open('../data/wk15/history_wk15.json', 'w') as fh:
    json.dump(HIST, fh, indent=1)
print('written to ../data/wk15/')
"""),

("markdown", "## Scoring\n\nRaw far-field R^2 first, then retention. Raw R^2 is "
             "not comparable across folds: the Week 13 per-cell MI implies "
             "Gaussian-equivalent ceilings of about 0.19-0.28 for L1 cells and "
             "0.39-0.58 for L2, so testing on L2 outscores testing on L1 before "
             "any model is fit. Retention divides that out."),

("code", """rows = []
RET = {}
for tag, f in FOLDS.items():
    test_far = f['masks']['test'] & FAR
    ctrl = f"C_{f['axis']}"
    r_t = [regression_metrics(y[test_far], PRED[tag][s][test_far])['r2']
           for s in range(len(SEEDS))]
    RET[tag] = retention(y, PRED[tag], PRED[ctrl], mask=test_far)
    RET[tag + '_vs_B0'] = retention(y, PRED[tag], PRED['B0'], mask=test_far)
    rows.append(dict(fold=tag, axis=f['axis'], n_far=int(test_far.sum()),
                     r2_raw=float(np.mean(r_t)),
                     r2_control=RET[tag]['r2_control'],
                     retention=RET[tag]['retention'],
                     se=RET[tag]['se'], defined=RET[tag]['defined']))
    if not RET[tag]['defined']:
        print(f"  {tag}: retention undefined -- {RET[tag]['reason']}")

from metrics import format_table
print(format_table(rows))
"""),

("markdown", "## Pre-registered readouts\n\nEach resolves to confirmed, "
             "falsified, or **unresolved**. Far-field test n runs 569-923, so "
             "unresolved is a live outcome, not a formality. Falsifications are "
             "reported as prominently as confirmations."),

("code", """def verdict(name, statement, got, want):
    # BUG FIXED: this mapped anything that was not `want` and not 'unresolved'
    # to FALSIFIED, so a prediction whose metric could not be COMPUTED was
    # recorded as falsified.  A prediction that was never evaluated is VOID.
    # Logging it as falsified is a pre-registration integrity failure, not a
    # cosmetic one, and it is the direction that flatters -- it makes an
    # unusable metric look like a finding.
    if got == 'undefined':
        ok = 'VOID -- metric not computable, prediction NOT evaluated'
    elif got == 'unresolved':
        ok = 'UNRESOLVED'
    elif got == want:
        ok = 'CONFIRMED'
    else:
        ok = 'FALSIFIED'
    print(f"{name}: {ok}")
    print(f"   {statement}")
    print(f"   got: {got}")
    return ok

# P15.1 -- raw R2 orders by TEST cell, not by training cell.
raw = {r['fold']: r['r2_raw'] for r in rows}
# P15.1 compares two raw R2 values.  If either is far below zero the ordering
# is between two failures and says nothing about ceilings, which is what the
# prediction was about.  Guard it rather than read a sign off garbage.
if min(raw['T1'], raw['T2']) < -1.0:
    verdict('P15.1', 'raw R2(T1, test L2) > raw R2(T2, test L1)',
            'undefined', 'a_greater')
    print(f"   both raw R2 are far below zero ({raw['T1']:.1f}, "
          f"{raw['T2']:.1f}); an ordering between two failures is not the "
          f"ceiling comparison this predicted")
else:
    verdict('P15.1', 'raw R2(T1, test L2) > raw R2(T2, test L1)',
            'a_greater' if raw['T1'] > raw['T2'] else 'b_greater', 'a_greater')

# P15.2 -- retention orders OPPOSITE to raw R2.
d = retention_difference(RET['T2'], RET['T1'])
verdict('P15.2', 'retention(T2: L2->L1) > retention(T1: L1->L2)',
        d['verdict'], 'a_greater')
print(f"   difference={d['difference']:.4f} pooled_se={d['pooled_se']:.4f} "
      f"n_sigma={d.get('n_sigma', float('nan')):.2f}")

# P15.3 -- interpolation beats extrapolation across energy.
worse = 'T3' if RET['T3']['retention'] < RET['T5']['retention'] else 'T5'
best_extrap = 'T5' if worse == 'T3' else 'T3'
d3 = retention_difference(RET['T4'], RET[best_extrap])
verdict('P15.3', 'retention(T4, interpolation) > both extrapolation folds',
        d3['verdict'], 'a_greater')

# P15.5 -- the near field is vacuous, reported as such rather than as a
# transfer failure.  A near-field number materially ABOVE the Week 12
# within-window R2 of -0.133 is a surprise needing explanation before it is
# reported, not after.
print()
for tag, f in FOLDS.items():
    tn = f['masks']['test'] & NEAR
    r = float(np.mean([regression_metrics(y[tn], PRED[tag][s][tn])['r2']
                       for s in range(len(SEEDS))]))
    flag = '  <-- ABOVE the W12 band, explain before reporting' if r > -0.05 else ''
    print(f"P15.5 {tag}: near-field R2 = {r:.4f} (n={int(tn.sum())}){flag}")
"""),

("markdown", "## Not run here\n\n`nearest_point_overlap` is declared not "
             "applicable to a level holdout (W15_PREREG.md §6a): with the full "
             "key set the overlap is zero by construction, and without it the "
             "binning compares parameterisation coordinates across families "
             "rather than positions. The recovered coordinates it documents as "
             "inputs are absent from this archive in any case.\n\n"
             "P15.6 is void on discovery: `gap` is an interval index, not a "
             "width, and the archive carries no per-row gap width. Risk 19 goes "
             "to Limitations."),
]


def build():
    nb = dict(cells=[], metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python",
                        name="python3"),
        language_info=dict(name="python")),
        nbformat=4, nbformat_minor=5)
    for kind, src in CELLS:
        cell = dict(cell_type=kind, metadata={}, source=src.splitlines(True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb['cells'].append(cell)
    return nb


def check_cells_compile(nb):
    """Every code cell must parse.  Cell source passes through a Python literal
    here and then JSON, and an escape surviving only one level lands as a real
    line break inside a string -- invisible in both files, a SyntaxError on
    execution.  "The builder wrote N cells" is a proxy; this is the condition.
    """
    bad = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        try:
            compile(''.join(c['source']), f'<cell {i}>', 'exec')
        except SyntaxError as e:
            bad.append((i, e))
    return bad


def sources(nb):
    return [''.join(c['source']) for c in nb['cells']]


def main():
    fresh = build()
    bad = check_cells_compile(fresh)
    if bad:
        for i, e in bad:
            print(f"cell {i} does not compile: {e}")
        print("REFUSING to write a notebook with cells that cannot run.")
        return 1
    if os.path.exists(NB):
        with open(NB) as f:
            existing = json.load(f)
        if sources(existing) != sources(fresh):
            if '--force' not in sys.argv:
                print(f"REFUSING to overwrite {NB}: cell source has diverged.")
                print(f"Fold the hand-edit back into {__file__} and re-run, "
                      f"or pass --force to discard the notebook's version.")
                return 1
            print(f"--force: overwriting a DIVERGED {os.path.basename(NB)}")
        else:
            # IDENTICAL SOURCE -> DO NOT WRITE.  Rewriting is a no-op on the
            # source and strips every executed output, which is a silent loss
            # of work: the exact failure mknb11.py's guard was built after
            # (33 cells / 23 outputs, 2026-08-16).  Both this builder and
            # mknb15b.py used to fall through to the write here.
            n_out = sum(len(c.get('outputs', [])) for c in existing['cells']
                        if c['cell_type'] == 'code')
            print(f"{os.path.basename(NB)}: already up to date "
                  f"({len(fresh['cells'])} cells, {n_out} executed outputs "
                  f"preserved) -- not rewriting")
            return 0
    with open(NB, 'w') as f:
        json.dump(fresh, f, indent=1)
    print(f"wrote {NB} ({len(fresh['cells'])} cells)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
