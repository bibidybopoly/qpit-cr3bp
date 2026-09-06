"""Which input direction does the divergence ride on?  Optional.

WHAT THE SUPPORT DIAGNOSTIC ALREADY SETTLED, AND WHAT IT DID NOT.

Between folds, distance off support tracks the damage: ordering the folds by
max|z| gives T1 45.3 > T2 8.0 > T3 7.9 > T5 4.7 > T4 2.6, and by RMSE ratio
gives T1 14.1 > T2 8.0 > T5 5.7 > T3 2.6 > T4 1.2 -- four of five concordant.

Within folds it does not, and the reason is degeneracy rather than absence.
T1 and T2 have max|z| equal to p99|z| (45.3 and 45.3; 8.0 and 8.0) with 100% of
rows beyond |z| = 5.  A family holdout is a RIGID covariate shift: every
held-out row is displaced by nearly the same amount, so a rank correlation
against max|z| has almost no variance to work with.  rho = +0.31 and +0.06 are
therefore uninformative, NOT evidence against off-support extrapolation.

T5 IS THE COUNTEREXAMPLE AND IT IS THE INTERESTING ONE.  Every T5 test row sits
within |z| = 4.7 -- the same as B0's in-distribution maximum, 0% beyond 5 --
and T5 is still 5.67x worse than its control with 42% of rows wrong by over
three decades.  Standardised distance therefore does NOT suffice.  Its worst
feature is C, and the hypothesis is that C is the reason: with a level held out
the model sees exactly TWO distinct C values in training, and the slope it fits
between two points is unconstrained by data, so a small standardised step
beyond them can move the output arbitrarily.  A feature at 1 sigma with two
training levels is not comparable to a feature at 1 sigma with a continuum
behind it.

THIS SCRIPT MEASURES THAT.  For each seed of each fold it substitutes every
feature in turn with its TRAINING median and re-predicts.  The feature whose
substitution recovers the most error is the direction the divergence rides on.

THE HEADLINE IS STABILITY, NOT THE PERCENTAGE.  A dominant direction that moves
between seeds is not a mechanism, so the argmax is reported per seed and the
run says outright whether it held.  The first version of this probe ran one
seed and could not have told the difference.

The `recovered` share is measured against each fold's OWN control RMSE on the
same rows, read from the saved predictions with no refit.  The single-seed
version divided by a hardcoded 1.2, which put its figures about a point off.

Substituting a state component is not physically meaningful -- it breaks the
state.  This is a SENSITIVITY probe, not a physical manipulation, and the
distinction belongs in any sentence written from it.

Costs 5 refits per fold, four folds, roughly 20-25 minutes total.  Nothing in
Week 15's claims depends on the outcome; it is mechanism, and mechanism has
been downgraded twice in this project (P3, kl_entropy) for being asserted
rather than measured.

Run from anywhere:  python3 notebooks/probe_divergence.py
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from splits import leave_one_level_out
from features import build_features, Standardizer, FEATURE_SET_W15
from models import train_mlp, predict, MLPConfig
from mi import make_windows, DEAD_ZONE_KM_W12

ARCH = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')
PRED = os.path.join(_HERE, '..', 'data', 'wk15', 'predictions_wk15.npz')
CFG = MLPConfig(max_epochs=3000, min_epochs=150, patience=50)
SEEDS = [0, 1, 2, 3, 4]

# T3 FIRST: it is the fold that can still change the write-up.  T3 and T5 are
# both energy extrapolations but their worst-|z| features differ (vz vs C).  If
# T3's divergence also rides on C, the two-training-levels account covers both
# extrapolation folds.  If it rides on vz, that account covers only E_c and the
# deflationary reading of P15.3 is overstated.
#
# T4 IS A NULL CONTROL.  Its excess over control is ~0.2 decades.  A probe that
# reports a confident dominant direction on a fold that barely diverges is
# manufacturing structure, so `_summarise` refuses percentages below
# MIN_EXCESS_DEC and this fold is the check on that refusal.
PROBE_FOLDS = [('T3', 'energy_id', 0), ('T5', 'energy_id', 2),
               ('T1', 'family_id', 1), ('T4', 'energy_id', 1)]
CONTROL_OF = {'T1': 'C_family', 'T2': 'C_family',
              'T3': 'C_energy', 'T4': 'C_energy', 'T5': 'C_energy'}
MIN_EXCESS_DEC = 0.30


def _summarise(rmse0, rmse_sub, rmse_control, names):
    """Aggregate the substitution probe over seeds.  Pure, so it is testable
    without a fit.

    `recovered` is the share of the fold's EXCESS OVER ITS OWN CONTROL that a
    substitution removes -- not over a hardcoded constant, which is what the
    single-seed version used and which put the reported figures about a point
    off.

    The headline is not the mean percentage.  It is whether the argmax feature
    is the SAME on every seed: a dominant direction that moves between seeds is
    not a mechanism.
    """
    rmse0 = np.asarray(rmse0, float)              # (n_seeds,)
    rmse_sub = np.asarray(rmse_sub, float)        # (n_seeds, n_features)
    excess = rmse0 - rmse_control
    if float(np.mean(excess)) < MIN_EXCESS_DEC:
        return dict(reportable=False,
                    excess=float(np.mean(excess)),
                    reason=(f"mean excess over control is "
                            f"{float(np.mean(excess)):.3f} decades, below "
                            f"{MIN_EXCESS_DEC}; there is no material "
                            f"divergence to attribute and any dominant "
                            f"direction reported here would be noise"))
    rec = (rmse0[:, None] - rmse_sub) / excess[:, None]
    arg = [names[int(np.argmax(rec[i]))] for i in range(rec.shape[0])]
    return dict(reportable=True, excess=float(np.mean(excess)),
                recovered_mean=rec.mean(axis=0), recovered_sd=rec.std(axis=0, ddof=1),
                argmax_per_seed=arg, argmax_stable=bool(len(set(arg)) == 1),
                rmse0_mean=float(rmse0.mean()), rmse0_sd=float(rmse0.std(ddof=1)))


def main():
    z = np.load(ARCH, allow_pickle=True)
    gid, fid, eid = z['group_id'], z['family_id'], z['energy_id']
    d_km, at_floor = z['d_km'], z['at_floor']
    floor = float(z['floor_km'])
    y = np.log10(np.maximum(d_km, floor))
    far = make_windows(d_km, ~at_floor, floor_km=floor,
                       boundary_km=DEAD_ZONE_KM_W12)['far']

    X_all, names, scale_mask = build_features(
        z['states'], z['C'], fid, [0, 1], include_family=False)
    assert tuple(names) == FEATURE_SET_W15

    P = np.load(PRED)

    for tag, keyname, lv in PROBE_FOLDS:
        key = eid if keyname == 'energy_id' else fid
        m, meta = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                      frac_val=0.1, seed=0)
        tr, va, te = m['train'], m['val'], m['test'] & far

        # Control RMSE on the SAME rows, from the saved predictions.  No refit,
        # and it is the fold's own denominator rather than a constant.
        ctrl = P[CONTROL_OF[tag]]
        rmse_ctrl = float(np.sqrt(np.mean((ctrl[:, te] - y[te]) ** 2)))

        rmse0, rmse_sub, dirty = [], [], []
        for seed in SEEDS:
            sc = Standardizer(scale_mask).fit(X_all[tr])
            model, hist = train_mlp(sc.transform(X_all[tr]), y[tr],
                                    sc.transform(X_all[va]), y[va],
                                    cfg=CFG, loss='mse', seed=seed)
            if hist['is_mean_predictor'] or hist['hit_max_epochs']:
                dirty.append(seed)
            base = predict(model, sc.transform(X_all), hist)
            rmse0.append(float(np.sqrt(np.mean((base[te] - y[te]) ** 2))))
            row = []
            for j in range(len(names)):
                Xs = X_all.copy()
                Xs[:, j] = np.median(X_all[tr, j])       # training median
                p_j = predict(model, sc.transform(Xs), hist)
                row.append(float(np.sqrt(np.mean((p_j[te] - y[te]) ** 2))))
            rmse_sub.append(row)

        r = _summarise(rmse0, rmse_sub, rmse_ctrl, names)
        print(f"\n=== {tag} ({keyname} level {lv} held out), "
              f"{len(SEEDS)} seeds, n_far={int(te.sum())} ===")
        if dirty:
            print(f"  !! seeds {dirty} are not clean fits -- probe unreliable")
        print(f"  RMSE transfer = {np.mean(rmse0):.3f} +/- "
              f"{np.std(rmse0, ddof=1):.3f}   control = {rmse_ctrl:.3f}   "
              f"excess = {r['excess']:.3f} decades")
        if not r['reportable']:
            print(f"  NOT REPORTABLE: {r['reason']}")
            continue
        print(f"  {'substituted':<14}{'recovered':>12}{'sd':>8}")
        order = np.argsort(-r['recovered_mean'])
        for j in order:
            print(f"  {names[j]:<14}{r['recovered_mean'][j]:>11.0%}"
                  f"{r['recovered_sd'][j]:>8.1%}")
        print(f"  argmax per seed: {r['argmax_per_seed']}")
        print(f"  DOMINANT DIRECTION STABLE ACROSS SEEDS: {r['argmax_stable']}")
        if not r['argmax_stable']:
            print("  -> a direction that moves between seeds is not a "
                  "mechanism; do not name one in the write-up")

    print("\n  Substituting a state component breaks the state, so this is a")
    print("  SENSITIVITY analysis, not a physical manipulation.  Say so in any")
    print("  sentence written from it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
