"""Is the post-hoc metric choice load-bearing?  No refitting.

THE RISK BEING TESTED.  Retention, Week 15's pre-registered primary metric, is
void: it is a ratio of R^2 on far-field-only rows, and within-window R^2 is
negative by construction.  The replacement, RMSE ratio, was chosen after the
results were seen.  The danger is not that it is post-hoc -- it is that a metric
was SELECTED from several candidates and the one that happened to reproduce the
pre-registered directions was kept.

That risk is testable.  If every reasonable computable metric gives the same
fold ordering, the choice is not load-bearing and post-hoc selection steered
nothing.  If they disagree, the choice is doing work and has to be defended on
principle rather than on convenience.

THE ARBITER IS ALREADY IN THE REPO.  `metrics.factor_table` is documented as
"THE ARBITER FOR WEEK 12 ... physical km, so no choice of the asinh scale delta
can move it".  It was built for exactly this situation: a comparison that must
not depend on a target-space choice.  Using a pre-established arbiter is
stronger than promoting a metric picked this week, so the factor table is
reported as the arbiter and RMSE ratio beside it as the compact summary.

DIRECTION IS NORMALISED so the orderings are comparable.  Every metric is turned
into a GOODNESS: higher means the transfer fit is closer to its own control on
identical rows.  For error metrics that is control/transfer; for
fraction-within-a-factor metrics it is transfer/control.

SEED SPREAD IS NOT IGNORED.  T5's far-field RMSE is 5.433 +/- 5.326 across
seeds -- a spread as large as its mean -- so any ordering involving T5 could be
an artefact of which seeds landed where.  Rather than pair seeds across folds,
which would be arbitrary since the fits are on different data, every ordering is
also evaluated over all 5 x 5 seed COMBINATIONS and reported as a fraction.

Run from anywhere:  python3 notebooks/metric_robustness.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from splits import leave_one_level_out
from metrics import regression_metrics, accuracy_vs_error_tolerance, FACTORS
from mi import make_windows, DEAD_ZONE_KM_W12

ARCH = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')
PRED = os.path.join(_HERE, '..', 'data', 'wk15', 'predictions_wk15.npz')

CONTROL_OF = {'T1': 'C_family', 'T2': 'C_family',
              'T3': 'C_energy', 'T4': 'C_energy', 'T5': 'C_energy'}


def _rmse(y, p):
    return regression_metrics(y, p)['rmse']


def _mae(y, p):
    return regression_metrics(y, p)['mae']


def _medae(y, p):
    return regression_metrics(y, p)['medae']


def _within(factor):
    """Fraction predicted within a factor of N in PHYSICAL km.  This is the
    factor_table quantity -- the Week 12 arbiter -- evaluated at one factor."""
    def f(y, p):
        tol = float(np.log10(factor))
        return float(accuracy_vs_error_tolerance(
            y, p, tols=[tol])['fraction_within'][0])
    return f


#: (name, function, kind).  `kind` fixes which direction is "better" so the
#: goodness scores are comparable across metrics.
METRICS = [
    ('RMSE ratio',           _rmse,        'error'),
    ('MAE ratio',            _mae,         'error'),
    ('median |err| ratio',   _medae,       'error'),
    ('within factor 3',      _within(3),   'fraction'),
    ('within factor 10',     _within(10),  'fraction'),
    ('within factor 100',    _within(100), 'fraction'),
]


def goodness(transfer_vals, control_vals, kind):
    """Higher = transfer closer to its control.  Mean of per-seed values on each
    side first, matching the project's rule that achieved R^2 is the mean of
    per-seed R^2 and never the R^2 of the seed-averaged prediction."""
    t, c = float(np.mean(transfer_vals)), float(np.mean(control_vals))
    if kind == 'error':
        return c / t if t != 0 else np.inf
    return t / c if c != 0 else np.inf


def pairwise_fraction(a_t, a_c, b_t, b_c, kind):
    """Fraction of the 5 x 5 seed combinations in which fold A outscores fold B.

    Seeds are NOT paired across folds -- the fits are on different rows, so a
    pairing would be arbitrary.  Every combination is evaluated instead.
    """
    hits = tot = 0
    for i in range(len(a_t)):
        for j in range(len(b_t)):
            ga = goodness([a_t[i]], a_c, kind)
            gb = goodness([b_t[j]], b_c, kind)
            hits += int(ga > gb)
            tot += 1
    return hits / tot


def main():
    z = np.load(ARCH, allow_pickle=True)
    P = np.load(PRED)
    gid, fid, eid = z['group_id'], z['family_id'], z['energy_id']
    floor = float(z['floor_km'])
    y = np.log10(np.maximum(z['d_km'], floor))
    far = make_windows(z['d_km'], ~z['at_floor'], floor_km=floor,
                       boundary_km=DEAD_ZONE_KM_W12)['far']

    folds = {}
    for tag, key, lv in (('T1', fid, 1), ('T2', fid, 0), ('T3', eid, 0),
                         ('T4', eid, 1), ('T5', eid, 2)):
        m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                   frac_val=0.1, seed=0)
        folds[tag] = m['test'] & far

    # per fold, per metric, per seed -- transfer and its control on identical rows
    vals = {}
    for tag, te in folds.items():
        ctrl = CONTROL_OF[tag]
        for name, fn, kind in METRICS:
            vals[(tag, name, 'T')] = [fn(y[te], P[tag][s][te])
                                      for s in range(P[tag].shape[0])]
            vals[(tag, name, 'C')] = [fn(y[te], P[ctrl][s][te])
                                      for s in range(P[ctrl].shape[0])]

    print("=== goodness (higher = transfer closer to its own control) ===")
    print(f"  {'metric':<20}" + "".join(f"{t:>9}" for t in folds))
    G = {}
    for name, fn, kind in METRICS:
        row = {}
        for tag in folds:
            row[tag] = goodness(vals[(tag, name, 'T')],
                                vals[(tag, name, 'C')], kind)
        G[name] = row
        print(f"  {name:<20}" + "".join(f"{row[t]:>9.3f}" for t in folds))

    print("\n=== do the pre-registered directions survive under each metric? ===")
    print("  P15.2: goodness(T2, train L2 -> test L1) > goodness(T1)")
    print("  P15.3: goodness(T4, interpolation) > both extrapolation folds\n")
    print(f"  {'metric':<20}{'P15.2':>8}{'5x5':>8}   "
          f"{'P15.3 vs T3':>12}{'5x5':>8}   {'P15.3 vs T5':>12}{'5x5':>8}")
    verdicts = []
    for name, fn, kind in METRICS:
        g = G[name]
        p2 = g['T2'] > g['T1']
        p3a, p3b = g['T4'] > g['T3'], g['T4'] > g['T5']
        f2 = pairwise_fraction(vals[('T2', name, 'T')], vals[('T2', name, 'C')],
                               vals[('T1', name, 'T')], vals[('T1', name, 'C')], kind)
        f3a = pairwise_fraction(vals[('T4', name, 'T')], vals[('T4', name, 'C')],
                                vals[('T3', name, 'T')], vals[('T3', name, 'C')], kind)
        f3b = pairwise_fraction(vals[('T4', name, 'T')], vals[('T4', name, 'C')],
                                vals[('T5', name, 'T')], vals[('T5', name, 'C')], kind)
        verdicts.append((name, p2, p3a and p3b))
        y_n = lambda b: 'holds' if b else 'FAILS'
        print(f"  {name:<20}{y_n(p2):>8}{f2:>8.0%}   "
              f"{y_n(p3a):>12}{f3a:>8.0%}   {y_n(p3b):>12}{f3b:>8.0%}")

    n2 = sum(v[1] for v in verdicts)
    n3 = sum(v[2] for v in verdicts)
    print(f"\n  P15.2 direction holds under {n2}/{len(verdicts)} metrics")
    print(f"  P15.3 direction holds under {n3}/{len(verdicts)} metrics")
    if n2 == len(verdicts) and n3 == len(verdicts):
        print("  -> the metric choice is NOT load-bearing.  RMSE ratio can be "
              "reported\n     as the compact summary and the factor table as the "
              "arbiter, with\n     the note that the orderings agree.")
    else:
        print("  -> METRICS DISAGREE.  The choice IS doing work and must be "
              "defended on\n     principle, not convenience.  Report the "
              "disagreement; do not pick the\n     metric that agrees with the "
              "pre-registration.")

    print("\n=== the arbiter, in physical km: fraction within a factor of N ===")
    print("  (transfer, then its control on the same rows)")
    for tag, te in folds.items():
        ctrl = CONTROL_OF[tag]
        print(f"\n  {tag}  n={int(te.sum())}  control={ctrl}")
        print("    " + "".join(f"{'x'+str(f):>9}" for f in FACTORS))
        for who, key in (('transfer', tag), ('control ', ctrl)):
            fr = []
            for f in FACTORS:
                tol = float(np.log10(f))
                v = [float(accuracy_vs_error_tolerance(
                        y[te], P[key][s][te], tols=[tol])['fraction_within'][0])
                     for s in range(P[key].shape[0])]
                fr.append(float(np.mean(v)))
            print(f"    {who}" + "".join(f"{v:>9.3f}" for v in fr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
