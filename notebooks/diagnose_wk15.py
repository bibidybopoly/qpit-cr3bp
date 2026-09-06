"""Is the Week 15 catastrophe a pipeline bug or a real result?  No refitting.

WHAT WE KNOW ALREADY.  Every fit trained fine (val_r2 0.47-0.63) and every
far-field test R2 came back negative, controls included (-0.87 to -2.03).

Part of that is EXPECTED and documented.  Week 12 measured within-window R2 at
-0.133 near and -0.887 far against 0.628 overall, and recorded it as restricted
range, not a bug: residual sd stays near 1.02 decades while the window's own
variance is small, so R2 = 1 - resid/window_var goes negative for a GOOD model.
The controls landing at -0.87 to -2.03 sit right in that range.

That alone voids retention as the pre-registration defines it -- a ratio of two
quantities that are negative by construction is not interpretable.

But it does NOT explain the transfer folds.  T1's -602 needs a residual sd near
20 decades against a target spanning 5.8.  Restricted range buys about -1.  So
something else is happening, and it is one of two things:

  (A) genuine catastrophic extrapolation.  The Standardizer is fit on training
      rows only, so for a family fold the L2 states are scaled by L1 moments and
      land far outside anything the network saw.  A 3x256 silu MLP can diverge
      arbitrarily off-distribution.  That would be a real and reportable result.

  (B) a pipeline defect -- misaligned rows, a scale applied twice, predictions
      returned in the wrong units.

The distinguishing check is B0, which is in-distribution and reproduces a known
number.  If B0's overall test R2 comes back near 0.628, the pipeline is sound
and (A) stands.  If B0 is also broken, nothing else here means anything.

Run from anywhere:  python3 notebooks/diagnose_wk15.py
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

from splits import grouped_split, leave_one_level_out
from metrics import regression_metrics
from mi import make_windows, DEAD_ZONE_KM_W12

ARCH = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')
PRED = os.path.join(_HERE, '..', 'data', 'wk15', 'predictions_wk15.npz')

W12_ALL_ROWS = 0.6269         # mean per-seed R2, all held-out rows
W12_MEASURED = 0.6280         # mean per-seed R2, measured rows (archive value)
W12_FAR_WINDOW = -0.887       # within-window R2, far field


def main():
    z = np.load(ARCH, allow_pickle=True)
    P = np.load(PRED)
    gid, fid, eid = z['group_id'], z['family_id'], z['energy_id']
    d_km, at_floor = z['d_km'], z['at_floor']
    floor = float(z['floor_km'])
    y = np.log10(np.maximum(d_km, floor))
    W = make_windows(d_km, ~at_floor, floor_km=floor,
                     boundary_km=DEAD_ZONE_KM_W12)
    far, near, meas = W['far'], W['near'], ~at_floor

    def r2_mean(name, mask):
        A = P[name]
        return float(np.mean([regression_metrics(y[mask], A[s][mask])['r2']
                              for s in range(A.shape[0])]))

    # ---- THE DECIDING CHECK -------------------------------------------------
    B0 = grouped_split(gid, frac_test=0.20, frac_val=0.10, seed=0)
    t = B0['test']
    print("=== B0, in-distribution, against known Week 12 values ===")
    for label, m, ref in (('all held-out rows', t, W12_ALL_ROWS),
                          ('measured rows', t & meas, W12_MEASURED),
                          ('far window only', t & far, W12_FAR_WINDOW),
                          ('near window only', t & near, -0.133)):
        got = r2_mean('B0', m)
        ok = abs(got - ref) < 0.15
        print(f"  {label:<20} n={int(m.sum()):>5}  R2={got:>10.4f}  "
              f"expect ~{ref:>7.3f}  {'OK' if ok else '<-- MISMATCH'}")
    print("\n  If the first two rows match, the pipeline is sound and the")
    print("  transfer collapse is a real extrapolation result, not a bug.\n")

    # ---- prediction scale, per fit -----------------------------------------
    print("=== prediction ranges (target spans "
          f"{y.min():.2f} to {y.max():.2f} in log10 km) ===")
    print(f"  {'fit':<10}{'min':>10}{'p01':>9}{'p50':>9}{'p99':>9}{'max':>12}"
          f"{'resid sd':>10}")
    for name in sorted(P.files):
        A = P[name].ravel()
        rs = float(np.std(P[name][0][t & meas] - y[t & meas]))
        print(f"  {name:<10}{A.min():>10.2f}{np.percentile(A,1):>9.2f}"
              f"{np.percentile(A,50):>9.2f}{np.percentile(A,99):>9.2f}"
              f"{A.max():>12.2f}{rs:>10.2f}")

    # ---- where the blow-up lives -------------------------------------------
    print("\n=== transfer folds: is the damage everywhere or in a tail? ===")
    FOLDS = [('T1', fid, 1), ('T2', fid, 0), ('T3', eid, 0),
             ('T4', eid, 1), ('T5', eid, 2)]
    for tag, key, lv in FOLDS:
        m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                   frac_val=0.1, seed=0)
        tf = m['test'] & far
        res = np.abs(P[tag][0][tf] - y[tf])
        wild = res > 3.0                      # >1000x error in km
        # R2 with the worst 1% of rows removed: if it recovers, the collapse is
        # a few divergent points, not a uniformly bad fit.
        keep = res <= np.percentile(res, 99)
        idx = np.flatnonzero(tf)
        trimmed = regression_metrics(y[idx[keep]], P[tag][0][idx[keep]])['r2']
        print(f"  {tag}: n={int(tf.sum())}  median|err|={np.median(res):.3f}  "
              f"max|err|={res.max():.1f}  rows>3 dex={int(wild.sum())} "
              f"({wild.mean():.1%})  R2 trimmed 1% = {trimmed:.3f}")

    print("\n=== residual sd on the SAME rows, transfer vs its control ===")
    print("  MSE is well defined where a ratio of negative R2 is not.")
    for tag, key, lv in FOLDS:
        m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                   frac_val=0.1, seed=0)
        tf = m['test'] & far
        ctrl = 'C_family' if tag in ('T1', 'T2') else 'C_energy'
        st = float(np.sqrt(np.mean((P[tag][:, tf] - y[tf]) ** 2)))
        sc = float(np.sqrt(np.mean((P[ctrl][:, tf] - y[tf]) ** 2)))
        sb = float(np.sqrt(np.mean((P['B0'][:, tf] - y[tf]) ** 2)))
        print(f"  {tag}: rmse transfer={st:>8.3f}  control={sc:>7.3f}  "
              f"B0={sb:>7.3f}   ratio t/c={st/sc:>7.2f}")
    # ---- is the divergence OFF-SUPPORT, or is the network unstable? --------
    # The Standardizer is fit on TRAINING rows only, so a family fold scales L2
    # states by L1 moments.  If the held-out level sits many sigma outside
    # anything the network saw, a silu MLP is roughly linear out there and the
    # output grows with the input -- which would produce exactly this.
    #
    # That is a HYPOTHESIS.  This measures it instead of asserting it: no
    # refitting is needed, because the Standardizer is deterministic given the
    # training mask.  If test rows sit at |z| ~ 4 like B0's and the model still
    # diverges, the hypothesis is WRONG and the instability is elsewhere.
    from features import build_features, Standardizer, FEATURE_SET_W15
    X_all, names, scale_mask = build_features(
        z['states'], z['C'], fid, [0, 1], include_family=False)

    print("\n=== how far outside training support do the test rows sit? ===")
    print(f"  {'fit':<10}{'max|z|':>9}{'p99|z|':>9}{'rows any|z|>5':>15}"
          f"{'worst feature':>16}")
    cases = [('B0', B0['train'], t & far)]
    for tag, key, lv in FOLDS:
        m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                   frac_val=0.1, seed=0)
        cases.append((tag, m['train'], m['test'] & far))
    support = {}
    for tag, tr, te in cases:
        Z = np.abs(Standardizer(scale_mask).fit(X_all[tr]).transform(X_all[te]))
        rowmax = Z.max(axis=1)
        support[tag] = rowmax
        j = int(np.argmax(Z.max(axis=0)))
        print(f"  {tag:<10}{Z.max():>9.1f}{np.percentile(rowmax,99):>9.1f}"
              f"{(rowmax > 5).mean():>14.1%}{names[j]:>16}")

    print("\n=== does per-row error track per-row distance off support? ===")
    print("  Spearman rho between max|z| and |prediction error| on the same rows.")
    for tag, key, lv in FOLDS:
        m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                   frac_val=0.1, seed=0)
        te = m['test'] & far
        err = np.abs(P[tag][0][te] - y[te])
        a = np.argsort(np.argsort(support[tag])).astype(float)
        b = np.argsort(np.argsort(err)).astype(float)
        rho = float(np.corrcoef(a, b)[0, 1])
        print(f"  {tag}: rho = {rho:+.3f}")
    print("\n  A high rho supports off-support extrapolation.  A rho near zero")
    print("  means the divergence is not explained by distance from training")
    print("  data and the mechanism is something else -- do not assume it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
