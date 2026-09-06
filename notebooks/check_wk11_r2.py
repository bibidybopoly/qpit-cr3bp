"""Settle `week11_m1_r2` = 0.6576.  Start-of-week item 4.  Archive hygiene.

The research log carries three values for one quantity -- 0.632 +/- 0.020
(Week 11, after the epoch budget moved to 3000/150/50), 0.6288 +/- 0.0214
(Week 12 reproduction on measured rows), 0.6269 (all held-out rows) -- and then
restates the puzzle unresolved under *Stuck on*.  No entry describes how
`results_wk12.json` aggregates seeds, so the log cannot settle it and the
archive has to.

Two mechanisms both land in the right place and cannot be told apart without
the stored predictions:

  1. Seed-averaged prediction (ensembling).  R2(mean prediction) - mean per-seed
     R2 equals the mean pointwise across-seed prediction variance over Var(y).
     Var(y) on measured held-out rows is 2.8231 (measured directly from the
     archive), so the observed 0.0288 gap implies an across-seed prediction sd
     of 0.285 decades, about 28% of the residual sd.  High but not implausible.
  2. Best of five seeds.  0.6576 sits 1.35 r2_sd above 0.6288.  The expected max
     of five draws is 1.16 sd above the mean and P(max >= 1.35 sd) ~ 0.37.
     Entirely unremarkable.

This script computes all three aggregations from the stored predictions and
reports which one equals 0.6576.  It asserts nothing about which it will be.

NOTE ON PLACEMENT: this is a one-off diagnostic, not a notebook builder and not
a test, so `notebooks/` is a guess.  Move it if there is a better home.

Run from anywhere:  python3 notebooks/check_wk11_r2.py
"""

import os
import sys
import numpy as np

# Anchored to THIS FILE, not the working directory, so the script runs from
# anywhere -- an IDE, the repo root, or notebooks/.  Same pattern as
# tests/test_splits.py.  The earlier cwd-relative version failed the moment it
# was launched from anywhere but notebooks/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from splits import grouped_split

ARCHIVE = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')
CANDIDATES = [os.path.join(_HERE, '..', 'data', 'wk11', 'predictions_wk11.npz'),
              os.path.join(_HERE, '..', 'data', 'wk12', 'predictions_wk12.npz')]

TARGET = 0.6576
KNOWN = {'week11_m1_r2 (results_wk12.json)': 0.6576,
         'Week 11 log line 1188': 0.632,
         'Week 12 reproduction, measured rows': 0.6288,
         'all held-out rows': 0.6269}


def r2(y, yhat):
    """1 - SSE/SST against the mean of THIS y.  Same convention as Week 12."""
    resid = y - yhat
    return 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())


def main():
    z = np.load(ARCHIVE, allow_pickle=True)
    gid, d_km, at_floor = z['group_id'], z['d_km'], z['at_floor']
    floor = float(z['floor_km'])
    y_all = np.log10(np.maximum(d_km, floor))

    m = grouped_split(gid, frac_test=0.20, frac_val=0.10, seed=0)
    test = m['test']
    n_test = int(test.sum())
    n_meas = int((test & ~at_floor).sum())
    print(f"split reconstruction: test={n_test} rows, measured={n_meas}, "
          f"censored={n_test - n_meas}")
    assert n_test == 4035, (
        f"expected 4035 held-out rows, got {n_test}; the split has drifted and "
        f"nothing below this is trustworthy")

    # Two candidate row populations.  Which one a predictions array covers is
    # read from its SHAPE, not assumed.
    meas_of_test = ~at_floor[test]
    pops = {n_test: ('all held-out', y_all[test], None),
            n_meas: ('measured only', y_all[test & ~at_floor], None)}

    found = []
    for path in CANDIDATES:
        print(f"\n=== {os.path.relpath(path, _HERE)} ===")
        if not os.path.exists(path):
            print("  absent -- skipping")
            continue
        p = np.load(path, allow_pickle=True)
        print(f"  {len(p.files)} keys")
        for k in sorted(p.files):
            print(f"    {k:<24} {str(p[k].shape):<14} {p[k].dtype}")

        # The reconstruction was previously justified by Var(y) agreeing with
        # the value implied by Week 12's R2 -- inference.  Where the archive
        # stores the split itself, check against it directly instead.
        if 'test_mask' in p.files:
            ok = np.array_equal(np.asarray(p['test_mask'], bool), test)
            print(f"\n  test_mask matches reconstruction exactly: {ok}")
            assert ok, ("stored test_mask disagrees with "
                        "grouped_split(group_id, 0.2, 0.1, seed=0); every "
                        "number below is aligned to the wrong rows")
        if 'y_test' in p.files:
            dy = float(np.abs(np.asarray(p['y_test'], float) - y_all[test]).max())
            print(f"  max |y_test - reconstructed y|: {dy:.3e}")
            assert dy < 1e-9, "stored y_test disagrees with log10(max(d_km, floor))"
        if 'at_floor_test' in p.files:
            ok = np.array_equal(np.asarray(p['at_floor_test'], bool),
                                at_floor[test])
            print(f"  at_floor_test matches: {ok}")
            assert ok, "stored censoring flags disagree with the archive"

        for k in sorted(p.files):
            a = np.asarray(p[k], dtype=float)
            if a.ndim != 2:
                continue
            n_seeds, n_rows = a.shape
            if n_rows not in pops:
                print(f"\n  {k}: {n_rows} columns matches no known population "
                      f"({n_test} or {n_meas}) -- skipped, do not guess")
                continue
            pop_name, y, _ = pops[n_rows]

            # Arrays stored over all held-out rows can also be scored on the
            # measured subset.  Week 12 reported 0.6288 +/- 0.0214 on measured
            # rows; without this view that number is unreachable from the
            # archive and cannot be cross-checked.
            views = [(pop_name, y, a)]
            if n_rows == n_test:
                views.append(('measured only', y_all[test & ~at_floor],
                              a[:, meas_of_test]))

            for vname, yv, av in views:
                per_seed = np.array([r2(yv, av[s]) for s in range(n_seeds)])
                agg = {'mean of per-seed R2': float(per_seed.mean()),
                       'R2 of seed-averaged prediction': r2(yv, av.mean(axis=0)),
                       'max per-seed R2': float(per_seed.max())}
                print(f"\n  {k}  ({n_seeds} seeds, {vname})")
                print(f"    per-seed: {np.round(per_seed, 4).tolist()}  "
                      f"sd={per_seed.std(ddof=1):.4f}")
                for name, v in agg.items():
                    tags = ''
                    for lbl, ref in KNOWN.items():
                        if abs(v - ref) < 5e-4:
                            tags += f'   <== {ref:.4f}  {lbl}'
                    print(f"    {name:<34} {v:.4f}{tags}")
                    if abs(v - TARGET) < 5e-4:
                        found.append((path, k, f'{name} ({vname})', v))

    print("\n=== verdict ===")
    if not found:
        print("  No aggregation of any stored array reproduces 0.6576.")
        print("  Item 4 stays open; it is not a seed-aggregation artifact.")
    else:
        for path, k, name, v in found:
            print(f"  0.6576 = {name} of {k} in {path}  ({v:.4f})")
        if len({f[2] for f in found}) > 1:
            print("  MORE THAN ONE aggregation matches -- report the ambiguity, "
                  "do not pick one.")
    print("\n  For the paper, quote the mean of per-seed R2.  Ensembling "
          "flatters and is not what Week 12 reported.")
    print("  Reference values:")
    for label, v in KNOWN.items():
        print(f"    {v:.4f}  {label}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
