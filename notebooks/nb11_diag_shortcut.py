"""Week 11 diagnostic -- WHICH feature carries the shortcut?

B1 scored R2 = 0.816 on (dist_L1, dist_L2, C). The Weekly Plan's rule says the
sampling left a location shortcut in. That rule is right, but "fix the dataset"
is not yet an instruction: a leak in C is a reprojection bug and a one-column
change, while a leak in position means the sampling geometry is wrong and the
dataset is regenerated. Those are a day apart, so the ablation comes first.

Run:  python nb11_diag_shortcut.py
One shot. Folded into the notebook once the answer is known; kept as its own
file so the Week 11 notebook is not edited on a hypothesis.

-----------------------------------------------------------------------------
HYPOTHESES AND PREDICTIONS, stated before running.

H1  THE LEAK IS ENERGY. `sample_offsets` steps off the surface in 6-D phase
    space; if those states were never reprojected onto the energy manifold then
    dC ~ grad(C).dr with |dr| = d, so C encodes the target directly. The
    observed C spread of 0.10 is two orders of magnitude wider than the ~1.4e-3
    spacing between sheets, which sheet identity cannot explain.

    P-A  C alone scores R2 > 0.6, and (dist_L1, dist_L2) alone scores < 0.2.
    P-B  C among source=='surface' rows takes ~6 distinct values (one per
         sheet), while C among source=='offset' rows spans a wide range.
         Surface samples sit ON the manifold, so their C IS the sheet energy.
    P-C  log10|C - C_sheet| against the target has slope ~1 in the linear
         regime -- the signature of a first-order energy displacement.

    Falsified if positions alone carry the score, or if surface rows show the
    same spread as offset rows (which would mean the family itself spans that
    energy range and C is honest).

H2  GROUPING IS VACUOUS. `group_ids` bins (family, energy, gap, alpha, theta);
    with fine alpha/theta binning nearly every row becomes its own group and
    the grouped split degenerates to a random one -- which would explain P2's
    zero gap without any statement about the data's correlation length.

    P-D  n_groups / n_rows > 0.5.
    Falsified if groups hold several rows each, in which case the zero gap is
    about 1-NN in 9-D, not about the key.
"""

import sys, os
import numpy as np

# Paths resolved against THIS FILE, not the working directory -- otherwise the
# script only runs when launched from inside notebooks/ and dies with a bare
# ModuleNotFoundError from anywhere else.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from features import jacobi_constant, Standardizer, MU_EARTH_MOON
from splits import grouped_split, group_census
from baselines import shortcut_features, collinear_points, fit_flexible
from metrics import regression_metrics

DATA = os.path.join(HERE, '..', 'data', 'qpit_dataset_w10.npz')
FLOOR_KM = 1.55e-4
SEED = 0


def load(path):
    raw = dict(np.load(path, allow_pickle=True))
    n = len(raw['states'])
    cols = {k: v for k, v in raw.items() if v.ndim >= 1 and len(v) == n}
    meta = {k: v for k, v in raw.items() if k not in cols}
    return cols, meta, n


def fit(Xf, y, m, tag, seed=SEED):
    sc = Standardizer().fit(Xf[m['train']])
    Z = sc.transform(Xf)
    p, info = fit_flexible(Z[m['train']], y[m['train']], Z[m['val']],
                           y[m['val']], Z[m['test']], seed=seed)
    r2 = regression_metrics(y[m['test']], p)['r2']
    print(f"  {tag:38s} R2 = {r2:+.4f}   ({info['selected']})")
    return r2


def main():
    D, META, n = load(DATA)
    print(f"{n} rows\n")

    # ---- target, and the check that score_C is the same thing -------------
    y = np.log10(np.maximum(D['d_km'].astype(float), FLOOR_KM))
    if 'score_C' in D:
        same = np.allclose(D['score_C'].astype(float), y, atol=1e-9)
        print(f"score_C == log10(max(d_km, {FLOOR_KM})):  {same}")
        if not same:
            bad = np.abs(D['score_C'] - y)
            print(f"  max |diff| {bad.max():.3e} on {int((bad > 1e-9).sum())} rows"
                  f" -- NOT the same column; stop and resolve this first")
        y = D['score_C'].astype(float)

    # ---- P-D: is the group key doing anything? ----------------------------
    print("\n--- H2: grouping")
    c = group_census(D['group_id'])
    ratio = c['n_groups'] / c['n_rows']
    for k in ('n_rows', 'n_groups', 'size_min', 'size_max', 'size_mean',
              'size_median', 'n_singleton'):
        print(f"  {k:14s} {c[k]}")
    print(f"  groups/rows    {ratio:.4f}")
    print(f"  P-D {'HELD -> grouping is VACUOUS, the key needs coarsening'
                  if ratio > 0.5 else
                  'falsified -> groups hold several rows; the zero B2 gap is '
                  'about 1-NN in 9-D, not the key'}")

    # ---- P-B: does C vary within a sheet? ---------------------------------
    print("\n--- H1: where the energy spread comes from")
    C = D['C'].astype(float)
    src = D['source']
    sheet = np.array([f"{a}e{b}" for a, b in zip(D['family'], D['energy_id'])])
    for s in np.unique(src):
        m = src == s
        print(f"  source={str(s):9s} n={m.sum():6d}  C {C[m].min():.6f} .. "
              f"{C[m].max():.6f}  spread {np.ptp(C[m]):.3e}  "
              f"distinct(9dp) {len(np.unique(np.round(C[m], 9)))}")
    # startswith, not equality: a source array saved with a too-narrow unicode
    # dtype silently truncates 'surface' to 'surfac' and the equality test then
    # finds no surface rows at all
    surf = np.char.startswith(src.astype(str), 'surf')
    if surf.any():
        print("\n  per-sheet energy, taken from surface rows (they ARE on the manifold):")
        C_sheet = {}
        for s in np.unique(sheet):
            m = surf & (sheet == s)
            if m.any():
                C_sheet[s] = float(np.median(C[m]))
                print(f"    {s:6s} n={m.sum():5d}  C = {C_sheet[s]:.9f}  "
                      f"spread within sheet {np.ptp(C[m]):.3e}")
    else:
        C_sheet = {s: float(np.median(C[sheet == s])) for s in np.unique(sheet)}
        print("\n  no surface rows; falling back to per-sheet median C")

    # ---- P-C: does |dC| track the target? ---------------------------------
    dC = np.abs(C - np.array([C_sheet.get(s, np.nan) for s in sheet]))
    ok = (dC > 0) & np.isfinite(dC) & (y > np.log10(FLOOR_KM) + 0.01)
    if ok.sum() > 50:
        ldC = np.log10(dC[ok])
        slope, icept = np.polyfit(y[ok], ldC, 1)
        r = np.corrcoef(y[ok], ldC)[0, 1]
        print(f"\n  log10|C - C_sheet| vs target, on {ok.sum()} rows:")
        print(f"    slope {slope:.4f}   corr {r:.4f}")
        print("    slope ~1 would mean dC is first-order in the displacement.")
        print("    Reported, not scored: rows pinned at the floor keep varying "
              "in dC\n    while y is constant, which biases the slope low. "
              "P-B and P-A carry\n    the verdict.")

    # ---- P-A: the ablation ------------------------------------------------
    print("\n--- H1: ablation, grouped split, same learner everywhere")
    m = grouped_split(D['group_id'], seed=SEED)
    pos, _ = shortcut_features(D['states'], C)          # dist_L1, dist_L2, C
    S = D['states'].astype(float)
    C_re = jacobi_constant(S, MU_EARTH_MOON)

    r2 = {}
    r2['B1 as run (distL1, distL2, C)'] = fit(pos, y, m, 'distL1 + distL2 + C')
    r2['positions only'] = fit(pos[:, :2], y, m, 'distL1 + distL2  (no energy)')
    r2['C only'] = fit(C[:, None], y, m, 'C alone')
    r2['recomputed C only'] = fit(C_re[:, None], y, m, 'jacobi(states) alone')
    r2['full state, no C'] = fit(S, y, m, 'x y z vx vy vz  (no energy)')
    r2['full state + C'] = fit(np.column_stack([S, C]), y, m,
                               'x y z vx vy vz + C')

    print("\n--- verdict")
    lead = 'C leaks' if (r2['C only'] > 0.6 and r2['positions only'] < 0.2) \
        else 'not a clean energy leak'
    print(f"  P-A {'HELD' if lead == 'C leaks' else 'FALSIFIED'}: {lead}")
    print(f"    C alone {r2['C only']:+.4f} vs positions alone "
          f"{r2['positions only']:+.4f}")
    print(f"  state without C: {r2['full state, no C']:+.4f}  "
          f"(this is what the model would score with the energy column removed)")
    print("\nIf C leaks, three fixes, in increasing cost:\n"
          "  (a) drop C from the features. Cheapest. Loses energy entirely, so\n"
          "      Week 15's train-on-one-energy/test-on-another becomes impossible.\n"
          "  (b) substitute the SHEET energy -- look C up from (family_id,\n"
          "      energy_id) instead of reading it off the state. Keeps energy as a\n"
          "      real feature, keeps Week 15 alive, removes the offset information.\n"
          "      A Week 11 feature change. This is the recommendation.\n"
          "  (c) reproject the offset samples onto the energy manifold and\n"
          "      regenerate. Correct at the source, but it is Week 10 work and\n"
          "      needs a scope call with Dr. Lang before it starts.")


if __name__ == '__main__':
    main()
