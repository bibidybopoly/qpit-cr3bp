"""Reproduce the numbers and figures reported in the paper from the archived data.

Usage, from the repository root:

    python reproduce/reproduce_paper.py

Nothing is retrained. Every number is recomputed from the saved predictions in
data/ (or read from the saved results files where the computation itself was
not archived), and the two figures are written to figures/. Needs only numpy,
scipy, and matplotlib. Section numbers refer to the paper.

Conventions used throughout, matching the paper:
  * Targets are log10 of the distance in km, clamped at the 0.155 m floor.
  * "Measured" rows are rows above the floor (at_floor == False).
  * Scores are the mean over seeds of per-seed scores, never the score of the
    seed-averaged prediction (the one exception, the dead-zone edge, says so).
  * The near field is measured rows below the 124.62 m dead-zone edge.
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from splits import grouped_split, leave_one_level_out   # noqa: E402
from metrics import dead_zone, residual_bins             # noqa: E402
from targets import to_log10_km                          # noqa: E402

DATA = os.path.join(ROOT, "data")
EDGE_KM = 0.12462          # dead-zone edge, Section 3.2
DELTAS_KM = {"output_floor": 0.006568514405910856, "boundary_over3": 0.041539407585797515,
             "boundary": 0.12461822275739255, "boundary_x3": 0.37385466827217767}


def load_json(*parts):
    with open(os.path.join(DATA, *parts)) as f:
        return json.load(f)


def r2(y, p):
    return 1.0 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def per_seed(fn, P, *args):
    return np.array([fn(*args, P[s]) for s in range(P.shape[0])])


def header(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78)


# --------------------------------------------------------------------------
# Data and the in-distribution split (Section 2.3.3)
# --------------------------------------------------------------------------
ds = np.load(os.path.join(DATA, "qpit_dataset_w10.npz"))
FLOOR = float(ds["floor_km"])
d_all = ds["d_km"]
y_all = np.log10(np.maximum(d_all, FLOOR))
af_all = ds["at_floor"]
group = ds["group_id"]

test = grouped_split(group, 0.2, 0.1, seed=0)["test"]
y, d, af = y_all[test], d_all[test], af_all[test]
meas = ~af
near = meas & (d < EDGE_KM)
far = meas & (d >= EDGE_KM)
p12 = np.load(os.path.join(DATA, "wk12", "predictions_wk12.npz"))
M1 = p12["boundary__M1"].astype(float)            # baseline network, 5 seeds

# --------------------------------------------------------------------------
header("Section 3.1 and Table 1: the baseline network")
# --------------------------------------------------------------------------
r2_all = per_seed(lambda yy, p: r2(yy, p), M1, y)
r2_meas = per_seed(lambda yy, p: r2(yy, p[meas]), M1, y[meas])
rmse_meas = per_seed(lambda yy, p: np.sqrt(np.mean((p[meas] - yy) ** 2)), M1, y[meas])
print(f"test rows {test.sum()}, measured {meas.sum()}")
print(f"R2 all rows      {r2_all.mean():.3f} +/- {r2_all.std(ddof=1):.3f}")
print(f"R2 measured      {r2_meas.mean():.3f} +/- {r2_meas.std(ddof=1):.3f}")
print(f"RMSE measured    {rmse_meas.mean():.2f} decades")
for f in (10, 2):
    tol = np.log10(f)
    a = np.mean([np.mean(np.abs(M1[s] - y) <= tol) for s in range(5)])
    m = np.mean([np.mean(np.abs(M1[s][meas] - y[meas]) <= tol) for s in range(5)])
    print(f"within x{f:<3d}      all {100 * a:.1f}%   measured {100 * m:.1f}%")
w11 = load_json("wk11", "results_wk11.json")["results"]
print("baselines (results_wk11.json):")
for k, name in (("B0 mean", "mean"), ("B1 shortcut", "location"),
                ("B2 1-NN (grouped)", "nearest-neighbor, group split"),
                ("B2 1-NN (random-row)", "nearest-neighbor, row split")):
    print(f"  {name:32s} R2 {w11[k]['r2']:+.3f}  (n = {int(w11[k]['n'])})")
counts = np.unique(group, return_counts=True)[1]
print(f"group census: {len(counts)} groups, median size {np.median(counts):.0f}")

p15 = np.load(os.path.join(DATA, "wk15", "predictions_wk15.npz"))
B0 = p15["B0"].astype(float)[:, test]             # seven features, no family columns
b0_all = per_seed(lambda yy, p: r2(yy, p), B0, y)
b0_meas = per_seed(lambda yy, p: r2(yy, p[meas]), B0, y[meas])
print(f"retrained without family columns: R2 all {b0_all.mean():.3f}, measured {b0_meas.mean():.3f}")

# --------------------------------------------------------------------------
header("Section 3.2: the near-field dead zone")
# --------------------------------------------------------------------------
ym = y[meas]
for nb in (12, 18, 24, 36):
    dz = dead_zone(ym, M1.mean(axis=0)[meas], n_bins=nb)   # seed-averaged
    print(f"seed-averaged, {nb} bins: constant {dz['constant_m']:.2f} m, "
          f"edge {dz['boundary_m']:.1f} m, slope {dz['slope']:.3f}, "
          f"self-consistent {dz['self_consistent']}")
consts = [dead_zone(ym, M1[s][meas], n_bins=12)["constant_m"] for s in range(5)]
slopes = [dead_zone(ym, M1[s][meas], n_bins=12)["slope"] for s in range(5)]
print(f"per seed, 12 bins: constant {min(consts):.1f} to {max(consts):.1f} m, "
      f"slope {min(slopes):.3f} to {max(slopes):.3f}")
edges = np.linspace(ym.min(), ym.max(), 13)
idx = np.clip(np.digitize(ym, edges) - 1, 0, 11)
R = M1[:, meas] - ym
print("12 equal bins of log true distance (measured rows):")
for b in range(12):
    m = idx == b
    print(f"  {1e3 * 10 ** edges[b]:10.2f} m to {1e3 * 10 ** edges[b + 1]:11.1f} m   "
          f"bias {R[:, m].mean():+.2f}   scatter {R[:, m].std(axis=1).mean():.2f}")
print(f"on-surface rows: mean prediction {1e3 * 10 ** M1[:, af].mean():.1f} m")


def p3(yy, P):
    """Locked P3 rule: three closest vs three farthest of 12 equal bins."""
    rb = [r for r in residual_bins(yy, P, n_bins=12) if r["n"] > 0]
    near_b, far_b = rb[:3], rb[-3:]
    nb_ = np.mean([r["bias"] for r in near_b]); fb_ = np.mean([r["bias"] for r in far_b])
    nr_ = np.mean([r["rmse"] for r in near_b]); fr_ = np.mean([r["rmse"] for r in far_b])
    return np.array([nb_, nr_, fr_, np.sqrt(max(nr_ ** 2 - nb_ ** 2, 0)),
                     np.sqrt(max(fr_ ** 2 - fb_ ** 2, 0))])


locked = np.mean([p3(y, M1[s]) for s in range(5)], axis=0)          # all rows, as locked
changed = np.mean([p3(y[meas], M1[s][meas]) for s in range(5)], axis=0)  # measured rows only
print(f"P3a as locked (RMSE, all rows): near {locked[1]:.2f} vs far {locked[2]:.2f} decades")
print(f"P3a as re-scored (scatter, measured rows): near {changed[3]:.2f} vs far {changed[4]:.2f}")
print(f"P3b bias, three closest bins: all rows {locked[0]:+.2f}, measured rows {changed[0]:+.2f}")

# --------------------------------------------------------------------------
header("Section 3.3 and Table 2: the two fixes")
# --------------------------------------------------------------------------
print("near-field fraction within x10, km (asinh cells inverted to log10 km)")
rows = {}
for key in sorted(p12.files):
    delta, cell = key.split("__")
    P = p12[key].astype(float)
    if cell in ("M2a", "M2c"):                   # asinh target
        P = np.array([to_log10_km(P[s], "asinh", FLOOR, DELTAS_KM[delta]) for s in range(len(P))])
    rows[key] = P
x10 = lambda P: np.array([np.mean(np.abs(P[s][near] - y[near]) <= 1.0) for s in range(len(P))])
r2m = lambda P: np.array([r2(y[meas], P[s][meas]) for s in range(len(P))])
tob = rows["boundary__M2b"]
tob_r2 = r2m(tob)
dead = tob_r2 < 0.05                              # the seed that never trained
print(f"Tobit seeds with R2 < 0.05 (untrained, excluded): {np.flatnonzero(dead).tolist()}")
tob_ok = tob[~dead]
pm = lambda v: f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}"
print(f"{'delta':>8s} {'baseline':>15s} {'asinh':>15s} {'Tobit':>15s} {'both':>15s}")
for dname in ("output_floor", "boundary_over3", "boundary", "boundary_x3"):
    print(f"{1e3 * DELTAS_KM[dname]:7.1f}m {pm(x10(M1)):>15s} {pm(x10(rows[dname + '__M2a'])):>15s} "
          f"{pm(x10(tob_ok)):>15s} {pm(x10(rows[dname + '__M2c'])):>15s}")
src_all = ds["source"]
print(f"rows at the floor: {af_all.sum()} of {af_all.size} ({100 * af_all.mean():.1f}%), "
      f"on-surface {(af_all & (src_all == 'surface')).sum()}")
both = rows["boundary__M2c"]
print(f"floor rows, median prediction: baseline {1e3 * 10 ** np.median(M1[:, af]):.1f} m, "
      f"both fixes {1e3 * 10 ** np.median(both[:, af]):.1f} m "
      f"({10 ** np.median(both[:, af]) / FLOOR:.0f} times the floor)")
print(f"R2 measured: baseline {r2m(M1).mean():.3f}; Tobit {tob_r2[~dead].mean():.3f} "
      f"(with the untrained seed {tob_r2.mean():.3f})")
for dname in ("output_floor", "boundary_over3", "boundary", "boundary_x3"):
    print(f"  delta {1e3 * DELTAS_KM[dname]:6.1f} m: R2 asinh {r2m(rows[dname + '__M2a']).mean():.3f}, "
          f"both {r2m(rows[dname + '__M2c']).mean():.3f}")
nb = lambda P: np.mean([np.mean(P[s][near] - y[near]) for s in range(len(P))])
print(f"near-field bias: baseline {nb(M1):+.2f}, asinh at 124.6 m {nb(rows['boundary__M2a']):+.2f}")
print("(The validation R2 of 0.85 against 0.62 in the asinh network's own units comes from the")
print(" Week 12 training logs; validation predictions were not archived.)")

# --------------------------------------------------------------------------
header("Section 3.4: mutual information (results_wk13_mi.json)")
# --------------------------------------------------------------------------
mi = load_json("wk13", "results_wk13_mi.json")
for w in ("near", "far"):
    r = mi[w]
    ks = ", ".join(f"k={k}: {v:+.4f}" for k, v in r["by_k"].items())
    print(f"{w}: n {r['n']}, MI {r['I_total']:+.4f} nats, shuffled 95th pct {r['null_p95']:.4f} ({ks})")
for w in ("near_thinned", "far_thinned"):
    r = mi["thinned"][w]
    print(f"{w}: n {r['n']}, MI {r['I_total']:+.4f}, shuffled 95th pct {r['null_p95']:.4f}")
print(f"far cells: {min(c['mi_nats'] for c in mi['far']['per_cell'].values()):.2f} to "
      f"{max(c['mi_nats'] for c in mi['far']['per_cell'].values()):.2f} nats")
print(f"Gaussian-equivalent R2 = 1 - exp(-2I): far {1 - np.exp(-2 * mi['far']['I_total']):.3f}, "
      f"near at most {1 - np.exp(-2 * mi['near']['null_p95']):.3f}")

# --------------------------------------------------------------------------
header("Section 3.5 and Table 3: triage (results_wk13_triage.json)")
# --------------------------------------------------------------------------
tri = load_json("wk13", "results_wk13_triage.json")
oc = tri["operating_characteristic"]
n_test = int(test.sum())
main = oc["dead-zone boundary"]
n_worth = main["n_worth"]
r95 = [r for r in main["by_recall"] if r["recall"] == 0.95][0]
S = r95["solves_avoided"]
ideal = (n_test - n_worth) / n_test
spec = (S * n_test - r95["missed_mean"]) / (n_test - n_worth)
print(f"worth solving: {n_worth} of {n_test} ({100 * n_worth / n_test:.1f}%)")
print(f"95% recall: S = {100 * S:.1f}% +/- {100 * r95['solves_avoided_sd']:.1f}%, "
      f"missed {r95['missed_mean']:.0f}, cap {1 / (1 - S):.2f}x")
print(f"ideal screen skips {100 * ideal:.1f}% -> {1 / (1 - ideal):.2f}x; network gets "
      f"{100 * S / ideal:.0f}%; specificity {100 * spec:.0f}%")
T_SOLVE, T_INFER = 2.594, 39.2e-6      # s; mean GMOS step and one prediction (time_solve.py)
real = T_SOLVE / ((1 - S) * T_SOLVE + T_INFER)
print(f"realized speedup {real:.4f}x vs cap {1 / (1 - S):.4f}x "
      f"({100 * (1 / (1 - S) - real) * (1 - S):.4f}% below); saved {S * T_SOLVE:.2f} s per candidate")
print("Table 3, fraction skipped (mean +/- sd over seeds):")
for name, block in oc.items():
    cells = "  ".join(f"{int(100 * r['recall'])}%: {100 * r['solves_avoided']:.1f}% +/- "
                      f"{100 * r['solves_avoided_sd']:.1f}%" for r in block["by_recall"])
    print(f"  {name:20s} {cells}")
# On-surface rows, per seed, with the triage rule of Section 2.3.5 recomputed.
from scipy.stats import binomtest                         # noqa: E402
src_te, fam_te = ds["source"][test], ds["family_id"][test]
worth = d <= EDGE_KM
surf = src_te == "surface"
other = worth & ~surf
rates, orates, k_l2 = [], [], []
for s in range(5):
    skip = M1[s] > np.quantile(M1[s][worth], 0.95)
    miss = worth & skip & surf
    rates.append(miss.sum()); orates.append((worth & skip & other).sum() / other.sum())
    k_l2.append((miss & (fam_te == 1)).sum())
    print(f"  seed {s}: on-surface rows skipped {miss.sum()} of {surf.sum()} "
          f"(L2 {k_l2[-1]}, L1 {miss.sum() - k_l2[-1]}; two-sided p "
          f"{binomtest(int(k_l2[-1]), int(miss.sum()), (surf & (fam_te == 1)).sum() / surf.sum()).pvalue:.3f})")
print(f"on-surface miss rate {100 * np.mean(rates) / surf.sum():.1f}% vs {100 * np.mean(orates):.1f}% "
      f"for the other worth-solving rows; L2 {sum(k_l2)} of {sum(rates)} misses, "
      f"L2 share of on-surface rows {(surf & (fam_te == 1)).sum()} of {surf.sum()}")

# --------------------------------------------------------------------------
header("Section 3.6 and Table 4: transfer")
# --------------------------------------------------------------------------
fam, en, C = ds["family_id"], ds["energy_id"], ds["C"]
E = np.unique(en)[np.argsort([C[en == e][0] for e in np.unique(en)])]
folds = {"T1": (fam, 1, "C_family", "L2 family"), "T2": (fam, 0, "C_family", "L1 family"),
         "T3": (en, E[0], "C_energy", f"C = {C[en == E[0]][0]:.4f}"),
         "T4": (en, E[1], "C_energy", f"C = {C[en == E[1]][0]:.4f}"),
         "T5": (en, E[2], "C_energy", f"C = {C[en == E[2]][0]:.4f}")}
meas_all = ~af_all
train0 = grouped_split(group, 0.2, 0.1, seed=0)["train"]
lo, hi = y_all.min(), y_all.max()
print(f"in-distribution baseline trained on {train0.sum()} rows")
print(f"{'fold':4s} {'held out':13s} {'rows':>5s} {'RMSE T':>7s} {'RMSE C':>7s} {'ratio':>6s} "
      f"{'>3 dec':>7s} {'x10 T/C':>13s} {'B0':>5s} {'R2 C':>6s} {'near R2 T':>10s}")
for f, (key, held, ctl, label) in folds.items():
    tmask = leave_one_level_out(key, held, group, seed=0)[0]["test"]
    fr = tmask & meas_all & (d_all >= EDGE_KM)
    nr = tmask & meas_all & (d_all < EDGE_KM)
    yy, yn = y_all[fr], y_all[nr]
    T, Cc, B = (p15[k].astype(float) for k in (f, ctl, "B0"))
    rms = lambda P: np.array([np.sqrt(np.mean((P[s][fr] - yy) ** 2)) for s in range(5)])
    rm = lambda P: rms(P).mean()
    w10 = lambda P: np.mean([np.mean(np.abs(P[s][fr] - yy) <= 1) for s in range(5)])
    off3 = np.mean(np.abs(T[:, fr] - yy) > 3)
    r2c = np.mean([r2(yy, Cc[s][fr]) for s in range(5)])
    r2n = np.mean([r2(yn, T[s][nr]) for s in range(5)])
    print(f"{f:4s} {label:13s} {fr.sum():5d} {rm(T):7.2f} {rm(Cc):7.2f} {rm(T) / rm(Cc):6.2f} "
          f"{100 * off3:6.1f}% {100 * w10(T):5.1f}%/{100 * w10(Cc):4.1f}% {rm(B):5.2f} {r2c:6.2f} {r2n:10.1f}")
    inr = (T[:, fr] >= lo) & (T[:, fr] <= hi)
    err = np.abs(T[:, fr] - yy)
    print(f"     RMSE T {rm(T):.2f} +/- {rms(T).std(ddof=1):.2f}, C {rm(Cc):.2f} +/- {rms(Cc).std(ddof=1):.2f}; "
          f"C - B0 {rm(Cc) - rm(B):.2f}; rows in B0's training set {100 * (fr & train0).sum() / fr.sum():.0f}%; "
          f">1 decade wrong inside the target range {100 * np.mean((err > 1) & inr):.1f}%; "
          f">3-decade errors outside it {100 * ((err > 3) & ~inr).sum() / (err > 3).sum():.0f}%")
    if f == "T1":
        Rt = T[:, fr] - yy
        t1 = (T[:, fr].min(), Rt.std(), np.mean([r2(yy, T[s][fr]) for s in range(5)]))
print(f"T1: lowest prediction {t1[0]:.0f} (log10 km), error sd pooled over seeds {t1[1]:.0f} "
      f"decades, far-field R2 {t1[2]:,.0f}")

# --------------------------------------------------------------------------
header("Section 3.7: training schedule (history_wk15.json)")
# --------------------------------------------------------------------------
hist = load_json("wk15", "history_wk15.json")
for name, fits in hist.items():
    be = [h["best_epoch"] for h in fits]
    capped = sum(h["hit_max_epochs"] for h in fits)
    print(f"{name:9s} best epochs {min(be):5d} to {max(be):5d}   at the 8,000 cap: {capped}")

# --------------------------------------------------------------------------
header("Figures")
# --------------------------------------------------------------------------
fig_dir = os.path.join(ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "make_fig_deadzone.py"),
                os.path.join(DATA, "qpit_dataset_w10.npz"),
                os.path.join(DATA, "wk12", "predictions_wk12.npz"),
                os.path.join(fig_dir, "fig1_dead_zone.png")], check=True)
subprocess.run([sys.executable, os.path.join(here, "make_fig_mi.py"),
                os.path.join(DATA, "wk13", "results_wk13_mi.json"),
                os.path.join(fig_dir, "fig2_mutual_information.png")], check=True)
