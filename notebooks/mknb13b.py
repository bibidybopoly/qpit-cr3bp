#!/usr/bin/env python3
"""Builds notebooks/wk13_triage.ipynb -- the Week 13 speedup benchmark.

Usage:
    python mknb13b.py            # write if absent, no-op if identical, fail if diverged
    python mknb13b.py --force    # overwrite regardless

Divergence is measured on CELL SOURCE ONLY, so executing the notebook does not
read as a hand-edit. Same contract as mknb13.py.
"""

import argparse
import json
import os
import sys

# Resolved against THIS FILE, not the working directory.  With a bare
# os.path.join("notebooks", ...) the builder only found its own notebook when
# run from the repo root; run from inside notebooks/ it wrote a NEW
# notebooks/notebooks/<name>.ipynb instead, which meant the divergence guard
# below never fired -- the path it checks did not exist, so `--force` was not
# needed to lose a hand-edit, only a different working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(_HERE, "wk13_triage.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [

md(r"""
# Week 13 — triage benchmark

Tracker row: *time classical QPIT computation on identical hardware, compare
against NN inference time, report speedup OR honestly report no speedup.*
**PHASE 4 EXIT GATE: paper writeable.**

## Which benchmark, and why the question is now settled

The handoff left two readings open. A **genuine warm start** seeds the Newton
solver from the prediction; **triage** predicts `d` and skips the solve when it
is large. P13.0 was pre-registered to decide between them by measuring
‖∇d‖ on the frozen M1 and checking how far it sits from the eikonal condition.

**P13.0 is retired, not skipped.** The MI test settles it more strongly than
the eikonal check could have: the features carry no information about `d` in
the near field that the estimator can detect at all — the point estimate sits
inside its own permutation null while a matched far-field control is nine times
above its own. A warm start needs to place a state accurately in precisely that
regime. There is nothing to place it with, whatever the gradient looks like.

So triage, and this notebook benchmarks triage.

## The raw speedup ratio is the wrong number, and reporting it would be misleading

An MLP forward pass against a 42-dimensional STM propagation plus a 186×188
Newton solve is four to six orders of magnitude. That number is real and it is
**irrelevant**, because triage does not replace the solve. It decides whether
to attempt one. With `S` the fraction of solves skipped and `T_infer ≪ T_solve`,

    total cost  =  (1 − S)·N·T_solve  +  N·T_infer
    speedup     =  T_solve / ((1 − S)·T_solve + T_infer)  ≈  1 / (1 − S)

**The achievable speedup is capped at 1/(1 − S) no matter how fast inference
is.** S = 0.5 gives 2×; S = 0.9 gives 10×. Quoting the inference ratio as the
speedup would overstate the result by orders of magnitude, and it is the single
easiest thing for a reviewer to catch.

So the benchmark is an accuracy measurement, and the timing enters only to
confirm `T_infer ≪ T_solve` — which is what makes the cap the binding
constraint rather than a lower bound.

## Pre-registration — fixed before the operating characteristic is examined

A row is **worth solving** if `d ≤ D_OP`. The decision is: skip when
`ŷ > τ`. At each target recall, τ is set to that quantile of ŷ among
worth-solving rows, so recall holds **by construction** and the reported
quantity is what it costs.

| ID | Statement | Arbiter |
|----|-----------|---------|
| **P13.4** | At 95% recall with `D_OP` = the Week 12 dead-zone boundary, solves avoided > 0.40 | Held-out rows, per seed, mean ± sd |
| **P13.5** | Solves avoided rises monotonically as `D_OP` falls | Three `D_OP` values |
| **P13.6** | *Arbiter.* If solves avoided at 95% recall < 0.20, triage buys nothing and the honest report is **no useful speedup** | Same |

**Balanced accuracy is not used and should not be quoted.** It weights a
wrongly-skipped near-field row equally against a wrongly-kept far one. Those
costs are not equal — the first is a missed torus, the second is one wasted
solve — so the reportable quantity is solves avoided *at fixed recall*, with the
missed count stated alongside.

Metrics are computed **per seed and then averaged**, never on the seed-averaged
prediction. Ensembling would flatter the result and it is not what Week 12
reported.

Censored rows are **included**: they sit on a surface, so they are the rows
triage most needs to keep, and excluding them would measure an easier problem
than the deployed one.
"""),

code(r"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join("..", "src"))
import mi
import constants
import splits

SEED = 20260827
RECALLS = (0.90, 0.95, 0.99)

DATA      = os.path.join("..", "data", "qpit_dataset_w10.npz")
PRED12    = os.path.join("..", "data", "wk12", "predictions_wk12.npz")
RESULTS12 = os.path.join("..", "data", "wk12", "results_wk12.json")
OUT       = os.path.join("..", "data", "wk13", "results_wk13_triage.json")

# Measure this yourself: one converge_torus call on a representative target,
# same hardware, warm cache. It enters only as a scalar. Leave as None to get
# the accuracy result without the wall-clock conversion.
T_SOLVE_SEC = None

rng = np.random.default_rng(SEED)
"""),

code(r"""
# ---- data and the frozen Week 12 predictions ------------------------------
# Same adapter as wk13_mi.ipynb, already verified there.
npz = np.load(DATA, allow_pickle=True)
d_km     = np.asarray(npz["d_km"], dtype=float).ravel()
censored = np.asarray(npz["at_floor"]).ravel().astype(bool)
group_id = np.asarray(npz["group_id"]).ravel()
FLOOR_KM = float(npz["floor_km"])
log_d_km = np.log10(np.maximum(d_km, FLOOR_KM))

res12 = json.load(open(RESULTS12))
p12 = np.load(PRED12, allow_pickle=True)
yhat_M1 = np.asarray(p12["boundary__M1"], dtype=float)
for kk in ("output_floor__M1", "boundary_over3__M1", "boundary_x3__M1"):
    assert np.allclose(np.asarray(p12[kk], float), yhat_M1), \
        f"{kk} differs from boundary__M1 -- M1 is not delta-independent"
n_seeds, n_test = yhat_M1.shape

_sp = res12["split"]
_masks = splits.grouped_split(group_id, frac_test=_sp["frac_test"],
                              frac_val=_sp["frac_val"], seed=_sp["seed"])
test_idx = np.flatnonzero(_masks["test"])
assert test_idx.size == n_test, "split reconstruction does not match the archive"

# Same verification as the MI notebook: if the rows were misaligned this number
# would not come back.
_m = np.isin(test_idx, np.flatnonzero(~censored))
_yt = log_d_km[test_idx[_m]]
_ss = float(np.sum((_yt - _yt.mean()) ** 2))
_r2 = np.mean([1.0 - float(np.sum((_yt - yhat_M1[s][_m]) ** 2)) / _ss
               for s in range(n_seeds)])
assert abs(_r2 - 0.6288) < 0.02, "split reconstruction is wrong"
print(f"{n_seeds} seeds x {n_test} held-out rows; overall M1 R2 {_r2:.4f}")
print(f"held-out rows on a surface (censored): {int(censored[test_idx].sum())}")
print("scorecard has no budget_ltd/dead_seeds column; M1's five seeds are "
      "clean by the r2_sd argument (0.0214 observed vs 0.35 expected if one "
      "seed were dead). M2b's 0.2865 IS one dead seed. M1 is what we use.")
"""),

code(r"""
def operating_characteristic(yhat, d_true, d_op, recalls=RECALLS):
    '''Solves avoided at fixed recall on the worth-solving class.

    tau is the recall-th quantile of yhat among worth-solving rows, so recall
    holds BY CONSTRUCTION and the reported quantity is its cost. Reporting it
    the other way round -- fixing tau and reading off recall -- makes the two
    error types trade against each other invisibly.
    '''
    worth = d_true <= d_op
    if worth.sum() < 20:
        return None
    rows = []
    for r in recalls:
        tau = float(np.quantile(yhat[worth], r))
        skip = yhat > tau
        rows.append({
            "recall_target": r,
            "recall_actual": float((worth & ~skip).sum() / worth.sum()),
            "tau_log10_km": tau,
            "solves_avoided": float(skip.mean()),
            "missed": int((worth & skip).sum()),
            "wasted_kept": int((~worth & ~skip).sum()),
            "speedup_cap": float(1.0 / max(1.0 - skip.mean(), 1e-9)),
        })
    return rows


d_test = d_km[test_idx]
D_OPS = [("dead-zone boundary", mi.DEAD_ZONE_KM_W12), ("1 km", 1.0), ("10 km", 10.0)]

oc = {}
for name, d_op in D_OPS:
    per_seed = [operating_characteristic(yhat_M1[s], d_test, d_op)
                for s in range(n_seeds)]
    if per_seed[0] is None:
        print(f"[{name}] too few worth-solving rows; skipped")
        continue
    agg = []
    for i, r in enumerate(RECALLS):
        sa = np.array([ps[i]["solves_avoided"] for ps in per_seed])
        ms = np.array([ps[i]["missed"] for ps in per_seed])
        agg.append({"recall": r, "solves_avoided": float(sa.mean()),
                    "solves_avoided_sd": float(sa.std(ddof=1)),
                    "missed_mean": float(ms.mean()),
                    "speedup_cap": float(1.0 / max(1.0 - sa.mean(), 1e-9))})
    oc[name] = {"d_op_km": d_op,
                "n_worth": int((d_test <= d_op).sum()),
                "frac_worth": float((d_test <= d_op).mean()),
                "by_recall": agg, "per_seed": per_seed}
    print(f"\n[{name}]  d_op = {d_op:g} km, "
          f"{oc[name]['n_worth']} of {d_test.size} rows worth solving "
          f"({100 * oc[name]['frac_worth']:.1f}%)")
    for a in agg:
        print(f"   recall {a['recall']:.0%}  ->  solves avoided "
              f"{a['solves_avoided']:.3f} +/- {a['solves_avoided_sd']:.3f}   "
              f"missed {a['missed_mean']:.0f}   speedup cap "
              f"{a['speedup_cap']:.2f}x")
"""),

code(r"""
# ---- inference cost -------------------------------------------------------
# The forward-pass cost is set by the architecture, not by the weights, so a
# freshly initialised network of the recorded shape times the same as the
# trained one. That avoids needing saved weights, which the archive does not
# carry.
cfg = res12["config"]
print(f"architecture from results_wk12.json: hidden {cfg['hidden']}, "
      f"{cfg['activation']}, device {cfg['device']}, batch {cfg['batch_size']}")
t_infer = None
try:
    import torch, torch.nn as nn
    dims = [9] + list(cfg["hidden"]) + [1]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    net = nn.Sequential(*layers[:-1]).eval()
    xb = torch.randn(n_test, 9)
    with torch.no_grad():
        net(xb[:64])                                   # warm up
        t0 = time.perf_counter()
        for _ in range(10):
            net(xb)
        t_infer = (time.perf_counter() - t0) / 10 / n_test
    print(f"inference: {t_infer * 1e6:.2f} us per row "
          f"({t_infer * n_test * 1e3:.1f} ms for all {n_test})")
except Exception as exc:
    print("could not time inference:", exc)
    print("  -> set t_infer by hand if you want the wall-clock conversion")
"""),

code(r"""
# ---- wall clock -----------------------------------------------------------
# T_INFER << T_SOLVE is what makes 1/(1-S) the BINDING cap rather than a lower
# bound. The ratio itself is not the speedup and must not be reported as one.
if T_SOLVE_SEC is None:
    print("T_SOLVE_SEC not set. Time one converge_torus call on a "
          "representative target -- same hardware, warm cache -- and put the "
          "number in cell 1. It enters only as a scalar; every accuracy result "
          "above stands without it.")
elif t_infer is None:
    print("inference timing unavailable; set t_infer by hand")
else:
    ratio = T_SOLVE_SEC / t_infer
    print(f"T_solve {T_SOLVE_SEC:.4g} s, T_infer {t_infer:.4g} s, "
          f"raw ratio {ratio:.3g}x")
    print(f"  overhead of screening every row: "
          f"{100 * t_infer / T_SOLVE_SEC:.4f}% of one solve -- negligible, "
          f"which is what makes the cap binding")
    for name, blk in oc.items():
        for a in blk["by_recall"]:
            S = a["solves_avoided"]
            realised = T_SOLVE_SEC / ((1 - S) * T_SOLVE_SEC + t_infer)
            print(f"  [{name}] recall {a['recall']:.0%}: realised speedup "
                  f"{realised:.2f}x against a cap of {a['speedup_cap']:.2f}x "
                  f"(raw inference ratio {ratio:.3g}x is NOT the speedup)")
"""),

code(r"""
# ---- verdicts -------------------------------------------------------------
key = "dead-zone boundary"
a95 = [a for a in oc[key]["by_recall"] if a["recall"] == 0.95][0]
S95 = a95["solves_avoided"]

P13_4 = bool(S95 > 0.40)
sa_by_dop = [oc[n]["by_recall"][RECALLS.index(0.95)]["solves_avoided"]
             for n, _ in D_OPS if n in oc]
P13_5 = bool(all(x >= y for x, y in zip(sa_by_dop, sa_by_dop[1:])))
P13_6_useful = bool(S95 >= 0.20)

print(f"P13.4  solves avoided at 95% recall = {S95:.3f} > 0.40  ->  "
      f"{'CONFIRMED' if P13_4 else 'FALSIFIED'}")
print(f"P13.5  monotone in D_OP: {[round(x, 3) for x in sa_by_dop]}  ->  "
      f"{'CONFIRMED' if P13_5 else 'FALSIFIED'}")
print(f"P13.6  {'TRIAGE IS USEFUL' if P13_6_useful else 'NO USEFUL SPEEDUP'}: "
      f"{S95:.3f} avoided at 95% recall, capping speedup at "
      f"{a95['speedup_cap']:.2f}x, at a cost of {a95['missed_mean']:.0f} "
      f"missed of {oc[key]['n_worth']} worth-solving rows")
print("\nThe claim this supports is a screening claim, not a solver-"
      "replacement claim. Pair it with the MI result: the model resolves WHICH "
      "DECADE a state sits in, which is what triage needs, and carries no "
      "detectable information about position within the near field, which is "
      "what a warm start would have needed.")
"""),

code(r"""
payload = {
    "week": 13, "component": "triage", "seed": SEED,
    "p13_0_status": ("retired: the MI result settles the benchmark reading "
                     "more strongly than the eikonal check would have"),
    "recalls": list(RECALLS),
    "d_ops_km": {n: v for n, v in D_OPS},
    "operating_characteristic": {n: {k: v for k, v in b.items() if k != "per_seed"}
                                 for n, b in oc.items()},
    "t_infer_sec": t_infer, "t_solve_sec": T_SOLVE_SEC,
    "note_speedup_cap": ("achievable speedup is 1/(1-S), NOT the "
                         "inference-to-solve ratio; triage decides whether to "
                         "attempt a solve, it does not replace one"),
    "verdicts": {"P13_4": P13_4, "P13_5": P13_5, "P13_6_useful": P13_6_useful},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(payload, fh, indent=2)
print("wrote", OUT)
"""),

# WHICH ROWS THE TRIAGE SKIPS -- FOLDED BACK IN FROM THE NOTEBOOK, 2026-09-11.
#
# The next two cells were hand-added to wk13_triage.ipynb and were never in
# this builder, so the notebook carried 10 cells against the builder's 8.  They
# are not scratch: they produce the 16-of-23 censored-miss figure that the
# Limitations item in docs/CUTS_AND_DISPOSITIONS.md rests on.  Run from the
# repo root, this builder would have deleted them silently -- Risk 21 with a
# live example.
#
# Folded back VERBATIM, with no explanatory markdown cell added, because an
# extra cell is itself a divergence and would force a regeneration that strips
# the notebook's eight executed outputs.  The explanation belongs here.
code(r"""
tau = np.quantile(yhat_M1[0][d_test <= mi.DEAD_ZONE_KM_W12], 0.95)
cen = censored[test_idx]
print(f"censored held-out rows kept at 95% recall: "
      f"{(yhat_M1[0][cen] <= tau).mean():.3f} of {cen.sum()}")
"""),

code(r"""
skip = yhat_M1[0] > tau
print(np.unique(np.asarray(npz["source"])[test_idx][cen & skip], return_counts=True))
print(np.unique(np.asarray(npz["family"])[test_idx][cen & skip], return_counts=True))
"""),

]


def build():
    return {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def sources(nb):
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    nb = build()
    if os.path.exists(NB_PATH) and not args.force:
        with open(NB_PATH) as fh:
            existing = json.load(fh)
        new_src, old_src = sources(nb), sources(existing)
        if new_src == old_src:
            print(f"{NB_PATH}: identical, no write")
            return 0
        print(f"{NB_PATH}: DIVERGED from mknb13b.py")
        print(f"  builder has {len(new_src)} cells, file has {len(old_src)}")
        for i, (a, b) in enumerate(zip(old_src, new_src)):
            if a == b:
                continue
            print(f"  first differing cell: {i} ({a[0]})")
            al, bl = a[1].splitlines(), b[1].splitlines()
            for j in range(max(len(al), len(bl))):
                x = al[j] if j < len(al) else "<absent>"
                y = bl[j] if j < len(bl) else "<absent>"
                if x != y:
                    print(f"    line {j}\n      file:    {x}\n      builder: {y}")
                    break
            break
        print("  Hand-edits belong back in mknb13b.py. --force to overwrite.")
        return 1
    with open(NB_PATH, "w") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
