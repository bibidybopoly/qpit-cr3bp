"""The Week 12 2x2: loss x target, on the frozen Week 11 dataset and split.

    M1   log10 + MSE     the frozen Week 11 baseline, reproduced exactly
    M2a  asinh + MSE     target change only
    M2b  log10 + Tobit   loss change only
    M2c  asinh + Tobit   both

M1 is nested in M2c: `losses.CensoredGaussianNLL` with no censored rows and a
fixed sigma is MSE up to a constant, and `targets.forward` on the log10 path
reproduces `sampling.label_and_pack` byte for byte.  So the four cells differ in
exactly the two intended ways and nothing else.

WHAT THIS MODULE DOES NOT DO.  It does not split.  The caller passes rows already
split by `src/splits.py` on `group_id`, because the split is a Week 11 decision
and re-deriving it here would let it drift silently.  It does not choose delta
either -- delta is swept over the anchors measured in Week 11 (the model's own
output floor at ~6.5 m and the dead-zone boundary at ~125 m) with a factor-of-3
band, and the sweep is a loop the caller writes.

METRICS ARE PER SEED, THEN AGGREGATED.  Averaging the seeds' predictions and
scoring that once would report an ENSEMBLE, which is a different and better
model than any cell contains.  Each seed is scored on its own and the spread is
reported, so `M2c - M1` can be read against it.

THE ARBITER IS `near`, NOT R^2.  delta = 125 m flattens 2.91 of the target's
5.81 decades, and it flattens exactly the decades the model already fails in.  A
cell that lifts R^2 while leaving the near-field factor-of-N column flat did not
fix the dead zone, it stopped being asked about it -- and that is a
falsification of the target change, not a win.  `evaluate_cell` therefore
returns `near` alongside `all` and `measured` in every factor table.

RISK 19 IS UNTOUCHED.  The censoring encoded here is the labeller's measured
reproducibility floor (0.155 m, `sampling.FLOOR_KM`).  It says nothing about
inter-torus gaps: `d_km` remains an upper bound on distance-to-nearest-QPIT
because only the computed surfaces exist, and the tightness of that bound in the
gaps is unknown.  Nothing in this file tests Risk 19 and results from it must
not be reported as if it did.
"""

from __future__ import annotations

import json
import os

import numpy as np

import metrics as MET
import targets as T
from models import MLPConfig, train_multi_seed

CELLS = (
    ('M1',  'log10', 'mse'),
    ('M2a', 'asinh', 'mse'),
    ('M2b', 'log10', 'tobit'),
    ('M2c', 'asinh', 'tobit'),
)

FROZEN_BASENAMES = ('results_wk11.json', 'predictions_wk11.npz')

D_MAX_KM = 100.0            # top of the sampler's drawn range
NEAR_FIELD_KM = 0.125       # dead-zone boundary, measured Week 11


# =========================================================================
# SECTION 1 -- TARGET CONSTRUCTION
# =========================================================================

def build_target(d_km, transform, floor_km, delta_km=None, at_floor=None):
    """z and the censoring mask, from ONE clamp.

    Returning both together is the point: a target built with one floor and a
    mask built with another produces censored rows that do not sit on z_L, and
    that is the failure `losses.check_censoring_consistency` exists to catch.
    Deriving them in the same call means they cannot disagree.

    `at_floor`, when given, is checked against the recomputed mask rather than
    used.  The dataset column and this file must agree about which rows are
    censored; if they do not, one of them is stale.
    """
    d = np.asarray(d_km, dtype=float).reshape(-1)
    z = T.forward(d, transform, floor_km, delta_km)
    censored = d < floor_km
    if at_floor is not None:
        a = np.asarray(at_floor).reshape(-1).astype(bool)
        if a.shape != censored.shape:
            raise ValueError(f"at_floor has {a.size} rows, expected {censored.size}")
        if not np.array_equal(a, censored):
            n = int((a != censored).sum())
            raise ValueError(
                f"at_floor disagrees with d_km < floor_km on {n} rows "
                f"(floor_km={floor_km:.6g}).  One of the dataset column and "
                "this floor value is stale; do not silence this.")
    return z, censored


def common_taus(floor_km, d_max_km=D_MAX_KM, n=60):
    """Threshold sweep in log10 km, spanning the sampler's drawn range.

    Fixed and shared by every cell.  The alternative -- letting each cell sweep
    its own observed range -- compares a log10 window against an asinh window
    and is the trap `metrics.accuracy_vs_threshold` warns about.
    """
    return np.linspace(np.log10(floor_km), np.log10(d_max_km), n)


def common_edges(floor_km, d_max_km=D_MAX_KM, n_bins=24):
    """Residual-bin edges in log10 km.  24 bins is one of the three
    resolutions at which the Week 11 dead-zone boundary was stable."""
    return np.linspace(np.log10(floor_km), np.log10(d_max_km), n_bins + 1)


# =========================================================================
# SECTION 2 -- EVALUATION IN A COMMON SPACE
# =========================================================================

def evaluate_cell(preds_by_seed, d_km_test, transform, floor_km,
                  delta_km=None, near_field_km=NEAR_FIELD_KM,
                  taus=None, edges=None):
    """Score one cell's per-seed predictions in log10 km.

    `preds_by_seed` is (n_seeds, n_test) in the CELL'S OWN target units, exactly
    as `models.predict` returns it.  Conversion to log10 km happens here and
    nowhere else, so no caller can compare two cells in mismatched units.
    """
    y = T.log10_forward(d_km_test, floor_km)          # the dataset's log_d_km
    censored = np.asarray(d_km_test, dtype=float).reshape(-1) < floor_km
    measured = ~censored
    near = measured & (y < np.log10(near_field_km))
    if not near.any():
        raise ValueError(
            f"near-field mask selects no rows: no measured test row lies below "
            f"{near_field_km:g} km.  That column is the arbiter for whether a "
            "target change fixed the dead zone, so it cannot be silently empty.")
    masks = {'all': None, 'measured': measured, 'near': near}

    taus = common_taus(floor_km) if taus is None else np.asarray(taus, float)
    edges = common_edges(floor_km) if edges is None else np.asarray(edges, float)

    per_seed = []
    for p_raw in np.atleast_2d(preds_by_seed):
        p = T.to_log10_km(p_raw, transform, floor_km, delta_km)
        curve = MET.accuracy_vs_threshold(y, p, taus=taus)
        per_seed.append(dict(
            # R^2 on MEASURED rows only: a clamped label is not a point
            # observation, so scoring against it rewards predicting the clamp.
            regression_measured=MET.regression_metrics(y[measured], p[measured]),
            regression_near=MET.regression_metrics(y[near], p[near]),
            threshold=MET.summarize_threshold_curve(curve),
            factors=MET.factor_table(y, p, masks=masks),
            bins=MET.residual_bins(y, p, edges=edges, mask=measured),
            frac_below_floor=float(np.mean(p <= np.log10(floor_km) + 1e-12)),
        ))

    return dict(
        n_seeds=len(per_seed), per_seed=per_seed,
        n_test=int(y.size),
        censored_frac=float(censored.mean()),
        near_frac=float(near.mean()),
        summary=_aggregate(per_seed),
    )


def _aggregate(per_seed):
    """mean and sd across seeds for the scalars that go in the scorecard."""
    def col(path):
        out = []
        for s in per_seed:
            v = s
            for k in path:
                v = v[k]
            out.append(float(v))
        return float(np.mean(out)), float(np.std(out, ddof=1)) if len(out) > 1 else 0.0

    keys = {
        'r2_measured': ('regression_measured', 'r2'),
        'rmse_measured': ('regression_measured', 'rmse'),
        'bias_near': ('regression_near', 'bias'),
        'rmse_near': ('regression_near', 'rmse'),
        'min_balanced_accuracy': ('threshold', 'min_balanced_accuracy'),
        'frac_below_floor': ('frac_below_floor',),
    }
    out = {}
    for name, path in keys.items():
        m, s = col(path)
        out[name], out[name + '_sd'] = m, s

    # factor tables averaged element-wise, keyed factor -> mask -> mean/sd
    ft = {}
    for i, row in enumerate(per_seed[0]['factors']):
        f = f"{row['factor']:g}"
        ft[f] = {}
        for mask_name in row:
            if mask_name in ('factor', 'log10_tol'):
                continue
            vals = [s['factors'][i][mask_name] for s in per_seed]
            ft[f][mask_name] = float(np.mean(vals))
            ft[f][mask_name + '_sd'] = (float(np.std(vals, ddof=1))
                                        if len(vals) > 1 else 0.0)
    out['factors'] = ft
    return out


# =========================================================================
# SECTION 3 -- THE RUN
# =========================================================================

def run_cell(name, transform, loss, X_tr, X_val, X_test,
             d_tr, d_val, d_test, floor_km, delta_km=None,
             cfg=None, seeds=(0, 1, 2, 3, 4),
             at_floor_tr=None, at_floor_val=None, verbose=False):
    """Fit one cell across seeds and score it.  X must already be standardised
    on TRAIN rows, and the split must already be group-disjoint."""
    if loss == 'tobit' and transform is None:
        raise ValueError("transform is required")
    z_tr, c_tr = build_target(d_tr, transform, floor_km, delta_km, at_floor_tr)
    z_val, c_val = build_target(d_val, transform, floor_km, delta_km, at_floor_val)

    kw = dict(loss=loss)
    if loss == 'tobit':
        kw.update(z_L=T.threshold_in_target_units(floor_km, transform, delta_km),
                  c_tr=c_tr, c_val=c_val)

    preds, hists = train_multi_seed(X_tr, z_tr, X_val, z_val, X_test,
                                    cfg=cfg, seeds=tuple(seeds),
                                    verbose=verbose, **kw)
    ev = evaluate_cell(preds, d_test, transform, floor_km, delta_km)
    # A seed stopped by the epoch budget is a LOWER BOUND, not a fit.  nb11's
    # first run had all three seeds end that way and reported the result as
    # though they had converged.  Counting it here puts it in the scorecard
    # instead of in a printed warning inside a 3000-line training log.
    ev.update(budget_limited=int(sum(bool(h['hit_max_epochs']) for h in hists)),
              mean_predictors=int(sum(bool(h.get('is_mean_predictor'))
                                      for h in hists)),
              cell=name, transform=transform, loss=loss,
              delta_km=None if delta_km is None else float(delta_km),
              floor_km=float(floor_km), seeds=list(seeds),
              censored_frac_train=float(c_tr.mean()),
              histories=[{k: v for k, v in h.items()
                          if k not in ('train_loss', 'val_loss', 'lr',
                                       'log_sigma')} for h in hists])
    return ev, preds


def run_2x2(X_tr, X_val, X_test, d_tr, d_val, d_test, floor_km, delta_km,
            cfg=None, seeds=(0, 1, 2, 3, 4), cells=CELLS,
            at_floor_tr=None, at_floor_val=None, verbose=False):
    """All four cells at one delta.  Returns (results dict, predictions dict).

    Sweeping delta means calling this once per delta; delta does not enter the
    two log10 cells, so M1 and M2b are identical across the sweep and only need
    running once.  The caller decides whether to spend the compute or reuse.
    """
    results, preds = {}, {}
    for name, transform, loss in cells:
        if verbose:
            print(f"--- {name}: {transform} + {loss} ---")
        ev, p = run_cell(name, transform, loss, X_tr, X_val, X_test,
                         d_tr, d_val, d_test, floor_km,
                         delta_km=(delta_km if transform == 'asinh' else None),
                         cfg=cfg, seeds=seeds, at_floor_tr=at_floor_tr,
                         at_floor_val=at_floor_val, verbose=verbose)
        results[name], preds[name] = ev, p
    return results, preds


# =========================================================================
# SECTION 4 -- OUTPUT
# =========================================================================

def _guard_not_frozen(path):
    """Week 11's results are the comparison point and are not to be rewritten."""
    if os.path.basename(path) in FROZEN_BASENAMES:
        raise ValueError(
            f"refusing to write {path}: that is a frozen Week 11 artefact")


def save_results(results, preds, out_dir='../data/wk12',
                 results_name='results_wk12.json',
                 preds_name='predictions_wk12.npz'):
    """Write the run.  Refuses the Week 11 filenames outright."""
    rp = os.path.join(out_dir, results_name)
    pp = os.path.join(out_dir, preds_name)
    for p in (rp, pp):
        _guard_not_frozen(p)
    os.makedirs(out_dir, exist_ok=True)
    with open(rp, 'w') as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=float)
    np.savez_compressed(pp, **{k: np.asarray(v) for k, v in preds.items()})
    return rp, pp


def scorecard(results, baseline='M1'):
    """The 2x2 as one table.  `d_r2` and `d_near10` are against the baseline
    cell; the seed sd is carried so a difference can be read against it.

    Ordered so the two Week 12 predictions are readable off it: M2b should sit
    within the seed spread of M1 (the Mills-ratio limit makes Tobit numerically
    almost MSE while mu is far above the floor), and the value of the loss
    change should appear as M2c - M2a rather than as M2b - M1.
    """
    base = results.get(baseline, {}).get('summary', {})
    rows = []
    for name in results:
        s = results[name]['summary']
        rows.append(dict(
            cell=name,
            transform=results[name]['transform'],
            loss=results[name]['loss'],
            r2_measured=s['r2_measured'],
            r2_sd=s['r2_measured_sd'],
            d_r2=s['r2_measured'] - base.get('r2_measured', float('nan')),
            bias_near=s['bias_near'],
            near10=s['factors']['10']['near'],
            d_near10=(s['factors']['10']['near']
                      - base.get('factors', {}).get('10', {}).get('near',
                                                                  float('nan'))),
            below_floor=s['frac_below_floor'],
            budget_ltd=results[name].get('budget_limited', 0),
            dead_seeds=results[name].get('mean_predictors', 0),
        ))
    return rows
