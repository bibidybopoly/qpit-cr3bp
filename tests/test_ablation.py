"""Tests for src/ablation.py.  Runs standalone.

The load-bearing one is `test_two_transforms_of_one_model_score_identically`.
If the same underlying predictor, expressed in two target parameterisations,
does not produce the same numbers in log10 km, then the 2x2 is comparing unit
systems rather than models and every cross-cell difference is an artefact.

Nothing here fits a network.  `run_cell` and `run_2x2` are thin loops over
`models.train_multi_seed`, which has its own tests; what is tested here is the
target/mask contract, the conversion, and the aggregation.
"""

import sys, os, json, shutil, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import targets as T
from ablation import (build_target, common_taus, common_edges, evaluate_cell,
                      scorecard, save_results, _aggregate, _guard_not_frozen,
                      CELLS, D_MAX_KM, NEAR_FIELD_KM)

from constants import FLOOR_KM
DELTA = 0.125


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def toy_d(n=600, seed=0, censored_frac=0.15):
    """Log-uniform distances over the sampler's drawn range, with a clamped
    block, mirroring the dataset."""
    rng = np.random.default_rng(seed)
    d = 10.0 ** rng.uniform(np.log10(FLOOR_KM), np.log10(D_MAX_KM), n)
    d[rng.choice(n, int(n * censored_frac), replace=False)] = FLOOR_KM * 0.1
    return d


# =========================================================================
# BUILD_TARGET -- ONE CLAMP, TWO OUTPUTS
# =========================================================================

def test_log10_target_reproduces_the_dataset_column():
    d = toy_d()
    z, c = build_target(d, 'log10', FLOOR_KM)
    assert np.array_equal(z, np.log10(np.maximum(d, FLOOR_KM)))
    assert np.array_equal(c, d < FLOOR_KM)


def test_censored_rows_land_exactly_on_the_threshold():
    """The invariant `losses.check_censoring_consistency` checks at fit time.
    Asserting it here means a failure is attributed to target construction
    rather than to the loss."""
    d = toy_d()
    for transform, delta in (('log10', None), ('asinh', DELTA)):
        z, c = build_target(d, transform, FLOOR_KM, delta)
        z_L = T.threshold_in_target_units(FLOOR_KM, transform, delta)
        assert c.any()
        assert np.allclose(z[c], z_L, rtol=0.0, atol=1e-15)
        assert np.all(z[~c] > z_L)


def test_censoring_boundary_matches_the_strict_inequality():
    """FOUND BY MUTATION.  The toy data put censored rows at 0.1x the floor, so
    nothing sat ON it and `<` vs `<=` was indistinguishable.  `sampling` sets
    at_floor with a strict `<`, so a row at exactly the floor is NOT censored
    even though its target IS clamped -- measure-zero, deliberate, and it has
    to stay that way or the flag and the clamp disagree."""
    d = np.array([FLOOR_KM * 0.5, FLOOR_KM, FLOOR_KM * 2])
    z, c = build_target(d, 'log10', FLOOR_KM)
    assert c.tolist() == [True, False, False]
    assert z[0] == z[1] == np.log10(FLOOR_KM)


def test_at_floor_column_is_checked_not_trusted():
    d = toy_d()
    good = d < FLOOR_KM
    build_target(d, 'log10', FLOOR_KM, at_floor=good)          # agrees
    bad = good.copy()
    bad[np.flatnonzero(~good)[0]] = True                        # one row off
    _raises(ValueError, build_target, d, 'log10', FLOOR_KM, at_floor=bad)


def test_at_floor_length_mismatch_raises():
    d = toy_d()
    _raises(ValueError, build_target, d, 'log10', FLOOR_KM,
            at_floor=(d < FLOOR_KM)[:-1])


def test_asinh_without_delta_raises():
    _raises(ValueError, build_target, toy_d(), 'asinh', FLOOR_KM)


# =========================================================================
# THE SHARED EVALUATION GRID
# =========================================================================

def test_taus_and_edges_span_the_drawn_range_independently_of_the_data():
    for grid in (common_taus(FLOOR_KM), common_edges(FLOOR_KM)):
        assert abs(grid[0] - np.log10(FLOOR_KM)) < 1e-12
        assert abs(grid[-1] - np.log10(D_MAX_KM)) < 1e-12
    assert len(common_edges(FLOOR_KM)) == 25   # 24 bins, a Week 11 stable choice


# =========================================================================
# THE CONVERSION -- THE ONE THAT MATTERS
# =========================================================================

def test_two_transforms_of_one_model_score_identically():
    """One predictor, two parameterisations, one reported space.

    A predictor off by a fixed factor in physical distance is an additive shift
    under log10 and a different function under asinh.  If `evaluate_cell` is
    converting correctly both must yield the same log10 km metrics -- and if it
    is not, this is the only test that would notice.
    """
    d = toy_d(n=800, seed=1)
    d_hat = np.maximum(d, FLOOR_KM) * 1.7                # same model, both cells

    a = evaluate_cell(T.log10_forward(d_hat, FLOOR_KM)[None, :], d,
                      'log10', FLOOR_KM)
    b = evaluate_cell(T.asinh_forward(d_hat, DELTA)[None, :], d,
                      'asinh', FLOOR_KM, DELTA)

    for key in ('r2_measured', 'rmse_measured', 'bias_near',
                'min_balanced_accuracy'):
        assert abs(a['summary'][key] - b['summary'][key]) < 1e-9
    for f in ('2', '10', '100'):
        for col in ('all', 'measured', 'near'):
            assert abs(a['summary']['factors'][f][col]
                       - b['summary']['factors'][f][col]) < 1e-12


def test_perfect_predictor_scores_perfectly_in_common_space():
    d = toy_d(n=400, seed=2)
    ev = evaluate_cell(T.log10_forward(d, FLOOR_KM)[None, :], d,
                       'log10', FLOOR_KM)
    assert abs(ev['summary']['r2_measured'] - 1.0) < 1e-12
    assert ev['summary']['factors']['1.3']['near'] == 1.0


def test_predictions_below_the_floor_are_censored_like_labels():
    """An asinh cell may predict negative distance.  It must be scored as a
    sub-floor label is stored, not dropped and not turned into nan."""
    d = toy_d(n=200, seed=3)
    ev = evaluate_cell(np.full((1, d.size), -4.0), d, 'asinh', FLOOR_KM, DELTA)
    assert np.isfinite(ev['summary']['r2_measured'])
    assert abs(ev['summary']['frac_below_floor'] - 1.0) < 1e-12


def test_empty_near_field_mask_raises_rather_than_reporting_nan():
    d = np.full(50, 10.0)                                # nothing near the surface
    _raises(ValueError, evaluate_cell,
            T.log10_forward(d, FLOOR_KM)[None, :], d, 'log10', FLOOR_KM)


def test_censored_and_near_fractions_are_reported():
    d = toy_d(n=1000, seed=4, censored_frac=0.2)
    ev = evaluate_cell(T.log10_forward(d, FLOOR_KM)[None, :], d,
                       'log10', FLOOR_KM)
    assert abs(ev['censored_frac'] - (d < FLOOR_KM).mean()) < 1e-12
    assert 0.0 < ev['near_frac'] < 1.0


# =========================================================================
# AGGREGATION ACROSS SEEDS
# =========================================================================

def test_metrics_are_per_seed_then_aggregated_not_scored_on_the_mean():
    """Two seeds that bracket the truth.  Their MEAN prediction is near-perfect,
    so scoring the averaged prediction reports an ENSEMBLE -- a better model than
    either cell contains.  The per-seed aggregate must be visibly worse.

    The two seeds do NOT score identically despite being symmetric offsets, and
    that is correct: `to_log10_km` clamps predictions at the floor exactly as
    the labeller clamps labels, so the downward seed is censored near the floor
    and the upward one is not.  The asymmetry is the censoring, not a bug.
    """
    d = toy_d(n=400, seed=5)
    y = T.log10_forward(d, FLOOR_KM)
    preds = np.vstack([y + 0.4, y - 0.4])
    ev = evaluate_cell(preds, d, 'log10', FLOOR_KM)
    ens = evaluate_cell(preds.mean(axis=0)[None, :], d, 'log10', FLOOR_KM)
    assert ev['n_seeds'] == 2
    assert ens['summary']['r2_measured'] > 0.999          # the ensemble
    assert ev['summary']['r2_measured'] < 0.99            # what is reported
    assert ens['summary']['r2_measured'] - ev['summary']['r2_measured'] > 0.04


def test_aggregate_reports_sample_sd_and_zero_for_one_seed():
    mk = lambda r2: dict(
        regression_measured=dict(r2=r2, rmse=0.0),
        regression_near=dict(bias=0.0, rmse=0.0),
        threshold=dict(min_balanced_accuracy=0.5),
        frac_below_floor=0.0,
        factors=[dict(factor=10.0, log10_tol=1.0, all=r2)])
    out = _aggregate([mk(0.6), mk(0.8)])
    assert abs(out['r2_measured'] - 0.7) < 1e-12
    assert abs(out['r2_measured_sd'] - np.std([0.6, 0.8], ddof=1)) < 1e-12
    assert _aggregate([mk(0.6)])['r2_measured_sd'] == 0.0


# =========================================================================
# SCORECARD AND OUTPUT
# =========================================================================

def test_scorecard_differences_are_against_the_named_baseline():
    d = toy_d(n=400, seed=6)
    y = T.log10_forward(d, FLOOR_KM)
    res = {
        'M1': dict(evaluate_cell((y + 0.5)[None, :], d, 'log10', FLOOR_KM),
                   transform='log10', loss='mse'),
        'M2c': dict(evaluate_cell(y[None, :], d, 'log10', FLOOR_KM),
                    transform='log10', loss='tobit'),
    }
    rows = {r['cell']: r for r in scorecard(res, baseline='M1')}
    assert abs(rows['M1']['d_r2']) < 1e-12
    assert rows['M2c']['d_r2'] > 0.0
    assert rows['M2c']['d_near10'] >= 0.0


def test_frozen_week11_filenames_are_refused():
    for name in ('results_wk11.json', 'predictions_wk11.npz'):
        _raises(ValueError, _guard_not_frozen, os.path.join('../data/wk11', name))
    _guard_not_frozen('../data/wk12/results_wk12.json')


def test_save_refuses_to_overwrite_week11():
    tmp = tempfile.mkdtemp()
    try:
        _raises(ValueError, save_results, {}, {}, out_dir=tmp,
                results_name='results_wk11.json')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_round_trips():
    tmp = tempfile.mkdtemp()
    try:
        d = toy_d(n=200, seed=7)
        res = {'M1': dict(evaluate_cell(T.log10_forward(d, FLOOR_KM)[None, :],
                                        d, 'log10', FLOOR_KM),
                          transform='log10', loss='mse')}
        rp, pp = save_results(res, {'M1': np.zeros((2, 5))}, out_dir=tmp)
        with open(rp) as fh:
            back = json.load(fh)
        assert abs(back['M1']['summary']['r2_measured'] - 1.0) < 1e-12
        assert np.load(pp)['M1'].shape == (2, 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cells_are_the_intended_two_by_two():
    spec = {c[0]: (c[1], c[2]) for c in CELLS}
    assert [c[0] for c in CELLS] == ['M1', 'M2a', 'M2b', 'M2c']
    assert spec['M1'] == ('log10', 'mse') and spec['M2c'] == ('asinh', 'tobit')
    assert spec['M2a'] == ('asinh', 'mse') and spec['M2b'] == ('log10', 'tobit')


def test_near_field_boundary_is_the_measured_one():
    """125 m, the Week 11 dead-zone boundary.  Hardcoded on purpose -- if it
    moves, the arbiter column changes meaning and the change should be a visible
    edit rather than a call-site default."""
    assert abs(NEAR_FIELD_KM - 0.125) < 1e-12
    assert abs(np.log10(NEAR_FIELD_KM / FLOOR_KM) - 2.907) < 5e-3


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
