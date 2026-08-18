"""Tests for src/metrics.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics import (regression_metrics, accuracy_vs_threshold,
                     summarize_threshold_curve, accuracy_vs_error_tolerance,
                     factor_table, residual_bins, dead_zone, mcc, balanced_accuracy,
                     format_table, FACTORS)


def _raises(exc, fn, *a, **kw):
    """Exception check without a pytest dependency -- this file runs standalone."""
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def test_perfect_prediction():
    y = np.linspace(-4, 3, 100)
    m = regression_metrics(y, y)
    assert abs(m['r2'] - 1.0) < 1e-12
    assert m['rmse'] == 0.0 and m['mae'] == 0.0 and m['bias'] == 0.0


def test_mean_predictor_scores_exactly_zero():
    """Defines the zero of R^2.  B0 predicts the TRAIN mean, so its R^2 on test
    is slightly negative; predicting the test mean gives exactly 0."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert abs(regression_metrics(y, np.full_like(y, y.mean()))['r2']) < 1e-12
    off = regression_metrics(y, np.full_like(y, y.mean() + 0.3))['r2']
    assert off < 0


def test_bias_sign_convention():
    y = np.zeros(10)
    assert regression_metrics(y, y + 0.5)['bias'] > 0     # over-prediction


def test_r2_vs_reference():
    y = np.linspace(0, 1, 50)
    m = regression_metrics(y, y, reference_mean=0.5)
    assert abs(m['r2_vs_reference'] - 1.0) < 1e-12


def test_threshold_curve_degenerates_to_one_at_both_ends():
    """The reason the mean over tau is meaningless and the MINIMUM is quoted."""
    rng = np.random.default_rng(1)
    y = rng.uniform(-4, 3, 800)
    p = y + rng.normal(scale=0.5, size=800)
    c = accuracy_vs_threshold(y, p, taus=np.array([-10.0, 0.0, 10.0]))
    assert c['accuracy'][0] == 1.0 and c['accuracy'][-1] == 1.0
    assert c['accuracy'][1] < 1.0


def test_threshold_curve_perfect_predictor():
    y = np.linspace(-4, 3, 200)
    c = accuracy_vs_threshold(y, y, n_tau=25)
    assert np.allclose(c['accuracy'], 1.0)
    assert np.allclose(c['positive_rate_true'], c['positive_rate_pred'])


def test_threshold_curve_is_invariant_under_monotone_reparameterisation():
    """WEEK 12.  `y < t` against `p < t` uses the same t on both sides, so any
    monotone map of the target leaves the curve alone -- PROVIDED taus are
    mapped too.  This is what makes it the fair headline across the 2x2, and
    why the `taus=None` default must never be used for a cross-cell comparison.
    """
    rng = np.random.default_rng(10)
    d = 10.0 ** rng.uniform(-4, 2, 600)                # km
    d_hat = d * 10.0 ** rng.normal(scale=0.3, size=600)
    taus_km = 10.0 ** np.linspace(-4, 2, 40)

    a = accuracy_vs_threshold(np.log10(d), np.log10(d_hat),
                              taus=np.log10(taus_km))
    b = accuracy_vs_threshold(np.arcsinh(d / 1e-2), np.arcsinh(d_hat / 1e-2),
                              taus=np.arcsinh(taus_km / 1e-2))
    assert np.allclose(a['accuracy'], b['accuracy'])
    assert np.allclose(a['balanced_accuracy'], b['balanced_accuracy'])


def test_summarize_excludes_the_degenerate_tails():
    rng = np.random.default_rng(2)
    y = rng.uniform(-4, 3, 1000)
    p = y + rng.normal(scale=0.4, size=1000)
    c = accuracy_vs_threshold(y, p, taus=np.linspace(-12, 12, 200))
    s = summarize_threshold_curve(c)
    assert s['n_tau_informative'] < 200
    assert s['min_balanced_accuracy'] < 1.0
    assert -4 < s['tau_at_min'] < 3
    assert s['min_balanced_accuracy'] <= s['median_balanced_accuracy']


def test_summarize_handles_no_informative_taus():
    y = np.linspace(0, 1, 20)
    c = accuracy_vs_threshold(y, y, taus=np.array([-5.0]))
    assert summarize_threshold_curve(c)['n_tau_informative'] == 0


def test_mcc_endpoints():
    assert abs(mcc(50, 50, 0, 0) - 1.0) < 1e-12
    assert abs(mcc(0, 0, 50, 50) + 1.0) < 1e-12
    assert mcc(50, 0, 50, 0) == 0.0          # constant predictor -> 0, not NaN


def test_balanced_accuracy_of_constant_predictor():
    """Everything called positive: accuracy can look high, balanced is 0.5."""
    assert abs(balanced_accuracy(90, 0, 10, 0) - 0.5) < 1e-12


def test_error_tolerance_curve_is_monotone_and_bounded():
    rng = np.random.default_rng(3)
    y = rng.normal(size=400)
    p = y + rng.normal(scale=0.3, size=400)
    c = accuracy_vs_error_tolerance(y, p, n_tol=40)
    f = c['fraction_within']
    assert np.all(np.diff(f) >= -1e-15)
    assert f[0] >= 0.0 and abs(f[-1] - 1.0) < 1e-12


def test_error_tolerance_matches_a_hand_count():
    y = np.zeros(5)
    p = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    c = accuracy_vs_error_tolerance(y, p, tols=np.array([0.1, 0.6]))
    assert c['fraction_within'].tolist() == [0.4, 0.8]


# =========================================================================
# WEEK 12 -- MASKS AND THE FACTOR TABLE
# =========================================================================

def test_mask_selects_by_label_and_reports_the_count():
    """Selection is by LABEL, so the same rows are chosen in every cell of the
    2x2.  Selecting on the prediction would let each cell score itself on a
    different subset."""
    y = np.array([-3.0, -1.0, 0.0, 1.0])
    p = y + 0.1
    c = accuracy_vs_error_tolerance(y, p, tols=[0.2], mask=y < -0.5)
    assert c['n'] == 2 and c['fraction_within'][0] == 1.0


def test_empty_mask_raises_rather_than_returning_nan():
    """The near-field column is the Week 12 arbiter; an empty one has to be
    loud, not a NaN in a table."""
    y = np.linspace(-3, 2, 20)
    _raises(ValueError, accuracy_vs_error_tolerance, y, y, tols=[0.1],
            mask=np.zeros(20, bool))
    _raises(ValueError, accuracy_vs_threshold, y, y, taus=[0.0],
            mask=np.zeros(20, bool))


def test_non_boolean_mask_raises():
    y = np.linspace(-3, 2, 20)
    _raises(TypeError, accuracy_vs_error_tolerance, y, y, tols=[0.1],
            mask=np.zeros(20, int))


def test_mask_reaches_every_curve_that_takes_one():
    """One `_apply_mask` behind all three, so the selection rule cannot drift
    between the curves that are read side by side."""
    y = np.linspace(-4, 2, 300)
    p = y + np.where(y < -2, 0.9, 0.0)
    near = y < -2
    assert accuracy_vs_threshold(y, p, taus=[-3.0], mask=near)['tau'].size == 1
    rows = residual_bins(y, p, n_bins=4, mask=~near)
    assert all(abs(r['bias']) < 1e-12 for r in rows if r['n'])
    assert sum(r['n'] for r in rows) == int((~near).sum())


def test_factor_table_shape_and_hand_count():
    y = np.zeros(4)
    # errors of 0, 0.2, 0.4, 1.5 dex -> within factor 2 (0.301 dex): 2 of 4
    p = np.array([0.0, 0.2, 0.4, 1.5])
    rows = factor_table(y, p)
    assert [r['factor'] for r in rows] == list(FACTORS)
    by_f = {r['factor']: r for r in rows}
    assert by_f[2.0]['all'] == 0.5
    assert by_f[3.0]['all'] == 0.75          # 0.477 dex admits the third
    assert by_f[100.0]['all'] == 1.0


def test_factor_table_columns_follow_the_masks_given():
    y = np.linspace(-4, 2, 200)
    p = y + 0.05
    rows = factor_table(y, p, masks={'all': None, 'near': y < -3})
    assert set(rows[0]) == {'factor', 'log10_tol', 'all', 'near'}
    assert rows[0]['near'] == 1.0            # 0.05 dex is inside factor 1.3


def test_factor_table_log10_tols_are_the_reported_ones():
    """The Week 11 table printed these; Week 12's must be readable against it."""
    expected = [0.114, 0.301, 0.477, 0.699, 1.000, 1.477, 2.000]
    got = [r['log10_tol'] for r in factor_table(np.zeros(3), np.zeros(3))]
    assert all(abs(a - b) < 5e-4 for a, b in zip(got, expected))


# =========================================================================

def test_residual_bins_detect_a_planted_near_field_bias():
    """The P3 signature, planted: over-prediction concentrated at small y."""
    rng = np.random.default_rng(4)
    y = rng.uniform(-4, 3, 4000)
    p = y + np.where(y < -2, 0.8, 0.0) + rng.normal(scale=0.05, size=4000)
    rows = residual_bins(y, p, n_bins=7)
    assert rows[0]['bias'] > 0.5
    assert abs(rows[-1]['bias']) < 0.1
    assert rows[0]['rmse'] > rows[-1]['rmse']
    assert sum(r['n'] for r in rows) == 4000


def test_residual_bins_slope_is_minus_one_for_a_constant_predictor():
    """WEEK 11's dead-zone signature, recorded as a test.  A model emitting a
    constant has bias = log10(c) - y, i.e. slope exactly -1 against the target.
    The measured -0.976 to -0.999 is what identified the saturation."""
    y = np.linspace(-4, -1, 2000)
    p = np.full_like(y, -2.2)                # constant 6.3 m in log10 km
    rows = [r for r in residual_bins(y, p, n_bins=10) if r['n']]
    mid = np.array([(r['bin_lo'] + r['bin_hi']) / 2 for r in rows])
    bias = np.array([r['bias'] for r in rows])
    slope = np.polyfit(mid, bias, 1)[0]
    # NOT exactly -1, and cannot be: bias uses the bin MEAN of y while the fit
    # regresses on the bin CENTRE, and the offset between them drifts linearly
    # across bins.  Here that puts the exact answer at -1.0005.  So -0.999 in
    # the Week 11 table is at the discretisation floor, not short of it.
    assert abs(slope + 1.0) < 2e-3


def test_dead_zone_recovers_a_planted_constant_output():
    """A model that emits 6.3 m below 125 m and tracks above it.  The detector
    must return slope -1, the constant, and the boundary."""
    y = np.linspace(-3.81, 2.0, 6000)
    b = np.log10(0.125)
    p = np.where(y < b, -2.2, y)
    dz = dead_zone(y, p, n_bins=12)
    assert dz['flat'] and dz['self_consistent']
    assert abs(dz['slope'] + 1.0) < 0.05
    assert abs(dz['constant_m'] - 6.3) < 0.5
    assert 0.5 < dz['boundary_m'] / 125.0 < 2.0


def test_dead_zone_self_consistency_catches_a_truncated_run():
    """The n_bins=36 case: the fixed tolerance truncates on bin noise, and the
    reported constant then sits OUTSIDE the range it was constant over.  The
    flag has to catch that without anyone eyeballing the table."""
    lo, hi = -3.81, -2.5
    # constant above the boundary is incoherent for a monotone predictor
    assert not (lo <= np.log10(6.2e-3) <= hi)
    y = np.linspace(-3.81, 2.0, 4000)
    p = np.where(y < np.log10(0.125), -2.2, y)
    for n_bins in (12, 18, 24):
        dz = dead_zone(y, p, n_bins=n_bins)
        assert dz['self_consistent'], n_bins


def test_dead_zone_reports_no_flat_prefix_when_the_model_tracks():
    y = np.linspace(-4, 2, 3000)
    dz = dead_zone(y, y + 0.02, n_bins=12)
    assert dz['flat'] is False and dz['k'] < 3


def test_dead_zone_honours_the_mask():
    """Week 11 scored this on measured rows only: 13% of the test set is pinned
    at the floor, all of it in the smallest-d bins, and left-censoring makes a
    large positive bias there mechanically."""
    y = np.linspace(-3.81, 2.0, 4000)
    p = np.where(y < np.log10(0.125), -2.2, y)
    floor = y <= -3.8
    assert dead_zone(y, p, n_bins=12, mask=~floor)['k'] >= 3


def test_regression_metrics_rejects_shape_mismatch():
    _raises(ValueError, regression_metrics, np.zeros(5), np.zeros(6))


def test_format_table_runs():
    assert 'bias' in format_table(residual_bins(np.linspace(0, 1, 50),
                                                np.linspace(0, 1, 50), n_bins=3))


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
