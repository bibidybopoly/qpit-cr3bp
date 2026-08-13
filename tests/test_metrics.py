"""Tests for src/metrics.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics import (regression_metrics, accuracy_vs_threshold,
                     summarize_threshold_curve, accuracy_vs_error_tolerance,
                     residual_bins, mcc, balanced_accuracy, format_table)


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


def test_regression_metrics_rejects_shape_mismatch():
    try:
        regression_metrics(np.zeros(5), np.zeros(6))
    except ValueError:
        return
    raise AssertionError("shape mismatch accepted")


def test_format_table_runs():
    assert 'bias' in format_table(residual_bins(np.linspace(0, 1, 50),
                                                np.linspace(0, 1, 50), n_bins=3))


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
