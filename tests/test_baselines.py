"""Tests for src/baselines.py.

The two that matter are at the bottom.  B1 is the Weekly Plan's dataset check
and B2 is the justification for grouping, and neither is worth anything unless
it has been shown to respond to the thing it is supposed to detect.  So a
synthetic dataset is built twice -- once clean, once with a location shortcut
deliberately planted -- and the diagnostics are required to tell them apart.

The synthetic fixture mimics the Week 10 schema's GEOMETRY, not its physics:
base points on a ring of varying amplitude around L2, states pushed off by a
log-uniform distance along a random direction, several siblings per base point.
That is enough to reproduce both failure modes.  It imports no CR3BP code.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baselines import (collinear_points, shortcut_features, MeanPredictor,
                       KNNRegressor, fit_flexible)
from features import MU_EARTH_MOON, jacobi_constant, Standardizer
from splits import grouped_split, random_row_split
from metrics import regression_metrics

LU = 384400.0


def test_collinear_points_match_published_earth_moon_values():
    L = collinear_points(MU_EARTH_MOON)
    assert abs(L['L1'] - 0.836915) < 1e-4, L['L1']
    assert abs(L['L2'] - 1.155682) < 1e-4, L['L2']
    assert abs(L['L3'] + 1.005063) < 1e-4, L['L3']
    assert L['L1'] < 1 - MU_EARTH_MOON < L['L2']


def test_collinear_points_are_roots():
    mu = MU_EARTH_MOON
    for x in collinear_points(mu).values():
        f = x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 \
            - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3
        assert abs(f) < 1e-10


def test_shortcut_features_shape_and_names():
    S = np.zeros((7, 6))
    X, names = shortcut_features(S, np.full(7, 3.06))
    assert X.shape == (7, 3)
    assert names == ['dist_L1', 'dist_L2', 'C']


def test_shortcut_uses_position_only():
    """Velocity must not enter: the question is whether LOCATION alone leaks."""
    rng = np.random.default_rng(0)
    S = rng.normal(size=(20, 6))
    T = S.copy(); T[:, 3:] = rng.normal(size=(20, 3))
    a, _ = shortcut_features(S, np.zeros(20))
    b, _ = shortcut_features(T, np.zeros(20))
    assert np.allclose(a, b)


def test_mean_predictor():
    y = np.array([1.0, 2.0, 6.0])
    p = MeanPredictor().fit(None, y).predict(np.zeros((4, 2)))
    assert p.shape == (4,) and np.allclose(p, 3.0)


def test_knn_recalls_its_own_training_points():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] * 2 - X[:, 1]
    m = KNNRegressor(k=1).fit(X, y)
    assert np.allclose(m.predict(X), y)


def test_knn_k_larger_than_training_set():
    X = np.zeros((3, 2)); y = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(KNNRegressor(k=25).fit(X, y).predict(X)).all()


# --------------------------------------------------------------------------
# synthetic fixture
# --------------------------------------------------------------------------

def make_synthetic(n_groups=400, siblings=6, planted=False, seed=0):
    """Ring of base points around L2; states offset by a log-uniform distance.

    planted=False : d is drawn independently of the base point, so location
                    carries no information about it -- the clean case.
    planted=True  : d is a deterministic function of the ring amplitude, which
                    the distance to L2 measures directly -- the failure mode
                    the Weekly Plan is checking for.
    """
    rng = np.random.default_rng(seed)
    xL2 = collinear_points(MU_EARTH_MOON)['L2']
    n = n_groups * siblings

    amp = rng.uniform(2e-5, 2.3e-3, n_groups)              # ~8 to 890 km
    th = rng.uniform(0, 2 * np.pi, n_groups)
    base = np.column_stack([
        xL2 + amp * np.cos(th), amp * 1.7 * np.sin(th), -amp * 0.9 * np.cos(th)])
    base = np.repeat(base, siblings, axis=0)
    amp_r = np.repeat(amp, siblings)

    if planted:
        d_km = 10.0 ** (np.log10(amp_r * LU) - 2.5)
    else:
        d_km = 10.0 ** rng.uniform(-3.8, 2.7, n)

    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pos = base + (d_km / LU)[:, None] * u
    vel = np.repeat(rng.normal(scale=0.35, size=(n_groups, 3)), siblings, axis=0)
    states = np.hstack([pos, vel])

    group_id = np.repeat(np.arange(n_groups), siblings)
    return dict(states=states, log_d_km=np.log10(d_km), d_km=d_km,
                group_id=group_id, family_id=np.zeros(n, dtype=int),
                jacobi=jacobi_constant(states), amp_km=amp_r * LU)


def _score_shortcut(D, seed=0):
    m = grouped_split(D['group_id'], seed=seed)
    X, _ = shortcut_features(D['states'], D['jacobi'])
    sc = Standardizer().fit(X[m['train']])
    Z = sc.transform(X)
    y = D['log_d_km']
    p, info = fit_flexible(Z[m['train']], y[m['train']],
                           Z[m['val']], y[m['val']], Z[m['test']], seed=seed)
    return regression_metrics(y[m['test']], p)['r2'], info


def test_shortcut_probe_fires_on_a_planted_shortcut():
    r2, _ = _score_shortcut(make_synthetic(planted=True, seed=3))
    assert r2 > 0.9, f"planted shortcut not detected (R2={r2:.3f})"


def test_shortcut_probe_stays_quiet_on_a_clean_dataset():
    r2, _ = _score_shortcut(make_synthetic(planted=False, seed=4))
    assert r2 < 0.25, f"clean dataset scored too well (R2={r2:.3f})"


def test_nearest_neighbour_gap_between_grouped_and_random_splits():
    """The measurement that justifies grouping.  1-NN should do markedly better
    under a row-random split, because siblings of every test row sit in train.
    """
    D = make_synthetic(planted=False, seed=5)
    y = D['log_d_km']
    out = {}
    for tag, m in (('grouped', grouped_split(D['group_id'], seed=0)),
                   ('random', random_row_split(len(y), seed=0))):
        sc = Standardizer().fit(D['states'][m['train']])
        Z = sc.transform(D['states'])
        p = KNNRegressor(k=1).fit(Z[m['train']], y[m['train']]).predict(Z[m['test']])
        out[tag] = regression_metrics(y[m['test']], p)['r2']
    assert out['random'] > out['grouped'] + 0.1, out


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
