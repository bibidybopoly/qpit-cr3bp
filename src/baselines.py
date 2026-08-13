"""Baselines for the Week 11 regressor.

  B0  mean predictor            -- defines the zero of R^2.
  B1  shortcut probe            -- (distance to L1, distance to L2, C) only.
  B2  nearest neighbour         -- run under BOTH splits.

B1 IS A TEST OF THE DATASET, NOT OF THE MODEL, so it must be fitted with a
FLEXIBLE learner.  The Weekly Plan asks this baseline to do POORLY: if a model
knowing only where a state sits relative to the libration points can recover
the distance, the sampling left a location shortcut in and the dataset needs
fixing.  But a weak learner fails on every dataset, so a weak learner's failure
proves nothing.  Fitting gradient-boosted trees and k-NN -- both of which can
represent an essentially arbitrary function of three variables given the sample
count here -- makes failure evidence of ABSENT INFORMATION rather than absent
capacity.  Reported score is the best over the learners tried, selected on the
validation split.

B2 UNDER BOTH SPLITS is the measurement that justifies grouping.  1-NN in
standardised state space is close to a pure memoriser; under a row-random split
it should score high because near-duplicates of every test row sit in train,
and under the grouped split it should fall.  The GAP between the two is the
empirical correlation length of the dataset, and it costs one extra fit.

The collinear libration points are solved here rather than imported so that the
notebook can cross-check them against `orbits`/`cr3bp`; two routes to the same
number is the project's standing verification pattern.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from features import MU_EARTH_MOON


def collinear_points(mu=MU_EARTH_MOON):
    """x-coordinates of L1, L2, L3 in the rotating frame.

    Root of  f(x) = x - (1-mu)(x+mu)/|x+mu|^3 - mu(x-1+mu)/|x-1+mu|^3,
    bracketed away from the two singularities at x = -mu and x = 1-mu.
    Earth-Moon reference: L1 ~ 0.83692, L2 ~ 1.15568, L3 ~ -1.00506.
    """
    def f(x):
        d1 = abs(x + mu)
        d2 = abs(x - 1.0 + mu)
        return x - (1 - mu) * (x + mu) / d1 ** 3 - mu * (x - 1 + mu) / d2 ** 3

    eps = 1e-8
    xL1 = brentq(f, -mu + eps, 1 - mu - eps, xtol=1e-14, rtol=1e-15)
    xL2 = brentq(f, 1 - mu + eps, 2.0, xtol=1e-14, rtol=1e-15)
    xL3 = brentq(f, -2.0, -mu - eps, xtol=1e-14, rtol=1e-15)
    return dict(L1=xL1, L2=xL2, L3=xL3)


def shortcut_features(states, jacobi, mu=MU_EARTH_MOON):
    """(|r - L1|, |r - L2|, C) -- the Weekly Plan's shortcut probe, literally.

    Position only.  Velocity is deliberately withheld: the question is whether
    LOCATION alone gives the distance away, and adding velocity would make this
    a different (and much stronger) baseline whose failure would say less.
    """
    s = np.atleast_2d(np.asarray(states, dtype=float))
    L = collinear_points(mu)
    r = s[:, :3]
    d1 = np.linalg.norm(r - np.array([L['L1'], 0.0, 0.0]), axis=1)
    d2 = np.linalg.norm(r - np.array([L['L2'], 0.0, 0.0]), axis=1)
    return np.column_stack([d1, d2, np.asarray(jacobi, float).reshape(-1)]), \
        ['dist_L1', 'dist_L2', 'C']


class MeanPredictor:
    """B0.  Predicts the training mean everywhere."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.mu_)


class KNNRegressor:
    """B2.  k-nearest-neighbour on whatever matrix it is handed.

    cKDTree rather than sklearn so this baseline has no optional dependency --
    it is the one that carries the grouped-vs-random argument and must always
    run.  Inverse-distance weighting with a floor so a coincident point does not
    divide by zero.
    """

    def __init__(self, k=1, weights='uniform'):
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.atleast_2d(np.asarray(X, dtype=float))
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        self.tree_ = cKDTree(self.X_)
        return self

    def predict(self, X):
        k = min(self.k, len(self.X_))
        d, i = self.tree_.query(np.atleast_2d(np.asarray(X, dtype=float)), k=k)
        if k == 1:
            d, i = d[:, None], i[:, None]
        if self.weights == 'distance':
            w = 1.0 / np.maximum(d, 1e-12)
            return np.sum(w * self.y_[i], axis=1) / np.sum(w, axis=1)
        return np.mean(self.y_[i], axis=1)


def _gbt(seed=0):
    """Gradient-boosted trees, or None if sklearn is unavailable."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:                                   # pragma: no cover
        return None
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_depth=None,
        min_samples_leaf=20, l2_regularization=1e-3, random_state=seed)


def fit_flexible(X_tr, y_tr, X_val, y_val, X_test, seed=0, knn_ks=(1, 5, 25)):
    """Fit several flexible learners, select on VALIDATION, predict on test.

    Returns (test predictions, info dict).  `info['candidates']` records every
    learner's validation RMSE so that "the baseline failed" can be read as
    "every one of these failed", which is the claim actually being made.
    """
    cands = []
    gbt = _gbt(seed)
    if gbt is not None:
        gbt.fit(X_tr, y_tr)
        cands.append(('gbt', gbt))
    for k in knn_ks:
        cands.append((f'knn{k}', KNNRegressor(k=k, weights='distance').fit(X_tr, y_tr)))

    scored = []
    for name, m in cands:
        rmse = float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))
        scored.append((rmse, name, m))
    scored.sort(key=lambda t: t[0])
    best_rmse, best_name, best = scored[0]
    return best.predict(X_test), dict(
        selected=best_name, val_rmse=best_rmse,
        candidates={n: r for r, n, _ in scored},
        sklearn_available=gbt is not None)
