"""Certification of the per-point local-distance target.

The distance is the regression label for the entire Phase 4 dataset, so it is
checked against cases whose answer is known independently: exactness on the
curve, invariance under reparameterisation, agreement with brute force, and
recovery of an imposed displacement.

Run: pytest tests/test_torus_probe.py -v   (or python tests/test_torus_probe.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qpit import evaluate_curve, node_angles                       # noqa: E402
from torus_probe import (curve_distance, curve_local_distance,     # noqa: E402
                         regression_target)

N = 31
from constants import LU             # one definition, project-wide


def _curve(seed=0, R=5.0e-4):
    """A band-limited closed curve in 6-D, resolvable at N nodes."""
    rng = np.random.default_rng(seed)
    th = node_angles(N)
    U = np.zeros((N, 6))
    U[:, 0] = 1.05 + R * np.cos(th)
    U[:, 1] = R * np.sin(th)
    U[:, 2] = -0.07 + 0.4 * R * np.cos(2 * th)
    U[:, 3] = 0.3 * R * np.sin(th)
    U[:, 4] = 0.39 + 0.3 * R * np.cos(th)
    U[:, 5] = 0.2 * R * np.sin(2 * th)
    return U


def test_points_on_the_curve_have_zero_distance():
    """Exactness. Points sampled off-node must still land at machine zero."""
    U = _curve()
    th = np.array([0.11, 1.7, 3.33, 4.9, 6.1])
    r = curve_local_distance(evaluate_curve(U, th), U)
    assert r['d6'].max() < 1e-12, r['d6'].max()
    # and the foot must be the angle we sampled at
    dth = np.mod(r['theta_foot'] - th + np.pi, 2 * np.pi) - np.pi
    assert np.abs(dth).max() < 1e-8, np.abs(dth).max()


def test_invariant_under_reparameterisation():
    """THE decisive test.

    Phase-shift a curve. It is the SAME SET, so the geometric distance is
    zero, while the node-wise distance is O(R * sigma) and reports pure
    relabelling as motion. This is the reason the target changed.
    """
    U = _curve()
    sigma = 0.35
    U_shift = evaluate_curve(U, node_angles(N) + sigma)

    node_max, _ = curve_distance(U_shift, U, normalise=False)
    geo = curve_local_distance(U_shift, U)['d6'].max()

    assert node_max > 1e-4, node_max          # node-wise sees a large motion
    assert geo < 1e-11, geo                   # geometric sees none
    assert node_max / max(geo, 1e-300) > 1e6


def test_matches_brute_force():
    """Agreement with a dense direct search, for off-curve points."""
    U = _curve()
    rng = np.random.default_rng(4)
    X = evaluate_curve(U, rng.uniform(0, 2*np.pi, 12))
    X = X + 3.0e-5 * rng.standard_normal(X.shape)

    got = curve_local_distance(X, U)['d6']
    th_b = np.linspace(0, 2*np.pi, 400001, endpoint=False)
    G = evaluate_curve(U, th_b)
    brute = np.array([np.linalg.norm(G - x, axis=1).min() for x in X])

    # brute force is limited by its own grid; refined answer must be <= it
    assert np.all(got <= brute * (1 + 1e-9) + 1e-14)
    assert np.max((brute - got) / brute) < 1e-3


def test_recovers_an_imposed_normal_displacement():
    """Displace along the local normal by delta; distance must return delta."""
    U = _curve()
    th = np.array([0.4, 2.2, 5.0])
    X0 = evaluate_curve(U, th)
    tang = evaluate_curve(U, th + 1e-6) - evaluate_curve(U, th - 1e-6)
    tang /= np.linalg.norm(tang, axis=1)[:, None]

    rng = np.random.default_rng(7)
    for delta in (1e-9, 1e-7, 1e-5):
        n = rng.standard_normal(X0.shape)
        n -= (np.sum(n * tang, axis=1)[:, None]) * tang       # project out tangent
        n /= np.linalg.norm(n, axis=1)[:, None]
        got = curve_local_distance(X0 + delta * n, U)['d6']
        assert np.allclose(got, delta, rtol=2e-3), (delta, got)


def test_position_only_target_is_degenerate():
    """Why the target is 6-D.

    Two states at the SAME point with different velocities must not receive
    the same label. d3 cannot tell them apart; d6 can.
    """
    U = _curve()
    x_on = evaluate_curve(U, [1.0])[0]
    x_off = x_on.copy()
    x_off[3:] += 1e-6                                  # velocity only
    r = curve_local_distance(np.vstack([x_on, x_off]), U)
    # position-only: both sit at the same point, which is ON the curve
    assert r['d3'].max() < 1e-11, r['d3']
    # 6-D: separates them. Note it comes in slightly BELOW the fixed-angle
    # value 1e-6*sqrt(3) -- the minimiser slides along the curve to trade
    # position error against velocity error, so the true minimum beats it.
    fixed_angle = 1e-6 * np.sqrt(3)
    assert r['d6'][1] <= fixed_angle * (1 + 1e-9), (r['d6'][1], fixed_angle)
    assert r['d6'][1] > 0.9 * fixed_angle, r['d6'][1]
    assert r['d6'][1] / max(r['d6'][0], 1e-300) > 1e4


def test_regression_target_floor_and_censor():
    d = np.array([1e-15, 1e-8, 1e-2, 1e2])
    log_d, cens = regression_target(d, floor=1e-12, censor=1e-1)
    assert np.isclose(log_d[0], -12.0)                 # floored, not -15
    assert np.isclose(log_d[1], -8.0)
    assert list(cens) == [False, False, False, True]


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
