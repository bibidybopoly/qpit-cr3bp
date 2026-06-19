"""Tests for the CR3BP module — verifies known physical properties."""

import numpy as np
from cr3bp import cr3bp_rhs, jacobi_constant, find_lagrange_points, MU_EARTH_MOON


def test_lagrange_points_match_published():
    """L1, L2, L3 positions should match published Earth-Moon values."""
    lps = find_lagrange_points(MU_EARTH_MOON)
    assert abs(lps['L1'][0] - 0.8369) < 1e-3
    assert abs(lps['L2'][0] - 1.1557) < 1e-3
    assert abs(lps['L3'][0] - (-1.0051)) < 1e-3


def test_lagrange_points_are_equilibria():
    """At a Lagrange point with zero velocity, the acceleration should be zero."""
    mu = MU_EARTH_MOON
    lps = find_lagrange_points(mu)
    for name, (x, y) in lps.items():
        state = [x, y, 0.0, 0.0]
        derivs = cr3bp_rhs(0, state, mu)
        # derivs[0] and derivs[1] are velocities (should be zero — they are)
        # derivs[2] and derivs[3] are accelerations (should be near zero)
        assert abs(derivs[2]) < 1e-10, f"{name}: ax = {derivs[2]}"
        assert abs(derivs[3]) < 1e-10, f"{name}: ay = {derivs[3]}"


def test_jacobi_constant_is_real_number():
    """Sanity check that Jacobi constant returns a finite number."""
    state = [0.5, 0.3, 0.1, 0.2]
    C = jacobi_constant(state, MU_EARTH_MOON)
    assert np.isfinite(C)