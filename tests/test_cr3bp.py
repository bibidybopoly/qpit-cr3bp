"""Tests for the CR3BP module — verifies known physical properties.

Run: pytest tests/test_cr3bp.py -v   (or python tests/test_cr3bp.py)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

from cr3bp import (cr3bp_rhs, integrate_with_stm, jacobi_constant,  # noqa: E402
                   find_lagrange_points, MU_EARTH_MOON)


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

def test_stm_matches_finite_differences():
    """The STM should predict perturbation growth, validated against finite differences."""
    from scipy.integrate import solve_ivp
    mu = MU_EARTH_MOON
    x0 = np.array([0.8, 0.0, 0.0, 0.3])
    t_final = 2.0

    sol_ref = integrate_with_stm(x0, (0, t_final), mu, t_eval=[t_final])
    ref_final = sol_ref.y[:4, -1]
    Phi = sol_ref.y[4:, -1].reshape(4, 4)

    epsilon = 1e-6
    for j in range(4):
        dx0 = np.zeros(4)
        dx0[j] = epsilon
        sol_pert = solve_ivp(cr3bp_rhs, (0, t_final), x0 + dx0, args=(mu,),
                             method='DOP853', rtol=1e-12, atol=1e-12, t_eval=[t_final])
        dx_actual = sol_pert.y[:, -1] - ref_final
        dx_predicted = Phi @ dx0
        assert np.max(np.abs(dx_actual - dx_predicted)) < 1e-6, f"STM failed for component {j}"


def test_mu_agrees_across_every_module_that_defines_it():
    """mu has three homes; they must hold the SAME double, not merely close ones.

    `features.py` keeps its own copy on purpose (it stays independent of
    cr3bp_3d so it can cross-check the C convention), and that copy was once
    truncated to 0.012150585609624 -- 23 ULP low, enough to shift a computed
    Jacobi constant by ~4e-16 and make an exact feature-C vs solver-C
    comparison fail for a reason nobody would find.  Exact equality, not
    np.isclose: the whole point is the last bits.
    """
    from cr3bp_3d import MU_EARTH_MOON as MU_3D
    from features import MU_EARTH_MOON as MU_FEAT

    assert MU_EARTH_MOON == MU_3D == MU_FEAT, (
        f"mu disagrees: cr3bp={MU_EARTH_MOON!r} cr3bp_3d={MU_3D!r} "
        f"features={MU_FEAT!r}")
    assert MU_EARTH_MOON == 0.01215058560962404


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")