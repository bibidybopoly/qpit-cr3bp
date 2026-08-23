import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cr3bp_3d import cr3bp_rhs_3d, integrate_with_stm_3d, MU_EARTH_MOON

def test_stm_3d_matches_finite_differences():
    """The 3D STM should predict perturbation growth, validated vs finite differences."""
    mu = MU_EARTH_MOON
    x0 = np.array([0.82, 0.0, 0.1, 0.0, 0.15, 0.05])
    t_final = 1.5
    sol_ref = integrate_with_stm_3d(x0, (0, t_final), mu, t_eval=[t_final])
    ref_final = sol_ref.y[:6, -1]
    Phi = sol_ref.y[6:, -1].reshape(6, 6)
    epsilon = 1e-6
    for j in range(6):
        dx0 = np.zeros(6); dx0[j] = epsilon
        sol_pert = solve_ivp(cr3bp_rhs_3d, (0, t_final), x0 + dx0, args=(mu,),
                             method='DOP853', rtol=1e-12, atol=1e-12, t_eval=[t_final])
        dx_actual = sol_pert.y[:, -1] - ref_final
        dx_predicted = Phi @ dx0
        assert np.max(np.abs(dx_actual - dx_predicted)) < 1e-6, f"3D STM failed component {j}"

if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
