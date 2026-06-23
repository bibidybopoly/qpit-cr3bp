"""Small Titan test: integrate several CR3BP trajectories and report Jacobi conservation.

This is a pipeline test, not science — it proves the full chain works on Titan:
module load -> venv -> import cr3bp -> compute -> write results.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from cr3bp import cr3bp_rhs, jacobi_constant, find_lagrange_points, MU_EARTH_MOON

mu = MU_EARTH_MOON
lps = find_lagrange_points(mu)

# A few test initial conditions near different Lagrange points
test_ics = {
    'near_L1': [lps['L1'][0] + 0.01, 0.0, 0.0, 0.0],
    'near_L4': [lps['L4'][0] + 0.05, lps['L4'][1], 0.0, 0.0],
    'near_L2': [lps['L2'][0] + 0.01, 0.0, 0.0, 0.0],
}

results = []
for name, ic in test_ics.items():
    sol = solve_ivp(
        cr3bp_rhs, (0, 50), ic, args=(mu,),
        method='DOP853', rtol=1e-12, atol=1e-12,
        t_eval=np.linspace(0, 50, 5000)
    )
    C_vals = np.array([jacobi_constant(s, mu) for s in sol.y.T])
    max_dev = np.abs(C_vals - C_vals[0]).max()
    results.append((name, sol.success, max_dev))
    print(f"{name}: success={sol.success}, max Jacobi deviation={max_dev:.2e}")

# Write results to a file we can retrieve
outpath = "titan_test_results.txt"
with open(outpath, 'w') as f:
    f.write("CR3BP Titan pipeline test results\n")
    f.write("=" * 40 + "\n")
    for name, success, dev in results:
        f.write(f"{name}: success={success}, max Jacobi deviation={dev:.2e}\n")

print(f"\nResults written to {outpath}")
print("Pipeline test complete.")