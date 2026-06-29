import numpy as np
import sys
sys.path.insert(0, '../src')
from cr3bp import cr3bp_jacobian, find_lagrange_points, integrate_with_stm, MU_EARTH_MOON

mu = MU_EARTH_MOON
lps = find_lagrange_points(mu)

# The L1 point as a full state (on x-axis, at rest)
L1_state = [lps['L1'][0], 0.0, 0.0, 0.0]

# Jacobian at L1
A_L1 = cr3bp_jacobian(L1_state, mu)
print("Jacobian at L1:")
print(A_L1)
print()

# Its eigenvalues
eigvals = np.linalg.eigvals(A_L1)
print("Eigenvalues at L1:")
for ev in eigvals:
    print(f"  {ev:.6f}")

# Get eigenvalues AND eigenvectors
eigvals, eigvecs = np.linalg.eig(A_L1)

# Find the eigenvalue with POSITIVE imaginary part (the +i*omega one)
imag_idx = np.argmax(eigvals.imag)   # index of the most-positive-imaginary eigenvalue
omega = eigvals[imag_idx].imag
print(f"Oscillation frequency omega = {omega:.6f}")
print(f"Linearized orbit period T = {2*np.pi/omega:.6f}")
print()

# The corresponding (complex) eigenvector
v = eigvecs[:, imag_idx]
print("Complex eigenvector for +i*omega:")
print(f"  x-component:  {v[0]:.6f}")
print(f"  y-component:  {v[1]:.6f}")
print(f"  vx-component: {v[2]:.6f}")
print(f"  vy-component: {v[3]:.6f}")

# Extract the real parts that define the symmetric initial condition
x_comp = v[0].real      # -0.105758
vy_comp = v[3].real     #  0.885435

# The ratio of vy0 to x-offset for the linearized Lyapunov orbit
ratio = vy_comp / x_comp
print(f"vy0 / delta_x ratio = {ratio:.6f}")

# Choose a small orbit: offset x from L1 by a small amount.
# Sign convention: let's offset to the LEFT of L1 (toward Earth), delta_x < 0,
# which makes vy0 positive. Small magnitude = small orbit = closer to linear.
delta_x = -0.001
x0 = lps['L1'][0] + delta_x
vy0 = ratio * delta_x

print(f"\nLinearized initial guess for a small L1 Lyapunov orbit:")
print(f"  x0  = {x0:.8f}   (L1 offset by {delta_x})")
print(f"  y0  = 0.0")
print(f"  vx0 = 0.0")
print(f"  vy0 = {vy0:.8f}")

# Sanity check: integrate this guess to the x-axis crossing and see the residual.
# If the linearization is good, the residual vx should be SMALL (unlike the 0.63 before).
from scipy.integrate import solve_ivp
from cr3bp import cr3bp_rhs

def y_axis_crossing(t, state, mu):
    return state[1]
y_axis_crossing.terminal = True
y_axis_crossing.direction = -1

state0 = [x0, 0.0, 0.0, vy0]
sol = solve_ivp(cr3bp_rhs, (0, 10), state0, args=(mu,),
                method='DOP853', rtol=1e-12, atol=1e-12,
                events=y_axis_crossing)

if len(sol.t_events[0]) > 0:
    state_cross = sol.y_events[0][0]
    t_cross = sol.t_events[0][0]
    print(f"\nHalf-period crossing at t = {t_cross:.6f} (linear prediction: {np.pi/omega:.6f})")
    print(f"  x at crossing  = {state_cross[0]:.8f}")
    print(f"  vx at crossing = {state_cross[2]:.8e}   <- residual to correct")
else:
    print("\nNo crossing found — check the guess.")

def correct_lyapunov_orbit(x0, vy0_guess, mu, tol=1e-11, max_iter=20, verbose=True):
    """Differential corrector for a planar Lyapunov orbit via single shooting.

    Fixes x0; adjusts vy0 so that vx = 0 at the perpendicular x-axis crossing
    (the half-period). Uses the STM for a Newton correction, with the
    moving-crossing-time correction term.

    Returns the corrected vy0, the half-period, and the converged crossing state.
    """
    vy0 = vy0_guess

    for iteration in range(max_iter):
        # Integrate WITH the STM until the x-axis crossing
        state0 = [x0, 0.0, 0.0, vy0]

        def crossing_event(t, aug, mu):
            return aug[1]   # y-component
        crossing_event.terminal = True
        crossing_event.direction = -1

        sol = integrate_with_stm(state0, (0, 10), mu, events=crossing_event)

        if len(sol.t_events[0]) == 0:
            raise RuntimeError("No x-axis crossing found during correction.")

        # State and STM at the crossing
        aug_cross = sol.y_events[0][0]
        state_cross = aug_cross[:4]
        Phi = aug_cross[4:].reshape(4, 4)

        vx_cross = state_cross[2]   # the residual we want to zero

        if verbose:
            print(f"  iter {iteration}: vy0={vy0:.12f}, residual vx={vx_cross:.3e}")

        # Converged?
        if abs(vx_cross) < tol:
            half_period = sol.t_events[0][0]
            if verbose:
                print(f"  Converged: vx residual {vx_cross:.2e} < tol {tol:.0e}")
            return vy0, half_period, state_cross

        # --- Newton correction with moving-time term ---
        # Derivatives at the crossing (from equations of motion)
        deriv = cr3bp_rhs(0, state_cross, mu)
        ydot = deriv[1]        # d(y)/dt at crossing
        vxdot = deriv[2]       # d(vx)/dt at crossing

        # Corrected sensitivity: d(vx_cross)/d(vy0)
        # Phi[2,3] = sensitivity of vx (idx 2) to initial vy (idx 3)
        # Phi[1,3] = sensitivity of y  (idx 1) to initial vy (idx 3)
        sensitivity = Phi[2, 3] - (vxdot / ydot) * Phi[1, 3]

        # Newton step
        delta_vy0 = -vx_cross / sensitivity
        vy0 = vy0 + delta_vy0

    raise RuntimeError(f"Did not converge in {max_iter} iterations.")