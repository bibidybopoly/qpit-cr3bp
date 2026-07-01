"""3D CR3BP dynamics: equations of motion, Jacobian, STM — for halo orbits."""

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH_MOON = 0.01215058560962404


def cr3bp_rhs_3d(t, state, mu):
    """3D CR3BP equations of motion. State = [x, y, z, vx, vy, vz]."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3   # purely gravitational, no Coriolis/centrifugal

    return [vx, vy, vz, ax, ay, az]


def jacobi_constant_3d(state, mu):
    """Jacobi constant in 3D."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    omega = 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2
    return 2*omega - (vx**2 + vy**2 + vz**2)


def cr3bp_jacobian_3d(state, mu):
    """The 6x6 Jacobian A = df/dx of the 3D CR3BP equations of motion.

    Block structure:
      [[ 0(3x3)    I(3x3)  ]
       [ Omega2    OmegaC  ]]
    where Omega2 is the potential Hessian (your hand derivation) and
    OmegaC is the Coriolis block.
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    r1_3, r1_5 = r1**3, r1**5
    r2_3, r2_5 = r2**3, r2**5

    # Potential Hessian Omega2 (your six derived entries)
    Oxx = 1 - (1-mu)*(1/r1_3 - 3*(x+mu)**2/r1_5) - mu*(1/r2_3 - 3*(x-1+mu)**2/r2_5)
    Oyy = 1 - (1-mu)*(1/r1_3 - 3*y**2/r1_5) - mu*(1/r2_3 - 3*y**2/r2_5)
    Ozz = -(1-mu)*(1/r1_3 - 3*z**2/r1_5) - mu*(1/r2_3 - 3*z**2/r2_5)
    Oxy = 3*(1-mu)*(x+mu)*y/r1_5 + 3*mu*(x-1+mu)*y/r2_5
    Oxz = 3*(1-mu)*(x+mu)*z/r1_5 + 3*mu*(x-1+mu)*z/r2_5
    Oyz = 3*(1-mu)*y*z/r1_5 + 3*mu*y*z/r2_5

    Omega2 = np.array([[Oxx, Oxy, Oxz],
                       [Oxy, Oyy, Oyz],
                       [Oxz, Oyz, Ozz]])

    # Coriolis block (your derivation)
    OmegaC = np.array([[0.0, 2.0, 0.0],
                       [-2.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])

    # Assemble 6x6
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)      # top-right: identity
    A[3:6, 0:3] = Omega2         # bottom-left: potential Hessian
    A[3:6, 3:6] = OmegaC         # bottom-right: Coriolis
    # top-left stays zeros
    return A

def cr3bp_rhs_with_stm_3d(t, aug_state, mu):
    """Augmented RHS: 6D state + 6x6 STM integrated together (42 components).

    aug_state[0:6]  = state [x, y, z, vx, vy, vz]
    aug_state[6:42] = flattened 6x6 STM
    """
    state = aug_state[:6]
    Phi = aug_state[6:].reshape(6, 6)

    state_dot = cr3bp_rhs_3d(t, state, mu)
    A = cr3bp_jacobian_3d(state, mu)
    Phi_dot = A @ Phi

    return np.concatenate([state_dot, Phi_dot.flatten()])


def integrate_with_stm_3d(state0, t_span, mu, **kwargs):
    """Integrate a 3D trajectory together with its 6x6 STM.

    Returns the solve_ivp solution. aug state in sol.y, shape (42, n_times):
    rows 0:6 are the trajectory, rows 6:42 the flattened STM.
    """
    aug0 = np.concatenate([state0, np.eye(6).flatten()])
    defaults = dict(method='DOP853', rtol=1e-12, atol=1e-12)
    defaults.update(kwargs)
    return solve_ivp(cr3bp_rhs_with_stm_3d, t_span, aug0, args=(mu,), **defaults)