"""CR3BP dynamics: equations of motion, Jacobi constant, Lagrange points."""

import numpy as np
from scipy.optimize import brentq


MU_EARTH_MOON = 0.01215058560962404


def cr3bp_rhs(t, state, mu):
    """Right-hand side of the planar CR3BP equations of motion.
    
    Parameters
    ----------
    t : float
        Time (unused — system is autonomous, but solve_ivp requires this signature).
    state : array_like, shape (4,)
        State vector [x, y, vx, vy] in the rotating frame.
    mu : float
        Mass parameter μ = m2 / (m1 + m2).

    Returns
    -------
    list of length 4
        Time derivative of the state: [vx, vy, ax, ay].
    """
    x, y, vx, vy = state

    # Distances to primaries
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    # Accelerations
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3
    ay = -2 * vx + y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3

    return [vx, vy, ax, ay]


def jacobi_constant(state, mu):
    """Compute the Jacobi constant.

    The Jacobi constant is the only known integral of motion for this system
    and is therefore our primary diagnostic for integrator correctness.
    It is also the parameter that labels families of QPITs in this project.

    Parameters
    ----------
    state : array_like, shape (4,)
        State vector [x, y, vx, vy].
    mu : float
        Mass parameter.

    Returns
    -------
    float
        The Jacobi constant. Conserved along any true trajectory.
    """
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    omega = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2
    return 2 * omega - (vx ** 2 + vy ** 2)

def find_lagrange_points(mu):
    """Locate all five Lagrange points.
    Uses bracketed root-finding (brentq) for the three collinear points;
    L4 and L5 are given analytically.

    Parameters
    ----------
    mu : float
        Mass parameter.

    Returns
    -------
    dict
        Mapping from 'L1' .. 'L5' to (x, y) coordinate tuples.
    """
    def collinear_eq(x):
        # Sign of (x+mu) and (x-1+mu) is handled by abs() in the radii.
        # The equation comes from setting ẍ = 0 with y=vx=vy=0.
        r1 = abs(x + mu)
        r2 = abs(x - 1 + mu)
        return x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3

    eps = 1e-6
    L1_x = brentq(collinear_eq, -mu + eps, 1 - mu - eps)   # between Earth and Moon
    L2_x = brentq(collinear_eq, 1 - mu + eps, 2.0)         # beyond Moon
    L3_x = brentq(collinear_eq, -2.0, -mu - eps)           # opposite side of Earth from Moon

    return {
        'L1': (L1_x, 0.0),
        'L2': (L2_x, 0.0),
        'L3': (L3_x, 0.0),
        'L4': (0.5 - mu, np.sqrt(3) / 2),
        'L5': (0.5 - mu, -np.sqrt(3) / 2),
    }

def effective_potential(x, y, mu):
    """Compute the effective potential Omega(x, y) in the rotating frame.

    Works on scalars or NumPy arrays (for grid evaluation).

    Parameters
    ----------
    x, y : float or ndarray
        Position coordinates in the rotating frame.
    mu : float
        Mass parameter.

    Returns
    -------
    float or ndarray
        The effective potential Omega at each point.
    """
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    return 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2

def cr3bp_jacobian(state, mu):
    """The 4x4 Jacobian A = df/dx of the planar CR3BP equations of motion.

    This is the matrix that drives the State Transition Matrix via the
    variational equation Phi_dot = A @ Phi. Derived by hand:
    - Top-left block: zeros (velocities don't depend on position)
    - Top-right block: identity (velocity is d/dt of position)
    - Bottom-right block: Coriolis terms [[0, 2], [-2, 0]]
    - Bottom-left block: second derivatives of the effective potential Omega

    Parameters
    ----------
    state : array_like, shape (4,)
        State vector [x, y, vx, vy].
    mu : float
        Mass parameter.

    Returns
    -------
    ndarray, shape (4, 4)
        The Jacobian matrix evaluated at this state.
    """
    x, y, vx, vy = state

    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)

    r1_3, r1_5 = r1 ** 3, r1 ** 5
    r2_3, r2_5 = r2 ** 3, r2 ** 5

    # Second derivatives of the effective potential (your hand derivation)
    Omega_xx = (1
                - (1 - mu) * (1.0 / r1_3 - 3 * (x + mu) ** 2 / r1_5)
                - mu * (1.0 / r2_3 - 3 * (x - 1 + mu) ** 2 / r2_5))

    Omega_yy = (1
                - (1 - mu) * (1.0 / r1_3 - 3 * y ** 2 / r1_5)
                - mu * (1.0 / r2_3 - 3 * y ** 2 / r2_5))

    Omega_xy = (3 * (1 - mu) * (x + mu) * y / r1_5
                + 3 * mu * (x - 1 + mu) * y / r2_5)

    A = np.array([
        [0.0,      0.0,      1.0,  0.0],
        [0.0,      0.0,      0.0,  1.0],
        [Omega_xx, Omega_xy, 0.0,  2.0],
        [Omega_xy, Omega_yy, -2.0, 0.0],
    ])
    return A

def cr3bp_rhs_with_stm(t, aug_state, mu):
    """Augmented RHS: integrate the 4-D state and the 4x4 STM together.

    The augmented state is a 20-vector:
      - aug_state[0:4]   = the trajectory state [x, y, vx, vy]
      - aug_state[4:20]  = the flattened 4x4 STM Phi

    The STM evolves by the variational equation Phi_dot = A(t) @ Phi,
    where A is the Jacobian evaluated at the current trajectory state.

    Parameters
    ----------
    t : float
        Time (unused; system is autonomous).
    aug_state : array_like, shape (20,)
        The augmented state: 4 trajectory components + 16 STM components.
    mu : float
        Mass parameter.

    Returns
    -------
    ndarray, shape (20,)
        Time derivative of the augmented state.
    """
    # Unpack
    state = aug_state[:4]
    Phi = aug_state[4:].reshape(4, 4)

    # Derivative of the trajectory (your existing equations of motion)
    state_dot = cr3bp_rhs(t, state, mu)

    # Derivative of the STM: Phi_dot = A @ Phi
    A = cr3bp_jacobian(state, mu)
    Phi_dot = A @ Phi

    # Repack into a 20-vector
    return np.concatenate([state_dot, Phi_dot.flatten()])

def integrate_with_stm(state0, t_span, mu, **kwargs):
    """Integrate a CR3BP trajectory together with its STM.

    Parameters
    ----------
    state0 : array_like, shape (4,)
        Initial trajectory state.
    t_span : tuple (t0, tf)
        Integration interval.
    mu : float
        Mass parameter.
    **kwargs : passed to solve_ivp (e.g. t_eval, rtol, atol).

    Returns
    -------
    The solve_ivp solution object. The augmented state is in sol.y,
    shape (20, n_times). Rows 0:4 are the trajectory, rows 4:20 the STM.
    """
    from scipy.integrate import solve_ivp
    # Initial augmented state: real state + identity STM (flattened)
    aug0 = np.concatenate([state0, np.eye(4).flatten()])
    defaults = dict(method='DOP853', rtol=1e-12, atol=1e-12)
    defaults.update(kwargs)
    return solve_ivp(cr3bp_rhs_with_stm, t_span, aug0, args=(mu,), **defaults)