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