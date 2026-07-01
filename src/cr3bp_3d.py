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
    """The 6x6 Jacobian A = df/dx of the 3D CR3BP equations of motion."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    r1_3, r1_5 = r1**3, r1**5
    r2_3, r2_5 = r2**3, r2**5

    Oxx = 1 - (1-mu)*(1/r1_3 - 3*(x+mu)**2/r1_5) - mu*(1/r2_3 - 3*(x-1+mu)**2/r2_5)
    Oyy = 1 - (1-mu)*(1/r1_3 - 3*y**2/r1_5) - mu*(1/r2_3 - 3*y**2/r2_5)
    Ozz = -(1-mu)*(1/r1_3 - 3*z**2/r1_5) - mu*(1/r2_3 - 3*z**2/r2_5)
    Oxy = 3*(1-mu)*(x+mu)*y/r1_5 + 3*mu*(x-1+mu)*y/r2_5
    Oxz = 3*(1-mu)*(x+mu)*z/r1_5 + 3*mu*(x-1+mu)*z/r2_5
    Oyz = 3*(1-mu)*y*z/r1_5 + 3*mu*y*z/r2_5

    Omega2 = np.array([[Oxx, Oxy, Oxz],
                       [Oxy, Oyy, Oyz],
                       [Oxz, Oyz, Ozz]])
    OmegaC = np.array([[0.0, 2.0, 0.0],
                       [-2.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = Omega2
    A[3:6, 3:6] = OmegaC
    return A


def cr3bp_rhs_with_stm_3d(t, aug_state, mu):
    """Augmented RHS: 6D state + 6x6 STM integrated together (42 components)."""
    state = aug_state[:6]
    Phi = aug_state[6:].reshape(6, 6)
    state_dot = cr3bp_rhs_3d(t, state, mu)
    A = cr3bp_jacobian_3d(state, mu)
    Phi_dot = A @ Phi
    return np.concatenate([state_dot, Phi_dot.flatten()])


def integrate_with_stm_3d(state0, t_span, mu, **kwargs):
    """Integrate a 3D trajectory together with its 6x6 STM."""
    aug0 = np.concatenate([state0, np.eye(6).flatten()])
    defaults = dict(method='DOP853', rtol=1e-12, atol=1e-12)
    defaults.update(kwargs)
    return solve_ivp(cr3bp_rhs_with_stm_3d, t_span, aug0, args=(mu,), **defaults)


# ============================================================
#  Richardson 3rd-order halo seed + differential corrector
# ============================================================

def _richardson_constants(mu, point='L1'):
    """All Richardson (1980) constants for the L1 or L2 halo construction.

    point : 'L1' or 'L2'. Only the collinear-point quintic and the Legendre
    coefficients c_n differ between the two; everything downstream (the a/b/d
    amplitude coefficients, l1, l2, ...) is identical once c2, c3, c4 are set.
    """
    # gamma: distance from the Moon to the libration point (Newton on the quintic)
    g = (mu/3.0)**(1.0/3.0)
    for _ in range(100):
        if point == 'L1':
            f  = g**5 - (3-mu)*g**4 + (3-2*mu)*g**3 - mu*g**2 + 2*mu*g - mu
            df = 5*g**4 - 4*(3-mu)*g**3 + 3*(3-2*mu)*g**2 - 2*mu*g + 2*mu
        elif point == 'L2':
            f  = g**5 + (3-mu)*g**4 + (3-2*mu)*g**3 - mu*g**2 - 2*mu*g - mu
            df = 5*g**4 + 4*(3-mu)*g**3 + 3*(3-2*mu)*g**2 - 2*mu*g - 2*mu
        else:
            raise ValueError(f"point must be 'L1' or 'L2', got {point!r}")
        g -= f/df
    if point == 'L1':
        cn = lambda n: (1.0/g**3)*(mu + (-1)**n*(1-mu)*g**(n+1)/(1-g)**(n+1))
    else:  # L2: expansion centered past the Moon -> (1-g)->(1+g); mu term gains (-1)^n
        cn = lambda n: ((-1)**n/g**3)*(mu + (1-mu)*g**(n+1)/(1+g)**(n+1))
    c2, c3, c4 = cn(2), cn(3), cn(4)

    # in-plane frequency lam(=omega_p): oscillatory root of the linear system
    b, cc = 2-c2, 1+c2-2*c2**2                 # beta^2 + b*beta + cc = 0
    beta_neg = (-b - np.sqrt(b*b - 4*cc))/2    # negative (oscillatory) root
    lam = np.sqrt(-beta_neg)
    k   = 2*lam/(lam**2 + 1 - c2)
    Delta = lam**2 - c2

    d1 = (3*lam**2/k)*(k*(6*lam**2-1) - 2*lam)
    d2 = (8*lam**2/k)*(k*(11*lam**2-1) - 2*lam)
    a21 = 3*c3*(k**2-2)/(4*(1+2*c2));           a22 = 3*c3/(4*(1+2*c2))
    a23 = -3*c3*lam/(4*k*d1)*(3*k**3*lam-6*k*(k-lam)+4)
    a24 = -3*c3*lam/(4*k*d1)*(2+3*k*lam)
    b21 = -3*c3*lam/(2*d1)*(3*k*lam-4);         b22 = 3*c3*lam/d1
    d21 = -c3/(2*lam**2)
    a31 = (-9*lam/(4*d2))*(4*c3*(k*a23-b21)+k*c4*(4+k**2)) \
          + ((9*lam**2+1-c2)/(2*d2))*(3*c3*(2*a23-k*b21)+c4*(2+3*k**2))
    a32 = (-9*lam/(4*d2))*(4*c3*(k*a24-b22)+k*c4) \
          - ((9*lam**2+1-c2)/(2*d2))*(3*c3*(k*b22+d21-2*a24)-c4)
    b31 = (3/(8*d2))*(8*lam*(3*c3*(k*b21-2*a23)-c4*(2+3*k**2)) \
          + (9*lam**2+1+2*c2)*(4*c3*(k*a23-b21)+k*c4*(4+k**2)))
    b32 = (1/d2)*(9*lam*(c3*(k*b22+d21-2*a24)+c4) \
          + (3/8)*(9*lam**2+1+2*c2)*(4*c3*(k*a24-b22)+k*c4))
    d31 = (3/(64*lam**2))*(4*c3*a24+c4)
    d32 = (3/(64*lam**2))*(4*c3*(a23-d21)+c4*(4+k**2))
    a1 = -1.5*c3*(2*a21+a23+5*d21) - (3/8)*c4*(12-k**2)
    a2 =  1.5*c3*(a24-2*a22) + (9/8)*c4
    den = 2*lam*(lam*(1+k**2)-2*k)
    s1 = (1.5*c3*(2*a21*(k**2-2)-a23*(k**2+2)-2*k*b21)-(3/8)*c4*(3*k**4-8*k**2+8))/den
    s2 = (1.5*c3*(2*a22*(k**2-2)+a24*(k**2+2)+2*k*b22+5*d21)+(3/8)*c4*(12-k**2))/den
    l1 = a1 + 2*lam**2*s1;  l2 = a2 + 2*lam**2*s2
    return dict(g=g, c2=c2, c3=c3, c4=c4, lam=lam, wz=np.sqrt(c2), k=k, Delta=Delta,
                l1=l1, l2=l2, a21=a21, a22=a22, a23=a23, a24=a24, a31=a31, a32=a32,
                b21=b21, b22=b22, b31=b31, b32=b32, d21=d21, d31=d31, d32=d32)


def richardson_halo_seed(Az, mu, point='L1', northern=True):
    """3rd-order analytic seed on the x-z plane (perpendicular crossing).

    point : 'L1' or 'L2'. Returns (state0, T_guess, (Ax, Az)),
    state0 = [x,0,z,0,vy,0].
    """
    C = _richardson_constants(mu, point)
    Ax2 = -(C['Delta'] + C['l2']*Az**2)/C['l1']     # amplitude constraint
    if Ax2 < 0:
        raise ValueError(f"no real A_x for A_z={Az}")
    Ax = np.sqrt(Ax2); dm = 1.0 if northern else -1.0
    g, lam = C['g'], C['lam']
    x_s = C['a21']*Ax**2 + C['a22']*Az**2 - Ax + (C['a23']*Ax**2 - C['a24']*Az**2) \
          + (C['a31']*Ax**3 - C['a32']*Ax*Az**2)
    z_s = dm*(Az - 2*C['d21']*Ax*Az + C['d32']*Az*Ax**2 - C['d31']*Az**3)
    yd  = C['k']*Ax + 2*(C['b21']*Ax**2 - C['b22']*Az**2) \
          + 3*(C['b31']*Ax**3 - C['b32']*Ax*Az**2)
    Lx = (1 - mu - g) if point == 'L1' else (1 - mu + g)
    seed = np.array([Lx + g*x_s, 0.0, g*z_s, 0.0, g*lam*yd, 0.0])
    T_guess = 2*np.pi/lam
    return seed, T_guess, (Ax, Az)


def halo_corrector(seed, mu, tol=1e-12, itmax=30, verbose=False):
    """x-z symmetric perpendicular-crossing corrector: fix Z0, vary (X0, vy0).

    Returns (state0_corrected, period_T, converged_bool).
    """
    s = seed.copy()
    ev = lambda t, S, mu: S[1]
    ev.terminal = True
    ev.direction = -1
    th = np.nan
    for it in range(itmax):
        aug0 = np.concatenate([s, np.eye(6).flatten()])
        sol = solve_ivp(cr3bp_rhs_with_stm_3d, [0, 10], aug0, args=(mu,), events=ev,
                        method='DOP853', rtol=1e-12, atol=1e-12, max_step=0.05)
        th = sol.t_events[0][0]
        Sf = sol.y_events[0][0]
        sf, Phi = Sf[:6], Sf[6:].reshape(6, 6)
        res = np.hypot(sf[3], sf[5])
        if verbose:
            print(f"  it{it}: T/2={th:.6f}  vx_f={sf[3]:+.3e}  vz_f={sf[5]:+.3e}  |res|={res:.3e}")
        if res < tol:
            return s, 2*th, True
        # accelerations at the crossing (your EOM already returns ax, az in slots 3, 5)
        acc = cr3bp_rhs_3d(0.0, sf, mu)
        xdd, zdd, yd = acc[3], acc[5], sf[4]
        # perpendicular-crossing correction with variable crossing time
        M  = np.array([[Phi[3, 0], Phi[3, 4]],
                       [Phi[5, 0], Phi[5, 4]]])
        M -= (1.0/yd) * np.array([[xdd], [zdd]]) @ np.array([[Phi[1, 0], Phi[1, 4]]])
        d  = np.linalg.solve(M, -np.array([sf[3], sf[5]]))
        s[0] += d[0]; s[4] += d[1]
    return s, 2*th, False


def halo_stability(state0, T, mu):
    """Monodromy eigenvalues + center rotation number (2nd torus frequency).

    Returns (eigenvalues, rotation_number_rad_or_None).
    """
    sol = integrate_with_stm_3d(state0, [0, T], mu)
    M = sol.y[6:, -1].reshape(6, 6)
    ev = np.linalg.eigvals(M)
    center = [e for e in ev if abs(abs(e) - 1) < 1e-3 and abs(e.imag) > 1e-6]
    rot = float(np.angle(center[0])) if center else None
    return ev, rot