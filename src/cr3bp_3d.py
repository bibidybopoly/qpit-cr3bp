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


def halo_corrector(seed, mu, tol=1e-12, itmax=30, verbose=False,
                   t_min=1e-6, t_max=10.0, step_cap=0.1,
                   divergence_factor=1e2, return_info=False):
    """x-z symmetric perpendicular-crossing corrector: fix Z0, vary (X0, vy0).

    Returns (state0_corrected, period_T, converged_bool), or with
    return_info=True a 4th element carrying the failure code and iteration log.

    ---------------------------------------------------------------------
    SILENT-WRONG-ANSWER BUG, FIXED.  The residual here is hypot(vx, vz) at the
    crossing.  Every seed this method accepts has the form [x, 0, z, 0, vy, 0],
    so vx = vz = 0 IDENTICALLY at t = 0.  If the event ever fires at t = 0 the
    residual is exactly zero, the convergence test passes, and the corrector
    returns the seed with T = 0 and an identity monodromy -- reported as
    converged.  Every downstream consumer then sees six unit multipliers and no
    centre pair.

    The old code made that reachable.  With `direction = -1` fixed, an iterate
    whose vy0 has gone negative has y DECREASING at t = 0, which is exactly the
    crossing direction being watched, so the terminal event fires immediately.
    And vy0 does go negative, because the undamped Newton step can leave the
    basin: on a seed 0.002 off the family in z0 the iteration ran
    x0 = 1.044 -> 1.105 -> 0.977 -> 4.210 with vy0 -> -1.296 in three steps,
    the last step having |d| = 3.8.

    Three changes, each addressing a distinct part of that chain:

    1. DIRECTION IS NOW ADAPTIVE: `direction = -sign(vy0)`.  The condition we
       actually want is "the next return to the x-z plane", and at t = 0 the
       trajectory leaves the plane with sign(vy0).  Watching for the opposite
       sign makes the t = 0 crossing STRUCTURALLY IMPOSSIBLE rather than merely
       guarded against.  For every seed with vy0 > 0 this is identical to the
       old `direction = -1`, so nothing that previously worked changes.
    2. THE STEP IS DAMPED to |d| <= step_cap.  Measured over L1 and L2 seeds
       spanning Az = 0.04 to 0.20, the largest step on a run that converges is
       1.44e-2, so the 0.1 default has ~7x headroom and never activates on a
       healthy run.  It only bites when Newton is already leaving the basin.
    3. DIVERGENCE AND DEGENERACY ARE REPORTED, not converted into a return
       value that looks like success: codes 'diverged', 'no_crossing',
       'degenerate_crossing', 'max_iter'.

    The seed shape is now checked too.  The whole method rests on the x-z
    mirror symmetry, which is only valid from a perpendicular crossing; handed
    a general state it would return a confident non-orbit.
    """
    s = np.asarray(seed, dtype=float).copy()
    if max(abs(s[1]), abs(s[3]), abs(s[5])) > 1e-12:
        raise ValueError(
            "halo_corrector requires a perpendicular x-z crossing "
            "[x, 0, z, 0, vy, 0]; got y=%.3e vx=%.3e vz=%.3e. The mirror "
            "symmetry this corrector exploits does not hold otherwise."
            % (s[1], s[3], s[5]))
    if s[4] == 0.0:
        raise ValueError("halo_corrector: vy0 = 0, the trajectory never "
                         "leaves the x-z plane and no crossing exists")

    th, best, log, code = np.nan, np.inf, [], 'max_iter'
    for it in range(itmax):
        def ev(t, S, mu):
            return S[1]
        ev.terminal = True
        ev.direction = -np.sign(s[4])     # excludes t = 0 by construction

        aug0 = np.concatenate([s, np.eye(6).flatten()])
        sol = solve_ivp(cr3bp_rhs_with_stm_3d, [0, t_max], aug0, args=(mu,),
                        events=ev, method='DOP853', rtol=1e-12, atol=1e-12,
                        max_step=0.05)
        if sol.t_events[0].size == 0:
            code = 'no_crossing'
            break
        th = sol.t_events[0][0]
        if th <= t_min:
            code = 'degenerate_crossing'   # unreachable now; kept as a tripwire
            break

        Sf = sol.y_events[0][0]
        sf, Phi = Sf[:6], Sf[6:].reshape(6, 6)
        res = float(np.hypot(sf[3], sf[5]))
        log.append(dict(it=it, th=float(th), res=res, x0=s[0], vy0=s[4]))
        if verbose:
            print(f"  it{it}: T/2={th:.6f}  vx_f={sf[3]:+.3e}  "
                  f"vz_f={sf[5]:+.3e}  |res|={res:.3e}")
        if res < tol:
            code = 'converged'
            break
        if res > divergence_factor * best:
            code = 'diverged'
            break
        best = min(best, res)

        acc = cr3bp_rhs_3d(0.0, sf, mu)
        xdd, zdd, yd = acc[3], acc[5], sf[4]
        M  = np.array([[Phi[3, 0], Phi[3, 4]],
                       [Phi[5, 0], Phi[5, 4]]])
        M -= (1.0/yd) * np.array([[xdd], [zdd]]) @ np.array([[Phi[1, 0], Phi[1, 4]]])
        d  = np.linalg.solve(M, -np.array([sf[3], sf[5]]))
        nrm = np.linalg.norm(d)
        if nrm > step_cap:
            d *= step_cap / nrm
            if verbose:
                print(f"        step damped: |d| {nrm:.3e} -> {step_cap:.3e}")
        s[0] += d[0]; s[4] += d[1]

    converged = (code == 'converged')
    T = 2*th if np.isfinite(th) else np.nan
    if return_info:
        return s, T, converged, dict(code=code, n_iter=len(log), log=log)
    return s, T, converged


def halo_stability(state0, T, mu, circle_tol=1e-4):
    """Floquet analysis of ONE periodic orbit: monodromy spectrum, grouped into
    reciprocal pairs and classified hyperbolic / trivial / center.

    Stateless by design. It classifies a single orbit and returns the full
    labeled spectrum *including eigenvectors*. Following the center pair
    continuously along a family (matching by eigenvector overlap, unwrapping
    theta) is the sweep's job, because only the sweep holds cross-orbit memory.

    Why not just "the complex pair on the circle"? The trivial (1,1) block is
    defective, so ~1e-12 integration error splits it off-axis into e^{+/-i eps}
    with eps ~ 1e-6 -- itself a complex pair on the circle. So we never key on
    imaginary part directly. Instead we use what is certain: the hyperbolic pair
    is real with |mu| far from 1; the trivial pair hugs +1; the center pair is
    what remains, and only *it* sits a finite angle from +1.

    Returns dict:
      'multipliers' : (6,) monodromy eigenvalues (raw Floquet multipliers)
      'pairs'       : list of 3 reciprocal-pair records, each
                        {'type': 'hyperbolic'|'trivial'|'center',
                         'mu'   : representative eigenvalue (Im >= 0 branch),
                         'vec'  : unit eigenvector for 'mu', complex (6,),
                         'mate' : reciprocal partner eigenvalue,
                         'dev'  : | |mu| - 1 |  (distance off the unit circle)}
      'center'      : the pair record tagged 'center', or None if the gate
                      (on the circle AND clear of +1) fails -- e.g. right at the
                      bifurcation where center collides with the trivial fuzz.
      'theta'       : arg(center.mu) in (0, pi], or None. Per-orbit and NOT
                      unwrapped -- the sweep unwraps it along the family.
      'hyp_on_circle' : True if the hyperbolic pair has drifted onto the circle
                      (a stability change, e.g. deep in the NRHO regime), which
                      makes the 3-way split ambiguous on this orbit.
    """
    sol = integrate_with_stm_3d(state0, [0, T], mu)
    M = sol.y[6:, -1].reshape(6, 6)
    vals, vecs = np.linalg.eig(M)

    # --- group into reciprocal pairs: mate of lambda is the eigenvalue nearest
    #     1/lambda (true for all three pair types, since |mu|=1 => 1/mu = conj mu,
    #     and the hyperbolic pair is literally (lambda, 1/lambda)).
    remaining = list(range(6))
    pair_idx = []
    while remaining:
        i = remaining.pop(0)
        inv = 1.0 / vals[i]
        j = min(remaining, key=lambda k: abs(vals[k] - inv))
        remaining.remove(j)
        pair_idx.append((i, j))

    # --- pair records; representative is the member with Im >= 0 (fixes theta
    #     branch to (0, pi] and gives a definite eigenvector to carry/track).
    pairs = []
    for i, j in pair_idx:
        rep, mate = (i, j) if vals[i].imag >= vals[j].imag else (j, i)
        v = vecs[:, rep]
        v = v / np.linalg.norm(v)
        pairs.append({'mu': vals[rep], 'vec': v, 'mate': vals[mate],
                      'dev': abs(abs(vals[rep]) - 1.0)})

    # --- classify. Hyperbolic = pair sitting farthest off the unit circle.
    hyp = max(pairs, key=lambda p: p['dev'])
    hyp['type'] = 'hyperbolic'
    rest = [p for p in pairs if p is not hyp]
    # of the two near-circle pairs, trivial is the one hugging +1.
    trivial = min(rest, key=lambda p: abs(p['mu'] - 1.0))
    trivial['type'] = 'trivial'
    center = [p for p in rest if p is not trivial][0]
    center['type'] = 'center'

    # --- gates. Center is valid only if it is ON the circle and CLEAR of +1
    #     (so it isn't the trivial fuzz). And flag if the "hyperbolic" pair has
    #     actually come onto the circle -- then this orbit's 3-way split is not
    #     trustworthy and the sweep should treat it with care.
    center_ok = center['dev'] < circle_tol and abs(center['mu'] - 1.0) > circle_tol
    hyp_on_circle = hyp['dev'] < circle_tol

    return {'multipliers': vals,
            'pairs': pairs,
            'center': center if center_ok else None,
            'theta': float(np.angle(center['mu'])) if center_ok else None,
            'hyp_on_circle': hyp_on_circle}