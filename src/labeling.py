"""Behavioral quasi-periodicity labelling for QPIT dataset generation.

LABEL DEFINITION B.  A state is labelled quasi-periodic by what its
trajectory DOES, not by which region of phase space it was drawn from.
This module supplies the indicators; thresholds are CALIBRATED from a
positive control, never hard-coded (see calibrate_thresholds).

Two indicators, chosen because each covers the other's blind spot:

  1. NAFF frequency drift.  Laskar's diffusion index: extract the dominant
     frequency from the first and second halves of a time series and take
     the difference.  Detects bounded chaos, which no escape test can see.
     Blind to an orbit that leaves the region before the window ends.

  2. Deviation growth.  Max distance from a reference point, late half over
     early half.  Detects hyperbolic escape for free from a trajectory
     already integrated.  Blind to bounded chaos.

PRIMARY SCORE IS THE STROBOSCOPIC ONE.  Sampled at multiples of T the
motion on a torus is theta_n = theta_0 + n*rho, so any observable carries
tones only at harmonics of rho, with coefficients decaying at the rate
qpit.fourier_tail measures.  That is a near-single-tone signal, so its
leakage floor is lower than the dense signal's, and the frequency it
returns is rho itself -- the invariant that labels the torus.  The dense
canonical-pair drift is retained as an INDEPENDENT cross-check.

All scores are returned continuous.  The binary label is derived last and
separately, so Week 12 can re-threshold without regenerating the dataset.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from cr3bp_3d import cr3bp_rhs_3d
from qpit import dft_matrix, wavenumbers

# Matches qpit._INTEGRATOR exactly.  Duplicated rather than imported because
# a private name is not an API; if qpit's tolerances change this must be
# updated deliberately, and the mismatch would show up as a raised noise
# floor in the positive control rather than silently.
_INTEGRATOR = dict(method='DOP853', rtol=1e-12, atol=1e-12)


# =========================================================================
# SECTION 1 -- NAFF
# =========================================================================

def _hann(n):
    """Laskar's Hann window on the centred grid, w = 1 + cos(2*pi*t/T).

    Integrates to T over the window, so (1/T)*sum(f*w)*dt returns the
    amplitude of a pure tone unbiased -- checked in test_labeling.
    """
    j = np.arange(n)
    return 1.0 + np.cos(2.0 * np.pi * (j / n - 0.5))


def naff_frequency(f, dt, n_bins_search=1):
    """Dominant frequency of a complex signal, by Laskar's NAFF.

    Maximises |phi(w)| = |(1/T) integral f(t) w(t) e^{-i w t} dt| over
    CONTINUOUS w, seeded by the FFT peak.  The Hann window buys frequency
    accuracy O(T^-4) against the O(T^-1) of a bare FFT bin -- the entire
    reason the drift of a regular orbit sits near 1e-12 rather than near
    the bin width, and therefore the entire reason the indicator separates.

    Returns (omega, amplitude).
    """
    f = np.asarray(f, dtype=complex)
    n = f.size
    if n < 16:
        raise ValueError(f"need at least 16 samples for NAFF, got {n}")
    T = n * dt
    t = -T / 2.0 + dt * np.arange(n)
    fw = f * _hann(n)

    spec = np.fft.fft(fw)
    freqs = 2.0 * np.pi * np.fft.fftfreq(n, dt)
    w0 = float(freqs[int(np.argmax(np.abs(spec)))])
    dw = 2.0 * np.pi / T                       # one FFT bin

    def neg_amp(om):
        return -np.abs(np.mean(fw * np.exp(-1j * om * t)))

    res = minimize_scalar(
        neg_amp,
        bounds=(w0 - n_bins_search * dw, w0 + n_bins_search * dw),
        method='bounded',
        options={'xatol': 1e-14 * max(1.0, abs(w0)) + 1e-16},
    )
    return float(res.x), float(-res.fun)


def frequency_drift(f, dt):
    """Laskar diffusion index |w1 - w2| between the two halves of f.

    The halves are windowed independently, so this is NOT sensitive to the
    signal's overall envelope -- only to genuine migration of the frequency.
    Verified against an imposed linear chirp in test_labeling: measured
    drift reproduces rate*T/2 to better than 0.1%.
    """
    f = np.asarray(f, dtype=complex)
    h = f.size // 2
    w1, a1 = naff_frequency(f[:h], dt)
    w2, a2 = naff_frequency(f[h:2 * h], dt)
    d = abs(w1 - w2)
    return dict(w1=w1, w2=w2, amp1=a1, amp2=a2,
                drift_abs=d, drift_rel=d / max(abs(w1), abs(w2), 1e-300))


# =========================================================================
# SECTION 2 -- PROPAGATION WITH DENSE AND STROBOSCOPIC OUTPUT
# =========================================================================

def propagate_sampled(x0, T, mu, n_periods, n_per_period=64):
    """One integration yielding BOTH sample sets.

    qpit.propagate takes no t_eval, so this exists rather than calling it
    n times.  Same integrator and tolerances.

    Returns (t_dense, X_dense, t_strobe, X_strobe).  The stroboscopic grid
    is the EXACT multiples of T -- placed in t_eval directly, not picked out
    of the dense grid by nearest-neighbour, which would inject a timing
    error of dt/2 into rho and swamp the drift signal.
    """
    x0 = np.asarray(x0, float)
    t_strobe = T * np.arange(n_periods + 1)
    t_dense = np.linspace(0.0, T * n_periods, n_periods * n_per_period + 1)
    t_eval = np.unique(np.concatenate([t_dense, t_strobe]))

    sol = solve_ivp(cr3bp_rhs_3d, [0.0, T * n_periods], x0, args=(mu,),
                    t_eval=t_eval, **_INTEGRATOR)
    if not sol.success:
        return None
    # exact membership, not tolerance matching: both grids came from t_eval
    idx_s = np.searchsorted(sol.t, t_strobe)
    idx_d = np.searchsorted(sol.t, t_dense)
    return sol.t[idx_d], sol.y[:, idx_d].T, sol.t[idx_s], sol.y[:, idx_s].T


# =========================================================================
# SECTION 3 -- OBSERVABLES
# =========================================================================

def curve_plane(U0):
    """Orthonormal (e1, e2) spanning the k=1 POSITION mode of a curve.

    The stroboscopic signal is a projection onto this plane, so every sample
    in a gap shares one basis and their scores are directly comparable.
    Positions only -- mixing position and velocity components would weight
    the projection by an arbitrary choice of units.

    Also returns 'conditioning', the sine of the angle between the raw
    Re/Im directions.  Near zero means the k=1 mode is nearly linear in
    position and the plane is ill-defined; the caller should fall back.
    """
    N = U0.shape[0]
    Ck = dft_matrix(N) @ U0
    k1 = int(np.where(wavenumbers(N) == 1)[0][0])
    c1 = Ck[k1, :3]
    a, b = np.real(c1).copy(), np.imag(c1).copy()
    na = np.linalg.norm(a)
    if na < 1e-300:
        raise ValueError("k=1 position mode vanishes; curve has no plane")
    e1 = a / na
    b_perp = b - (b @ e1) * e1
    nb = np.linalg.norm(b_perp)
    cond = float(nb / max(np.linalg.norm(b), 1e-300))
    if nb < 1e-14:
        raise ValueError(f"k=1 mode nearly linear (conditioning {cond:.2e})")
    return e1, b_perp / nb, cond


def stroboscopic_signal(X_strobe, centre, e1, e2):
    """Complex signal from stroboscopic iterates: dominant tone at rho.

    Centre is subtracted BEFORE projection.  Without it the constant offset
    (x ~ 1.15 near L2) is the largest Fourier component and the FFT peak
    lands on zero frequency.
    """
    w = X_strobe[:, :3] - np.asarray(centre, float)[:3]
    return (w @ e1) + 1j * (w @ e2)


def canonical_signals(X_dense):
    """Three complex signals from the canonical (q, p) pairs.

    Rotating-frame momenta: px = vx - y, py = vy + x, pz = vz.

    Each component is mean-removed and scaled to unit variance before being
    combined.  A q + i*p with mismatched amplitudes is a MIXTURE of e^{+iwt}
    and e^{-iwt} rather than a clean tone, which raises the leakage floor;
    equalising the variances makes the signal as near-circular as the orbit
    allows.  This is a linear change of the observable and cannot change
    which frequencies are present, only their relative weight.
    """
    x, y, z, vx, vy, vz = X_dense.T
    pairs = [(x, vx - y), (y, vy + x), (z, vz)]
    out = []
    for q, p in pairs:
        q = q - q.mean()
        p = p - p.mean()
        sq, sp = q.std(), p.std()
        if sq < 1e-14 or sp < 1e-14:      # e.g. z == 0 on a planar orbit
            out.append(None)
            continue
        out.append(q / sq + 1j * (p / sp))
    return out


def deviation_growth(X_strobe, ref):
    """max||x - ref|| over the late half divided by the early half.

    A torus point revisits the same curve, so the ratio is ~1.  A point
    displaced along the unstable bundle grows by R per stroboscopic period,
    so the ratio is astronomically large well before the window ends.
    Returns (ratio, max_dev_overall).
    """
    d = np.linalg.norm(X_strobe[:, :3] - np.asarray(ref, float)[:3], axis=1)
    h = d.size // 2
    early = max(d[:h].max(), 1e-300)
    return float(d[h:].max() / early), float(d.max())


# =========================================================================
# SECTION 4 -- DIAGNOSTICS AND LABEL
# =========================================================================

def max_usable_periods(R_u, eps=1e-12, safety=0.5):
    """Stroboscopic periods a trajectory can be trusted to SHADOW the torus.

    MEASURED, NOT ASSUMED.  These tori are partially hyperbolic: the
    unstable bundle amplifies any displacement by R_u per stroboscopic
    period, and integrator error eps is such a displacement.  It reaches
    O(1) after

        n_max = log(1/eps) / log(R_u)

    and past that the trajectory is no longer near the torus it started on,
    so ANY long-time diagnostic is measuring the escape, not the torus.

    Measured on the L2 halo family at C ~ 3.15: R_u = 1212, giving
    n_max = 3.9 periods at rtol 1e-12.  Torus R_u reproduces the halo's
    |lambda_u| to four decimals, so this can be read off the halo before any
    torus is computed.  Confirmed empirically: at n_periods=60 all three
    experimental arms escaped, INCLUDING the positive control.

    The consequence is a hard ceiling on the NAFF window, hence on the
    frequency resolution, hence on the drift floor.  It is not a compute
    budget question and cannot be bought off with a longer run.
    """
    R_u = float(R_u)
    if R_u <= 1.0:
        return np.inf                    # stable torus: no shadowing limit
    return safety * np.log(1.0 / eps) / np.log(R_u)


def qp_diagnostics(x0, T, mu, centre, e1, e2, n_periods=200,
                   n_per_period=64, escape_radius=1.0, R_u=None):
    """Full continuous score set for one initial condition.

    Never returns a binary label -- see qp_label.  Every score is stored so
    the dataset can be re-thresholded without regeneration.

    Pass R_u to have the window checked against the shadowing bound.  The
    check WARNS rather than raises, and records n_max, so an over-long run
    is visible in the dataset instead of silently producing drift values
    that describe escape trajectories.
    """
    out = dict(ok=False, reason='', drift_strobe=np.nan, rho_est=np.nan,
               drift_dense=np.nan, dev_ratio=np.nan, max_dev=np.nan,
               n_periods=n_periods, n_per_period=n_per_period,
               n_max=np.inf, window_ok=True)
    if R_u is not None:
        out['n_max'] = max_usable_periods(R_u)
        out['window_ok'] = bool(n_periods <= out['n_max'])

    res = propagate_sampled(x0, T, mu, n_periods, n_per_period)
    if res is None:
        out['reason'] = 'integration_failed'
        return out
    t_d, X_d, t_s, X_s = res

    if not np.all(np.isfinite(X_s)):
        out['reason'] = 'nonfinite'
        return out

    out['dev_ratio'], out['max_dev'] = deviation_growth(X_s, centre)
    if out['max_dev'] > escape_radius:
        out['reason'] = 'escaped'
        out['ok'] = True          # a real measurement, not a failure
        return out

    f_s = stroboscopic_signal(X_s, centre, e1, e2)
    ds = frequency_drift(f_s, 1.0)          # dt = 1 stroboscopic period
    out['drift_strobe'] = ds['drift_abs']
    out['rho_est'] = float(np.mod(ds['w1'], 2 * np.pi))

    dense = [s for s in canonical_signals(X_d) if s is not None]
    dt = float(t_d[1] - t_d[0])
    # MAX over the three pairs, not mean: if ANY canonical coordinate shows
    # frequency migration the orbit is not quasi-periodic.  Conservative,
    # and it removes the need to guess which pair carries the centre mode.
    out['drift_dense'] = float(max(frequency_drift(s, dt)['drift_abs']
                                   for s in dense))
    out['ok'] = True
    return out


def calibrate_thresholds(control_diags, quantile=99.0, margin=10.0):
    """Set cutoffs from the POSITIVE control, not by fiat.

    The floor is not machine epsilon -- it is set by spectral leakage from
    the harmonics of rho and by integrator error, both of which depend on
    n_periods and on rtol.  Measuring it on states known to lie on tori is
    the only defensible way to place the cutoff, and it must be re-measured
    whenever the window length or tolerances change.
    """
    def q(key):
        v = np.array([d[key] for d in control_diags
                      if d['ok'] and np.isfinite(d[key])])
        if v.size == 0:
            raise ValueError(f"no usable control values for '{key}'")
        return float(np.percentile(v, quantile) * margin)
    return dict(drift_strobe=q('drift_strobe'), drift_dense=q('drift_dense'),
                dev_ratio=3.0, n_control=len(control_diags),
                quantile=quantile, margin=margin)


def qp_label(diag, thr):
    """Binary label from continuous scores.  Conjunctive: all must pass."""
    if not diag['ok']:
        return None                                   # unmeasured, not negative
    if diag['reason'] == 'escaped':
        return False
    return bool(diag['drift_strobe'] <= thr['drift_strobe']
                and diag['drift_dense'] <= thr['drift_dense']
                and diag['dev_ratio'] <= thr['dev_ratio'])
