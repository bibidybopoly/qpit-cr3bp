"""Certification of the labelling indicators on systems with known answers.

Deliberately independent of the CR3BP code.  The indicator has to be
trustworthy BEFORE it is pointed at the tori, because on the tori there is
no ground truth to check it against -- that is the whole reason the gap
experiment exists.  So it is certified here against (a) analytic signals
and (b) the Chirikov standard map, where regularity is known from K vs the
critical value 0.971635.

Run: pytest tests/test_labeling.py -v   (or python tests/test_labeling.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from labeling import naff_frequency, frequency_drift   # noqa: E402


# =========================================================================
# ANALYTIC SIGNALS
# =========================================================================

def test_naff_recovers_known_frequency():
    """Frequency accuracy must beat the FFT bin by many orders."""
    N, npers = 8000, 200
    dt = 2 * np.pi * npers / N
    t = dt * np.arange(N)
    w_true = 1.0
    f = np.exp(1j * w_true * t) + 0.3 * np.exp(1j * 2.121320344 * t)
    w_est, amp = naff_frequency(f, dt)
    bin_width = 2 * np.pi / (N * dt)
    assert abs(w_est - w_true) < 1e-9, abs(w_est - w_true)
    assert abs(w_est - w_true) < 1e-6 * bin_width
    assert abs(amp - 1.0) < 1e-3          # Hann normalisation unbiased


def test_drift_floor_on_clean_signal():
    """A genuinely quasi-periodic signal must sit near the floor.

    The floor is leakage-limited, not machine epsilon -- which is exactly
    why calibrate_thresholds measures it rather than assuming a value.
    """
    N, dt = 16000, 0.05
    t = dt * np.arange(N)
    f = np.exp(1j * t) + 0.4 * np.exp(1j * 0.6180339887 * t)
    assert frequency_drift(f, dt)['drift_abs'] < 1e-5


def test_drift_is_linear_in_imposed_chirp():
    """Drift must MEASURE frequency migration, not merely be large for it."""
    N, dt = 16000, 0.05
    t = dt * np.arange(N)
    T = N * dt
    for rate in (1e-6, 1e-4):
        f = (np.exp(1j * (t + 0.5 * rate * t ** 2))
             + 0.4 * np.exp(1j * 0.618 * t))
        got = frequency_drift(f, dt)['drift_abs']
        expected = rate * T / 2.0        # half-window centres differ by T/2
        assert abs(got - expected) / expected < 0.01, (rate, got, expected)


# =========================================================================
# STANDARD MAP
# =========================================================================

def _standard_map(x0, y0, K, n):
    """Iterate and accumulate the max Lyapunov exponent via the tangent map."""
    x, y = x0, y0
    xs = np.empty(n)
    xs[0] = x
    v = np.array([1.0, 0.0])
    s = 0.0
    for i in range(1, n):
        c = K * np.cos(2 * np.pi * x)
        dx, dy = v
        dy2 = dy + c * dx
        v = np.array([dx + dy2, dy2])
        nv = float(np.hypot(*v))
        s += np.log(nv)
        v /= nv
        y = y + (K / (2 * np.pi)) * np.sin(2 * np.pi * x)
        x = (x + y) % 1.0
        xs[i] = x
    return xs, s / (n - 1)


def _drift(xs):
    return frequency_drift(np.exp(2j * np.pi * xs), 1.0)['drift_abs']


def test_regular_circles_below_chaotic_sea():
    """Orders of magnitude of separation, not a marginal difference."""
    reg, _ = _standard_map(0.10, 0.31, 0.50, 8192)
    cha, _ = _standard_map(0.10, 0.31, 2.00, 8192)
    d_reg, d_cha = _drift(reg), _drift(cha)
    assert d_reg < 1e-9, d_reg
    assert d_cha > 1e-6, d_cha
    assert d_cha / d_reg > 1e3


def test_drift_and_lyapunov_agree_including_embedded_islands():
    """The decisive test.

    At K=2.0 part of the phase space is chaotic sea and part is regular
    islands.  An indicator judged against the REGION label would look like
    it produced false positives on the island orbits.  Judged against an
    independently computed Lyapunov exponent it agrees.

    This is the gap experiment in miniature on a system where the answer is
    known: sampling from a region and inheriting the region's label is what
    breaks, and that is precisely why label definition B is behavioural.
    """
    rng = np.random.default_rng(1)
    drifts, mles = [], []
    for _ in range(50):
        xs, mle = _standard_map(rng.uniform(0, 1), rng.uniform(0.10, 0.45),
                                2.0, 8192)
        drifts.append(_drift(xs))
        mles.append(mle)
    drifts, mles = np.array(drifts), np.array(mles)

    by_drift = drifts > 1.84e-8      # regular-p99 cutoff, calibrated at K=0.5
    by_mle = mles > 0.02
    assert (by_drift == by_mle).sum() == 50, (by_drift != by_mle).sum()

    # and the two groups are genuinely distinct objects, not a marginal split
    assert mles[~by_drift].max() < 0.01
    assert mles[by_drift].min() > 0.15


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
