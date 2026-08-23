"""Densify a torus family so the labelling surface has a bounded error.

WHY DENSIFY AT ALL.  The regression label is distance to the nearest torus.
Computing it by Newton per sample caps the dynamic range at the probe's
faithful regime (34.2 m on L2 halo #29, measured) and costs ~2.4 s per
sample.  Computing it geometrically against a known family has neither
limit -- but only if the family is treated as a CONTINUOUS SURFACE.

THE ERROR BOUND, AND THE ONE IT IS NOT.  Two different quantities:

  (a) distance from a point to the nearest member of a FINITE set of
      curves.  A point midway between members spaced g apart sits ~g/2 from
      both, while its true nearest torus is essentially at zero distance --
      the gap experiment found tori at 80/80 candidates.  So enumeration
      overestimates by g/2, and 10 cm accuracy would need 20 cm spacing,
      i.e. thousands of members.  Not viable.

  (b) distance from a point to the INTERPOLATED SURFACE
      S = {u_alpha(theta)}.  Here the error is how far the interpolant sits
      from a true invariant curve, which the gap experiment measured
      directly: displacement = c * g^2 with a fitted exponent of 2.003 and
      c = 2.6e-3 km / (62.5 km)^2 = 6.656e-7 /km.

Label against (b), not (a).  Then

    g = 12.3 km  ->  error 1e-4 km = 10 cm
    g = 10.0 km  ->  error 6.7e-5 km = 6.7 cm
    g =  5.0 km  ->  error 1.7e-5 km = 1.7 cm

and famC's 17 -> 515 km span needs ~50 members rather than ~62000.

WHAT THIS DOES NOT COVER.  The bound is for the centre-manifold family at
ONE Jacobi constant.  Points off that level set are a different question,
and a multi-energy dataset needs one densified family per energy.
"""

from __future__ import annotations

import os

import numpy as np

from qpit import curve_amplitude
from torus_probe import probe_gap

from constants import LU                     # re-exported: see constants.py

# Measured in nb10_gap_experiment: midpoint displacement vs gap, exponent
# 2.003, 2.6e-3 km at a 62.5 km gap.  Re-measure if the family changes.
GAP_COEFF_PER_KM = 2.6e-3 / 62.5 ** 2


def interp_error_km(gap_km):
    """Bound on the label error from treating the family as interpolable."""
    return GAP_COEFF_PER_KM * np.asarray(gap_km, float) ** 2


def required_gap_km(target_error_km):
    """Amplitude spacing needed for a given label accuracy."""
    return float(np.sqrt(target_error_km / GAP_COEFF_PER_KM))


def _as_rec(m):
    return dict(U0=np.asarray(m['U0'], float), T=float(m['T']),
                rho=float(m['rho']), C=float(m['C']),
                amplitude=float(m['amplitude']))


def densify_once(members, mu, target_gap_km, max_new=None, verbose=True, **kw):
    """One densification pass: bisect every gap wider than target_gap_km.

    Bisection rather than even subdivision because the interpolation error
    falls as g^2 -- halving a gap cuts its error by 4, so the widest gaps
    dominate the bound and are worth attacking first.  A pass that fails to
    converge on a gap leaves it alone; the gap survives into the reported
    bound rather than being silently dropped.

    Returns (members_sorted_by_amplitude, report).
    """
    members = sorted((_as_rec(m) for m in members),
                     key=lambda m: m['amplitude'])
    gaps = [( (members[i+1]['amplitude'] - members[i]['amplitude']) * LU, i)
            for i in range(len(members) - 1)]
    wide = sorted([g for g in gaps if g[0] > target_gap_km], reverse=True)
    if max_new is not None:
        wide = wide[:max_new]

    added, failed = [], []
    for gap_km, i in wide:
        p = probe_gap(members[i], members[i+1], mu, [0.5], **kw)[0]
        if p['converged']:
            added.append(_as_rec(p['result']))
            if verbose:
                print(f"    gap {gap_km:7.2f} km at A={members[i]['amplitude']*LU:7.2f}"
                      f" -> new member A={p['result']['amplitude']*LU:7.2f} km",
                      flush=True)
        else:
            failed.append((gap_km, i, p['code']))
            if verbose:
                print(f"    gap {gap_km:7.2f} km FAILED ({p['code']})", flush=True)

    out = sorted(members + added, key=lambda m: m['amplitude'])
    g = np.diff([m['amplitude'] for m in out]) * LU
    return out, dict(n_before=len(members), n_added=len(added),
                     n_failed=len(failed), failed=failed,
                     max_gap_km=float(g.max()) if g.size else 0.0,
                     error_km=float(interp_error_km(g.max())) if g.size else 0.0)


def densify(members, mu, target_gap_km, max_passes=8, checkpoint=None,
            verbose=True, **kw):
    """Repeated bisection until every gap is under target, or passes run out.

    Checkpointed: N is NOT constant across members (continue_torus_family
    adapts Fourier resolution), so the archive stores a ragged object array.
    A plain np.array() on the curve list raises.
    """
    if checkpoint and os.path.exists(checkpoint):
        members = load_family(checkpoint)
        if verbose:
            print(f"  resumed {len(members)} members", flush=True)

    for k in range(max_passes):
        members, rep = densify_once(members, mu, target_gap_km,
                                    verbose=verbose, **kw)
        if verbose:
            print(f"  pass {k}: {rep['n_before']} -> {len(members)} members, "
                  f"max gap {rep['max_gap_km']:.2f} km, "
                  f"error bound {rep['error_km']*1000:.1f} m", flush=True)
        if checkpoint:
            save_family(checkpoint, members, mu)
        if rep['n_added'] == 0:
            break
    return members, rep


def save_family(path, members, mu, **extra):
    U = np.empty(len(members), dtype=object)
    for i, m in enumerate(members):
        U[i] = np.asarray(m['U0'], float)
    np.savez(path, U=U,
             T=np.array([m['T'] for m in members]),
             rho=np.array([m['rho'] for m in members]),
             C=np.array([m['C'] for m in members]),
             A=np.array([m['amplitude'] for m in members]),
             N=np.array([np.asarray(m['U0']).shape[0] for m in members]),
             mu=mu, **extra)


def load_family(path):
    z = np.load(path, allow_pickle=True)
    return [dict(U0=np.asarray(z['U'][i], float), T=float(z['T'][i]),
                 rho=float(z['rho'][i]), C=float(z['C'][i]),
                 amplitude=float(z['A'][i])) for i in range(len(z['A']))]
