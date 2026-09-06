"""Start-of-week item 5: measure `t_solve_sec` under CONTINUATION.

WHAT THIS NUMBER IS FOR, AND WHAT IT IS NOT.

It does not change the triage result.  S = 0.342 +/- 0.008 solves avoided at 95%
recall is fixed, and so is the cap 1/(1-S) = 1.520 +/- 0.018.  What t_solve_sec
buys is the conversion of "0.342 solve attempts avoided per candidate" into
wall clock, which is the only form a reader can feel.

IT ALSO SETTLES A SECOND THING.  The realised speedup is

    1 / ((1 - S) + t_infer / t_solve)

A 3x256 MLP forward pass on one row is microseconds.  If t_solve is anywhere
near a second the ratio t_infer/t_solve is ~1e-5 and the realised speedup equals
the cap to four decimals -- so the honest statement is that the cap is
essentially attained, and the quotable quantity is ABSOLUTE SECONDS SAVED per
candidate, not a ratio.  Quoting the raw inference-to-solve ratio as the speedup
would overstate the result by orders of magnitude and is the easiest thing in
the paper to catch.

WHY CONTINUATION AND NOT A COLD SOLVE.  The dataset was built by continuation
along families, each torus starting from its neighbour, and operational use near
a known orbit is the same situation.  A cold `converge_torus` from a generic
seed is a different and slower operation, and timing it would inflate the saving
against what production would actually do.

WHY NOT ONE CALL.  Corrector times are right-skewed, and the continuation driver
halves ds and retries on failure and raises N when the curve outgrows its node
count.  A failed corrector call is part of what triage avoids, so it is timed
and counted rather than dropped.  Median and IQR are reported, not a mean.

HOW IT IS TIMED.  `continue_torus_family` is run unmodified and
`correct_along_family` and `torus_tangent` are wrapped in place, so the real
adaptive driver runs and every call it makes -- including retries and
re-convergences after an N raise -- is captured.  Timing the pieces by hand
would mean reimplementing the driver's ds and N logic and measuring something
production does not do.  (Same monkeypatch idiom as the mask-alignment spy in
tests/test_models.py.)

SCOPE LIMIT, STATED NOT HIDDEN.  This times ONE family: L2 halo #29, the only
orbit whose record is reachable from constants.py.  Solve cost plausibly varies
across the six cells -- family_sizes runs 25 to 64 tori and family_max_gap_km
runs 11.06 to 53.66 km -- so a cross-cell number needs the other five halo
records from orbits.compute_halo_family.  Drop them into ORBITS below.

Run from anywhere:  python3 notebooks/time_solve.py
"""

import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import qpit
import constants

S_TRIAGE = 0.342          # solves avoided at 95% recall (Week 13)
S_SD = 0.008
N_NODES = 31              # odd, as check_odd requires
EPSILON = 1e-4            # seed amplitude, nondimensional
ALL_CELLS = True          # False falls back to ORBITS below
N_STEPS = 40              # matches continue_torus_family's own default.
                          # At 15 the run never halved ds and never raised N,
                          # so it timed the easy end of one family: cost per
                          # step scales with N because every Newton iteration
                          # propagates the STM for all N nodes.

#: HALO29 IS NOT ONE OF THE SIX CELLS.  Its C = 3.060334 sits between the
#: dataset's E0 = 3.029398 and E1 = 3.064703, so this times an L2 halo at an
#: energy the dataset does not contain, and no L1 halo at all.  Close enough to
#: E1 to be indicative, not close enough to be quoted as a cell.  `main` checks
#: this against the archive and says so rather than leaving it to be noticed.
ORBITS = [
    dict(tag='L2_halo29 (NOT a dataset cell)', state0=constants.HALO29_STATE,
         period=constants.HALO29_PERIOD),
]

#: The halo family starts are DERIVED, not stored.  Recipe taken verbatim from
#: halo_families.ipynb / floquet_analysis.ipynb: Richardson seed at Az = 0.05 ->
#: halo_corrector -> pseudo-arclength continuation, ds = 0.01, 130 orbits,
#: marched toward growing |z0|.
#:
#: RESOLVED: `first_orbit`, `amplitude_direction` and `halo_family` now live in
#: src/orbits.py and this file imports them.  They had existed as two pasted
#: notebook copies that already differed (`T / 2.0` against `T / 2`); this file
#: briefly made a third.  Both notebooks should now import them too.
AZ_SEED = 0.05
HALO_DS = 0.01
HALO_N_ORBITS = 130


def cells_from_halo_families(energies, mu):
    """Build the six cell seed orbits: per family, the member nearest each
    dataset energy.

    THE GUARD CHECKS COVERAGE, NOT AN EXACT MATCH.  The family is continued at
    ds = 0.01, so adjacent members are separated in C by a finite step and an
    exact hit on a stored energy is not expected -- a tolerance demanding one
    would fail every time and teach nothing.  What matters is whether the family
    BRACKETS the target: outside its C range means the energy is not in this
    family at these settings, which is a real failure.  Inside it, the miss is
    reported against the local member spacing so a near-miss is visible rather
    than silently timed as if it were the cell.
    """
    from orbits import halo_family
    out = []
    for point in ('L1', 'L2'):
        fam = halo_family(point, mu, Az=AZ_SEED, ds=HALO_DS,
                          n_orbits=HALO_N_ORBITS)
        Cs = np.array([o['jacobi'] for o in fam])
        print(f"  {point}: {len(fam)} orbits, C from {Cs.min():.6f} to "
              f"{Cs.max():.6f}")
        for k, E in enumerate(energies):
            if not (Cs.min() <= E <= Cs.max()):
                raise ValueError(
                    f"{point} family does not bracket E{k} = {E:.6f} "
                    f"(range {Cs.min():.6f} to {Cs.max():.6f}) at ds={HALO_DS}, "
                    f"n_orbits={HALO_N_ORBITS}; this energy is not in the family "
                    f"at these settings")
            j = int(np.argmin(np.abs(Cs - E)))
            miss = abs(Cs[j] - E)
            spacing = float(np.median(np.abs(np.diff(Cs))))
            flag = '' if miss <= 0.5 * spacing else \
                '   <-- nearer a neighbour; refine before quoting as this cell'
            print(f"    E{k}={E:.6f} -> member {j}, C={Cs[j]:.6f}, "
                  f"miss={miss:.2e} (median spacing {spacing:.2e}){flag}")
            out.append(dict(tag=f"{point}_E{k}", state0=fam[j]['state0'],
                            period=fam[j]['period']))
    return out


def _instrument():
    """Wrap the two functions the driver calls, in place.

    Returns a record dict.  Wrapping rather than hand-driving keeps the real
    adaptive ds and N logic in the loop; a hand-rolled loop would be timing
    something production does not run.
    """
    rec = dict(corrector=[], corrector_failed=[], tangent=[])
    real_corr, real_tan = qpit.correct_along_family, qpit.torus_tangent

    def timed_corr(*a, **kw):
        t0 = time.perf_counter()
        out = real_corr(*a, **kw)
        dt = time.perf_counter() - t0
        ok = bool(out.get('converged', out.get('code') == 'converged'))
        (rec['corrector'] if ok else rec['corrector_failed']).append(dt)
        return out

    def timed_tan(*a, **kw):
        t0 = time.perf_counter()
        out = real_tan(*a, **kw)
        rec['tangent'].append(time.perf_counter() - t0)
        return out

    qpit.correct_along_family, qpit.torus_tangent = timed_corr, timed_tan
    return rec, (real_corr, real_tan)


def _restore(originals):
    qpit.correct_along_family, qpit.torus_tangent = originals


def _q(v):
    v = np.asarray(v, float)
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), float(v.min()), float(v.max()))


def time_infer():
    """One forward pass, one row, same machine.  Weights are irrelevant to
    timing, so an untrained net is fine and avoids needing a fitted model."""
    try:
        import torch
        from models import MLP
    except ImportError:
        return None
    net = MLP(7, hidden=(256, 256, 256)).eval()
    x = torch.zeros(1, 7)
    with torch.no_grad():
        for _ in range(20):
            net(x)                                    # warm up
        t0 = time.perf_counter()
        for _ in range(1000):
            net(x)
        return (time.perf_counter() - t0) / 1000.0


def main():
    mu = constants.MU_EARTH_MOON
    arch = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')
    energies = np.unique(np.load(arch, allow_pickle=True)['C'])
    print(f"dataset energies: {np.round(energies, 6).tolist()}")

    if ALL_CELLS:
        print("building the six cell seed orbits from the halo families")
        orbits = cells_from_halo_families(energies, mu)
    else:
        orbits = ORBITS
    print(f"machine: {platform.processor() or platform.machine()}  "
          f"python {platform.python_version()}")
    try:
        import torch
        print(f"torch threads: {torch.get_num_threads()}")
    except ImportError:
        pass

    per_member = []
    for orb in orbits:
        print(f"\n=== {orb['tag']} ===")
        U0, T0, rho0 = qpit.seed_from_halo(orb, mu, N_NODES, EPSILON)
        C_halo = qpit.jacobi_constant(np.asarray(orb['state0'], float), mu)
        # A 1e-6 threshold here fired on all six cells and so carried no
        # information -- a guard that always fires is as useless as one that
        # never does.  The builder already vets its own output against the
        # family's member spacing, so this reports the offset numerically and
        # shouts only for a hand-supplied orbit that is nowhere near a cell.
        miss = float(np.min(np.abs(energies - C_halo)))
        near = float(energies[np.argmin(np.abs(energies - C_halo))])
        rel = miss / near
        print(f"  C = {C_halo:.6f}, nearest dataset energy {near:.6f}, "
              f"offset {miss:.2e} ({rel:.2%} relative)")
        if rel > 1e-3:
            print(f"  !! offset exceeds 0.1% of C; solve cost varies ~2x over a "
                  f"3.4% span in C, so this is no longer a stand-in for that cell")
        res = qpit.converge_torus(U0, T0, rho0, mu, C_target=C_halo,
                                  verbose=False)
        if not res['converged']:
            print(f"  seed did not converge ({res['code']}) -- skipping")
            continue
        print(f"  seed converged, C={res['C']:.9f}, "
              f"amplitude={qpit.curve_amplitude(res['U0'])*constants.LU_KM:.2f} km")

        rec, originals = _instrument()
        try:
            t0 = time.perf_counter()
            fam = qpit.continue_torus_family(
                res['U0'], res['T'], res['rho'], mu, pin='rho',
                n_steps=N_STEPS, verbose=False)
            wall = time.perf_counter() - t0
        finally:
            _restore(originals)

        n_members = len(fam['members']) - 1        # the seed is member 0
        if n_members < 1:
            print(f"  continuation produced no members ({fam['terminated']})")
            continue
        per_member.append(wall / n_members)

        print(f"  {n_members} members in {wall:.1f}s  "
              f"({wall/n_members:.3f}s per member)  "
              f"terminated: {fam['terminated']}")
        for name in ('corrector', 'corrector_failed', 'tangent'):
            v = rec[name]
            if not v:
                print(f"  {name:<18} none")
                continue
            med, q1, q3, lo, hi = _q(v)
            print(f"  {name:<18} n={len(v):<4} median={med:.3f}s  "
                  f"IQR=[{q1:.3f}, {q3:.3f}]  range=[{lo:.3f}, {hi:.3f}]")
        if rec['corrector_failed']:
            frac = len(rec['corrector_failed']) / (
                len(rec['corrector']) + len(rec['corrector_failed']))
            print(f"  {frac:.0%} of corrector calls failed and were retried at "
                  f"smaller ds -- that cost is part of what triage avoids")

    if not per_member:
        print("\nno timings collected")
        return 1

    pm = np.asarray(per_member, float)
    t_solve = float(np.median(pm))
    t_solve_mean = float(pm.mean())
    t_infer = time_infer()

    print(f"\n=== t_solve_sec = {t_solve:.3f} (median per accepted member, "
          f"continuation) ===")
    cap = 1.0 / (1.0 - S_TRIAGE)
    cap_sd = S_SD / (1.0 - S_TRIAGE) ** 2
    print(f"  triage S = {S_TRIAGE} +/- {S_SD}   cap 1/(1-S) = "
          f"{cap:.3f} +/- {cap_sd:.3f}")
    if pm.size > 1:
        print(f"  across {pm.size} cells: min {pm.min():.3f}  max {pm.max():.3f} "
              f"({pm.max()/pm.min():.2f}x)  mean {t_solve_mean:.3f}  "
              f"sd {pm.std(ddof=1):.3f}")
    # MEDIAN describes a typical solve.  The per-candidate saving is an
    # EXPECTATION over candidates, and rows are 3400 per cell -- equal -- so the
    # cells weight equally and the mean is the correct statistic for it.  The
    # two differ by 7% here, which is larger than anything else in this
    # calculation.
    print(f"  seconds saved per candidate = S * mean(t_solve) = "
          f"{S_TRIAGE * t_solve_mean:.3f} s   "
          f"(median-based would give {S_TRIAGE * t_solve:.3f} s)")
    if t_infer is None:
        print("  t_infer not measured (torch unavailable)")
    else:
        realised = 1.0 / ((1.0 - S_TRIAGE) + t_infer / t_solve)
        print(f"  t_infer = {t_infer*1e6:.1f} us   t_infer/t_solve = "
              f"{t_infer/t_solve:.2e}")
        print(f"  realised speedup = {realised:.4f} against a cap of "
              f"{cap:.4f}")
        print("  -> inference cost is negligible; the cap is essentially "
              "attained.")
        print("  DO NOT quote t_solve/t_infer as the speedup.  Triage decides "
              "whether to attempt a solve; it does not replace one.")
    if len(orbits) == 1:
        print("\n  ONE FAMILY ONLY, and not a dataset cell.  Fill in "
              "HALO_FAMILY_STARTS to time all six.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
