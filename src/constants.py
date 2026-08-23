"""The single definition of every physical constant in this project.

WHY THIS EXISTS.  These numbers had 23 literal homes between them: FLOOR_KM in
8 files, LU in 10 (plus `LU_KM` in features.py as an 11th spelling of the same
number), mu in 3, the L2 halo #29 state in 4.  All of them agreed except mu,
where `features.py` carried `0.012150585609624` -- 23 ULP below the value in
`cr3bp.py` -- which shifted a computed Jacobi constant by ~4e-16 and made an
exact feature-C vs solver-C comparison impossible for no reason anyone would
have found.  That is Risk 20's shape, and one copy per file is how it recurs.

NOTHING IS RE-EXPORTED AWAY.  Every module that used to define one of these
now imports it from here and keeps exposing it under its old name, so
`from cr3bp import MU_EARTH_MOON` and `from sampling import FLOOR_KM` still
work exactly as before.  No notebook, script or test call site had to change.

This module imports nothing from the project, so it can never be part of an
import cycle.
"""

from __future__ import annotations

# =========================================================================
# CR3BP system parameters
# =========================================================================

#: Earth-Moon mass parameter, mu = m2 / (m1 + m2).
#: Re-exported by cr3bp, cr3bp_3d and features.  `features.py` keeps its own
#: `jacobi_constant` implementation deliberately independent of `cr3bp_3d`
#: (see its module docstring) -- that independence is about the CONVENTION,
#: not about holding a different number, so it takes mu from here.
MU_EARTH_MOON: float = 0.01215058560962404

#: Characteristic length: the Earth-Moon distance, in km.  Nondimensional
#: CR3BP positions are multiplied by this to report in km.
LU: float = 384400.0

#: Alias kept because `features.py` spells it `LU_KM`.
LU_KM: float = LU


# =========================================================================
# Labelling
# =========================================================================

#: Measured reproducibility floor, in km (0.155 m).
#:
#: NOT a tolerance and not a choice: re-deriving a torus by a different route
#: lands 2-155 mm from the surface, so the surface is not determined below
#: this however finely the family is densified.  It is the sigma_3 ~ 2.0e-4
#: result -- the curves themselves are the limit, not the spacing.  The Week 11
#: target is log10(max(d, FLOOR_KM)) and `at_floor` is d < FLOOR_KM, so this
#: number defines both the target and its censoring indicator.
FLOOR_KM: float = 1.55e-4


# =========================================================================
# The project's reference orbit
# =========================================================================

#: L2 halo #29 -- the Week 8 target orbit, [x, y, z, vx, vy, vz], nondimensional.
#:
#: C = 3.060334, rho/2pi = 0.293229, hyperbolic multiplier 146.4.  Hardcoded
#: rather than rebuilt from `orbits.compute_halo_family` so the suite stays fast
#: and independent of continuation settings.  Regenerated from a Richardson L2
#: seed (Az = 0.03) corrected by `halo_corrector`, then pseudo-arclength
#: continuation at ds = 0.004; it came out at index 75 of that family, so the
#: INDEX is not portable but the orbit is.  Southern branch (z0 < 0); the family
#: is symmetric in z.
#:
#: If this ever needs refreshing, the identifying pair is (C, rho/2pi), not the
#: index.  Treat as read-only -- it is a tuple so a caller cannot mutate the
#: shared object by accident.
HALO29_STATE: tuple = (1.043805201996, 0.0, -0.075586294085,
                       0.0, 0.392206924669, 0.0)

#: Period of HALO29_STATE, nondimensional.
HALO29_PERIOD: float = 3.131511429611

#: Jacobi constant of HALO29_STATE under this project's convention
#: (C = 2U - v^2, with the mu(1-mu)/2 term OMITTED).  Including that term would
#: give 3.072337.  `tests/test_features.py` pins the convention.
HALO29_C: float = 3.060334
