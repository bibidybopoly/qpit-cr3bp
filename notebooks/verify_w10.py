"""Artifact reproducibility for `qpit_dataset_w10.npz`.  Register item 1.

THE ITEM WAS OVERSTATED AND THIS RECORDS THE CORRECTION.  The disposition
register said "if qpit_dataset_w10.npz cannot be regenerated from a script, that
is an artifact-reproducibility problem for arXiv and JEI".  It can be.
`notebooks/nb10_dataset_generation.ipynb` builds it end to end and the sampling
seed is recorded in the notebook -- `np.random.default_rng(20260810)`, created
once before the family loop, so the draw order is fixed.  What is missing is a
`mknb10.py` BUILDER, which is a repo-convention gap, not a reproducibility one.

WHAT THIS SCRIPT DOES INSTEAD.  A full regeneration rebuilds and densifies six
torus families and costs hours, so nobody runs it casually and it therefore
never gets run.  This verifies, in seconds, every column that can be re-derived
from other stored columns under the SHIPPED `sampling.py`.  If those all agree,
the archive and the code that claims to have produced it are consistent, which
is the property a reviewer actually wants and the one that rots silently.

WHAT IT CANNOT CHECK, STATED RATHER THAN IMPLIED.  `d_km` itself, `alpha`,
`theta`, `gap`, `amplitude_km`, `family_sizes` and `family_max_gap_km` all
depend on the six densified families and on `torus_field.surface_distance`.
Re-deriving them means the full rebuild.  This script reports them as unchecked
rather than passing over them.

TWO FINDINGS RECORDED HERE BECAUSE THEY HAVE NOWHERE ELSE TO LIVE.

1. `label_and_pack` produces `alpha_found` and `theta_found`; the notebook's
   `np.savez_compressed` call writes an explicit allowlist of keys and those two
   are not on it.  So the recovered surface coordinates are computed and then
   discarded at write time.  `splits.nearest_point_overlap` documents exactly
   those two columns as its inputs, so the dataset's writer and that instrument
   disagree about what the archive contains.  Adding two names to the allowlist
   fixes it at the next regeneration -- and regeneration is cut, so it does not
   get fixed.  This is a second, independent reason that instrument is not
   applicable in Week 15.

2. `jacobi` in the archive is NOT a per-state Jacobi constant.  `label_and_pack`
   writes `jacobi=np.full(n, C)`, the sheet energy broadcast, so `jacobi` and
   `C` are byte-identical columns.  The audit note in `src/features.py` compares
   the stored sheet energy against a RECOMPUTED per-state Jacobi; the recomputed
   one is not in the archive at all.  Anything warning against confusing the two
   stored columns is warning against a confusion that cannot happen.

Run from anywhere:  python3 notebooks/verify_w10.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

ARCH = os.path.join(_HERE, '..', 'data', 'qpit_dataset_w10.npz')

#: From nb10_dataset_generation.ipynb.  Recorded here so the regeneration recipe
#: survives independently of the notebook's outputs.
RECIPE = dict(
    seed=20260810,                 # cell 9, created ONCE before the family loop
    n_offset=2600, n_surface=400, n_bundle=400,
    n_nodes=31, eps_seed=1e-4, n_steps=60,
    a_target_km=400.0, target_gap_km=12.3,
    q_max=5, n_energy=3, min_sep=0.025,
    label_b_km=1.0,
    family_cache='../data/families/{tag}.npz',
)

#: Depend on the six densified families and on torus_field.surface_distance.
#: Only a full rebuild checks these.
UNCHECKABLE = ('d_km', 'alpha', 'theta', 'gap', 'amplitude_km', 'd_drawn_km',
               'states', 'family_sizes', 'family_max_gap_km', 'random_dir')


def main():
    z = np.load(ARCH, allow_pickle=True)
    floor = float(z['floor_km'])
    ok = True

    print("=== re-derived from other stored columns, under shipped sampling.py ===")
    try:
        from sampling import group_ids
        src = "sampling.group_ids"
    except ImportError as e:
        print(f"  cannot import sampling ({e}); it needs torus_field. "
              f"group_id check SKIPPED -- this is the important one, so fix "
              f"the import rather than accepting the skip.")
        group_ids, src = None, None

    checks = []
    if group_ids is not None:
        g = group_ids(z['family_id'], z['energy_id'], z['gap'],
                      z['alpha'], z['theta'])
        checks.append((f"group_id == {src}(...)", np.array_equal(g, z['group_id'])))
    checks += [
        ("score_C == log10(max(d_km, floor_km))",
         float(np.abs(z['score_C'] - np.log10(np.maximum(z['d_km'], floor))).max()) < 1e-12),
        ("at_floor == d_km < floor_km",
         np.array_equal(z['at_floor'], z['d_km'] < floor)),
        ("label_B == d_km <= label_B_km",
         np.array_equal(z['label_B'], z['d_km'] <= float(z['label_B_km']))),
        ("jacobi == C  (both are the sheet energy)",
         np.array_equal(z['jacobi'], z['C'])),
        ("C is one-to-one with energy_id",
         all(len(np.unique(z['C'][z['energy_id'] == e])) == 1
             for e in np.unique(z['energy_id']))
         and len(np.unique(z['C'])) == len(np.unique(z['energy_id']))),
    ]
    for name, passed in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    print("\n=== sample counts against the recipe ===")
    cell = z['family_id'] * 3 + z['energy_id']
    for arm, key in (('offset', 'n_offset'), ('surface', 'n_surface'),
                     ('bundle', 'n_bundle')):
        want = int(z[key])
        got = [int(((z['source'] == arm) & (cell == c)).sum()) for c in range(6)]
        passed = all(gg == want for gg in got) and want == RECIPE[key]
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {arm:<8} {want} per cell "
              f"(archive and recipe agree)  actual {got}")

    n_expect = 6 * (RECIPE['n_offset'] + RECIPE['n_surface'] + RECIPE['n_bundle'])
    passed = len(z['d_km']) == n_expect
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  total rows {len(z['d_km'])} "
          f"== 6 x (2600+400+400) = {n_expect}")

    tags = [f"{'L1' if f == 0 else 'L2'}_E{e}" for f in range(2) for e in range(3)]
    passed = list(z['family_tags']) == tags
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  family_tags follow the "
          f"(family_id, energy_id) nesting of the build loop")

    print("\n=== NOT checked here; only a full rebuild covers these ===")
    print("  " + ", ".join(UNCHECKABLE))

    print("\n=== regeneration recipe ===")
    for k, v in RECIPE.items():
        print(f"  {k:<16} {v}")
    print("  Full rebuild: run notebooks/nb10_dataset_generation.ipynb.")
    print("  `build()` RESUMES from " + RECIPE['family_cache'] + " when present,")
    print("  so a from-scratch reproduction must clear that directory first or")
    print("  it silently reuses cached families instead of rebuilding them.")

    print(f"\n{'ALL CHECKS PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
