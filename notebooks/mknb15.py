"""Builder for nb15_gates.ipynb -- Week 15 adapter cell and pre-registered gates.

Covers G15.1 (group atomicity), G15.2 (energy ordering by C), the cell census,
and G15.4 (far-field test n per fold).  No fitting.  The fits are a later
notebook; this one is the gate that decides whether they are worth running.

NOTE ON SHARED HELPERS: this builder is self-contained -- it constructs the
notebook JSON itself rather than importing a common helper.  If notebooks/ has
a shared builder module, this should use it instead; say so and I will match it.

Divergence is measured on CELL SOURCE ONLY, so executing the notebook does not
read as a hand-edit.  Any hand-edit goes back into this file.
"""

import json
import os
import sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nb15_gates.ipynb')

CELLS = [

("markdown", """# Week 15 -- gates

Pre-registration: `W15_PREREG.md`. Nothing here fits a model. G15.1 and G15.2
can still change the week's design; G15.4 decides which folds are resolvable.
"""),

("code", """# --- ADAPTER CELL -------------------------------------------------------
# Repo coupling lives here and nowhere else.  Prints the archive's real
# contents and asserts before anything downstream runs.  Four wrong assumptions
# were caught this way in Week 13, each in one round trip.

import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..', 'src'))

from splits import (groups_within_levels, _straddling_groups,
                    leave_one_level_out, group_census, split_report)
import constants

ARCHIVE = '../data/qpit_dataset_w10.npz'
z = np.load(ARCHIVE, allow_pickle=True)

print(f"{ARCHIVE}: {len(z.files)} keys")
for k in sorted(z.files):
    a = z[k]
    print(f"  {k:<22} {str(a.shape):<14} {a.dtype}")

REQUIRED = ['group_id', 'family_id', 'energy_id', 'C', 'd_km', 'at_floor',
            'family_tags', 'floor_km', 'gap', 'family_max_gap_km']
missing = [k for k in REQUIRED if k not in z.files]
assert not missing, f"archive is missing required columns: {missing}"

gid, fid, eid = z['group_id'], z['family_id'], z['energy_id']
C, d_km, at_floor = z['C'], z['d_km'], z['at_floor']
tags = z['family_tags']

# floor_km is carried by the frozen dataset itself; read it from the archive
# and cross-check against constants.py.  That is the drift check, not shadowing.
FLOOR_KM = float(z['floor_km'])
assert abs(FLOOR_KM - constants.FLOOR_KM) < 1e-12, (
    f"archive floor_km {FLOOR_KM} disagrees with constants.py "
    f"{constants.FLOOR_KM}")

n = len(gid)
assert all(z[k].shape[0] == n for k in ('family_id', 'energy_id', 'C', 'd_km')), \\
    "per-row columns disagree on length"
print(f"\\nrows={n}  measured={int((~at_floor).sum())}  "
      f"censored={int(at_floor.sum())}  FLOOR_KM={FLOOR_KM}")
"""),

("markdown", "## G15.1 -- group atomicity\n\nNo `group_id` may straddle a level "
             "boundary on either key. A straddler between two *non*-held-out "
             "levels passes `groups_disjoint` on every seed of that fold while "
             "breaking a different one, so this is checked upfront, "
             "seed-independently, and not inferred from a split."),

("code", """G15_1 = {}
for name, key in (('family_id', fid), ('energy_id', eid)):
    bad = _straddling_groups(gid, key)
    G15_1[name] = dict(atomic=bool(groups_within_levels(gid, key)),
                       n_straddling=int(len(bad)),
                       examples=bad[:5].tolist())
    print(f"  {name:<10} atomic={G15_1[name]['atomic']}  "
          f"straddling={G15_1[name]['n_straddling']}")

assert all(v['atomic'] for v in G15_1.values()), \\
    "G15.1 FAILED -- a level holdout would leak; stop and redesign the split"
print("\\nG15.1 PASS")
"""),

("markdown", "## G15.2 -- energy ordering by C\n\n`C` is one-to-one with "
             "`energy_id` but not guaranteed monotone in it. The middle energy "
             "is identified by sorted `C` value. P15.3's "
             "interpolation/extrapolation reading is void if this is skipped."),

("code", """assert all(len(np.unique(C[eid == e])) == 1 for e in np.unique(eid)), \\
    "C is not constant within energy_id; the conditioning cells are not what "\\
    "the pre-registration assumes"

order = sorted((float(C[eid == e][0]), int(e)) for e in np.unique(eid))
E_a, E_b, E_c = (e for _, e in order)
ROLES = {E_a: 'E_a low  (extrapolation)',
         E_b: 'E_b MID  (interpolation)',
         E_c: 'E_c high (extrapolation)'}
for c, e in order:
    print(f"  C={c:.6f}  energy_id={e}   {ROLES[e]}")

MONOTONE = [e for _, e in order] == sorted(e for _, e in order)
print(f"\\nenergy_id monotone in C: {MONOTONE}")
print("G15.2 PASS -- folds are keyed by C rank, not by energy_id")
"""),

("markdown", "## Cell census\n\nSix conditioning cells (family x energy). "
             "Group-size structure drives the divergence between group "
             "fractions and row fractions."),

("code", """cells = fid * 3 + eid
u, counts = np.unique(cells, return_counts=True)
for i, t in enumerate(tags):
    print(f"  {t}: {counts[i]} rows")
cen = group_census(gid)
print(f"\\ngroups={cen['n_groups']}  median size={cen['size_median']}  "
      f"max={cen['size_max']}  singletons={cen['n_singleton']}")
"""),

("markdown", "## G15.4 -- far-field test n per fold\n\nTransfer is evaluated in "
             "the far field only; the near field carries no detectable "
             "information in distribution, so a near-field transfer "
             "measurement would measure the transfer of nothing. A fold with "
             "far-field test n below 500 is reported unresolved by "
             "construction."),

("code", """# The dead-zone boundary is NOT carried by the archive, so it comes from
# constants.py.  `DEAD_ZONE_KM` was a guessed name and does not exist; the
# candidates below are also guesses, and if none hits, this prints the real
# float attributes so the right name can be filled in once.
#
# The name is not the guard.  Re-deriving the boundary as the midpoint of the
# log10 span gives 0.124302 km and 8909 far-field rows -- one row off Week 13's
# 8908 -- so a plausible-looking derivation silently moves the window and makes
# Week 15 incomparable to the Week 13 MI result.  The assertion on the COUNT is
# the guard, because that is the condition that actually matters.

# RESOLVED: the boundary is not in constants.py -- it is DEAD_ZONE_KM_W12 in
# src/mi.py, and mi.make_windows is the function Week 13 built the windows with.
# Use it rather than re-deriving anything here.
#
# An unbounded `d_km > boundary` gives the same 8908 rows on this dataset, but
# only by luck: make_windows bounds the far window ABOVE at boundary^2/floor =
# 100.194 km and the data stops at 99.683 km, so the bound misses binding by
# half a percent.  A slightly longer tail and the two definitions diverge
# silently.
#
# The count assertion stays.  make_windows is the source; the count is the
# guard, and it is set equality rather than a size match: no measured row lies
# in (0.124390749, 0.124654774], so any boundary returning 8908 selects the
# identical rows.

from mi import make_windows, DEAD_ZONE_KM_W12

BOUNDARY_KM = DEAD_ZONE_KM_W12
W = make_windows(d_km, ~at_floor, floor_km=FLOOR_KM, boundary_km=BOUNDARY_KM)
far = W['far']
print(f"dead-zone boundary = {BOUNDARY_KM} km   from mi.DEAD_ZONE_KM_W12")
print(f"  near n={W['n_near']}  far n={W['n_far']}  "
      f"below floor={W['n_below_floor']}")
print(f"  near spans {W['near_decades']:.4f} decades, far realised "
      f"{W['far_decades_realised']:.4f}, shortfall={W['span_shortfall']}")
N_FAR_WK13 = 8908
print(f"far-field measured rows overall: {int(far.sum())}  "
      f"(Week 13 matched the MI windows at n = {N_FAR_WK13})")
assert int(far.sum()) == N_FAR_WK13, (
    f"far-field window gives {int(far.sum())} rows against Week 13's "
    f"{N_FAR_WK13}; the window definition has drifted and Week 15 far-field "
    f"numbers are not comparable to the Week 13 MI result")

FOLDS = [('T1  train L1 -> test L2', fid, 1),
         ('T2  train L2 -> test L1', fid, 0),
         ('T3  hold out E_a', eid, E_a),
         ('T4  hold out E_b', eid, E_b),
         ('T5  hold out E_c', eid, E_c)]

G15_4 = {}
for label, key, lv in FOLDS:
    m, meta = leave_one_level_out(key, lv, gid, frac_test=0.2,
                                  frac_val=0.1, seed=0)
    n_far = int((m['test'] & far).sum())
    G15_4[label] = dict(meta=meta, far_test_n=n_far,
                        resolvable=bool(n_far >= 500))
    print(f"  {label:<24} test={meta['test_rows']:>5}  far={n_far:>5}  "
          f"train={meta['train_rows']:>6}  excl={meta['excluded_rows']:>5}  "
          f"resolvable={n_far >= 500}")

# split_report per fold (G15.5).  A level holdout shows a large KS statistic by
# construction; the number quantifies how different the held-out region is.
print()
for label, key, lv in FOLDS:
    m, _ = leave_one_level_out(key, lv, gid, frac_test=0.2, frac_val=0.1, seed=0)
    r = split_report(np.log10(np.maximum(d_km, FLOOR_KM)), m, group_id=gid)
    print(f"  {label:<24} KS(train,test)={r['ks_stat_train_test']:.4f}")
"""),

("markdown", "## Not run here\n\n`nearest_point_overlap` is declared not "
             "applicable to a level holdout (W15_PREREG.md §6a) and is not "
             "called. The recovered coordinates it documents as its inputs "
             "(`alpha_found`, `theta_found`) are absent from this archive in "
             "any case."),
]


def build():
    nb = dict(cells=[], metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python",
                        name="python3"),
        language_info=dict(name="python")),
        nbformat=4, nbformat_minor=5)
    for kind, src in CELLS:
        cell = dict(cell_type=kind, metadata={}, source=src.splitlines(True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb['cells'].append(cell)
    return nb


def sources(nb):
    """Cell source only.  Executing a notebook changes outputs and execution
    counts; that must not read as a hand-edit."""
    return [''.join(c['source']) for c in nb['cells']]


def check_cells_compile(nb):
    """Every code cell must parse.  Cell source passes through two levels of
    quoting on the way here -- a Python literal in this file, then JSON -- and
    an escape that survives only one of them lands as a real line break inside
    a string. That is invisible in this file, invisible in the notebook JSON,
    and shows up as a SyntaxError on execution.  Compiling is the condition;
    "the builder ran and wrote N cells" is a proxy for it and was not enough.
    """
    bad = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        try:
            compile(src, f'<cell {i}>', 'exec')
        except SyntaxError as e:
            bad.append((i, e))
    return bad


def main():
    fresh = build()

    bad = check_cells_compile(fresh)
    if bad:
        for i, e in bad:
            print(f"cell {i} does not compile: {e}")
        print("REFUSING to write a notebook with cells that cannot run.")
        return 1

    if os.path.exists(NB):
        with open(NB) as f:
            existing = json.load(f)
        if sources(existing) != sources(fresh):
            if '--force' not in sys.argv:
                print(f"REFUSING to overwrite {NB}: cell source has diverged.")
                print(f"Fold the hand-edit back into {__file__} and re-run, "
                      f"or pass --force to discard the notebook's version.")
                return 1
            print(f"--force: overwriting a DIVERGED {os.path.basename(NB)}")
        else:
            # IDENTICAL SOURCE -> DO NOT WRITE.  Rewriting is a no-op on the
            # source and strips every executed output, which is a silent loss
            # of work: the exact failure mknb11.py's guard was built after
            # (33 cells / 23 outputs, 2026-08-16).  Both this builder and
            # mknb15b.py used to fall through to the write here.
            n_out = sum(len(c.get('outputs', [])) for c in existing['cells']
                        if c['cell_type'] == 'code')
            print(f"{os.path.basename(NB)}: already up to date "
                  f"({len(fresh['cells'])} cells, {n_out} executed outputs "
                  f"preserved) -- not rewriting")
            return 0
    with open(NB, 'w') as f:
        json.dump(fresh, f, indent=1)
    print(f"wrote {NB} ({len(fresh['cells'])} cells)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
