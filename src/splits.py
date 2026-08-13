"""Train/val/test splitting for the Week 11 regressor.

WHY GROUPED.  `sampling.group_ids` bins (family, energy, gap, alpha, theta) --
a patch of the family surface.  Samples inside one bin are constructed from
nearly the same base point along nearly the same normal, so they are
near-duplicates, not independent draws.  Splitting on rows puts near-duplicates
on both sides of the boundary and the test score measures memorisation.

WHAT THE GROUPED SPLIT ACTUALLY ASKS.  "Does the field interpolate to unseen
patches of sheets it has already seen."  Holding out a whole energy or a whole
family is a strictly harder question and it is Week 15, not Week 11.  Saying so
explicitly matters, because the two get conflated in write-ups and the harder
claim is the one a reader will assume.

THE LEAK GROUPING DOES NOT CLOSE.  For a large offset the nearest surface point
is not the base point it was constructed from: `alpha_found` != `alpha`.  So a
test sample's nearest torus can lie inside a training patch even though its own
group is held out.  That is not fixable by regrouping -- the label is defined by
the found point, not the drawn one -- so it gets MEASURED instead, by
`nearest_point_overlap`, and reported as a number beside the test score.

RANDOM-ROW SPLIT IS PROVIDED ON PURPOSE.  It is the wrong split, and running
the nearest-neighbour baseline under both is the experiment that measures how
dependent the rows are.  The gap between the two is the evidence for grouping.
"""

import numpy as np

try:
    from scipy.stats import ks_2samp as _ks_2samp
except ImportError:                                   # pragma: no cover
    _ks_2samp = None


def grouped_split(group_id, frac_test=0.20, frac_val=0.10, seed=0):
    """Split by GROUP.  Returns dict of boolean row masks: train, val, test.

    `frac_val` is a fraction of the WHOLE dataset (not of train), carved off as
    its own set of groups, so the early-stopping set is as group-disjoint from
    train as the test set is.  Early stopping on a row-random validation set
    would select the epoch that best memorises training patches.

    Fractions are over GROUPS; row fractions will differ because groups vary in
    size.  `split_report` prints both -- do not assume they agree.
    """
    g = np.asarray(group_id).reshape(-1)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)

    n = len(uniq)
    n_test = int(round(frac_test * n))
    n_val = int(round(frac_val * n))
    if n_test + n_val >= n:
        raise ValueError(f"frac_test+frac_val leaves no training groups "
                         f"({n} groups available)")
    test_g = set(order[:n_test].tolist())
    val_g = set(order[n_test:n_test + n_val].tolist())

    in_test = np.fromiter((v in test_g for v in g), bool, len(g))
    in_val = np.fromiter((v in val_g for v in g), bool, len(g))
    masks = dict(train=~(in_test | in_val), val=in_val, test=in_test)
    _assert_partition(masks, len(g))
    return masks


def random_row_split(n_rows, frac_test=0.20, frac_val=0.10, seed=0):
    """Deliberately-wrong row-wise split, for the contamination comparison."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_test = int(round(frac_test * n_rows))
    n_val = int(round(frac_val * n_rows))
    masks = {k: np.zeros(n_rows, dtype=bool) for k in ('train', 'val', 'test')}
    masks['test'][order[:n_test]] = True
    masks['val'][order[n_test:n_test + n_val]] = True
    masks['train'][order[n_test + n_val:]] = True
    _assert_partition(masks, n_rows)
    return masks


def _assert_partition(masks, n):
    total = masks['train'].astype(int) + masks['val'].astype(int) \
        + masks['test'].astype(int)
    if not np.all(total == 1):
        raise AssertionError("split masks are not a partition of the rows")
    if len(masks['train']) != n:
        raise AssertionError("mask length mismatch")


def groups_disjoint(group_id, masks):
    """True iff no group appears in more than one part.  Use in tests."""
    g = np.asarray(group_id).reshape(-1)
    sets = [set(np.unique(g[masks[k]]).tolist()) for k in ('train', 'val', 'test')]
    return not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def group_census(group_id, masks=None):
    """Group-size statistics.  A long tail of singleton groups means the split
    fractions over rows and over groups diverge badly, which is worth seeing
    before it shows up as a surprising test-set size."""
    g = np.asarray(group_id).reshape(-1)
    _, counts = np.unique(g, return_counts=True)
    out = dict(n_rows=len(g), n_groups=len(counts),
               size_min=int(counts.min()), size_max=int(counts.max()),
               size_mean=float(counts.mean()),
               size_median=float(np.median(counts)),
               n_singleton=int((counts == 1).sum()))
    if masks is not None:
        for k, m in masks.items():
            out[f'{k}_rows'] = int(m.sum())
            out[f'{k}_groups'] = int(len(np.unique(g[m])))
            out[f'{k}_row_frac'] = float(m.mean())
    return out


def split_report(y, masks, group_id=None, strata=None):
    """Is the test set a fair sample of the same distribution?

    A grouped split can accidentally hold out a systematically different region
    -- e.g. all the largest-amplitude patches -- and then a poor test score is a
    coverage artefact rather than a modelling result.  The KS statistic on the
    target and the per-stratum counts are the cheap check.
    """
    y = np.asarray(y, dtype=float)
    rep = {}
    for k, m in masks.items():
        rep[f'{k}_n'] = int(m.sum())
        rep[f'{k}_y_mean'] = float(y[m].mean()) if m.any() else np.nan
        rep[f'{k}_y_std'] = float(y[m].std()) if m.any() else np.nan
        rep[f'{k}_y_min'] = float(y[m].min()) if m.any() else np.nan
        rep[f'{k}_y_max'] = float(y[m].max()) if m.any() else np.nan
    if _ks_2samp is not None and masks['train'].any() and masks['test'].any():
        ks = _ks_2samp(y[masks['train']], y[masks['test']])
        rep['ks_stat_train_test'] = float(ks.statistic)
        rep['ks_pvalue_train_test'] = float(ks.pvalue)
    if group_id is not None:
        rep.update(group_census(group_id, masks))
    if strata is not None:
        strata = np.asarray(strata)
        for lv in np.unique(strata):
            for k, m in masks.items():
                rep[f'stratum_{lv}_{k}'] = int((m & (strata == lv)).sum())
    return rep


def nearest_point_overlap(coords_train, coords_test, n_bins=64,
                          keys_train=None, keys_test=None, ranges=None):
    """Fraction of TEST samples whose nearest surface point falls in a bin that
    also contains a TRAIN sample.  This is the residual contamination the group
    key cannot remove.

    Pass the RECOVERED coordinates (`alpha_found`, `theta_found`) as columns of
    `coords_*`, and (family_id, energy_id) as `keys_*` so overlap is only
    counted within the same sheet.  Binning is done on the recovered
    coordinates directly rather than by re-deriving a group id, because whether
    `alpha_found` is a global or within-gap coordinate does not then matter --
    only that the same convention is used on both sides.

    A number near 1 does not invalidate the split.  It bounds how much of the
    test score could be near-neighbour recall, and it is the number to quote
    beside the headline metric.
    """
    A = np.atleast_2d(np.asarray(coords_train, dtype=float))
    B = np.atleast_2d(np.asarray(coords_test, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError("train/test coordinate arrays differ in width")
    if ranges is None:
        lo = np.minimum(A.min(axis=0), B.min(axis=0))
        hi = np.maximum(A.max(axis=0), B.max(axis=0))
    else:
        lo, hi = (np.asarray(r, dtype=float) for r in ranges)
    span = np.where(hi > lo, hi - lo, 1.0)

    def binned(M, keys):
        """Tuple keys, not a mixed-radix integer.

        An earlier version packed bins and sheet keys into one integer with a
        radix taken from each side's OWN key array -- so train (keys all 0) and
        test (keys all 1) were encoded under different radices and collided at
        random.  Tuples cannot fail that way, and 2e4 rows do not need the
        speed.
        """
        idx = np.clip(((M - lo) / span * n_bins).astype(np.int64), 0, n_bins - 1)
        if keys is None:
            return [tuple(row) for row in idx]
        k = np.atleast_2d(np.asarray(keys))
        if k.shape[0] != len(M):
            k = k.T
        return [tuple(a) + tuple(b) for a, b in zip(idx, k)]

    occupied = set(binned(A, keys_train))
    hit = np.fromiter((v in occupied for v in binned(B, keys_test)),
                      bool, len(B))
    return dict(overlap_fraction=float(hit.mean()),
                n_test=int(len(B)),
                n_bins_occupied_train=int(len(occupied)))


def subsample_groups(group_id, mask, fraction, seed=0):
    """Restrict `mask` to a random `fraction` of the groups it contains.

    Used by the learning curve.  Subsampling GROUPS rather than rows keeps each
    fold's effective sample size honest -- dropping rows inside a retained
    group removes near-duplicates, which is not the same as having less data.
    """
    g = np.asarray(group_id).reshape(-1)
    present = np.unique(g[mask])
    rng = np.random.default_rng(seed)
    keep = set(rng.permutation(present)[:max(1, int(round(fraction * len(present))))].tolist())
    return mask & np.fromiter((v in keep for v in g), bool, len(g))
