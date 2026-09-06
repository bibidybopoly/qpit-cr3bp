"""Tests for src/splits.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from splits import (grouped_split, random_row_split, groups_disjoint,
                    group_census, split_report, nearest_point_overlap,
                    subsample_groups, leave_one_level_out,
                    groups_within_levels, _assert_partition)


def _groups(n_groups=200, size=7):
    return np.repeat(np.arange(n_groups), size)


def test_grouped_split_is_a_partition():
    g = _groups()
    m = grouped_split(g, seed=0)
    total = m['train'].astype(int) + m['val'].astype(int) + m['test'].astype(int)
    assert np.all(total == 1)


def test_no_group_on_two_sides():
    g = _groups()
    for seed in range(5):
        assert groups_disjoint(g, grouped_split(g, seed=seed))


def test_group_fractions_are_respected():
    g = _groups(1000, 3)
    m = grouped_split(g, frac_test=0.2, frac_val=0.1, seed=0)
    n_test = len(np.unique(g[m['test']]))
    n_val = len(np.unique(g[m['val']]))
    assert n_test == 200 and n_val == 100


def test_row_fraction_diverges_when_groups_are_uneven():
    """The reason group_census reports both.  Uneven groups make the row split
    fraction differ from the group split fraction; assuming they agree is how a
    surprising test-set size gets explained away as noise."""
    sizes = np.array([1] * 400 + [50] * 100)
    g = np.repeat(np.arange(len(sizes)), sizes)
    m = grouped_split(g, frac_test=0.2, frac_val=0.0, seed=3)
    assert abs(m['test'].mean() - 0.2) > 0.02


def test_determinism_and_seed_sensitivity():
    g = _groups()
    a = grouped_split(g, seed=7)['test']
    b = grouped_split(g, seed=7)['test']
    c = grouped_split(g, seed=8)['test']
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_val_is_group_disjoint_from_train():
    g = _groups()
    m = grouped_split(g, frac_val=0.15, seed=1)
    assert not (set(np.unique(g[m['train']])) & set(np.unique(g[m['val']])))


def test_grouped_split_refuses_impossible_fractions():
    try:
        grouped_split(_groups(10, 2), frac_test=0.8, frac_val=0.3)
    except ValueError:
        return
    raise AssertionError("impossible fractions accepted")


def test_random_row_split_is_a_partition_and_breaks_groups():
    g = _groups()
    m = random_row_split(len(g), seed=0)
    total = m['train'].astype(int) + m['val'].astype(int) + m['test'].astype(int)
    assert np.all(total == 1)
    assert not groups_disjoint(g, m), "row split should contaminate groups"


def test_group_census_counts():
    g = np.repeat([0, 1, 2], [1, 5, 10])
    c = group_census(g)
    assert c['n_rows'] == 16 and c['n_groups'] == 3
    assert c['size_min'] == 1 and c['size_max'] == 10 and c['n_singleton'] == 1


def test_split_report_flags_a_skewed_holdout():
    """A grouped split that holds out a systematically different region should
    show up as a large KS statistic, not be discovered later as a bad score."""
    g = np.repeat(np.arange(100), 5)
    y = np.repeat(np.linspace(0, 1, 100), 5)
    m = dict(train=g < 80, val=np.zeros(len(g), bool), test=g >= 80)
    r = split_report(y, m, group_id=g)
    assert r['ks_stat_train_test'] > 0.9
    fair = grouped_split(g, seed=0)
    assert split_report(y, fair, group_id=g)['ks_stat_train_test'] < 0.25


def test_nearest_point_overlap_endpoints():
    a = np.linspace(0, 1, 500)[:, None]
    assert nearest_point_overlap(a, a, n_bins=32)['overlap_fraction'] == 1.0
    lo = np.linspace(0.0, 0.4, 200)[:, None]
    hi = np.linspace(0.6, 1.0, 200)[:, None]
    r = nearest_point_overlap(lo, hi, n_bins=32, ranges=(np.array([0.]), np.array([1.])))
    assert r['overlap_fraction'] == 0.0


def test_nearest_point_overlap_respects_keys():
    """Same coordinates, different sheet -- must not count as overlap."""
    c = np.linspace(0, 1, 100)[:, None]
    same = nearest_point_overlap(c, c, n_bins=16,
                                 keys_train=np.zeros((100, 1), int),
                                 keys_test=np.zeros((100, 1), int))
    other = nearest_point_overlap(c, c, n_bins=16,
                                  keys_train=np.zeros((100, 1), int),
                                  keys_test=np.ones((100, 1), int))
    assert same['overlap_fraction'] == 1.0
    assert other['overlap_fraction'] == 0.0


def test_subsample_groups_keeps_whole_groups():
    g = _groups(100, 4)
    m = np.ones(len(g), bool)
    s = subsample_groups(g, m, 0.25, seed=0)
    kept = np.unique(g[s])
    assert len(kept) == 25
    for k in kept:
        assert s[g == k].all(), "a retained group was partially dropped"


def test_subsample_groups_is_monotone_in_fraction():
    g = _groups(200, 2)
    m = np.ones(len(g), bool)
    ns = [subsample_groups(g, m, f, seed=0).sum() for f in (0.1, 0.25, 0.5, 1.0)]
    assert ns == sorted(ns)



# --------------------------------------------------------------------------
# Week 15: leave_one_level_out
# --------------------------------------------------------------------------

def _levelled(n_groups=120, size=5, n_levels=2):
    """Groups atomic in `key`: every group sits wholly inside one level."""
    g = np.repeat(np.arange(n_groups), size)
    k = g % n_levels
    return g, k


def _straddling(n_groups=120, size=5, n_levels=3, bad_group=7):
    """Same, but one group has rows in two levels.  The defect the upfront
    atomicity check exists to reject."""
    g = np.repeat(np.arange(n_groups), size)
    k = g % n_levels
    idx = np.where(g == bad_group)[0]
    k[idx[:2]] = (k[idx[0]] + 1) % n_levels
    return g, k


def test_lolo_is_a_four_way_partition():
    g, k = _levelled()
    m, _ = leave_one_level_out(k, 1, g)
    assert set(m) == {'train', 'val', 'test', 'excluded'}
    total = sum(m[x].astype(int) for x in m)
    assert np.all(total == 1)


def test_lolo_all_four_parts_are_group_disjoint():
    g, k = _levelled()
    for seed in range(5):
        assert groups_disjoint(g, leave_one_level_out(k, 0, g, seed=seed)[0])


def test_lolo_val_contains_no_held_out_rows():
    """The contamination that would not show up in the test score.  Early
    stopping on held-out-level rows selects the epoch that best fits the
    regime the test set exists to measure."""
    g, k = _levelled()
    for seed in range(5):
        m, _ = leave_one_level_out(k, 1, g, seed=seed)
        assert not (k[m['val']] == 1).any()
        assert not (k[m['train']] == 1).any()


def test_lolo_test_and_excluded_are_only_held_out_rows():
    g, k = _levelled()
    m, _ = leave_one_level_out(k, 1, g)
    assert (k[m['test']] == 1).all()
    assert (k[m['excluded']] == 1).all()


def test_lolo_excluded_is_the_remainder_of_the_level():
    """`excluded` is the matched-n control's training stock.  If it is empty
    the control cannot be scored on the transfer run's test rows."""
    g, k = _levelled()
    m, _ = leave_one_level_out(k, 1, g, frac_test=0.2)
    lvl = (k == 1)
    assert np.array_equal(m['test'] | m['excluded'], lvl)
    assert m['excluded'].sum() > 0


def test_lolo_raises_on_a_straddling_group():
    g, k = _straddling()
    try:
        leave_one_level_out(k, 1, g)
    except ValueError as e:
        assert 'straddle' in str(e)
        return
    raise AssertionError("straddling group accepted")


def test_groups_within_levels_agrees_with_the_splitter():
    g, k = _levelled()
    assert groups_within_levels(g, k) is True
    gb, kb = _straddling()
    assert groups_within_levels(gb, kb) is False


def test_groups_within_levels_returns_a_bool_not_a_container():
    """A non-empty dict or array of offenders is truthy, so `if
    groups_within_levels(...)` at a call site would pass on exactly the
    datasets it is meant to reject."""
    gb, kb = _straddling()
    assert isinstance(groups_within_levels(gb, kb), bool)


def test_straddler_between_two_other_levels_evades_groups_disjoint():
    """Why atomicity is checked upfront rather than left to groups_disjoint.

    A group straddling levels 0 and 1 is invisible on the level-2 fold: it
    sits wholly in the training pool, lands in exactly one part, and passes
    on every seed.  Week 15 runs five folds off one dataset, so the fold that
    is checked is not the fold that breaks.
    """
    g = np.repeat(np.arange(120), 5)
    k = g % 3
    idx = np.where(g == 7)[0]                 # group 7 is level 1
    k[idx[:2]] = 0                            # straddles 0 and 1, NOT 2
    assert not groups_within_levels(g, k)
    assert set(k[g == 7].tolist()) == {0, 1}

    in_lvl = (k == 2)                          # the fold that cannot see it
    lvl, pool = np.unique(g[in_lvl]), np.unique(g[~in_lvl])
    for seed in range(8):
        tg = set(np.random.default_rng([seed, 0])
                 .permutation(lvl)[:int(round(0.2 * len(lvl)))].tolist())
        vg = set(np.random.default_rng([seed, 1])
                 .permutation(pool)[:int(round(0.1 * len(pool)))].tolist())
        it = np.fromiter((v in tg for v in g), bool, len(g))
        iv = np.fromiter((v in vg for v in g), bool, len(g))
        m = dict(test=in_lvl & it, excluded=in_lvl & ~it,
                 val=~in_lvl & iv, train=~in_lvl & ~iv)
        assert groups_disjoint(g, m), "expected the defect to be invisible here"


def test_lolo_determinism_and_seed_sensitivity():
    g, k = _levelled()
    a = leave_one_level_out(k, 1, g, seed=7)[0]['test']
    b = leave_one_level_out(k, 1, g, seed=7)[0]['test']
    c = leave_one_level_out(k, 1, g, seed=8)[0]['test']
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_frac_test_does_not_perturb_the_val_set():
    """What the two RNG streams buy.  A sensitivity check on test size must
    not silently move the early-stopping set."""
    g, k = _levelled()
    v1 = leave_one_level_out(k, 1, g, frac_test=0.2, seed=3)[0]['val']
    v2 = leave_one_level_out(k, 1, g, frac_test=0.4, seed=3)[0]['val']
    t1 = leave_one_level_out(k, 1, g, frac_test=0.2, seed=3)[0]['test']
    t2 = leave_one_level_out(k, 1, g, frac_test=0.4, seed=3)[0]['test']
    assert np.array_equal(v1, v2), "val moved when only frac_test changed"
    assert not np.array_equal(t1, t2)


def test_lolo_generic_key_handles_three_levels():
    """`energy_id` has three levels; `family_id` has two.  Same code path."""
    g, k = _levelled(n_groups=120, size=5, n_levels=3)
    for lv in (0, 1, 2):
        m, meta = leave_one_level_out(k, lv, g)
        assert (k[m['test']] == lv).all()
        assert meta['levels'] == [0, 1, 2]


def test_lolo_rejects_an_unknown_level():
    g, k = _levelled()
    try:
        leave_one_level_out(k, 9, g)
    except ValueError as e:
        assert 'not among levels' in str(e)
        return
    raise AssertionError("unknown level accepted")


def test_lolo_rejects_a_single_level_key():
    g = np.repeat(np.arange(50), 4)
    k = np.zeros(len(g), int)
    try:
        leave_one_level_out(k, 0, g)
    except ValueError as e:
        assert 'at least two levels' in str(e)
        return
    raise AssertionError("single-level key accepted")


def test_lolo_rejects_frac_test_that_empties_excluded():
    g, k = _levelled()
    try:
        leave_one_level_out(k, 1, g, frac_test=1.0)
    except ValueError as e:
        assert 'excluded' in str(e)
        return
    raise AssertionError("frac_test=1.0 accepted, leaving no control stock")


def test_lolo_rejects_frac_test_that_selects_nothing():
    g, k = _levelled(n_groups=10, size=4)
    try:
        leave_one_level_out(k, 1, g, frac_test=0.01)
    except ValueError as e:
        assert 'no groups' in str(e)
        return
    raise AssertionError("frac_test selecting zero groups accepted")


def test_lolo_rejects_frac_val_that_empties_train():
    g, k = _levelled()
    try:
        leave_one_level_out(k, 1, g, frac_val=1.0)
    except ValueError as e:
        assert 'no training groups' in str(e)
        return
    raise AssertionError("frac_val=1.0 accepted")


def test_lolo_rejects_length_mismatch():
    g, k = _levelled()
    try:
        leave_one_level_out(k[:-3], 1, g)
    except ValueError as e:
        assert 'differ in length' in str(e)
        return
    raise AssertionError("mismatched key/group_id accepted")


def test_lolo_meta_fractions_are_against_their_own_pools():
    """`meta` fractions are against the level and the pool; `group_census`
    reports fractions of the WHOLE dataset.  Reading one as the other is how
    a correct split gets called a broken one."""
    g, k = _levelled(n_groups=200, size=5)
    m, meta = leave_one_level_out(k, 1, g, frac_test=0.2, frac_val=0.1)
    assert abs(meta['test_row_frac_of_level'] - 0.2) < 0.05
    census = group_census(g, m)
    assert abs(census['test_row_frac'] - 0.1) < 0.05      # half as large
    assert meta['test_rows'] + meta['excluded_rows'] == int((k == 1).sum())
    assert meta['test_groups'] + meta['val_groups'] > 0


def test_lolo_group_fractions_are_over_groups_not_rows():
    g, k = _levelled(n_groups=200, size=5)
    m, meta = leave_one_level_out(k, 1, g, frac_test=0.2)
    assert len(np.unique(g[m['test']])) == meta['test_groups']


def test_lolo_uneven_groups_diverge_from_the_row_fraction():
    sizes = np.array([1] * 300 + [40] * 100)
    g = np.repeat(np.arange(len(sizes)), sizes)
    k = np.repeat(np.arange(len(sizes)) % 2, sizes)
    m, meta = leave_one_level_out(k, 1, g, frac_test=0.2, seed=2)
    assert abs(meta['test_row_frac_of_level'] - 0.2) > 0.02


def test_assert_partition_names_the_offending_mask():
    n = 10
    bad = dict(train=np.ones(n, bool), val=np.zeros(n - 1, bool),
               test=np.zeros(n, bool))
    try:
        _assert_partition(bad, n)
    except AssertionError as e:
        assert "'val'" in str(e)
        return
    raise AssertionError("wrong-length mask accepted")


def test_assert_partition_still_rejects_a_non_partition():
    n = 10
    bad = dict(train=np.ones(n, bool), val=np.ones(n, bool),
               test=np.zeros(n, bool))
    try:
        _assert_partition(bad, n)
    except AssertionError as e:
        assert 'partition' in str(e)
        return
    raise AssertionError("overlapping masks accepted")


def test_groups_disjoint_examines_a_fourth_part():
    """The generalisation.  The previous version iterated a hardcoded triple,
    so an overlap involving `excluded` was invisible."""
    g = np.repeat(np.arange(20), 3)
    m = dict(train=g < 10, val=np.zeros(len(g), bool),
             test=g >= 15, excluded=(g >= 5) & (g < 15))
    assert not groups_disjoint(g, m)
    m3 = {x: m[x] for x in ('train', 'val', 'test')}
    assert groups_disjoint(g, m3), "three-part view should look clean"


def test_straddling_groups_rejects_length_mismatch():
    """`leave_one_level_out` checks lengths itself, so the check inside
    `_straddling_groups` is unreachable from that path.  `groups_within_levels`
    is a public entry point and reaches it."""
    g, k = _levelled()
    try:
        groups_within_levels(g, k[:-3])
    except ValueError as e:
        assert 'differ in length' in str(e)
        return
    raise AssertionError("mismatched lengths accepted")


def test_lolo_group_counts_round_rather_than_truncate():
    """126 groups, 63 per level: 0.2 * 63 = 12.6.  Rounding gives 13 groups,
    truncation 12.  Every earlier fixture has an exact product, so truncation
    would go unnoticed there."""
    g, k = _levelled(n_groups=126, size=5)
    assert len(np.unique(g[k == 1])) == 63
    m, meta = leave_one_level_out(k, 1, g, frac_test=0.2)
    assert meta['test_groups'] == 13


def test_lolo_val_fraction_is_against_the_training_pool():
    """`frac_val` is a fraction of the groups OUTSIDE the held-out level, not
    of the whole dataset.  Over the dataset it would be twice as large here."""
    g, k = _levelled(n_groups=200, size=5)
    m, meta = leave_one_level_out(k, 1, g, frac_test=0.2, frac_val=0.1)
    assert abs(meta['val_row_frac_of_pool'] - 0.1) < 0.03
    assert meta['val_groups'] == 10


def test_assert_partition_rejects_an_uncovered_row():
    """A row belonging to NO part is as wrong as one belonging to two, and
    `total <= 1` would accept it."""
    g = np.repeat(np.arange(20), 3)
    bad = dict(train=g < 5, val=np.zeros(len(g), bool), test=g >= 15)
    try:
        _assert_partition(bad, len(g))
    except AssertionError as e:
        assert 'partition' in str(e)
        return
    raise AssertionError("uncovered rows accepted")

if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
