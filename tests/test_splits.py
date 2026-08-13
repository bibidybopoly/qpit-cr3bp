"""Tests for src/splits.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from splits import (grouped_split, random_row_split, groups_disjoint,
                    group_census, split_report, nearest_point_overlap,
                    subsample_groups)


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


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
