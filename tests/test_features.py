"""Tests for src/features.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import constants as C                            # noqa: E402

from features import (jacobi_constant, assert_no_leakage, build_features,
                      Standardizer, MU_EARTH_MOON, FEATURE_ALLOWLIST,
                      assert_feature_set, assert_sheet_energy,
                      FEATURE_SET_W11, FEATURE_SET_W15)

# L2 halo #29 -- the project's target orbit, C recorded as 3.060334.
HALO29 = np.array(C.HALO29_STATE)
HALO29_C = C.HALO29_C


def _toy(n=60, levels=(0, 1)):
    """States plus a SHEET energy column: three values, constant within
    energy_id, as the Week 10 archive stores it."""
    rng = np.random.default_rng(0)
    states = rng.normal(size=(n, 6))
    fid = np.array([levels[i % len(levels)] for i in range(n)])
    eid = np.arange(n) % 3
    sheet_C = np.array([3.029398, 3.064703, 3.130607])[eid]
    return states, sheet_C, fid, eid


def test_jacobi_matches_halo29():
    """Pins the convention: C = 2U - v^2 with mu(1-mu)/2 OMITTED.

    Including the constant would give 3.072337, off by 0.012003.  If this test
    fails against cr3bp_3d's value, the two modules disagree on the convention
    and the audit's C comparison would be reading that offset, not a
    reprojection error.
    """
    C = float(jacobi_constant(HALO29[None, :], MU_EARTH_MOON)[0])
    assert abs(C - HALO29_C) < 1e-6, C
    with_const = C + MU_EARTH_MOON * (1 - MU_EARTH_MOON) / 2
    assert abs(with_const - HALO29_C) > 1e-3, "constant term is not detectable"


def test_jacobi_vectorises_and_is_velocity_even():
    S = np.tile(HALO29, (5, 1))
    S[:, 3:] *= np.array([[1], [-1], [1], [-1], [1]])
    C = jacobi_constant(S)
    assert C.shape == (5,)
    assert np.allclose(C, C[0])          # C depends on |v|^2 only


def test_jacobi_zero_velocity_is_2U():
    s = np.array([[0.5, 0.3, 0.1, 0.0, 0.0, 0.0]])
    mu = 0.1
    x, y, z = s[0, :3]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    expect = 2 * (0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2)
    assert abs(float(jacobi_constant(s, mu)[0]) - expect) < 1e-14


def test_assert_no_leakage_catches_every_stem():
    leaks = ['d_km', 'log_d_km', 'd_drawn_km', 'at_floor', 'alpha',
             'alpha_found', 'theta', 'theta_found', 'gap', 'amplitude_km',
             'group_id', 'source', 'random_dir', 'energy_id']
    for name in leaks:
        try:
            assert_no_leakage(['x', 'y', name])
        except ValueError:
            continue
        raise AssertionError(f"leak not caught: {name}")


def test_assert_no_leakage_passes_allowlist():
    assert assert_no_leakage(['x', 'y', 'z', 'vx', 'vy', 'vz',
                              'C', 'family_0', 'family_1'])


def test_build_features_shape_names_and_onehot():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(20, 6))
    fid = np.array([0] * 10 + [1] * 10)
    X, names, mask = build_features(S, jacobi_constant(S), fid, [0, 1])
    assert X.shape == (20, 9)
    assert names == ['x', 'y', 'z', 'vx', 'vy', 'vz', 'C', 'family_0', 'family_1']
    assert np.all(X[:, 7:].sum(axis=1) == 1)          # exactly one hot
    assert np.array_equal(X[:10, 7], np.ones(10))
    assert mask.tolist() == [True] * 7 + [False] * 2   # one-hot not scaled
    assert 'family_onehot' in FEATURE_ALLOWLIST


def test_build_features_onehot_layout_is_stable_under_subset():
    """A single-family fold must not silently drop a column and shift the rest."""
    rng = np.random.default_rng(1)
    S = rng.normal(size=(6, 6))
    fid = np.zeros(6, dtype=int)
    X, names, _ = build_features(S, jacobi_constant(S), fid, [0, 1])
    assert X.shape[1] == 9 and names[-1] == 'family_1'
    assert np.all(X[:, 8] == 0)


def test_build_features_rejects_unknown_level():
    S = np.zeros((3, 6))
    try:
        build_features(S, np.zeros(3), np.array([0, 1, 7]), [0, 1])
    except ValueError:
        return
    raise AssertionError("unknown family level accepted")


def test_build_features_reduced_drops_the_onehot():
    states, Csheet, fid, _ = _toy()
    X, names, mask = build_features(states, Csheet, fid, [0, 1],
                                    include_family=False)
    assert tuple(names) == FEATURE_SET_W15
    assert X.shape == (60, 7)
    assert mask.shape == (7,) and mask.all(), \
        "every column is continuous once the one-hot is gone, so all scale"


def test_build_features_default_still_gives_the_week11_layout():
    """`include_family` defaults to True, so every pre-Week-15 call site must
    produce exactly what it did before or its model changed underneath it."""
    states, Csheet, fid, _ = _toy()
    X, names, mask = build_features(states, Csheet, fid, [0, 1])
    assert tuple(names) == FEATURE_SET_W11
    assert X.shape == (60, 9)
    assert mask.tolist() == [True] * 7 + [False, False]


def test_reduced_and_full_agree_on_the_shared_columns():
    states, Csheet, fid, _ = _toy()
    Xf, _, _ = build_features(states, Csheet, fid, [0, 1])
    Xr, _, _ = build_features(states, Csheet, fid, [0, 1], include_family=False)
    assert np.array_equal(Xf[:, :7], Xr), \
        "dropping the one-hot must not disturb the columns before it"


def test_family_levels_still_validated_when_the_onehot_is_dropped():
    states, Csheet, fid, _ = _toy()
    try:
        build_features(states, Csheet, fid, [0], include_family=False)
    except ValueError as e:
        assert 'absent from family_levels' in str(e)
        return
    raise AssertionError("unseen family level accepted")


def test_reduced_set_has_no_dead_column_on_a_single_family():
    """The consequence of the layout stability pinned above, and the reason
    Week 15 drops the one-hot.  Train on one family and `family_1` is constant
    zero: zero gradient, weights stuck at initialisation, and at test time both
    columns flip.  The reduced set has no such column."""
    states, Csheet, fid, _ = _toy(levels=(0,))
    X, names, _ = build_features(states, Csheet, fid, [0, 1])
    assert 'family_1' in [n for i, n in enumerate(names) if X[:, i].std() == 0]
    Xr, names_r, _ = build_features(states, Csheet, fid, [0, 1],
                                    include_family=False)
    assert not [n for i, n in enumerate(names_r) if Xr[:, i].std() == 0]


def test_reduced_set_passes_the_leakage_check():
    assert assert_no_leakage(list(FEATURE_SET_W15))


def test_assert_feature_set_accepts_and_names_the_difference():
    states, Csheet, fid, _ = _toy()
    _, names, _ = build_features(states, Csheet, fid, [0, 1],
                                 include_family=False)
    assert assert_feature_set(names, FEATURE_SET_W15) is True
    try:
        assert_feature_set(names, FEATURE_SET_W11)
    except ValueError as e:
        assert 'family_0' in str(e) and 'missing' in str(e)
        return
    raise AssertionError("mismatched feature set accepted")


def test_assert_feature_set_is_order_sensitive():
    """Same columns permuted is a different model, and a caller that reorders
    would otherwise pass."""
    try:
        assert_feature_set(['y', 'x', 'z', 'vx', 'vy', 'vz', 'C'],
                           FEATURE_SET_W15)
    except ValueError:
        return
    raise AssertionError("permuted feature order accepted")


def test_assert_sheet_energy_accepts_the_stored_sheet_column():
    _, Csheet, _, eid = _toy()
    assert assert_sheet_energy(Csheet, eid) is True


def test_assert_sheet_energy_rejects_the_per_state_column():
    """The audit's condition.  Per-state Jacobi scores R2 = +0.7976 alone
    against -0.0001 for the sheet column, because the unreprojected offset is
    proportional to the target.  Feeding it manufactures the shortcut."""
    _, Csheet, _, eid = _toy()
    rng = np.random.default_rng(1)
    per_state = Csheet + rng.normal(0, 1e-3, size=Csheet.shape)
    try:
        assert_sheet_energy(per_state, eid)
    except ValueError as e:
        assert 'not constant within energy_id' in str(e)
        return
    raise AssertionError("per-state energy column accepted")


def test_assert_sheet_energy_catches_a_single_perturbed_row():
    """Constancy within energy_id, not a distinct-value count: one bad row in
    one sheet is enough, and a count-based proxy would miss it."""
    _, Csheet, _, eid = _toy()
    bad = Csheet.copy()
    bad[np.where(eid == 1)[0][0]] += 1e-9
    try:
        assert_sheet_energy(bad, eid)
    except ValueError:
        return
    raise AssertionError("single perturbed row accepted")


def test_standardizer_fits_on_train_only():
    rng = np.random.default_rng(2)
    tr = rng.normal(size=(200, 3))
    te = rng.normal(loc=50.0, size=(50, 3))       # wildly different moments
    sc = Standardizer().fit(tr)
    assert np.allclose(sc.transform(tr).mean(axis=0), 0, atol=1e-12)
    assert np.allclose(sc.transform(tr).std(axis=0), 1, atol=1e-12)
    Z = sc.transform(te)
    assert Z.mean() > 5, "test moments were used -- scaler saw the test set"


def test_standardizer_leaves_masked_columns_alone():
    X = np.column_stack([np.arange(10.0), np.tile([0.0, 1.0], 5)])
    sc = Standardizer(mask=np.array([True, False])).fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z[:, 1], X[:, 1])            # one-hot preserved
    assert abs(Z[:, 0].mean()) < 1e-12


def test_standardizer_survives_constant_column():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    Z = Standardizer().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


def test_standardizer_scales_all_seven_reduced_columns():
    """The reduced `scale_mask` is all True, unlike the Week 11 one."""
    states, Csheet, fid, _ = _toy()
    X, _, mask = build_features(states, Csheet, fid, [0, 1],
                                include_family=False)
    Z = Standardizer(mask).fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_standardizer_survives_a_constant_energy_column():
    """Distinct from `test_standardizer_survives_constant_column`: that one
    calls Standardizer directly with no mask.  This routes the all-True reduced
    mask from `build_features` through a single-sheet energy column, which is
    what a per-cell fit would see."""
    states, Csheet, fid, _ = _toy()
    X, _, mask = build_features(states, np.full_like(Csheet, 3.064703), fid,
                                [0, 1], include_family=False)
    Z = Standardizer(mask).fit_transform(X)
    assert np.isfinite(Z).all()


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
