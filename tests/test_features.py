"""Tests for src/features.py.  No CR3BP imports; runs standalone."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import constants as C                            # noqa: E402

from features import (jacobi_constant, assert_no_leakage, build_features,
                      Standardizer, MU_EARTH_MOON, FEATURE_ALLOWLIST)

# L2 halo #29 -- the project's target orbit, C recorded as 3.060334.
HALO29 = np.array(C.HALO29_STATE)
HALO29_C = C.HALO29_C


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


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
