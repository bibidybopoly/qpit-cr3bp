"""Tests for src/gap_sample.py — the Week 10 gap-experiment sampler.

Run: pytest tests/test_gap_sample.py -v   (or python tests/gap_sample.py)

This module had no test file and was never imported by any test, despite
producing the treatment and both control arms the gap experiment rests on.

Where a test needs a torus with real dynamics (the unstable bundle) it uses one
converged member, built once.  Everything else uses a SYNTHETIC curve: the
bookkeeping-and-energy invariants -- row counts, arm labels, alpha bounds,
exact reprojection -- do not depend on the curve being genuinely invariant, and
saying so keeps the fixture cost honest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import qpit                                                    # noqa: E402
from cr3bp_3d import MU_EARTH_MOON as MU                       # noqa: E402
from gap_sample import (reproject_energy, phase_offset, sample_gap,  # noqa: E402
                        sample_on_torus, sample_off_manifold,
                        resonance_distance)
import constants as C                            # noqa: E402

HALO = dict(state0=list(C.HALO29_STATE), period=C.HALO29_PERIOD)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def _synthetic_record(seed=0, N=15, C=None):
    """A curve-shaped array with the right dtype and shape, NOT an invariant
    torus.  Legitimate for the bookkeeping and energy-exactness checks; never
    used where the dynamics matter.  N is odd: `qpit.check_odd` enforces that
    convention on every curve, synthetic or not."""
    rng = np.random.default_rng(seed)
    th = qpit.node_angles(N)
    U = np.column_stack([
        1.04 + 0.01 * np.cos(th), 0.006 * np.sin(th), -0.075 + 0.01 * np.sin(th),
        0.004 * np.cos(th), 0.392 + 0.004 * np.sin(th), 0.003 * np.cos(th)])
    U = U + 1e-5 * rng.standard_normal(U.shape)
    if C is None:
        C = qpit.mean_jacobi(U, MU)
    return dict(U0=U, T=3.13, rho=1.842, C=float(C))


_REAL = {}


def real_torus():
    """One converged torus, built once.  Only `sample_off_manifold` needs it,
    because the unstable bundle is only defined on a genuine invariant curve."""
    if 'rec' not in _REAL:
        U0, T, rho = qpit.seed_from_halo(HALO, MU, 15, 1e-4)
        out = qpit.converge_torus(U0, T, rho, MU, tol=1e-12, itmax=15,
                                  verbose=False)
        assert out['converged'], out['code']
        _REAL['rec'] = dict(U0=out['U0'], T=out['T'], rho=out['rho'],
                            C=float(qpit.mean_jacobi(out['U0'], MU)))
    return _REAL['rec']


# ---------------------------------------------------------------------------
# 1. Energy reprojection -- the arm-separating invariant
# ---------------------------------------------------------------------------

def test_reprojection_lands_exactly_on_the_target_level_set():
    """Exact to roundoff, not approximately.

    The whole design rests on energy carrying NO signal between the arms: if
    reprojection left a residual that correlated with alpha, the gap arm would
    differ from the controls in energy as well as in position and the
    experiment would not be measuring what it claims.
    """
    rng = np.random.default_rng(3)
    rec = _synthetic_record()
    for _ in range(40):
        x = rec["U0"][rng.integers(15)] + 1e-4 * rng.standard_normal(6)
        C_target = rec['C'] + rng.uniform(-1e-3, 1e-3)
        x_new, rel = reproject_energy(x, MU, C_target)
        assert x_new is not None
        assert abs(qpit.jacobi_constant(x_new, MU) - C_target) < 1e-12
        assert rel >= 0.0


def test_reprojection_preserves_position_and_velocity_direction():
    """Position must not move -- Omega depends on position only, so moving it
    would change the level set being targeted."""
    rec = _synthetic_record()
    x = rec['U0'][0].copy()
    x_new, _ = reproject_energy(x, MU, rec['C'] - 1e-4)
    assert np.array_equal(x_new[:3], x[:3])
    v, v_new = x[3:], x_new[3:]
    cos = (v @ v_new) / (np.linalg.norm(v) * np.linalg.norm(v_new))
    assert abs(cos - 1.0) < 1e-12, "velocity direction rotated"


def test_reprojection_does_not_mutate_its_argument():
    rec = _synthetic_record()
    x = rec['U0'][0].copy()
    before = x.copy()
    reproject_energy(x, MU, rec['C'] - 1e-3)
    assert np.array_equal(x, before), "reproject_energy wrote through its input"


def test_reprojection_refuses_the_forbidden_region_instead_of_returning_nonsense():
    """v^2_new <= 0 means the target energy is unreachable at that position.
    Returning None is what lets the samplers drop the row; a silent sqrt of a
    negative would give nan states that survive into the dataset."""
    rec = _synthetic_record()
    x, rel = reproject_energy(rec['U0'][0], MU, 1e6)     # far outside
    assert x is None and np.isnan(rel)
    x, rel = reproject_energy(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), MU, 3.0)
    assert x is None and np.isnan(rel)                   # zero velocity


# ---------------------------------------------------------------------------
# 2. Phase alignment
# ---------------------------------------------------------------------------

def test_phase_offset_is_zero_for_a_curve_against_itself():
    U = _synthetic_record()['U0']
    sigma, at_sigma, at_zero = phase_offset(U, U, n_scan=361)
    assert abs(sigma) < 1e-9
    assert at_sigma < 1e-9 and at_zero < 1e-9


def test_phase_offset_recovers_a_known_rotation():
    """A curve rotated by s must report sigma = -s: the scan asks how far to
    shift u_b to line it back up with u_a."""
    U = _synthetic_record()['U0']
    N = U.shape[0]
    s = 2 * np.pi * 3 / N                      # an exact node shift
    U_rot = qpit.evaluate_curve(U, qpit.node_angles(N) + s)
    sigma, at_sigma, at_zero = phase_offset(U, U_rot, n_scan=721)
    assert abs((-s - sigma + np.pi) % (2 * np.pi) - np.pi) < 0.02, sigma
    assert at_sigma < at_zero, "aligned mismatch is not below the unaligned one"


# ---------------------------------------------------------------------------
# 3. The three arms
# ---------------------------------------------------------------------------

def test_gap_arm_respects_the_open_interval_and_the_mean_energy():
    """alpha strictly inside (0,1), and every row on the MEAN of the two
    members' C -- both are what keep the treatment arm from being contaminated
    by the positive control."""
    a, b = _synthetic_record(0), _synthetic_record(1)
    rng = np.random.default_rng(11)
    rows = sample_gap(a, b, MU, n_alpha=6, n_theta=3, rng=rng)
    assert len(rows) > 0
    C_mean = 0.5 * (a['C'] + b['C'])
    for r in rows:
        assert r['arm'] == 'gap'
        assert 0.15 <= r['alpha'] <= 0.85
        assert 0.0 < r['alpha'] < 1.0
        assert r['C_target'] == C_mean
        assert abs(qpit.jacobi_constant(r['x0'], MU) - C_mean) < 1e-12
        assert r['T'] == 0.5 * (a['T'] + b['T'])
        assert 0.0 <= r['theta'] <= 2 * np.pi
    assert len({r['alpha_id'] for r in rows}) == 6
    assert len({r['alpha'] for r in rows}) == 6, "alpha must vary between draws"


def test_gap_arm_shares_alpha_within_a_draw():
    """The unit of replication is alpha, not theta -- every theta in one
    alpha_id must carry the SAME alpha, or the clustered CI is computed over
    the wrong groups and understates the interval by sqrt(n_theta)."""
    rows = sample_gap(_synthetic_record(0), _synthetic_record(1), MU,
                      n_alpha=4, n_theta=5, rng=np.random.default_rng(2))
    by_id = {}
    for r in rows:
        by_id.setdefault(r['alpha_id'], set()).add(r['alpha'])
    for aid, alphas in by_id.items():
        assert len(alphas) == 1, f"alpha_id {aid} spans {len(alphas)} alphas"


def test_positive_control_sits_on_the_member_at_its_own_energy():
    rec = _synthetic_record(5)
    rows = sample_on_torus(rec, MU, n_alpha=4, n_theta=3,
                           rng=np.random.default_rng(7))
    assert len(rows) == 12
    for r in rows:
        assert r['arm'] == 'pos'
        assert np.isnan(r['alpha'])
        assert r['reproj_rel'] == 0.0
        assert r['C_target'] == rec['C']
        assert r['T'] == rec['T']


def test_negative_control_displaces_along_the_bundle_and_keeps_the_energy():
    """The displacement must leave the curve but not the level set: if it
    changed C, the negative control would be separable from the treatment by
    energy alone and would stop being a control."""
    rec = real_torus()
    rows = sample_off_manifold(rec, MU, n_alpha=3, n_theta=4,
                               rng=np.random.default_rng(4), d=1e-6)
    assert len(rows) > 0
    nodes = set(np.round(qpit.node_angles(rec['U0'].shape[0]), 12))
    for r in rows:
        assert r['arm'] == 'neg'
        assert abs(qpit.jacobi_constant(r['x0'], MU) - rec['C']) < 1e-12
        assert r['bundle_R'] > 1.0, "unstable bundle growth factor must exceed 1"
        assert round(r['theta'], 12) in nodes, "displacement must be at a node"
    # displaced, but only just: d = 1e-6 plus the reprojection correction
    U = rec['U0']
    for r in rows[:8]:
        assert np.min(np.linalg.norm(U - r['x0'], axis=1)) < 1e-4


def test_samplers_are_reproducible_from_the_seed():
    a, b = _synthetic_record(0), _synthetic_record(1)
    r1 = sample_gap(a, b, MU, 4, 2, np.random.default_rng(99))
    r2 = sample_gap(a, b, MU, 4, 2, np.random.default_rng(99))
    assert len(r1) == len(r2)
    for p, q in zip(r1, r2):
        assert np.array_equal(p['x0'], q['x0'])
        assert p['alpha'] == q['alpha']


# ---------------------------------------------------------------------------
# 4. Resonance proximity
# ---------------------------------------------------------------------------

def test_resonance_distance_is_exact_on_rationals():
    for p, q in ((1, 2), (1, 3), (2, 5), (0, 1), (3, 4)):
        d, arg = resonance_distance(2 * np.pi * p / q, q_max=5)
        assert d < 1e-12, (p, q, d, arg)
        assert arg[1] <= q


def test_resonance_distance_is_bounded_by_the_farey_spacing():
    """With q <= q_max the worst case is half the smallest gap between
    neighbouring fractions, so nothing can exceed 1/(2*q_max)."""
    rng = np.random.default_rng(0)
    for q_max in (3, 5, 8):
        for _ in range(200):
            d, (p, q) = resonance_distance(rng.uniform(0, 20), q_max=q_max)
            assert 0.0 <= d <= 1.0 / (2 * q_max) + 1e-12, (d, q_max)
            assert 1 <= q <= q_max
            assert 0 <= p <= q


def test_resonance_distance_is_2pi_periodic_in_rho():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = rng.uniform(0, 2 * np.pi)
        base = resonance_distance(r)[0]
        for k in (-3, -1, 1, 4):
            assert abs(resonance_distance(r + k * 2 * np.pi)[0] - base) < 1e-12


def test_resonance_distance_tightens_as_q_max_grows():
    rng = np.random.default_rng(2)
    for _ in range(50):
        r = rng.uniform(0, 2 * np.pi)
        ds = [resonance_distance(r, q_max=q)[0] for q in (2, 4, 8, 16)]
        assert all(b <= a + 1e-15 for a, b in zip(ds, ds[1:])), ds


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
