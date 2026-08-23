"""Tests for src/sampling.py — the Phase 4 dataset sampler.

Run: pytest tests/test_sampling.py -v   (or python tests/test_sampling.py)

sampling.py had no test file, and only `sample_on_surface` was reached at all
(from inside a function body in test_torus_field.py).  `label_and_pack`, which
produces the targets every Week 11/12 result is trained against, and
`group_ids`, which produces the key the train/test split is made on, were both
untested.

The fixture family is SYNTHETIC — smooth closed curves of increasing amplitude,
not converged tori.  That is legitimate here because everything this module
does is geometry on the interpolant: `surface_point`, `surface_distance` and
the group key never ask whether a curve is invariant.  `sample_bundle` is the
exception and uses one real converged torus, because the unstable bundle is
only defined on a genuine one.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import qpit                                                       # noqa: E402
from cr3bp_3d import MU_EARTH_MOON as MU                          # noqa: E402
from torus_field import surface_distance                          # noqa: E402
from sampling import (LU, FLOOR_KM, amplitude_to_cell, surface_point,  # noqa: E402
                      normal_directions, sample_offsets,
                      sample_on_surface, sample_bundle, group_ids,
                      label_and_pack)
import constants as C                            # noqa: E402

HALO = dict(state0=list(C.HALO29_STATE), period=C.HALO29_PERIOD)


def family(M=6, N=31, r0=1e-3, r1=3e-3):
    """A synthetic family: closed curves of strictly increasing amplitude.

    Geometry only — see the module docstring.  N is odd because
    `qpit.check_odd` enforces that on every curve.
    """
    th = qpit.node_angles(N)
    base = np.array([1.0438, 0.0, -0.0756, 0.0, 0.3922, 0.0])
    shape = np.column_stack([np.cos(th), np.sin(th), 0.7 * np.sin(th),
                             0.3 * np.cos(th), 0.5 * np.sin(th),
                             0.2 * np.cos(th)])
    out = []
    for a in np.linspace(r0, r1, M):
        out.append(dict(U0=base + a * shape, amplitude=float(a),
                        T=3.13, rho=1.842))
    return out


_REAL = {}


def real_torus():
    if 'rec' not in _REAL:
        U0, T, rho = qpit.seed_from_halo(HALO, MU, 15, 1e-4)
        out = qpit.converge_torus(U0, T, rho, MU, tol=1e-12, itmax=15,
                                  verbose=False)
        assert out['converged'], out['code']
        _REAL['rec'] = dict(U0=out['U0'], T=out['T'], rho=out['rho'],
                            amplitude=float(out['amplitude']))
    return _REAL['rec']


# ---------------------------------------------------------------------------
# 1. Amplitude inversion — keeps the densification's spacing out of the measure
# ---------------------------------------------------------------------------

def test_amplitude_inverts_back_to_the_amplitude_drawn():
    """(i, alpha) must reconstruct A_target, or the sample sits at a different
    amplitude than the one drawn and the uniform-in-amplitude design is void."""
    mem = family()
    A = np.array([m['amplitude'] for m in mem])
    rng = np.random.default_rng(0)
    A_t = rng.uniform(A[0], A[-1], 400)
    i, alpha = amplitude_to_cell(mem, A_t)
    recon = (1 - alpha) * A[i] + alpha * A[i + 1]
    assert np.allclose(recon, A_t, rtol=1e-12, atol=1e-15)
    assert np.all((0.0 <= alpha) & (alpha <= 1.0))
    assert np.all((0 <= i) & (i <= len(A) - 2))


def test_amplitude_inversion_clamps_outside_the_family_span():
    mem = family()
    A = np.array([m['amplitude'] for m in mem])
    i, alpha = amplitude_to_cell(mem, np.array([A[0] - 1.0, A[-1] + 1.0]))
    assert i[0] == 0 and alpha[0] == 0.0
    assert i[1] == len(A) - 2 and alpha[1] == 1.0


# ---------------------------------------------------------------------------
# 2. Surface geometry
# ---------------------------------------------------------------------------

def test_surface_point_is_the_linear_interpolant_with_exact_alpha_tangent():
    mem = family()
    th = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
    for a_val in (0.0, 0.37, 1.0):
        alpha = np.full(th.size, a_val)
        x, t_th, t_al = surface_point(mem, 2, alpha, th)
        A_ = qpit.evaluate_curve(mem[2]['U0'], th)
        B_ = qpit.evaluate_curve(mem[3]['U0'], th)
        assert np.allclose(x, (1 - a_val) * A_ + a_val * B_, atol=1e-14)
        # t_alpha is exact because the interpolant is linear in alpha
        assert np.allclose(t_al, B_ - A_, atol=1e-15)


def test_surface_point_theta_tangent_matches_a_finer_difference():
    """t_theta is a 1e-6 central difference; check it against a 10x finer one.
    A wrong tangent does not fail loudly — it silently biases the offset
    directions, which is the bias this module exists to remove."""
    mem = family()
    th = np.linspace(0.1, 6.0, 7)
    alpha = np.full(th.size, 0.4)
    _, t_th, _ = surface_point(mem, 1, alpha, th)
    h = 1e-5
    fine = (0.6 * (qpit.evaluate_curve(mem[1]['U0'], th + h)
                   - qpit.evaluate_curve(mem[1]['U0'], th - h))
            + 0.4 * (qpit.evaluate_curve(mem[2]['U0'], th + h)
                     - qpit.evaluate_curve(mem[2]['U0'], th - h))) / (2 * h)
    assert np.allclose(t_th, fine, rtol=1e-5, atol=1e-9)


def test_normal_directions_are_unit_and_clear_of_the_alpha_tangent():
    mem = family()
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 200)
    alpha = rng.uniform(0, 1, 200)
    _, t_th, t_al = surface_point(mem, 2, alpha, th)
    V = normal_directions(t_th, t_al, rng)

    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12)
    e_al = t_al / np.linalg.norm(t_al, axis=1)[:, None]
    assert np.abs(np.sum(V * e_al, axis=1)).max() < 1e-10


def test_normal_directions_still_carry_a_theta_tangent_component_KNOWN_DEFECT():
    """PINS A KNOWN BUG.  Delete this test when the bug is fixed.

    `normal_directions` projects the tangents out one after the other:

        for T in (t_th, t_al):
            V = V - (V . T_hat) T_hat

    Removing the t_al component RE-INTRODUCES a t_th component, because t_al is
    not orthogonal to t_th.  Only the tangent projected out LAST is actually
    cleared.  On the real family `data/families/L2_E0.npz` the two surface
    tangents sit at |cos| ~ 0.46 (median over 4000 draws), and the residual
    overlap with t_theta after "projection" is median 0.12, p95 0.44, max 0.87.

    Consequence: a step of length d along such a direction lands SHORT of d --
    median 0.9%, but the 5th percentile is 14% short.  The labels themselves
    are fine (they are measured, not assumed), but the distance DISTRIBUTION is
    pulled toward the surface relative to the log-uniform draw, and the
    `random_dir` flag no longer cleanly separates the two direction regimes.

    The fix is to orthonormalise the basis before projecting:

        e1 = _normalise(t_th)
        e2 = _normalise(t_al - (t_al . e1) e1)
        for T in (e1, e2): V = V - (V . T) T

    That changes what `mknb10` generates, so it has to be taken together with a
    decision to regenerate the Week 10 dataset -- which is why it is recorded
    here rather than applied.
    """
    mem = family()
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 400)
    alpha = rng.uniform(0, 1, 400)
    _, t_th, t_al = surface_point(mem, 2, alpha, th)
    V = normal_directions(t_th, t_al, rng)

    e_th = t_th / np.linalg.norm(t_th, axis=1)[:, None]
    resid = np.abs(np.sum(V * e_th, axis=1))
    assert resid.max() > 1e-3, (
        "the theta-tangent residual is gone -- if normal_directions was fixed, "
        "delete this test and restore the orthogonality assertion above")

    # and the tangents really are non-orthogonal, which is what causes it
    e_al = t_al / np.linalg.norm(t_al, axis=1)[:, None]
    assert np.abs(np.sum(e_th * e_al, axis=1)).max() > 0.1


# ---------------------------------------------------------------------------
# 3. The samplers
# ---------------------------------------------------------------------------

def test_sample_on_surface_lands_exactly_on_the_surface():
    """d = 0 by construction — these anchor the near-surface end of the target
    distribution.

    The bar is the measured reproducibility floor: anything below FLOOR_KM is
    indistinguishable from on-surface by construction.  Most rows come back at
    exactly 0 and the worst is ~2e-6 km, a bit over two orders below the floor.
    A tighter bar would be measuring `surface_distance`'s own theta-grid
    resolution rather than the sampler.
    """
    mem = family()
    out = sample_on_surface(mem, 60, np.random.default_rng(5))
    d_km = surface_distance(out['states'], mem)['d'] * LU
    assert d_km.max() < 0.1 * FLOOR_KM, d_km.max()
    assert np.median(d_km) == 0.0
    assert np.all(out['d_drawn_km'] == 0.0)
    assert np.all(out['source'] == 'surface')
    assert not out['random_dir'].any()


def test_drawn_distance_comes_back_as_the_measured_label():
    """The module's central claim, end to end: draw d, construct the state,
    label it geometrically, and get d back.

    "Drawing d first also makes every sample a free regression test: the label
    comes back and should equal what was drawn."  With the tangents projected
    out the docstring puts the agreement at ~2%.  This is the test that fails
    if the tangent projection, the interpolant or the labeller drifts apart.
    """
    mem = family()
    out = sample_offsets(mem, 300, np.random.default_rng(7),
                         d_lo_km=1e-2, d_hi_km=50.0, frac_random_dir=0.0)
    lab = surface_distance(out['states'], mem)
    rel = np.abs(lab['d'] * LU - out['d_drawn_km']) / out['d_drawn_km']
    assert np.median(rel) < 0.02, f"median relative error {np.median(rel):.4f}"
    assert np.percentile(rel, 95) < 0.10, np.percentile(rel, 95)


def test_random_directions_land_short_and_are_flagged():
    """A uniformly random 6-D direction has a large tangential component, so it
    lands CLOSER than the drawn distance.  That is expected and is exactly why
    `random_dir` is recorded per row rather than assumed away."""
    mem = family()
    out = sample_offsets(mem, 400, np.random.default_rng(9),
                         d_lo_km=1e-1, d_hi_km=50.0, frac_random_dir=1.0)
    assert out['random_dir'].all()
    lab = surface_distance(out['states'], mem)
    ratio = (lab['d'] * LU) / out['d_drawn_km']
    assert np.median(ratio) < 0.99, np.median(ratio)
    assert np.all(ratio <= 1.0 + 1e-6), "a sample cannot land further than it stepped"


def test_sample_offsets_reports_consistent_columns():
    mem = family()
    n = 120
    out = sample_offsets(mem, n, np.random.default_rng(3))
    for k in ('states', 'gap', 'alpha', 'theta', 'amplitude', 'd_drawn_km',
              'random_dir', 'source'):
        assert len(out[k]) == n, k
    assert out['states'].shape == (n, 6)
    assert np.all((0.0 <= out['alpha']) & (out['alpha'] <= 1.0))
    assert np.all((0.0 <= out['theta']) & (out['theta'] <= 2 * np.pi))
    assert np.all(out['d_drawn_km'] >= FLOOR_KM * (1 - 1e-9))
    assert np.all(out['d_drawn_km'] <= 100.0 * (1 + 1e-9))
    assert np.all(out['source'] == 'offset')


def test_drawn_distances_are_log_uniform_not_uniform():
    """The target is log10 d, so distance must be drawn LOG-uniformly.

    Drawing it uniformly still respects the bounds and still round-trips
    through the labeller, so every other test here passes -- but it puts
    almost nothing in the near-surface decades, which is the region the whole
    dataset exists to cover.  Checked on the distribution of log10 d: uniform
    spacing in the log leaves the decade counts flat, uniform spacing in d
    piles ~90% of the rows into the top decade.
    """
    mem = family()
    out = sample_offsets(mem, 4000, np.random.default_rng(17),
                         d_lo_km=1e-3, d_hi_km=100.0)
    lg = np.log10(out['d_drawn_km'])
    counts, _ = np.histogram(lg, bins=5, range=(-3.0, 2.0))
    frac = counts / counts.sum()
    assert frac.min() > 0.15, f"decades are not evenly filled: {frac}"
    assert frac.max() < 0.25, f"distance is not log-uniform: {frac}"
    # a uniform draw would put the median at ~50 km; log-uniform puts it at ~0.3
    assert np.median(out['d_drawn_km']) < 1.0, np.median(out['d_drawn_km'])


def test_sample_bundle_uses_node_angles_and_the_requested_distances():
    """The bundle is a per-node object, so theta must come from the nodes.
    Uses one real converged torus: the unstable bundle is not defined on a
    curve that is not invariant."""
    rec = real_torus()
    mem = [dict(rec, amplitude=rec['amplitude'] * s) for s in (1.0, 1.05, 1.1)]
    out = sample_bundle(mem, MU, 30, np.random.default_rng(2),
                        d_lo_km=1e-2, d_hi_km=1.0, n_members=3)
    N = rec['U0'].shape[0]
    nodes = set(np.round(2 * np.pi * np.arange(N) / N, 12))
    assert out['states'].shape[1] == 6
    assert np.all(out['source'] == 'bundle')
    assert np.all(out['alpha'] == 0.0)
    assert not out['random_dir'].any()
    assert set(np.round(out['theta'], 12)) <= nodes
    assert np.all(out['d_drawn_km'] >= 1e-2 * (1 - 1e-9))
    assert np.all(out['d_drawn_km'] <= 1.0 * (1 + 1e-9))
    assert np.all(out['gap'] <= len(mem) - 2), "gap index must stay in range"


# ---------------------------------------------------------------------------
# 4. The group key — what the Week 11 train/test split is made on
# ---------------------------------------------------------------------------

def test_group_id_is_deterministic_and_bins_neighbours_together():
    """Two samples from the same cell must share a group, or near-duplicates
    end up on both sides of the train/test boundary and the split leaks."""
    g1 = group_ids(1, 2, np.array([3]), np.array([0.11]), np.array([0.30]))
    g2 = group_ids(1, 2, np.array([3]), np.array([0.12]), np.array([0.32]))
    assert g1[0] == g2[0]
    assert group_ids(1, 2, np.array([3]), np.array([0.11]),
                     np.array([0.30]))[0] == g1[0]


def test_group_id_separates_every_coordinate_it_bins():
    base = dict(fam_id=1, energy_id=2, gap=np.array([3]),
                alpha=np.array([0.1]), theta=np.array([0.1]))

    def g(**kw):
        return int(group_ids(**{**base, **kw})[0])

    assert g() != g(fam_id=2)
    assert g() != g(energy_id=3)
    assert g() != g(gap=np.array([4]))
    assert g() != g(alpha=np.array([0.9]))
    assert g() != g(theta=np.array([6.0]))


def test_group_id_bin_edges_and_clipping():
    """alpha = 1.0 and theta = 2pi are attainable and must not fall off the
    end of their bins into a group of their own."""
    n_a, n_t = 4, 8
    top = group_ids(0, 0, np.array([0]), np.array([1.0]),
                    np.array([2 * np.pi]), n_a, n_t)
    just_below = group_ids(0, 0, np.array([0]), np.array([0.999]),
                           np.array([2 * np.pi - 1e-9]), n_a, n_t)
    assert top[0] == just_below[0]

    alpha = np.array([0.0, 0.24, 0.26, 0.99])
    ids = group_ids(0, 0, np.zeros(4, int), alpha, np.zeros(4), n_a, n_t)
    assert ids[0] == ids[1] and ids[1] != ids[2] and ids[2] != ids[3]


def test_group_ids_are_vectorised_elementwise():
    rng = np.random.default_rng(0)
    gap = rng.integers(0, 5, 50)
    alpha, theta = rng.random(50), rng.uniform(0, 2 * np.pi, 50)
    all_at_once = group_ids(1, 1, gap, alpha, theta)
    one_by_one = np.array([group_ids(1, 1, gap[k:k + 1], alpha[k:k + 1],
                                     theta[k:k + 1])[0] for k in range(50)])
    assert np.array_equal(all_at_once, one_by_one)


# ---------------------------------------------------------------------------
# 5. Assembly
# ---------------------------------------------------------------------------

def test_label_and_pack_produces_a_consistent_row_block():
    mem = family()
    rng = np.random.default_rng(13)
    parts = [sample_offsets(mem, 80, rng, d_lo_km=1e-3, d_hi_km=20.0),
             sample_on_surface(mem, 40, rng)]
    D = label_and_pack(parts, mem, fam_id=2, energy_id=1, family_name='famC',
                       C=3.0603)
    n = 120

    for k, v in D.items():
        assert len(v) == n, f"column {k} has {len(v)} rows, expected {n}"
    assert D['states'].shape == (n, 6)
    assert np.all(D['d_km'] >= 0.0)
    assert np.all(np.isfinite(D['d_km']))
    assert np.all(D['family_id'] == 2) and np.all(D['energy_id'] == 1)
    assert np.all(D['family'] == 'famC')
    assert np.array_equal(D['C'], D['jacobi'])
    assert (D['source'] == 'surface').sum() == 40


def test_label_and_pack_target_is_the_clamped_log_and_the_floor_flag_agrees():
    """log_d_km = log10(max(d, floor)) and at_floor = d < floor.

    These two lines are the definition of the Week 11 target and its censoring
    indicator.  `tests/test_targets.py` pins the transform in isolation; this
    pins that the packer applies the SAME rule, which is where a floor
    mismatch between the sampler and the target module would show up.
    """
    mem = family()
    rng = np.random.default_rng(21)
    parts = [sample_offsets(mem, 60, rng, d_lo_km=1e-6, d_hi_km=10.0),
             sample_on_surface(mem, 40, rng)]
    D = label_and_pack(parts, mem, 0, 0, 'famC', 3.06)

    assert np.array_equal(D['log_d_km'],
                          np.log10(np.maximum(D['d_km'], FLOOR_KM)))
    assert np.array_equal(D['at_floor'], D['d_km'] < FLOOR_KM)
    assert D['at_floor'].sum() > 0, "the on-surface rows should be at the floor"
    assert D['log_d_km'][D['at_floor']].min() == np.log10(FLOOR_KM)
    assert np.all(D['log_d_km'] >= np.log10(FLOOR_KM) - 1e-12)


def test_label_and_pack_honours_a_custom_floor():
    mem = family()
    rng = np.random.default_rng(22)
    parts = [sample_offsets(mem, 50, rng, d_lo_km=1e-6, d_hi_km=1.0)]
    D = label_and_pack(parts, mem, 0, 0, 'famC', 3.06, floor_km=1e-2)
    assert np.array_equal(D['at_floor'], D['d_km'] < 1e-2)
    assert D['log_d_km'].min() >= np.log10(1e-2) - 1e-12


def test_label_and_pack_group_ids_match_a_direct_call():
    mem = family()
    rng = np.random.default_rng(31)
    parts = [sample_offsets(mem, 70, rng), sample_on_surface(mem, 30, rng)]
    D = label_and_pack(parts, mem, 4, 2, 'famC', 3.06)
    expect = group_ids(4, 2, np.concatenate([p['gap'] for p in parts]),
                       np.concatenate([p['alpha'] for p in parts]),
                       np.concatenate([p['theta'] for p in parts]))
    assert np.array_equal(D['group_id'], expect)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
