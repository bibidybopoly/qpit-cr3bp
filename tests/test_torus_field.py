"""Certification of the surface-distance label.

This is the regression target for the whole Phase 4 dataset, so it is
checked against quantities known independently of it: exactness on the
family, closed-form recovery of alpha, absolute recovery of an imposed
displacement, agreement with a brute-force search, and agreement with the
INDEPENDENT probe measurement in the band where both are valid.

Needs ../data/famC_dense.npz (densify.densify at target_gap_km=12.3).

THAT FILE IS NOT IN THE REPO -- data/ is gitignored -- so on a fresh clone this
file used to raise FileNotFoundError from all ten tests.  Nine red tests that
mean "the fixture is missing" are worse than no tests: they hide whether the
certification actually passes.  It now SKIPS instead, via stdlib
`unittest.SkipTest`, which pytest reports as a skip and the standalone runner
below prints as one.  No pytest import is added.

Do NOT point FAM at a different family to make it run.  PROBE_CAL (1.1433) and
the tolerances below were measured on famC specifically; substituting
data/families/L2_E0.npz, for instance, passes 9 of 10 and fails the probe-band
agreement at 0.9885 against a 0.99 bar -- a calibration mismatch reported as a
label error.

Run: pytest tests/test_torus_field.py -v   (or python tests/test_torus_field.py)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cr3bp_3d import MU_EARTH_MOON as MU                        # noqa: E402
from densify import interp_error_km, load_family, required_gap_km  # noqa: E402
from qpit import evaluate_curve                                  # noqa: E402
from torus_field import _gap_distance, label_states, surface_distance  # noqa: E402
from torus_probe import curve_local_distance, probe_curve        # noqa: E402

from constants import LU, INTERP_FLOOR_KM   # one definition, project-wide
FAM = os.path.join(os.path.dirname(__file__), '..', 'data', 'famC_dense.npz')
PROBE_CAL = 1.1433          # measured on unstable-bundle displacements


def _fam():
    if not os.path.exists(FAM):
        raise unittest.SkipTest(
            f"needs {os.path.normpath(FAM)}, which is not in the repo (data/ is "
            f"gitignored). Regenerate with densify.densify(members, mu, "
            f"target_gap_km=12.3) then densify.save_family(). Do not substitute "
            f"another family: PROBE_CAL and the tolerances here are famC's.")
    return load_family(FAM)


# =====================================================================
# ERROR BUDGET ARITHMETIC
# =====================================================================

def test_gap_and_error_are_inverses():
    for target in (1e-3, 1e-4, 1e-5):
        g = required_gap_km(target)
        assert np.isclose(interp_error_km(g), target, rtol=1e-12)
    # monotone, and the quadratic scaling the gap experiment measured
    assert np.isclose(interp_error_km(2 * 12.3) / interp_error_km(12.3), 4.0)


def test_densified_family_meets_its_bound():
    mem = _fam()
    A = np.array([m['amplitude'] for m in mem]) * LU
    assert np.all(np.diff(A) > 0), "members must be sorted by amplitude"
    assert np.diff(A).max() <= 12.3, np.diff(A).max()
    assert interp_error_km(np.diff(A).max()) * 1000 <= 100.0     # 10 cm


# =====================================================================
# EXACTNESS
# =====================================================================

def test_points_on_a_member_are_on_the_surface():
    mem = _fam()
    rng = np.random.default_rng(0)
    X = evaluate_curve(mem[30]['U0'], rng.uniform(0, 2 * np.pi, 12))
    assert surface_distance(X, mem)['d'].max() * LU < 1e-9      # < 1 micron


def test_gap_boundary_points_are_not_stranded():
    """Regression: gap selection must be revisited after the coarse pass.

    A point built at alpha ~ 0 sits at a gap BOUNDARY. One side reports it
    at a clipped alpha, the other interior; at coarse resolution the clipped
    side can win, and refining only the winner can never recover because the
    clipped endpoint really is that gap's nearest point. This stranded an
    exactly-on-surface point at 84 m.
    """
    from sampling import sample_on_surface
    mem = _fam()
    X = sample_on_surface(mem, 400, np.random.default_rng(0))['states']
    d_m = surface_distance(X, mem, n_grid=512)['d'] * LU * 1000
    assert d_m.max() < 1.0, d_m.max()          # millimetres, not metres
    assert (d_m > 0.01).sum() == 0, (d_m > 0.01).sum()


def test_interpolants_are_on_the_surface_and_alpha_is_recovered():
    """The distinction from a finite SET of curves.

    A midpoint interpolant is g/2 from both bracketing members, so a
    nearest-member label would return ~5 km here. The surface returns zero,
    and hands back alpha = 0.5 in closed form.
    """
    mem = _fam()
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 12)
    for a in (0.25, 0.5, 0.75):
        X = ((1 - a) * evaluate_curve(mem[30]['U0'], th)
             + a * evaluate_curve(mem[31]['U0'], th))
        r = surface_distance(X, mem)
        assert r['d'].max() * LU < 1e-6, (a, r['d'].max() * LU)
        assert np.allclose(r['alpha'], a, atol=1e-3), (a, r['alpha'])

    half_gap_km = 0.5 * (mem[31]['amplitude'] - mem[30]['amplitude']) * LU
    assert half_gap_km > 1.0        # a nearest-MEMBER label would return this


def test_alpha_minimisation_matches_brute_force():
    """Closed-form alpha* against a dense direct search over alpha."""
    mem = _fam()
    rng = np.random.default_rng(2)
    U_a, U_b = mem[20]['U0'], mem[21]['U0']
    th = np.array([0.3, 2.0, 4.5])
    X = evaluate_curve(U_a, th) + 2e-6 * rng.standard_normal((3, 6))

    D, al = _gap_distance(X, U_a, U_b, th)
    closed = np.diagonal(D)

    grid = np.linspace(0.0, 1.0, 20001)
    A_, B_ = evaluate_curve(U_a, th), evaluate_curve(U_b, th)
    brute = np.array([
        np.linalg.norm(X[i] - ((1 - grid)[:, None] * A_[i]
                               + grid[:, None] * B_[i]), axis=1).min()
        for i in range(3)])
    assert np.allclose(closed, brute, rtol=1e-6), (closed, brute)


# =====================================================================
# ABSOLUTE RECOVERY
# =====================================================================

def test_recovers_imposed_normal_displacement():
    """Displace a whole member curve along its normal; get the size back.

    A random 6-D direction has a tangential component the surface absorbs,
    so the recovered distance is <= the imposed one. Projecting the
    displacement off the surface's local tangent directions makes the
    recovery exact, which is the sharper claim.
    """
    mem = _fam()
    m = mem[30]
    N = m['U0'].shape[0]
    th_nodes = np.linspace(0, 2 * np.pi, N, endpoint=False)
    tang_th = (evaluate_curve(m['U0'], th_nodes + 1e-6)
               - evaluate_curve(m['U0'], th_nodes - 1e-6))
    tang_a = mem[31]['U0'] - m['U0']                 # family direction
    rng = np.random.default_rng(3)

    for delta in (1e-9, 1e-8, 1e-7):
        V = rng.standard_normal((N, 6))
        for T in (tang_th, tang_a):                  # project out both tangents
            Tn = T / np.linalg.norm(T, axis=1)[:, None]
            V -= np.sum(V * Tn, axis=1)[:, None] * Tn
        V /= np.linalg.norm(V, axis=1)[:, None]
        d = surface_distance(m['U0'] + delta * V, mem)['d'].max()
        assert abs(d - delta) / delta < 2e-2, (delta, d)


def test_random_direction_never_overestimates():
    mem = _fam()
    m = mem[25]
    N = m['U0'].shape[0]
    rng = np.random.default_rng(4)
    for delta in (1e-9, 1e-8, 1e-7, 1e-6):
        V = rng.standard_normal((N, 6))
        V /= np.linalg.norm(V, axis=1)[:, None]
        d = surface_distance(m['U0'] + delta * V, mem)['d'].max()
        assert d <= delta * (1 + 1e-6), (delta, d)


# =====================================================================
# AGREEMENT WITH THE INDEPENDENT PROBE
# =====================================================================

def test_agrees_with_probe_in_the_overlap_band():
    """Two unrelated measurements of the same quantity.

    The probe solves the invariance equation; the surface is pure geometry
    against the family. They are valid together between the surface's 10 cm
    bound and the probe's 34.2 m ceiling.

    The residual ~10% offset is the PROBE's: its 1.1433 calibration was
    measured on unstable-bundle displacements and does not transfer exactly
    to random directions. The surface needs no calibration constant, which
    is why it is the reference and the probe is the cross-check.
    """
    mem = _fam()
    m = mem[30]
    N = m['U0'].shape[0]
    rng = np.random.default_rng(5)
    for delta in (3e-9, 1e-8):
        V = rng.standard_normal((N, 6))
        V /= np.linalg.norm(V, axis=1)[:, None]
        U_c = m['U0'] + delta * V

        d_surf = surface_distance(U_c, mem)['d'].max()
        p = probe_curve(U_c, m['T'], m['rho'], MU, C_target=m['C'],
                        offgrid=False)
        assert p['converged'], p['code']
        d_probe = curve_local_distance(U_c, p['result']['U0'])['d6'].max() / PROBE_CAL

        # Random direction, max over nodes: the tangential component is
        # absorbed by the surface, so d_surf sits just BELOW delta. The
        # earlier 1e-3 equality was reading a bug in which gap selection was
        # never revisited after the coarse pass, inflating the answer.
        assert d_surf <= delta * (1 + 1e-6), (d_surf, delta)
        assert d_surf / delta > 0.99, (d_surf, delta)
        assert 0.85 < d_surf / d_probe < 1.30              # probe within ~15%
        assert abs(d_surf - d_probe) * LU * 1000 < 100.0   # under 100 mm


# =====================================================================
# LABEL PACKAGING
# =====================================================================

def test_label_floor_is_the_interpolation_bound():
    mem = _fam()
    rng = np.random.default_rng(6)
    X = evaluate_curve(mem[30]['U0'], rng.uniform(0, 2 * np.pi, 8))
    lab = label_states(X, mem, floor_km=INTERP_FLOOR_KM)
    assert np.all(lab['at_floor'])
    assert np.allclose(lab['log_d_km'], np.log10(INTERP_FLOOR_KM))


def test_label_states_defaults_to_the_interpolation_floor_not_FLOOR_KM():
    """THE TWO FLOORS ARE DIFFERENT NUMBERS AND MUST NOT BE SWAPPED.

    `INTERP_FLOOR_KM` (1e-4) is the surface's interpolation error at the
    12.3 km design gap and falls as g^2 under densification.  `FLOOR_KM`
    (1.55e-4) is the measured reproducibility floor and does not fall at all.
    `label_states` clamps at the first; `sampling.label_and_pack`, which
    produced the shipped dataset, clamps at the second.  They are close
    enough (1.55x apart) that a swap would look like a rounding difference
    rather than a wrong label, which is exactly why this is pinned.
    """
    import constants as C
    import inspect
    assert C.INTERP_FLOOR_KM != C.FLOOR_KM
    assert abs(C.INTERP_FLOOR_KM - interp_error_km(12.3)) < 1e-6
    default = inspect.signature(label_states).parameters['floor_km'].default
    assert default == C.INTERP_FLOOR_KM

    mem = _fam()
    rng = np.random.default_rng(6)
    X = evaluate_curve(mem[30]['U0'], rng.uniform(0, 2 * np.pi, 8))
    assert np.allclose(label_states(X, mem)['log_d_km'],
                       np.log10(C.INTERP_FLOOR_KM))


def test_pointwise_gap_distance_is_the_diagonal_of_the_full_one():
    """`_gap_distance_at` replaced `np.diagonal(_gap_distance(...))` in the
    golden-section refinement.  The claim is EXACTNESS, not approximation --
    same arithmetic on the same values -- so this asserts equality at rtol=0
    and would fail on any reordering that changed the result by an ulp.
    """
    from torus_field import _gap_distance_at
    mem = _fam()
    rng = np.random.default_rng(11)
    P = 40
    th = rng.uniform(0, 2 * np.pi, P)
    X = evaluate_curve(mem[12]['U0'], th) + 1e-6 * rng.standard_normal((P, 6))
    U_a, U_b = mem[12]['U0'], mem[13]['U0']
    D, al = _gap_distance(X, U_a, U_b, th)
    d1, a1 = _gap_distance_at(X, U_a, U_b, th)
    assert np.allclose(np.diagonal(D), d1, rtol=0, atol=0)
    assert np.allclose(np.diagonal(al), a1, rtol=0, atol=0)
    assert d1.shape == (P,) and a1.shape == (P,)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = skipped = 0
    for fn in fns:
        try:
            fn()
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP  {fn.__name__}: {e}")
            continue
        ok += 1
        print(f"PASS  {fn.__name__}")
    print(f"\n{ok}/{len(fns)} passed"
          + (f", {skipped} skipped" if skipped else ""))
