"""Tests for the spectral core of src/qpit.py — SECTIONS 1-3 and the helpers.

Run: pytest tests/test_qpit_spectral.py -v  (or python tests/test_qpit_spectral.py)

27 of qpit.py's 47 public names were never called by any test.  This file covers
the ones everything else is built on: the DFT pair, the rotation operator and
its rho-derivative, spectral differentiation, flattening, the continuation state
vector, the resolution monitor and resampling.

These are what qpit.py's own comments call "the exactness core of GMOS".  A bug
here does not raise -- it produces a slightly wrong Jacobian, a family that
drifts off the true branch, or a residual that measures truncation instead of
invariance.  Every test below is an analytic identity, so it holds at any N and
does not encode a recorded number.

Exactness assertions pass `rtol=0` deliberately.  `np.allclose` defaults to
rtol=1e-5, which on values of order k (~15 here) swamps any atol you set: a
mutation putting a 1e-7 relative error into the spectral derivative slipped
through the first version of this file for exactly that reason.  Where the
claim is "exact to roundoff", the tolerance has to be absolute.

The existing tests/test_qpit_week8.py and tests/test_qpit_integration.py cover
the solver end of the module; this is deliberately the other end.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import qpit                                                      # noqa: E402
import constants as C                                            # noqa: E402
from qpit import (check_odd, node_angles, wavenumbers,            # noqa: E402
                  rho_from_normalised, rho_to_normalised,
                  dft_matrix, idft_matrix, rotation_operator,
                  d_rotation_d_rho, spectral_derivative_operator,
                  apply_node_operator, flatten_curve, unflatten_curve,
                  evaluate_curve, pack_X, unpack_X, fourier_tail,
                  resample_curve, stability_index, latitudinal_tangent,
                  curve_amplitude)

ODD = (5, 15, 31)          # every test that can run at several N does


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def band_limited(N, kmax=3, seed=0):
    """A curve that is EXACTLY band-limited below the Nyquist of N.

    Spectral exactness claims only hold for band-limited data, so the fixture
    has to be genuinely band-limited rather than merely smooth -- otherwise a
    passing test is measuring aliasing tolerance, not exactness.
    """
    assert 2 * kmax + 1 <= N
    rng = np.random.default_rng(seed)
    th = node_angles(N)
    U = np.zeros((N, 6))
    for k in range(1, kmax + 1):
        a, b = rng.standard_normal(6), rng.standard_normal(6)
        U += np.cos(k * th)[:, None] * a + np.sin(k * th)[:, None] * b
    return U + rng.standard_normal(6)          # plus a k = 0 offset


# ---------------------------------------------------------------------------
# 1. Conventions
# ---------------------------------------------------------------------------

def test_odd_N_is_enforced_everywhere_it_matters():
    """Convention 2. An even N makes the symmetric wavenumber set asymmetric,
    so the k and -k contributions stop being conjugates and the rotation
    operator picks up an imaginary part."""
    for N in ODD:
        check_odd(N)
    for N in (4, 16, 32):
        _raises(ValueError, check_odd, N)
        _raises(ValueError, node_angles, N)
        _raises(ValueError, wavenumbers, N)


def test_wavenumbers_are_symmetric_and_ordered():
    for N in ODD:
        k = wavenumbers(N)
        assert k.shape == (N,)
        assert k[0] == -(N - 1) // 2 and k[-1] == (N - 1) // 2
        assert np.array_equal(k, np.sort(k))
        assert k.sum() == 0
        assert np.array_equal(k, -k[::-1])      # symmetric about zero


def test_node_angles_span_one_period_without_repeating_the_endpoint():
    for N in ODD:
        th = node_angles(N)
        assert th[0] == 0.0
        assert th[-1] < 2 * np.pi
        assert np.allclose(np.diff(th), 2 * np.pi / N)


def test_rho_unit_conversion_round_trips():
    """The Floquet tables are normalised, every GMOS formula is in radians.
    Getting this backwards is a factor of 2pi in the rotation number."""
    for r in (0.0, 0.1, 0.293229, 0.5, 1.0, -0.37):
        assert abs(rho_to_normalised(rho_from_normalised(r)) - r) < 1e-15
    assert abs(rho_from_normalised(0.5) - np.pi) < 1e-15
    assert abs(rho_from_normalised(1.0) - 2 * np.pi) < 1e-15


# ---------------------------------------------------------------------------
# 2. The DFT pair
# ---------------------------------------------------------------------------

def test_dft_and_idft_are_exact_inverses():
    for N in ODD:
        I = idft_matrix(N) @ dft_matrix(N)
        assert np.allclose(I, np.eye(N), rtol=0, atol=1e-12)
        assert np.allclose(dft_matrix(N) @ idft_matrix(N), np.eye(N), rtol=0, atol=1e-12)


def test_dft_puts_the_1_over_N_on_the_forward_transform():
    """Which side carries 1/N is a convention, and `fourier_tail` and
    `resample_curve` both assume this one."""
    N = 15
    v = np.ones((N, 1))
    C = dft_matrix(N) @ v
    k = wavenumbers(N)
    assert abs(C[k == 0, 0][0] - 1.0) < 1e-12          # mean, not N * mean
    assert np.abs(C[k != 0]).max() < 1e-12


def test_dft_recovers_the_coefficients_of_a_known_mode():
    """A single cos(k theta) must land as 1/2 on +k and 1/2 on -k."""
    N, k0 = 15, 3
    th = node_angles(N)
    v = np.cos(k0 * th)[:, None]
    C = dft_matrix(N) @ v
    k = wavenumbers(N)
    assert abs(C[k == k0, 0][0] - 0.5) < 1e-12
    assert abs(C[k == -k0, 0][0] - 0.5) < 1e-12
    assert np.abs(C[np.abs(k) != k0]).max() < 1e-12


def test_a_real_curve_has_conjugate_symmetric_coefficients():
    """C_{-k} = conj(C_k) is what makes every operator below come out REAL.
    If it fails, the wavenumber indexing is wrong."""
    for N in ODD:
        C = dft_matrix(N) @ band_limited(N, kmax=(N - 1) // 4 or 1)
        k = wavenumbers(N)
        for kk in range(1, (N - 1) // 2 + 1):
            assert np.allclose(C[k == kk], np.conj(C[k == -kk]), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Rotation — the operator the whole residual is built on
# ---------------------------------------------------------------------------

def test_rotation_by_zero_is_the_identity():
    for N in ODD:
        assert np.allclose(rotation_operator(N, 0.0), np.eye(N), rtol=0, atol=1e-12)


def test_rotations_compose_and_are_2pi_periodic():
    """R(a)R(b) = R(a+b) and R(2pi) = I. A rotation operator that failed these
    would still look plausible on a single application."""
    N = 15
    a, b = 0.7, -1.9
    assert np.allclose(rotation_operator(N, a) @ rotation_operator(N, b),
                       rotation_operator(N, a + b), rtol=0, atol=1e-12)
    assert np.allclose(rotation_operator(N, 2 * np.pi), np.eye(N), rtol=0, atol=1e-10)
    assert np.allclose(rotation_operator(N, 0.4) @ rotation_operator(N, -0.4),
                       np.eye(N), rtol=0, atol=1e-12)


def test_rotation_matches_evaluating_the_curve_at_shifted_angles():
    """The operator and the interpolant must be the same object. This is the
    identity that makes the invariance residual free of discretisation error."""
    for N in ODD:
        U = band_limited(N, kmax=(N - 1) // 4 or 1)
        for rho in (0.0, 0.37, 2.5, -1.1):
            lhs = rotation_operator(N, rho) @ U
            rhs = evaluate_curve(U, node_angles(N) + rho)
            assert np.allclose(lhs, rhs, rtol=0, atol=1e-11), (N, rho)


def test_rotation_by_one_node_is_an_exact_permutation():
    """Shifting by exactly 2pi/N must permute the nodes, with no leakage --
    the sharpest available check on the DFT normalisation."""
    N = 15
    R = rotation_operator(N, 2 * np.pi / N)
    P = np.roll(np.eye(N), -1, axis=0)
    assert np.allclose(R, P, rtol=0, atol=1e-10)


def test_rotation_preserves_the_mean_and_the_norm():
    """Both are exact: the mean is the k = 0 coefficient, which a phase shift
    cannot touch, and the node norm is Parseval's identity with |C_k| fixed."""
    N = 31
    U = band_limited(N, kmax=7)
    for rho in (0.3, 1.7, -2.2):
        V = rotation_operator(N, rho) @ U
        assert np.allclose(V.mean(axis=0), U.mean(axis=0), rtol=0, atol=1e-12)
        assert abs(np.linalg.norm(V) - np.linalg.norm(U)) < 1e-10


def test_amplitude_is_invariant_only_under_whole_node_rotations():
    """`curve_amplitude` is a max over the N NODES, not over the continuous
    curve, so an arbitrary rotation moves the sample points and legitimately
    changes it.  A rotation by a whole node spacing permutes the nodes and
    must leave it exactly alone.  Worth pinning because it is easy to assume
    the amplitude is a property of the torus rather than of its sampling --
    it is used for step limiting and collapse detection, where that
    distinction decides whether a step is accepted.
    """
    N = 31
    U = band_limited(N, kmax=7)
    step = 2 * np.pi / N
    for m in (1, 5, -3):
        V = rotation_operator(N, m * step) @ U
        assert abs(curve_amplitude(V) - curve_amplitude(U)) < 1e-10
    moved = curve_amplitude(rotation_operator(N, 0.3) @ U)
    assert abs(moved - curve_amplitude(U)) > 1e-6, (
        "a non-node rotation left the node-max unchanged -- suspicious")


def test_d_rotation_d_rho_matches_a_finite_difference():
    """This supplies the dF/drho column of the Jacobian. A wrong factor here
    still converges -- more slowly, to the same place -- so it hides."""
    N = 15
    for rho in (0.0, 0.6, -1.3):
        h = 1e-6
        fd = (rotation_operator(N, rho + h) - rotation_operator(N, rho - h)) / (2 * h)
        assert np.allclose(d_rotation_d_rho(N, rho), fd, rtol=0, atol=1e-7)


def test_d_rotation_d_rho_is_the_derivative_of_the_curve_it_rotates():
    """d/drho [R(rho) U] at rho = 0 is exactly dU/dtheta."""
    N = 31
    U = band_limited(N, kmax=7)
    assert np.allclose(d_rotation_d_rho(N, 0.0) @ U,
                       spectral_derivative_operator(N) @ U, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Spectral differentiation
# ---------------------------------------------------------------------------

def test_spectral_derivative_is_exact_on_a_pure_mode():
    """d/dtheta cos(k th) = -k sin(k th), to roundoff -- not to O(h^2).
    Exactness is the entire reason for the spectral representation."""
    for N in ODD:
        th = node_angles(N)
        for k in range(1, (N - 1) // 2 + 1):
            D = spectral_derivative_operator(N)
            assert np.allclose(D @ np.cos(k * th), -k * np.sin(k * th),
                               rtol=0, atol=1e-10)
            assert np.allclose(D @ np.sin(k * th), k * np.cos(k * th),
                               rtol=0, atol=1e-10)


def test_spectral_derivative_kills_a_constant():
    for N in ODD:
        assert np.abs(spectral_derivative_operator(N) @ np.ones(N)).max() < 1e-12


def test_second_derivative_is_the_square_of_the_operator():
    N = 31
    th = node_angles(N)
    D = spectral_derivative_operator(N)
    assert np.allclose(D @ (D @ np.cos(4 * th)), -16 * np.cos(4 * th),
                       rtol=0, atol=1e-9)


def test_latitudinal_tangent_is_the_spectral_derivative_of_the_curve():
    for N in (15, 31):
        U = band_limited(N, kmax=(N - 1) // 4)
        assert np.allclose(latitudinal_tangent(U),
                           spectral_derivative_operator(N) @ U, rtol=0, atol=1e-12)


def test_latitudinal_tangent_matches_a_finite_difference_of_the_interpolant():
    """Independent of the operator: differentiate `evaluate_curve` numerically."""
    N = 31
    U = band_limited(N, kmax=6)
    h = 1e-5
    th = node_angles(N)
    fd = (evaluate_curve(U, th + h) - evaluate_curve(U, th - h)) / (2 * h)
    assert np.allclose(latitudinal_tangent(U), fd, rtol=1e-6, atol=1e-8)


def test_apply_node_operator_acts_on_nodes_and_never_mixes_components():
    """The 6 state components must stay independent -- an operator that mixed
    them would couple x to vy and still produce plausible numbers."""
    N = 15
    U = band_limited(N)
    M = spectral_derivative_operator(N)
    out = apply_node_operator(M, U)
    assert out.shape == U.shape
    for a in range(6):
        assert np.allclose(out[:, a], M @ U[:, a], rtol=0, atol=1e-14)
    # perturbing one component must leave the other five untouched
    U2 = U.copy(); U2[:, 2] += 1.0
    d = apply_node_operator(M, U2) - out
    assert np.abs(d[:, [0, 1, 3, 4, 5]]).max() < 1e-14


# ---------------------------------------------------------------------------
# 5. Flattening and the continuation state vector
# ---------------------------------------------------------------------------

def test_flatten_round_trips_and_uses_the_documented_index():
    """Entry 6j + a is component a of node j.  Every Jacobian row index in
    Section 6 assumes exactly this."""
    N = 15
    U = band_limited(N)
    u = flatten_curve(U)
    assert u.shape == (6 * N,)
    assert np.array_equal(unflatten_curve(u), U)
    for j in (0, 1, 7, N - 1):
        for a in range(6):
            assert u[6 * j + a] == U[j, a]


def test_pack_and_unpack_round_trip():
    N = 15
    U, T, rho = band_limited(N), C.HALO29_PERIOD, 1.842
    X = pack_X(U, T, rho)
    assert X.shape == (6 * N + 2,)
    U2, T2, r2 = unpack_X(X)
    assert np.array_equal(U2, U) and T2 == T and r2 == rho
    assert X[-2] == T and X[-1] == rho          # T before rho, not after


def test_pack_X_accepts_integer_and_numpy_scalars():
    N = 5
    X = pack_X(band_limited(N, kmax=2), np.float64(3), 2)
    _, T, rho = unpack_X(X)
    assert isinstance(T, float) and isinstance(rho, float)


# ---------------------------------------------------------------------------
# 6. The resolution monitor
# ---------------------------------------------------------------------------

def test_fourier_tail_is_zero_for_a_curve_with_no_high_modes():
    N = 31
    assert fourier_tail(band_limited(N, kmax=3)) < 1e-12


def test_fourier_tail_is_normalised_against_the_fundamental_not_k0():
    """The docstring's own correction: dividing by k = 0 reports a tail about
    1e4 times healthier than it is for a small torus near the Moon, and a
    refinement trigger set on it would never fire.

    Constructed so the two normalisations differ by a known factor: a curve
    with a large k = 0 offset and a small fundamental.
    """
    N = 15
    th = node_angles(N)
    kmax = (N - 1) // 2
    U = np.zeros((N, 6))
    U += 1.0                                   # k = 0 offset, order 1
    U += 1e-5 * np.cos(th)[:, None]            # fundamental, order 1e-5
    U += 1e-9 * np.cos(kmax * th)[:, None]     # the tail

    tail = fourier_tail(U)
    # against the fundamental: 1e-9 / 1e-5 = 1e-4.  Against k=0 it would be
    # ~1e-9 -- five orders healthier, and below any sane refinement trigger.
    assert 1e-5 < tail < 1e-3, tail
    assert tail > 1e-6, "tail looks normalised against k=0, not the fundamental"


def test_fourier_tail_grows_with_high_mode_content():
    N = 31
    th = node_angles(N)
    base = np.cos(th)[:, None] * np.ones(6)
    prev = 0.0
    for amp in (1e-10, 1e-8, 1e-6, 1e-4):
        U = base + amp * np.cos(((N - 1) // 2) * th)[:, None]
        t = fourier_tail(U)
        assert t > prev
        prev = t


# ---------------------------------------------------------------------------
# 7. Resampling
# ---------------------------------------------------------------------------

def test_resample_to_the_same_N_is_an_independent_copy():
    N = 15
    U = band_limited(N)
    V = resample_curve(U, N)
    assert np.array_equal(V, U)
    V[0, 0] += 1.0
    assert U[0, 0] != V[0, 0], "resample_curve returned a view, not a copy"


def test_resample_is_exact_up_and_back_for_a_band_limited_curve():
    """Padding adds modes that were zero; truncating removes them again. Any
    error here is interpolation error entering through the back door."""
    U = band_limited(15, kmax=3)
    up = resample_curve(U, 31)
    assert up.shape == (31, 6)
    assert np.allclose(resample_curve(up, 15), U, rtol=0, atol=1e-11)


def test_resampling_up_agrees_with_the_interpolant_at_the_new_nodes():
    U = band_limited(15, kmax=3)
    up = resample_curve(U, 31)
    assert np.allclose(up, evaluate_curve(U, node_angles(31)), rtol=0, atol=1e-11)


def test_resampling_preserves_the_retained_spectrum():
    U = band_limited(15, kmax=3)
    up = resample_curve(U, 31)
    C_old = dft_matrix(15) @ U
    C_new = dft_matrix(31) @ up
    k_old, k_new = wavenumbers(15), wavenumbers(31)
    for k in range(-3, 4):
        assert np.allclose(C_new[k_new == k], C_old[k_old == k], rtol=0,
                           atol=1e-12), k
    # and the added modes are empty
    assert np.abs(C_new[np.abs(k_new) > 7]).max() < 1e-11


def test_resample_rejects_an_even_target():
    _raises(ValueError, resample_curve, band_limited(15), 16)


# ---------------------------------------------------------------------------
# 8. Stability index
# ---------------------------------------------------------------------------

def test_stability_index_is_one_on_the_unit_circle():
    """nu == 1 means stable; anything else from a purely rotational spectrum
    would mean the unstable radius is being misread."""
    for th in (0.0, 0.9, 2.4):
        DG = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        assert abs(stability_index(DG) - 1.0) < 1e-12


def test_stability_index_matches_its_definition_and_is_reciprocal_symmetric():
    """nu = (R + 1/R)/2 is invariant under R -> 1/R, which is what makes it a
    property of the reciprocal PAIR rather than of one eigenvalue."""
    for R in (1.5, 10.0, 146.4, 2674.0):
        DG = np.diag([R, 1.0 / R])
        expected = 0.5 * (R + 1.0 / R)
        assert abs(stability_index(DG) - expected) < 1e-9
        assert abs(stability_index(np.diag([1.0 / R, R])) - expected) < 1e-9


def test_stability_index_never_dips_below_one():
    rng = np.random.default_rng(0)
    for _ in range(50):
        DG = rng.standard_normal((6, 6))
        assert stability_index(DG) >= 1.0 - 1e-12


def test_stability_index_reads_the_modulus_not_the_real_part():
    """A purely imaginary unstable pair has |lambda| = R with Re = 0; reading
    the real part would report it as stable."""
    R = 5.0
    DG = np.array([[0.0, -R], [R, 0.0]])       # eigenvalues +/- iR
    assert abs(stability_index(DG) - 0.5 * (R + 1 / R)) < 1e-12


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
