"""Tests for the tangents, phase constraints and amplitude gradient in
src/qpit.py, plus two leftovers in targets.py and models.py.

Run: pytest tests/test_qpit_constraints.py -v
     (or python tests/test_qpit_constraints.py)

`test_qpit_spectral.py` covers the exactness core and `test_qpit_week8.py` the
converged solver.  This file takes the layer between them — the pieces the
Newton system is assembled from:

    longitudinal_tangent    dU/dtheta_0, from the PDE invariance equation
    phase_constraints       the two scalars that kill the symmetry directions
    mean_jacobi_row         the amplitude-pinning row of the Jacobian
    propagate_curve         the N-node stroboscopic map

None of these needs a CONVERGED torus: they are functions of whatever curve
they are handed, and the identities below hold on any curve at all.  That is
what makes them cheap to test, and it is also why they were skipped — they sit
next to code that does need one.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import constants as C                                            # noqa: E402
import qpit                                                      # noqa: E402
from qpit import (node_angles, latitudinal_tangent,              # noqa: E402
                  longitudinal_tangent, phase_constraints,
                  mean_jacobi, mean_jacobi_row, flatten_curve,
                  propagate_curve, vector_field, propagate,
                  jacobi_constant, curve_amplitude)
from cr3bp_3d import MU_EARTH_MOON as MU                          # noqa: E402
import targets as T                                               # noqa: E402

N = 15
TT = 3.13          # a stroboscopic time; nothing here needs it to be a period
RHO = 1.842


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def curve(N=N, A=1e-3, seed=0):
    """A smooth closed curve near the L2 halo.  NOT invariant — every identity
    tested here is a property of the formulas, not of the dynamics."""
    rng = np.random.default_rng(seed)
    th = node_angles(N)
    U = np.tile(np.array(C.HALO29_STATE), (N, 1))
    U[:, 0] += A * np.cos(th)
    U[:, 1] += A * np.sin(th)
    U[:, 2] += 0.4 * A * np.sin(th)
    U[:, 3] += 0.2 * A * np.cos(th)
    U[:, 4] += 0.3 * A * np.sin(th)
    U[:, 5] += 0.1 * A * np.cos(th)
    return U + 1e-9 * rng.standard_normal(U.shape)


# ---------------------------------------------------------------------------
# 1. The longitudinal tangent
# ---------------------------------------------------------------------------

def test_longitudinal_tangent_satisfies_the_pde_it_was_derived_from():
    """omega_0 du/dtheta_0 + omega_1 du/dtheta_1 = f(u), with omega_0 = 2pi/T
    and omega_1 = rho/T.  Rearranged, that IS the definition — so substituting
    the returned tangent back into the PDE must reproduce the vector field."""
    U = curve()
    t0 = longitudinal_tangent(U, TT, RHO, MU)
    t1 = latitudinal_tangent(U)
    f = np.array([vector_field(U[j], MU) for j in range(U.shape[0])])
    lhs = (2 * np.pi / TT) * t0 + (RHO / TT) * t1
    assert np.allclose(lhs, f, rtol=0, atol=1e-12)


def test_longitudinal_tangent_reduces_to_the_flow_when_rho_is_zero():
    """With no rotation the stroboscopic map is the flow itself, so
    du/dtheta_0 is T*f/(2pi)."""
    U = curve()
    f = np.array([vector_field(U[j], MU) for j in range(U.shape[0])])
    got = longitudinal_tangent(U, TT, 0.0, MU)
    assert np.allclose(got, TT * f / (2 * np.pi), rtol=0, atol=1e-14)


def test_longitudinal_tangent_scales_linearly_in_T_and_rho():
    U = curve()
    a = longitudinal_tangent(U, TT, RHO, MU)
    b = longitudinal_tangent(U, 2 * TT, RHO, MU)
    c = longitudinal_tangent(U, TT, 2 * RHO, MU)
    t1 = latitudinal_tangent(U)
    assert np.allclose(b - a, TT * np.array(
        [vector_field(U[j], MU) for j in range(U.shape[0])]) / (2 * np.pi),
        rtol=0, atol=1e-13)
    assert np.allclose(c - a, -RHO * t1 / (2 * np.pi), rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# 2. The phase constraints — and the Week 8 fix they carry
# ---------------------------------------------------------------------------

def test_phase_constraints_vanish_when_the_curve_is_its_own_reference():
    """<U - U_ref, tan> with U == U_ref is identically zero, whatever the
    tangents are.  Anything else means the difference is not being taken."""
    U = curve()
    f0, f1 = phase_constraints(U, U, TT, RHO, MU)
    assert f0 == 0.0 and f1 == 0.0


def test_phase_constraints_use_the_difference_not_the_curve_WEEK8_FIX():
    """THE constraint is <U - U_ref, tan>, NOT <U, tan>.

    The docstring records why this hid: for the LATITUDINAL tangent the two
    forms coincide identically, because <U_ref, D_theta U_ref> = 0 by
    antisymmetry.  For the LONGITUDINAL one they do not — except at a halo
    seed, where a perpendicular x-z crossing makes x0 . f(x0) vanish exactly
    (measured 2.5e-16), so the wrong form is invisible on iteration one and
    switches on the moment Newton moves off the crossing.

    This test constructs the case that separates them: U_ref a curve whose
    longitudinal tangent is NOT orthogonal to it, and U a different curve.
    """
    U_ref = curve(seed=1)
    U = curve(seed=2, A=1.3e-3)

    tan0 = longitudinal_tangent(U_ref, TT, RHO, MU)
    tan1 = latitudinal_tangent(U_ref)
    f0, f1 = phase_constraints(U, U_ref, TT, RHO, MU)

    assert abs(f0 - np.sum((U - U_ref) * tan0)) < 1e-18
    assert abs(f1 - np.sum((U - U_ref) * tan1)) < 1e-18

    # the wrong form <U, tan> would give a different number for tan0 ...
    wrong0 = float(np.sum(U * tan0))
    assert abs(wrong0 - f0) > 1e-6 * max(abs(f0), 1.0), (
        "the two forms agree here — this fixture cannot separate them")

    # ... while for tan1 the two coincide, exactly as the docstring says.
    # That is why the latitudinal constraint never exposed the bug.
    assert abs(float(np.sum(U_ref * tan1))) < 1e-12


def test_latitudinal_self_overlap_vanishes_by_antisymmetry():
    """<U, D_theta U> = N * sum_k i*k*|C_k|^2 = 0 for a real curve.  This is
    the identity the docstring leans on; check it directly rather than only
    through its consequence."""
    for n in (15, 31):
        U = curve(N=n, seed=3)
        assert abs(float(np.sum(U * latitudinal_tangent(U)))) < 1e-12


def test_phase_constraints_are_linear_in_the_displacement():
    """Both constraints are inner products, so doubling the displacement
    doubles them.

    TOLERANCES DIFFER BY DESIGN.  f0 is a well-conditioned O(|d|) number and
    holds to 1e-9 at every scale tested.  f1 is ~1e-16 -- it is zero up to the
    antisymmetry identity above -- so its value is set by the cancellation in
    (U_ref + d) - U_ref, which loses the low bits of d whenever |d| << |U_ref|.
    Asserting 1e-9 on f1 would be asserting a property of the roundoff.
    """
    U_ref = curve(seed=4)
    for scale, rtol1 in ((1e-5, 1e-2), (1e-3, 1e-4)):
        d = scale * curve(seed=5)
        a = np.array(phase_constraints(U_ref + d, U_ref, TT, RHO, MU))
        b = np.array(phase_constraints(U_ref + 2 * d, U_ref, TT, RHO, MU))
        assert abs(b[0] - 2 * a[0]) <= 1e-9 * abs(2 * a[0])
        assert abs(b[1] - 2 * a[1]) <= rtol1 * abs(2 * a[1]) + 1e-20
        assert abs(a[1]) < 1e-14, "f1 should be at the antisymmetry floor"


def test_phase_constraints_kill_the_two_symmetry_directions():
    """A displacement ALONG a tangent must produce a non-zero constraint in
    that slot — that is what 'kills the direction' means.  If either came back
    zero the corresponding symmetry would stay unconstrained and Newton's
    system would keep its nullspace."""
    U_ref = curve(seed=6)
    t0 = longitudinal_tangent(U_ref, TT, RHO, MU)
    t1 = latitudinal_tangent(U_ref)
    eps = 1e-6
    f0, _ = phase_constraints(U_ref + eps * t0, U_ref, TT, RHO, MU)
    _, f1 = phase_constraints(U_ref + eps * t1, U_ref, TT, RHO, MU)
    assert abs(f0) > 1e-12 and abs(f1) > 1e-12
    assert abs(f0 - eps * float(np.sum(t0 * t0))) < 1e-18
    assert abs(f1 - eps * float(np.sum(t1 * t1))) < 1e-18


# ---------------------------------------------------------------------------
# 3. The amplitude-pinning row
# ---------------------------------------------------------------------------

def test_mean_jacobi_row_is_the_gradient_of_mean_jacobi():
    """This row is what pins amplitude in the Newton system.  Without it the
    constant curve v == x_halo solves the invariance equation for any rho and
    Newton can bleed epsilon to zero -- so a wrong gradient does not fail
    loudly, it converges to the periodic orbit you started from."""
    U = curve(seed=7)
    g = mean_jacobi_row(U, MU)
    assert g.shape == (6 * U.shape[0],)

    h = 1e-7
    u = flatten_curve(U).copy()
    fd = np.zeros_like(u)
    for i in range(u.size):
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        fd[i] = (mean_jacobi(up.reshape(-1, 6), MU)
                 - mean_jacobi(um.reshape(-1, 6), MU)) / (2 * h)
    scale = max(1.0, np.abs(fd).max())
    assert np.abs(g - fd).max() / scale < 1e-6


def test_mean_jacobi_row_carries_the_one_over_N_averaging():
    """It is the gradient of the MEAN, not of the sum: doubling N with the same
    shape must leave the per-node contribution scaled by 1/N, or the pin row
    changes weight with the discretisation."""
    for n in (15, 31):
        U = curve(N=n, seed=8)
        g = mean_jacobi_row(U, MU).reshape(n, 6)
        node0 = qpit.jacobi_gradient(U[0], MU)
        assert np.allclose(g[0], node0 / n, rtol=0, atol=1e-14)


def test_mean_jacobi_is_the_common_value_on_a_constant_curve():
    """Every node of a constant curve carries the same C, so the mean is it."""
    x = np.array(C.HALO29_STATE)
    U = np.tile(x, (N, 1))
    assert abs(mean_jacobi(U, MU) - jacobi_constant(x, MU)) < 1e-14
    assert curve_amplitude(U) < 1e-15


# ---------------------------------------------------------------------------
# 4. The stroboscopic map over the nodes
# ---------------------------------------------------------------------------

def test_propagate_curve_is_the_per_node_propagation():
    """The nodes do not interact -- that is the property that makes this the
    natural place to parallelise on Titan, and it has to actually hold."""
    U = curve(N=7, A=1e-4, seed=9)
    U1 = propagate_curve(U, 0.35, MU)
    assert U1.shape == U.shape
    for j in range(U.shape[0]):
        assert np.allclose(U1[j], propagate(U[j], 0.35, MU), rtol=0, atol=1e-11)


def test_propagate_curve_with_stm_returns_one_stm_per_node():
    U = curve(N=5, A=1e-4, seed=10)
    U1, stms = propagate_curve(U, 0.35, MU, with_stm=True)
    assert U1.shape == U.shape
    assert len(stms) == U.shape[0]
    for Phi in stms:
        assert Phi.shape == (6, 6)
        assert np.all(np.isfinite(Phi))
        # the CR3BP flow is symplectic, so every node STM has |det| = 1
        assert abs(abs(np.linalg.det(Phi)) - 1.0) < 1e-6


def test_propagate_curve_states_agree_with_and_without_the_stm():
    """Carrying the STM changes the integrated dimension from 6 to 42, which
    changes DOP853's step sequence.  The states must still agree to integrator
    tolerance -- a larger gap means the augmented system is not the same
    trajectory."""
    U = curve(N=5, A=1e-4, seed=11)
    a = propagate_curve(U, 0.35, MU)
    b, _ = propagate_curve(U, 0.35, MU, with_stm=True)
    assert np.allclose(a, b, rtol=0, atol=1e-9)


def test_propagate_curve_over_zero_time_is_the_identity():
    U = curve(N=5, A=1e-4, seed=12)
    assert np.allclose(propagate_curve(U, 0.0, MU), U, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Leftovers: targets.inverse and models.count_parameters
# ---------------------------------------------------------------------------

def test_target_inverse_round_trips_both_transforms():
    rng = np.random.default_rng(0)
    d = 10.0 ** rng.uniform(np.log10(C.FLOOR_KM) + 0.1, 2, 500)
    assert np.allclose(T.inverse(T.forward(d, 'log10', C.FLOOR_KM), 'log10'),
                       d, rtol=1e-12)
    for delta in (1e-3, 6.5e-3, 0.125):
        z = T.forward(d, 'asinh', C.FLOOR_KM, delta)
        assert np.allclose(T.inverse(z, 'asinh', delta), d, rtol=1e-9)


def test_target_inverse_demands_delta_for_asinh():
    """asinh without a delta is not a transform, it is a guess -- and a wrong
    delta silently rescales every prediction in the 2x2."""
    _raises(ValueError, T.inverse, np.array([1.0]), 'asinh')
    _raises(ValueError, T.inverse, np.array([1.0]), 'log')


def test_target_inverse_may_return_negative_km_under_asinh():
    """"May return negative km under 'asinh'; that is the transform being odd,
    not a bug."  Pinned so nobody clamps it here instead of in to_log10_km."""
    out = T.inverse(np.array([-1.0, -0.5]), 'asinh', 1e-3)
    assert np.all(out < 0)
    clamped = T.to_log10_km(np.array([-1.0, -0.5]), 'asinh', C.FLOOR_KM, 1e-3)
    assert np.all(np.isfinite(clamped))
    assert np.all(clamped == np.log10(C.FLOOR_KM))


def test_count_parameters_counts_only_trainable_tensors():
    import torch
    from models import MLP, count_parameters
    m = MLP(4, hidden=(8, 8))
    expected = sum(p.numel() for p in m.parameters())
    assert count_parameters(m) == expected
    assert isinstance(count_parameters(m), int)

    # freezing a layer must reduce the count by exactly that layer's size
    first = next(iter(m.parameters()))
    n_first = first.numel()
    first.requires_grad_(False)
    assert count_parameters(m) == expected - n_first

    # a learn_sigma model carries exactly one extra scalar
    a = count_parameters(MLP(4, hidden=(8,), learn_sigma=False))
    b = count_parameters(MLP(4, hidden=(8,), learn_sigma=True))
    assert b - a == 1, (a, b)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
