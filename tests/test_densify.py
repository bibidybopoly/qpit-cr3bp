"""Tests for src/densify.py — family densification and the family archive.

Run: pytest tests/test_densify.py -v   (or python tests/test_densify.py)

`interp_error_km`, `required_gap_km` and `load_family` were reached from
test_torus_field.py; `densify_once`, `densify` and `save_family` were not
reached at all.

`densify_once` calls `probe_gap`, which is Newton per gap and slow, so the
tests here exercise the paths that do NOT need it: the error model, the archive
round-trip (whose whole subtlety is ragged N), the no-op path, and the report
the caller reads to decide whether the bound was met.  A gap-bisecting run is
covered through a stub probe rather than by paying for a real one.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import densify as D                                            # noqa: E402
from densify import (LU, GAP_COEFF_PER_KM, interp_error_km,     # noqa: E402
                     required_gap_km, densify_once, densify,
                     save_family, load_family)


def _member(A_km, N=15, seed=0):
    """A family record with the right shape.  Amplitude is what densify sorts
    and bisects on; the curve itself is only carried through."""
    rng = np.random.default_rng(seed)
    th = 2 * np.pi * np.arange(N) / N
    U = np.column_stack([1.04 + (A_km / LU) * np.cos(th),
                         (A_km / LU) * np.sin(th),
                         -0.075 + 0.5 * (A_km / LU) * np.sin(th),
                         0.1 * (A_km / LU) * np.cos(th),
                         0.392 + 0.2 * (A_km / LU) * np.sin(th),
                         0.05 * (A_km / LU) * np.cos(th)])
    U = U + 1e-12 * rng.standard_normal(U.shape)
    return dict(U0=U, T=3.13, rho=1.842, C=3.0603, amplitude=A_km / LU)


# ---------------------------------------------------------------------------
# 1. The error model
# ---------------------------------------------------------------------------

def test_error_model_reproduces_the_documented_table():
    """The module docstring commits to specific numbers; pin them.

    g = 12.3 km -> 10 cm is the value the Week 10 densification target was
    chosen from, and it is quoted in nb10 and in the labelling discussion.  If
    GAP_COEFF_PER_KM is ever re-measured, these move together and the docstring
    has to move with them.
    """
    assert abs(interp_error_km(12.3) - 1e-4) < 5e-6      # ~10 cm
    assert abs(interp_error_km(10.0) - 6.7e-5) < 2e-6    # ~6.7 cm
    assert abs(interp_error_km(5.0) - 1.7e-5) < 1e-6     # ~1.7 cm
    assert abs(GAP_COEFF_PER_KM - 2.6e-3 / 62.5 ** 2) < 1e-18


def test_error_model_is_quadratic_in_the_gap():
    """Exponent 2.003 was FITTED, and the whole densification argument (halving
    a gap cuts its error by four) rests on it being 2."""
    g = np.array([1.0, 2.0, 4.0, 8.0])
    e = interp_error_km(g)
    assert np.allclose(e[1:] / e[:-1], 4.0, rtol=1e-12)
    assert interp_error_km(0.0) == 0.0


def test_required_gap_inverts_the_error_model():
    for target in (1e-5, 1e-4, 1e-3, 1e-2):
        g = required_gap_km(target)
        assert abs(interp_error_km(g) - target) < 1e-15 * max(1.0, target)
    assert isinstance(required_gap_km(1e-4), float)


def test_interp_error_is_vectorised():
    out = interp_error_km([1.0, 2.0, 3.0])
    assert isinstance(out, np.ndarray) and out.shape == (3,)


# ---------------------------------------------------------------------------
# 2. The archive — ragged N is the whole point
# ---------------------------------------------------------------------------

def test_family_archive_round_trips_with_ragged_curve_resolutions():
    """N is NOT constant across a family: continue_torus_family adapts the
    Fourier resolution as it marches, so the curve list is ragged and a plain
    np.array() on it raises.  This is the documented reason `save_family`
    builds an object array by hand, and the one thing that would silently
    break the archive."""
    mem = [_member(10.0, N=15), _member(20.0, N=31), _member(30.0, N=63)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fam.npz')
        save_family(p, mem, 0.0121505856)
        back = load_family(p)

    assert len(back) == 3
    assert [m['U0'].shape[0] for m in back] == [15, 31, 63]
    for a, b in zip(mem, back):
        assert np.allclose(a['U0'], b['U0'], atol=0, rtol=0)
        assert a['T'] == b['T'] and a['rho'] == b['rho']
        assert a['C'] == b['C'] and a['amplitude'] == b['amplitude']


def test_archive_records_mu_and_extra_columns():
    mem = [_member(10.0), _member(20.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fam.npz')
        save_family(p, mem, 0.0121505856, energy_id=2, label='L2_E0')
        z = np.load(p, allow_pickle=True)
        assert float(z['mu']) == 0.0121505856
        assert int(z['energy_id']) == 2
        assert str(z['label']) == 'L2_E0'
        assert list(z['N']) == [15, 15]


def test_loaded_family_is_sorted_by_amplitude_when_saved_that_way():
    """`surface_distance` documents that members must be sorted by amplitude."""
    mem = [_member(a) for a in (5.0, 15.0, 40.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fam.npz')
        save_family(p, mem, 0.012)
        A = [m['amplitude'] for m in load_family(p)]
    assert A == sorted(A)


# ---------------------------------------------------------------------------
# stub probe — keeps the bisection tests off the Newton path
# ---------------------------------------------------------------------------

def _install_stub_probe():
    """Replace `probe_gap` with a bookkeeping stub.

    The tests above are about WHICH gaps get bisected and in what order, not
    about whether the corrector converges -- torus_probe has its own tests for
    that.  The stub records each call and returns the midpoint amplitude.
    """
    calls = []
    real = D.probe_gap

    def stub(rec_a, rec_b, mu, alphas, **kw):
        gap_km = (rec_b['amplitude'] - rec_a['amplitude']) * LU
        calls.append(dict(gap_km=round(gap_km, 9)))
        mid = 0.5 * (rec_a['amplitude'] + rec_b['amplitude'])
        res = dict(rec_a)
        res['amplitude'] = mid
        return [dict(converged=True, code='ok', result=res)]

    D.probe_gap = stub
    return calls, real


# ---------------------------------------------------------------------------
# 3. densify_once — the parts that do not need a real probe
# ---------------------------------------------------------------------------

def test_densify_once_is_a_no_op_when_every_gap_is_already_small_enough():
    """Nothing to bisect means no probe calls at all -- and the report still
    has to describe the family truthfully, because that report is what the
    caller checks the error bound against."""
    mem = [_member(a) for a in (10.0, 20.0, 30.0)]
    out, rep = densify_once(mem, 0.012, target_gap_km=50.0, verbose=False)

    assert len(out) == 3
    assert rep['n_added'] == 0 and rep['n_failed'] == 0
    assert rep['n_before'] == 3
    assert abs(rep['max_gap_km'] - 10.0) < 1e-9
    assert abs(rep['error_km'] - interp_error_km(10.0)) < 1e-18


def test_densify_once_sorts_by_amplitude_regardless_of_input_order():
    mem = [_member(a) for a in (30.0, 10.0, 20.0)]
    out, _ = densify_once(mem, 0.012, target_gap_km=1e9, verbose=False)
    A = [m['amplitude'] for m in out]
    assert A == sorted(A)


def test_densify_once_normalises_records_to_plain_floats_and_arrays():
    mem = [dict(U0=_member(10.0)['U0'].tolist(), T=np.float64(3.13),
                rho=np.float64(1.842), C=np.float64(3.06),
                amplitude=np.float64(10.0 / LU)),
           _member(20.0)]
    out, _ = densify_once(mem, 0.012, target_gap_km=1e9, verbose=False)
    for m in out:
        assert isinstance(m['U0'], np.ndarray) and m['U0'].dtype == float
        assert type(m['T']) is float and type(m['amplitude']) is float


def test_densify_once_bisects_the_widest_gaps_first():
    """Bisection order matters: error falls as g^2, so the widest gap dominates
    the bound and `max_new` must spend its budget there, not on the first gap
    in index order."""
    calls, real = _install_stub_probe()
    try:
        mem = [_member(a) for a in (0.0, 5.0, 100.0, 110.0, 400.0)]
        out, rep = densify_once(mem, 0.012, target_gap_km=10.0, max_new=2,
                                verbose=False)
    finally:
        D.probe_gap = real
    assert rep['n_added'] == 2
    widths = sorted([c['gap_km'] for c in calls], reverse=True)
    assert widths == [290.0, 95.0], widths        # not 5.0 or 10.0
    assert len(out) == 7


def test_densify_stops_once_a_pass_adds_nothing():
    calls, real = _install_stub_probe()
    try:
        mem = [_member(a) for a in (10.0, 20.0, 30.0)]
        out, rep = densify(mem, 0.012, target_gap_km=100.0, max_passes=8,
                           verbose=False)
    finally:
        D.probe_gap = real
    assert rep['n_added'] == 0
    assert len(calls) == 0, "no gap was wide enough to probe"
    assert len(out) == 3


def test_densify_checkpoints_every_pass_and_resumes_from_one():
    calls, real = _install_stub_probe()
    mem = [_member(a) for a in (10.0, 20.0)]
    try:
      with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'ck.npz')
        densify(mem, 0.012, target_gap_km=100.0, max_passes=2,
                checkpoint=p, verbose=False)
        assert os.path.exists(p), "checkpoint was not written"
        assert len(load_family(p)) == 2

        # a second call with a checkpoint present must resume from it, not
        # from the members it was handed
        out, _ = densify([_member(999.0)], 0.012, target_gap_km=100.0,
                         max_passes=1, checkpoint=p, verbose=False)
    finally:
        D.probe_gap = real
    assert len(out) == 2, "densify ignored the checkpoint it was given"


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
