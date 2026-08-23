"""Tests for src/models.py.  Runs standalone.

There was no test file for this module before Week 12.  These cover the wiring
the Tobit path adds, chosen on one criterion: every failure mode here is one
that completes without raising and reads as a result.  A mis-indexed censoring
mask, an unstandardised threshold, and weight decay on log_sigma all produce a
finished run with a plausible-looking number.

Fits use a tiny config throughout; nothing here is testing convergence.
"""

import sys, os, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import models as M
from models import MLP, MLPConfig, train_mlp, predict, _param_groups

from constants import FLOOR_KM
Z_L = math.log10(FLOOR_KM)          # raw target units, log10 km


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}, none raised")


def tiny(**kw):
    base = dict(hidden=(8,), batch_size=16, max_epochs=3, min_epochs=1,
                patience=99, lr=1e-2)
    base.update(kw)
    return MLPConfig(**base)


def toy(n=96, n_feat=4, censored_frac=0.25, seed=0):
    """Targets in log10 km with a clamped block, mirroring the dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    y = rng.uniform(Z_L, 2.0, n)
    c = np.zeros(n, dtype=bool)
    c[rng.choice(n, int(n * censored_frac), replace=False)] = True
    y[c] = Z_L                       # clamped rows carry exactly the threshold
    return X, y, c


# =========================================================================
# M1 IS UNTOUCHED
# =========================================================================

def test_mse_path_has_no_sigma_parameter():
    """M1 and M2a must share the module exactly, so state_dict keys cannot
    gain one."""
    a = set(MLP(4, hidden=(8,)).state_dict())
    assert a == set(MLP(4, hidden=(8,), learn_sigma=False).state_dict())
    assert not any('log_sigma' in k for k in a)
    assert 'log_sigma' in MLP(4, hidden=(8,), learn_sigma=True).state_dict()


def test_mse_path_ignores_masks_and_reports_no_threshold():
    X, y, c = toy()
    _, h = train_mlp(X, y, X, y, cfg=tiny(), loss='mse', c_tr=c, c_val=c)
    assert h['loss'] == 'mse' and h['z_L_std'] is None
    assert math.isnan(h['log_sigma_final'])


def test_mse_fits_are_deterministic_for_a_seed():
    X, y, _ = toy()
    m1, h1 = train_mlp(X, y, X, y, cfg=tiny(), seed=3)
    m2, h2 = train_mlp(X, y, X, y, cfg=tiny(), seed=3)
    assert np.allclose(predict(m1, X, h1), predict(m2, X, h2))


# =========================================================================
# THE STANDARDISATION PATH
# =========================================================================

def test_z_L_is_standardised_with_the_target_scaling():
    X, y, c = toy()
    _, h = train_mlp(X, y, X, y, cfg=tiny(), loss='tobit', z_L=Z_L,
                     c_tr=c, c_val=c)
    assert abs(h['z_L'] - Z_L) < 1e-12
    assert abs(h['z_L_std'] - (Z_L - h['y_mean']) / h['y_std']) < 1e-12


def test_standardisation_uses_all_rows_not_just_uncensored():
    """Deliberate: the scaling must depend on the target alone, so that within
    a column of the 2x2 only the loss changes."""
    X, y, c = toy()
    _, h = train_mlp(X, y, X, y, cfg=tiny(), loss='tobit', z_L=Z_L,
                     c_tr=c, c_val=c)
    assert abs(h['y_mean'] - float(y.mean())) < 1e-12
    assert abs(h['y_std'] - float(y.std())) < 1e-12
    assert abs(h['y_mean'] - float(y[~c].mean())) > 1e-6


def test_unstandardised_threshold_would_be_caught():
    """The bug the guard exists for, reached the only way it still can -- a
    target whose censored rows do not sit on the declared threshold."""
    X, y, c = toy()
    y[c] = Z_L + 0.5                        # clamp value and z_L disagree
    _raises(ValueError, train_mlp, X, y, X, y, cfg=tiny(), loss='tobit',
            z_L=Z_L, c_tr=c, c_val=c)


def test_tobit_without_threshold_raises():
    X, y, c = toy()
    _raises(ValueError, train_mlp, X, y, X, y, cfg=tiny(), loss='tobit',
            c_tr=c, c_val=c)


def test_bad_loss_name_raises():
    X, y, _ = toy()
    _raises(ValueError, train_mlp, X, y, X, y, cfg=tiny(), loss='tobin')


# =========================================================================
# MASK ALIGNMENT -- THE SILENT ONE
# =========================================================================

def test_censoring_mask_stays_aligned_under_shuffling():
    """A mis-indexed mask labels each row with another row's censoring status
    and nothing raises.  The invariant that catches it: within any batch, every
    row the mask calls censored must carry exactly z_L.  Asserted on every
    batch of every epoch by substituting a spy for the loss class.
    """
    seen = {'batches': 0}
    real = M.CensoredGaussianNLL

    class Spy(real):
        def forward(self, mu, z_obs, log_sigma, censored):
            if censored.any():
                dev = float((z_obs[censored] - self.z_L).abs().max())
                assert dev < 1e-5, f"mask/target misalignment, dev={dev:.3g}"
            seen['batches'] += 1
            return super().forward(mu, z_obs, log_sigma, censored)

    M.CensoredGaussianNLL = Spy
    try:
        X, y, c = toy(n=96, censored_frac=0.3)
        train_mlp(X, y, X, y, cfg=tiny(max_epochs=3), loss='tobit', z_L=Z_L,
                  c_tr=c, c_val=c)
    finally:
        M.CensoredGaussianNLL = real
    assert seen['batches'] >= 3 * (96 // 16)


def test_mask_length_and_dtype_are_checked():
    X, y, c = toy()
    _raises(ValueError, train_mlp, X, y, X, y, cfg=tiny(), loss='tobit',
            z_L=Z_L, c_tr=c[:-1], c_val=c)
    _raises(TypeError, train_mlp, X, y, X, y, cfg=tiny(), loss='tobit',
            z_L=Z_L, c_tr=c, c_val=c.astype(int))


# =========================================================================
# SIGMA
# =========================================================================

def test_log_sigma_is_in_a_zero_decay_group():
    m = MLP(4, hidden=(8,), learn_sigma=True)
    groups = _param_groups(m, weight_decay=1e-5)
    assert len(groups) == 2 and groups[1]['weight_decay'] == 0.0
    assert len(groups[1]['params']) == 1
    assert groups[1]['params'][0] is m.log_sigma
    assert all(p is not m.log_sigma for p in groups[0]['params'])


def test_mse_path_makes_one_group_only():
    """An empty second group is avoided rather than passed to Adam."""
    assert len(_param_groups(MLP(4, hidden=(8,)), 1e-5)) == 1


def test_log_sigma_is_trained_and_recorded():
    X, y, c = toy()
    _, h = train_mlp(X, y, X, y, cfg=tiny(max_epochs=5, log_sigma_init=0.5),
                     loss='tobit', z_L=Z_L, c_tr=c, c_val=c)
    assert len(h['log_sigma']) == h['epochs_run']
    assert math.isfinite(h['log_sigma_final'])
    assert abs(h['log_sigma_final'] - 0.5) > 1e-9
    assert h['config']['log_sigma_init'] == 0.5


# =========================================================================
# HISTORY AND PREDICTIONS
# =========================================================================

def test_best_val_loss_is_the_tobit_nll_and_not_mse():
    """Stopping a Tobit fit on MSE selects the wrong epoch.

    FOUND BY MUTATION.  The first version of this only asserted that a Tobit
    run and an MSE run report DIFFERENT val losses -- which stays true when the
    validation loss is swapped to MSE, because a Tobit-trained model scored
    with MSE still differs from an MSE-trained one.  It survived the mutation.
    `train_mlp` returns the best-epoch state, so `best_val_loss` must equal the
    training loss recomputed on that exact model.  Checking which of the two it
    equals is the assertion that has teeth.
    """
    import torch
    from losses import CensoredGaussianNLL
    X, y, c = toy(censored_frac=0.4)
    m, h = train_mlp(X, y, X, y, cfg=tiny(), loss='tobit', z_L=Z_L,
                     c_tr=c, c_val=c)
    yz = torch.as_tensor((y - h['y_mean']) / h['y_std'], dtype=torch.float32)
    with torch.no_grad():
        mu = m(torch.as_tensor(X, dtype=torch.float32))
        tob = float(CensoredGaussianNLL(z_L=h['z_L_std'])(
            mu, yz, m.log_sigma, torch.as_tensor(c)))
        mse = float(torch.nn.MSELoss()(mu, yz))
    assert abs(h['best_val_loss'] - tob) < 1e-5
    assert abs(h['best_val_loss'] - mse) > 1e-3


def test_a_mean_predictor_fit_is_flagged():
    """Wk12 M2b seed 2 collapsed, reported early_stopped=True, and was averaged
    into a five-seed band as a fit.  Forced here with a lr of 0 so the model
    never leaves initialisation."""
    X, y, c = toy()
    _, h = train_mlp(X, y, X, y, cfg=tiny(lr=0.0, max_epochs=2),
                     loss='tobit', z_L=Z_L, c_tr=c, c_val=c)
    assert h['is_mean_predictor'] is True
    assert h['val_r2'] < 0.05


def test_a_real_fit_is_not_flagged():
    """The flag must not fire on an ordinary short fit, or it is useless."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3))
    y = 2.0 * X[:, 0] - X[:, 1]            # trivially learnable
    _, h = train_mlp(X, y, X, y, cfg=tiny(max_epochs=200, min_epochs=50))
    assert h['val_r2'] > 0.5
    assert h['is_mean_predictor'] is False


def test_censored_fractions_are_recorded():
    X, y, c = toy(n=100, censored_frac=0.25)
    _, h = train_mlp(X, y, X, y, cfg=tiny(), loss='tobit', z_L=Z_L,
                     c_tr=c, c_val=c)
    assert abs(h['censored_frac_train'] - c.mean()) < 1e-6


def test_predictions_come_back_in_target_units():
    """A standardised return would sit near zero mean and unit sd; these
    targets run from log10(FLOOR) to 2, so the scales are distinguishable."""
    X, y, _ = toy()
    m, h = train_mlp(X, y, X, y, cfg=tiny())
    p = predict(m, X, h)
    assert p.shape == y.shape and np.isfinite(p).all()
    assert abs(p.mean() - y.mean()) < abs(y.mean())


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for name, fn in fns:
        fn(); ok += 1; print(f"  ok  {name}")
    print(f"{ok}/{len(fns)} passed")
