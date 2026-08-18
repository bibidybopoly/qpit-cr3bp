"""Baseline MLP regressor on the distance-to-nearest-QPIT target.

SMOOTH ACTIVATION ON PURPOSE.  The default is SiLU.  Prediction P3 was that the
network cannot follow the target into the near field, because grad(log d) ~ 1/d
along the normal while the network's gradient is bounded by the product of its
weight norms.  Stating that against a smooth network makes it clean; a ReLU
network is also Lipschitz but its kinks invite the objection that the failure is
an artefact of the activation.  `tanh` is available for the same reason.

P3 HELD, AND WEEK 12 IS THE RESPONSE TO IT.  The Week 11 dead zone is a genuine
saturation: bias-versus-target slope -0.98 against a predicted -1 for a constant
output, a constant of ~6.5 m, and a boundary at ~125 m stable across three bin
resolutions.  That is 2.90 of the target's 5.81 decades.  Week 12 tests two
changes against it -- a bounded-slope target (`src/targets.py`) and a censored
likelihood (`src/losses.py`) -- as a 2x2 with this module held fixed.

TARGET IS STANDARDISED INSIDE, PREDICTIONS COME BACK IN THE TARGET'S OWN UNITS.
Optimising a target with mean ~0 and sd ~1 is a numerical convenience with no
bearing on the result; returning standardised predictions to a caller who then
computes RMSE would silently rescale every reported number, so the inverse
transform is not optional and lives here.

STANDARDISATION IS A FUNCTION OF THE TARGET, NOT OF THE LOSS.  y_mean and y_std
come from ALL training rows including censored ones, under every loss.
Excluding censored rows would be defensible on its own terms and is wrong here:
it would make the scaling depend on which loss is running, so within a column of
the 2x2 the target would rescale when only the loss was meant to change.  The
censored pile-up costs the same conditioning in all four cells, so it does not
confound the comparison.

z_L IS STANDARDISED HERE, AND ONLY HERE.  `train_mlp` takes the threshold in raw
target units and builds the loss itself.  Letting a caller construct
`CensoredGaussianNLL` and pass it in would put the raw threshold beyond the
reach of the only function that knows y_mean and y_std, and an unstandardised
z_L does not raise -- it produces a large constant penalty and a permanent
downward gradient on every censored row, and reads as a bad result.

EARLY STOPPING ON A GROUP-DISJOINT VALIDATION SET (see `src/splits.py`).
Stopping on a row-random validation set selects the epoch that best memorises
training patches.  The validation loss uses the SAME loss the fit is minimising;
stopping a Tobit fit on MSE would select the wrong epoch.  One consequence:
`best_val_loss` is not comparable across cells with different losses, so cells
are compared on test metrics in a common space, never on history.

SEEDING.  `seed` fixes torch init, the shuffling RNG, and dropout.  Runs are
reproducible on the same machine and library build; they are NOT guaranteed
bitwise identical across BLAS versions.  Week 11's frozen numbers used three
seeds, which gives SEM ~ 0.012 against an observed spread of 0.020 -- too coarse
to separate the 2x2 cells.  Week 12 uses five.  That is still a seed band, not
the CI protocol; the sweep it was once bundled with is parked.
"""

from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from losses import CensoredGaussianNLL, check_censoring_consistency

LOSSES = ('mse', 'tobit')


@dataclass
class MLPConfig:
    """PATIENCE IS IN EPOCHS, AND AN EPOCH IS NOT A FIXED AMOUNT OF WORK.

    An earlier default of patience=30 stopped a fit after ~95 gradient steps on
    a 2.5k-row training set (five batches per epoch) and returned a model that
    was still, measurably, a mean predictor -- and it did so without any error,
    reporting `early_stopped=True` as though it had converged.  On a larger set
    the same number would have been harmless.  `min_epochs` now floors it and
    `hit_max_epochs` flags the opposite failure, so neither end is silent.

    `log_sigma_init` lives here rather than as a call argument so it lands in
    `hist['config']` and is recoverable from a results file.  The default of 0.0
    is sigma = 1 in standardised units, i.e. the residual scale of a model that
    explains nothing.  That OVERSTATES sigma for a fit like M1 (R^2 ~ 0.63 puts
    the residual sd near 0.61), and overstating it is the safe direction: a
    larger sigma softens the censored branch early, when mu is still far from
    the floor.  Pass 0.5 * log(M1 best_val_loss) to start it where the data is.

    THE STOPPING DEFAULTS ARE THE ONES THE FROZEN RESULTS WERE PRODUCED WITH.
    They were previously 600 / 80 / 25, which is what the RETRACTED Week 11 run
    used: all three seeds ended at the 600-epoch ceiling (best 589/598/595) with
    validation loss still falling, so its R^2 was a lower bound and its seed
    spread (0.17) exceeded the M1-B1 gap (0.13).  nb11 cell 20 overrode them to
    3000 / 150 / 50 and the frozen numbers come from that override; nb12 had to
    carry the same override forward in `CFG_BASE`.  Leaving the budget-limited
    values as the defaults meant a bare `MLPConfig()` silently reproduced the
    retracted configuration and reported `early_stopped=True` as though it had
    converged.  The defaults now match the frozen runs, so the notebook
    overrides are no-ops and the failure mode is gone.  Architecture, lr and
    batch size are unchanged -- only the stopping rule moved.
    """
    hidden: tuple = (256, 256, 256)
    activation: str = 'silu'          # 'silu' | 'tanh' | 'relu'
    dropout: float = 0.0
    lr: float = 2e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 3000            # was 600 -- budget-limited, see above
    min_epochs: int = 60              # no early stop before this
    patience: int = 150               # was 80; epochs without val improvement
    min_delta: float = 1e-5
    lr_factor: float = 0.5            # ReduceLROnPlateau
    lr_patience: int = 50             # was 25
    min_lr: float = 1e-5
    log_sigma_init: float = 0.0       # tobit only; see above
    device: str = 'cpu'
    extra: dict = field(default_factory=dict)


_ACT = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU}


class MLP(nn.Module):
    """`learn_sigma=False` leaves this module byte-identical to Week 11 --
    same submodules, same state_dict keys -- so M1 and M2a share it exactly and
    the target change is the only difference between them."""

    def __init__(self, n_in, hidden=(256, 256, 256), activation='silu',
                 dropout=0.0, learn_sigma=False, log_sigma_init=0.0):
        super().__init__()
        if activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), _ACT[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        if learn_sigma:
            # A single global scale, not a second head.  Heteroscedastic
            # sigma(x) is a real improvement and a known early-training
            # instability -- the model buys loss by inflating sigma on hard
            # rows -- so it is deliberately not this week.
            self.log_sigma = nn.Parameter(
                torch.tensor(float(log_sigma_init), dtype=torch.float32))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _tensor(a, device):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def _mask(a, n, device, name):
    if a is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    m = np.asarray(a).reshape(-1)
    if m.dtype != bool:
        raise TypeError(f"{name} must be a boolean array, got {m.dtype}")
    if m.size != n:
        raise ValueError(f"{name} has {m.size} rows, expected {n}")
    return torch.as_tensor(m, dtype=torch.bool, device=device)


def _param_groups(model, weight_decay):
    """`log_sigma` gets its own group at zero decay.

    Adam applies weight_decay to everything in `model.parameters()`, and decay
    on log_sigma is a silent prior pulling sigma toward 1 in standardised units.
    Small, invisible, and it would bias exactly the parameter the censored
    branch is most sensitive to.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name == 'log_sigma' else decay).append(p)
    groups = [dict(params=decay, weight_decay=weight_decay)]
    if no_decay:
        groups.append(dict(params=no_decay, weight_decay=0.0))
    return groups


def train_mlp(X_tr, y_tr, X_val, y_val, cfg=None, seed=0, verbose=False,
              loss='mse', z_L=None, c_tr=None, c_val=None):
    """Fit one MLP.  X must already be standardised on TRAIN rows.

    `loss='mse'` with the default masks reproduces the Week 11 baseline exactly.

    `loss='tobit'` additionally requires `z_L`, the left-censoring threshold in
    RAW target units -- `targets.threshold_in_target_units(...)` -- and the
    boolean masks `c_tr` / `c_val` marking clamped rows (the dataset's
    `at_floor` column, split the same way as the rows).  z_L is standardised
    here; do not pre-standardise it.

    Returns (model, history dict).  `history['y_mean'], ['y_std']` are the
    target scaling; `predict` needs them and applies them for you.
    """
    cfg = cfg or MLPConfig()
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(cfg.device)

    y_tr = np.asarray(y_tr, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_std = y_std if y_std > 0 else 1.0

    Xt, Xv = _tensor(X_tr, dev), _tensor(X_val, dev)
    yt = _tensor((y_tr - y_mean) / y_std, dev)
    yv = _tensor((y_val - y_mean) / y_std, dev)
    ct = _mask(c_tr, yt.numel(), dev, 'c_tr')
    cv = _mask(c_val, yv.numel(), dev, 'c_val')

    if loss == 'tobit':
        if z_L is None:
            raise ValueError("loss='tobit' requires z_L in raw target units")
        z_L_std = (float(z_L) - y_mean) / y_std
        # Fails loudly if z_L and the target disagree about units, or if an
        # asinh target was built without the floor clamp.  Once per fit.
        check_censoring_consistency(yt, z_L_std, ct)
        check_censoring_consistency(yv, z_L_std, cv)
        lossf = CensoredGaussianNLL(z_L=z_L_std)
    else:
        z_L_std = None
        lossf = nn.MSELoss()
    needs_sigma = getattr(lossf, 'needs_sigma', False)

    model = MLP(Xt.shape[1], cfg.hidden, cfg.activation, cfg.dropout,
                learn_sigma=needs_sigma,
                log_sigma_init=cfg.log_sigma_init).to(dev)
    opt = torch.optim.Adam(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)

    def _loss(pred, target, mask):
        if needs_sigma:
            return lossf(pred, target, model.log_sigma, mask)
        return lossf(pred, target)

    n = Xt.shape[0]
    best = np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, since = 0, 0
    hist = dict(train_loss=[], val_loss=[], lr=[], log_sigma=[])

    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        tot = 0.0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            # ct is indexed by the SAME idx as Xt and yt -- if these ever fall
            # out of step the censored rows are labelled by another row's mask
            # and nothing raises.  tests/test_models.py asserts the alignment.
            l = _loss(model(Xt[idx]), yt[idx], ct[idx])
            l.backward()
            opt.step()
            tot += l.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(_loss(model(Xv), yv, cv).item())
        hist['train_loss'].append(tot / n)
        hist['val_loss'].append(vl)
        hist['lr'].append(opt.param_groups[0]['lr'])
        hist['log_sigma'].append(
            float(model.log_sigma) if needs_sigma else float('nan'))
        sched.step(vl)

        if vl < best - cfg.min_delta:
            best, best_epoch, since = vl, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience and ep + 1 >= cfg.min_epochs:
                break
        if verbose and ep % 50 == 0:
            print(f"  ep {ep:4d}  train {tot/n:.5f}  val {vl:.5f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    model.eval()
    ran = len(hist['val_loss'])
    hist.update(best_epoch=best_epoch, best_val_loss=best, epochs_run=ran,
                early_stopped=(since >= cfg.patience and ran >= cfg.min_epochs),
                hit_max_epochs=(ran >= cfg.max_epochs),
                y_mean=y_mean, y_std=y_std, seed=seed, config=asdict(cfg),
                loss=loss, z_L=None if z_L is None else float(z_L),
                z_L_std=z_L_std,
                log_sigma_final=(float(model.log_sigma) if needs_sigma
                                 else float('nan')),
                censored_frac_train=float(ct.float().mean()),
                censored_frac_val=float(cv.float().mean()))
    if hist['hit_max_epochs'] and verbose:
        print("  WARNING: hit max_epochs without early stopping -- the budget, "
              "not convergence, ended this fit")
    return model, hist


def predict(model, X, hist, device='cpu'):
    """Predictions in the ORIGINAL target units.

    That is log10 km on the log10 cells and asinh units on the asinh cells --
    NOT a common space.  Route through `targets.to_log10_km` before any metric
    that compares cells; R^2 on two different targets is not one number.
    """
    model.eval()
    with torch.no_grad():
        z = model(_tensor(X, torch.device(device))).cpu().numpy()
    return z * hist['y_std'] + hist['y_mean']


def train_multi_seed(X_tr, y_tr, X_val, y_val, X_test, cfg=None,
                     seeds=(0, 1, 2), verbose=False, **kw):
    """Fit once per seed; return per-seed test predictions and histories.

    The default of three is Week 11's, kept so a bare call still reproduces the
    frozen baseline.  Week 12 passes seeds explicitly: three gives SEM ~ 0.012
    against a spread of 0.020, which cannot resolve the 2x2 cells.  Five can.
    `**kw` forwards `loss`, `z_L`, `c_tr`, `c_val` to `train_mlp`.
    """
    preds, hists = [], []
    for s in seeds:
        m, h = train_mlp(X_tr, y_tr, X_val, y_val, cfg=cfg, seed=s,
                         verbose=verbose, **kw)
        preds.append(predict(m, X_test, h, (cfg or MLPConfig()).device))
        hists.append(h)
        if verbose:
            print(f"seed {s}: best val {h['best_val_loss']:.5f} "
                  f"@ epoch {h['best_epoch']} ({h['epochs_run']} run)"
                  + (f"  log_sigma {h['log_sigma_final']:+.4f}"
                     if h['loss'] == 'tobit' else ''))
    return np.array(preds), hists


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
