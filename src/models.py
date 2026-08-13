"""Baseline MLP regressor on log10 distance-to-nearest-QPIT.

SMOOTH ACTIVATION ON PURPOSE.  The default is SiLU.  Prediction P3 for this
week is that the network cannot follow the target into the near field, because
grad(log d) ~ 1/d along the normal while the network's gradient is bounded by
the product of its weight norms.  Stating that prediction against a smooth
network makes it clean; a ReLU network is also Lipschitz but its kinks invite
the objection that the failure is an artefact of the activation.  `tanh` is
available for the same reason and should give the same qualitative answer --
if it does not, P3 is wrong and that is the interesting outcome.

TARGET IS STANDARDISED INSIDE, PREDICTIONS COME BACK IN log10 km.  Optimising a
target with mean ~0 and sd ~1 is a numerical convenience with no bearing on the
result; returning standardised predictions to a caller who then computes RMSE
would silently rescale every reported number, so the inverse transform is not
optional and lives here rather than in the notebook.

EARLY STOPPING ON A GROUP-DISJOINT VALIDATION SET (see `src/splits.py`).
Stopping on a row-random validation set selects the epoch that best memorises
training patches, which is exactly the quantity this week is trying not to
measure.

SEEDING.  `seed` fixes torch init, the shuffling RNG, and dropout.  Runs are
reproducible on the same machine and library build; they are NOT guaranteed
bitwise identical across BLAS versions, so multi-seed spread is a sanity band,
not a substitute for the Week 12 confidence-interval protocol.
"""

from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    """PATIENCE IS IN EPOCHS, AND AN EPOCH IS NOT A FIXED AMOUNT OF WORK.

    An earlier default of patience=30 stopped a fit after ~95 gradient steps on
    a 2.5k-row training set (five batches per epoch) and returned a model that
    was still, measurably, a mean predictor -- and it did so without any error,
    reporting `early_stopped=True` as though it had converged.  On a larger set
    the same number would have been harmless.  `min_epochs` now floors it and
    `hit_max_epochs` flags the opposite failure, so neither end is silent.
    """
    hidden: tuple = (256, 256, 256)
    activation: str = 'silu'          # 'silu' | 'tanh' | 'relu'
    dropout: float = 0.0
    lr: float = 2e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 600
    min_epochs: int = 60              # no early stop before this
    patience: int = 80                # epochs without val improvement
    min_delta: float = 1e-5
    lr_factor: float = 0.5            # ReduceLROnPlateau
    lr_patience: int = 25
    min_lr: float = 1e-5
    device: str = 'cpu'
    extra: dict = field(default_factory=dict)


_ACT = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU}


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 256, 256), activation='silu',
                 dropout=0.0):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _tensor(a, device):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def train_mlp(X_tr, y_tr, X_val, y_val, cfg=None, seed=0, verbose=False):
    """Fit one MLP.  X must already be standardised on TRAIN rows.

    Returns (model, history dict).  `history['y_mean'], ['y_std']` are the
    target scaling; `predict` needs them and applies them for you.
    """
    cfg = cfg or MLPConfig()
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

    model = MLP(Xt.shape[1], cfg.hidden, cfg.activation, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    lossf = nn.MSELoss()

    n = Xt.shape[0]
    best = np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, since = 0, 0
    hist = dict(train_loss=[], val_loss=[], lr=[])

    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), device=dev)
        tot = 0.0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xv), yv).item())
        hist['train_loss'].append(tot / n)
        hist['val_loss'].append(vl)
        hist['lr'].append(opt.param_groups[0]['lr'])
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
                y_mean=y_mean, y_std=y_std, seed=seed, config=asdict(cfg))
    if hist['hit_max_epochs'] and verbose:
        print("  WARNING: hit max_epochs without early stopping -- the budget, "
              "not convergence, ended this fit")
    return model, hist


def predict(model, X, hist, device='cpu'):
    """Predictions in the ORIGINAL target units (log10 km)."""
    model.eval()
    with torch.no_grad():
        z = model(_tensor(X, torch.device(device))).cpu().numpy()
    return z * hist['y_std'] + hist['y_mean']


def train_multi_seed(X_tr, y_tr, X_val, y_val, X_test, cfg=None,
                     seeds=(0, 1, 2), verbose=False):
    """Fit once per seed; return per-seed test predictions and histories.

    Three seeds is a SANITY BAND, not a confidence interval: it says whether
    "the baseline failed" survives re-initialisation.  The CI protocol
    (multi-seed with the hyperparameter sweep) is Week 12.
    """
    preds, hists = [], []
    for s in seeds:
        m, h = train_mlp(X_tr, y_tr, X_val, y_val, cfg=cfg, seed=s,
                         verbose=verbose)
        preds.append(predict(m, X_test, h, (cfg or MLPConfig()).device))
        hists.append(h)
        if verbose:
            print(f"seed {s}: best val {h['best_val_loss']:.5f} "
                  f"@ epoch {h['best_epoch']} ({h['epochs_run']} run)")
    return np.array(preds), hists


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
