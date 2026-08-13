"""Metrics for the Week 11 regressor, and the two curves that are the
week's deliverable.

ON R^2.  The target `log_d_km` was drawn LOG-UNIFORM by design (see
`src/sampling.py`), so the variance R^2 is measured against is a property of a
sampling choice, not of phase space.  Widening the drawn range would raise R^2
without the model improving.  R^2 is therefore reported for comparability and
the BASELINES are the meaningful comparison; a bare R^2 does not belong in the
abstract.

ON THE THRESHOLD CURVE.  Research Question row 9 specifies classification
recovered by thresholding the regression: true class (d < tau), predicted class
(d_hat < tau), agreement swept over tau.  This DEGENERATES TO 1 at both ends --
as tau leaves the data range every sample is one class and any predictor is
right.  Quoting the mean or the max over tau is therefore meaningless; the
honest scalar is the MINIMUM over tau, and it needs balanced accuracy or MCC
alongside because the class prior at each tau is set by the log-uniform draw
rather than by anything physical.

ON THE ERROR-TOLERANCE CURVE.  Fraction of samples with |y_hat - y| <= tol, in
log10 km.  This is the curve the warm-start story wants: an RMSE of 0.2 in
log10 is a typical factor-of-1.6 error in the distance handed to a solver, and
that framing is legible in a way R^2 is not.
"""

import numpy as np


def regression_metrics(y_true, y_pred, reference_mean=None):
    """R^2, RMSE, MAE, median |err|, and signed bias.

    R^2 is against the TEST-set mean (the sklearn convention), so a predictor
    emitting exactly the test mean scores 0 and one emitting the TRAIN mean
    scores slightly below 0.  Pass `reference_mean` to score against a fixed
    baseline instead; both are reported when it is given.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    r = p - y
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = dict(
        n=int(y.size),
        r2=float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        rmse=float(np.sqrt(np.mean(r ** 2))),
        mae=float(np.mean(np.abs(r))),
        medae=float(np.median(np.abs(r))),
        bias=float(np.mean(r)),
    )
    if reference_mean is not None:
        ss_ref = float(np.sum((y - reference_mean) ** 2))
        out['r2_vs_reference'] = float('nan') if ss_ref == 0 \
            else 1.0 - ss_res / ss_ref
    return out


def _confusion(true_pos, pred_pos):
    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))
    return tp, tn, fp, fn


def mcc(tp, tn, fp, fn):
    """Matthews correlation.  0 when a margin is empty -- the convention that
    scores a constant predictor at chance rather than leaving a NaN."""
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float(num / den)


def balanced_accuracy(tp, tn, fp, fn):
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return float(np.nanmean([tpr, tnr]))


def accuracy_vs_threshold(y_true, y_pred, taus=None, n_tau=60):
    """THE deliverable curve.  Classification recovered by thresholding.

    `taus` are in the units of the target (log10 km).  Default sweeps the
    observed range of `y_true`.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if taus is None:
        taus = np.linspace(y.min(), y.max(), n_tau)
    taus = np.asarray(taus, dtype=float)

    acc, bal, mc, prior, pred_rate = [], [], [], [], []
    for t in taus:
        tp, tn, fp, fn = _confusion(y < t, p < t)
        acc.append((tp + tn) / y.size)
        bal.append(balanced_accuracy(tp, tn, fp, fn))
        mc.append(mcc(tp, tn, fp, fn))
        prior.append(float(np.mean(y < t)))
        pred_rate.append(float(np.mean(p < t)))
    return dict(tau=taus, accuracy=np.array(acc),
                balanced_accuracy=np.array(bal), mcc=np.array(mc),
                positive_rate_true=np.array(prior),
                positive_rate_pred=np.array(pred_rate))


def summarize_threshold_curve(curve, prior_lo=0.05, prior_hi=0.95):
    """Collapse the curve to quotable scalars.

    Restricted to taus where the true positive rate lies in [prior_lo,
    prior_hi], because outside that window the classes have collapsed and the
    accuracy is an artefact of the prior rather than a measure of anything.
    """
    m = ((curve['positive_rate_true'] >= prior_lo)
         & (curve['positive_rate_true'] <= prior_hi))
    if not m.any():
        return dict(n_tau_informative=0)
    i = int(np.argmin(curve['balanced_accuracy'][m]))
    idx = np.flatnonzero(m)[i]
    return dict(
        n_tau_informative=int(m.sum()),
        tau_range=(float(curve['tau'][m].min()), float(curve['tau'][m].max())),
        min_balanced_accuracy=float(curve['balanced_accuracy'][idx]),
        tau_at_min=float(curve['tau'][idx]),
        accuracy_at_min=float(curve['accuracy'][idx]),
        min_mcc=float(np.min(curve['mcc'][m])),
        median_balanced_accuracy=float(np.median(curve['balanced_accuracy'][m])),
    )


def accuracy_vs_error_tolerance(y_true, y_pred, tols=None, n_tol=60):
    """Fraction with |y_hat - y| <= tol.  Monotone non-decreasing in tol."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    err = np.abs(p - y)
    if tols is None:
        tols = np.linspace(0.0, max(err.max(), 1e-12), n_tol)
    tols = np.asarray(tols, dtype=float)
    frac = np.array([float(np.mean(err <= t)) for t in tols])
    return dict(tol=tols, fraction_within=frac)


def residual_bins(y_true, y_pred, edges=None, n_bins=12):
    """Bias and RMSE as a function of the TARGET's own magnitude.

    This is the P3 check: grad(log d) ~ 1/d in the normal direction, so the
    target's Lipschitz constant is unbounded as d -> 0 while a smooth-activation
    MLP has bounded gradients.  The prediction is heteroscedastic residuals with
    a SYSTEMATIC POSITIVE BIAS in the smallest-d bins -- the network cannot
    follow the target down and over-predicts distance near the surface.

    Binning is on the target, not on phase-space location.  The phase-space
    error map is Week 12.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if edges is None:
        edges = np.linspace(y.min(), y.max(), n_bins + 1)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        r = p[m] - y[m]
        rows.append(dict(
            bin_lo=float(edges[b]), bin_hi=float(edges[b + 1]),
            n=int(m.sum()),
            bias=float(r.mean()) if m.any() else np.nan,
            rmse=float(np.sqrt(np.mean(r ** 2))) if m.any() else np.nan,
            mae=float(np.mean(np.abs(r))) if m.any() else np.nan))
    return rows


def format_table(rows, cols=None, float_fmt='{:>10.4f}'):
    """Plain-text table for notebook output.  No dependency on pandas."""
    if not rows:
        return '(empty)'
    cols = cols or list(rows[0].keys())

    def cell(v):
        return float_fmt.format(v).strip() if isinstance(v, float) else str(v)

    # width from the CONTENT, not a fixed 10 -- a long model name used to
    # overflow its column and shear every row after it out of alignment
    widths = [max(len(c), max(len(cell(r.get(c, ''))) for r in rows))
              for c in cols]
    out = ['  '.join(c.rjust(w) for c, w in zip(cols, widths))]
    for r in rows:
        out.append('  '.join(cell(r.get(c, '')).rjust(w)
                             for c, w in zip(cols, widths)))
    return '\n'.join(out)
