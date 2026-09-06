"""Metrics for the regressor, and the curves that are the deliverable.

ON R^2.  The target `log_d_km` was drawn LOG-UNIFORM by design (see
`src/sampling.py`), so the variance R^2 is measured against is a property of a
sampling choice, not of phase space.  Widening the drawn range would raise R^2
without the model improving.  R^2 is therefore reported for comparability and
the BASELINES are the meaningful comparison; a bare R^2 does not belong in the
abstract.

ON R^2 ACROSS TARGETS -- WEEK 12.  A second, sharper version of the same problem
now applies.  R^2 is scale-invariant under an AFFINE change of target, and
asinh(d/delta) is not affine in log10 d: it agrees with it up to a constant in
the far field and departs from it below delta.  So R^2 on an asinh target and
R^2 on a log10 target are not one number, and a cell of the 2x2 can post a
higher R^2 purely by flattening the region it fails in.  Every cross-cell
comparison goes through `targets.to_log10_km` first, and the arbiter is
`factor_table` in physical km, which no choice of delta can move.

ON THE THRESHOLD CURVE.  Research Question row 9 specifies classification
recovered by thresholding the regression: true class (d < tau), predicted class
(d_hat < tau), agreement swept over tau.  This DEGENERATES TO 1 at both ends --
as tau leaves the data range every sample is one class and any predictor is
right.  Quoting the mean or the max over tau is therefore meaningless; the
honest scalar is the MINIMUM over tau, and it needs balanced accuracy or MCC
alongside because the class prior at each tau is set by the log-uniform draw
rather than by anything physical.

The curve compares `y < t` against `p < t` with the SAME t on both sides, so it
is invariant under any monotone reparameterisation of the target -- but only if
`taus` is passed EXPLICITLY.  The `None` default sweeps the observed range of
whatever units were handed in, which would silently compare a log10 sweep in one
cell against an asinh sweep in another.  Pass taus in log10 km.

It is also valid on censored rows.  A clamped row carries log10(FLOOR) and is
genuinely below every tau above the floor, so the class label is correct; only
tau = log10(FLOOR) exactly is degenerate.  The full test set can be used.

ON THE ERROR-TOLERANCE CURVE.  Fraction of samples with |y_hat - y| <= tol, in
log10 km.  This is the curve the warm-start story wants: an RMSE of 0.2 in
log10 is a typical factor-of-1.6 error in the distance handed to a solver, and
that framing is legible in a way R^2 is not.  `factor_table` prints it at
round factors.

ON MASKS.  `mask` selects rows by the LABEL, never by the prediction, so the
same rows are chosen in every cell and the comparison stays fair.  Selecting on
the prediction would make each cell score itself on a different subset.
"""

import numpy as np

FACTORS = (1.3, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)


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


def _apply_mask(y, p, mask):
    """Subset by label-derived mask.  Shared by every curve below so the
    selection rule cannot drift between them."""
    if mask is None:
        return y, p
    m = np.asarray(mask).reshape(-1)
    if m.dtype != bool:
        raise TypeError(f"mask must be boolean, got {m.dtype}")
    if m.shape != y.shape:
        raise ValueError(f"mask shape {m.shape} vs {y.shape}")
    if not m.any():
        raise ValueError("mask selects no rows")
    return y[m], p[m]


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


def accuracy_vs_threshold(y_true, y_pred, taus=None, n_tau=60, mask=None):
    """THE deliverable curve.  Classification recovered by thresholding.

    `taus` are in the units of the target (log10 km).  PASS THEM EXPLICITLY for
    any cross-cell comparison; the default sweeps the observed range of whatever
    was handed in, which is not the same window in two different target units.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    y, p = _apply_mask(y, p, mask)
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
    The window is computed from the LABELS, which are the same rows in every
    cell, so it selects the same physical taus across the 2x2.
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


def accuracy_vs_error_tolerance(y_true, y_pred, tols=None, n_tol=60, mask=None):
    """Fraction with |y_hat - y| <= tol.  Monotone non-decreasing in tol.

    Both arguments must already be in log10 km -- route asinh-cell predictions
    through `targets.to_log10_km` first.  Unlike the threshold curve this is NOT
    invariant under reparameterisation: |y_hat - y| in asinh units and in log10
    km are different quantities.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    y, p = _apply_mask(y, p, mask)
    err = np.abs(p - y)
    if tols is None:
        tols = np.linspace(0.0, max(err.max(), 1e-12), n_tol)
    tols = np.asarray(tols, dtype=float)
    frac = np.array([float(np.mean(err <= t)) for t in tols])
    return dict(tol=tols, fraction_within=frac, n=int(y.size))


def factor_table(y_true, y_pred, factors=FACTORS, masks=None):
    """Fraction of rows whose predicted distance is within a factor of N.

    THE ARBITER FOR WEEK 12.  Physical km, so no choice of the asinh scale
    delta can move it -- which is what a raw R^2 comparison across targets
    cannot promise.  `masks` is a dict of name -> boolean array (or None for
    all rows); the near-field column, restricted to measured rows below the
    dead-zone boundary, is the one that says whether a target change fixed the
    near field or merely stopped asking about it.

    Rows are ready for `format_table`.
    """
    masks = masks or {'all': None}
    rows = []
    for f in factors:
        tol = float(np.log10(f))
        row = dict(factor=float(f), log10_tol=tol)
        for name, m in masks.items():
            c = accuracy_vs_error_tolerance(y_true, y_pred, tols=[tol], mask=m)
            row[name] = float(c['fraction_within'][0])
        rows.append(row)
    return rows


def residual_bins(y_true, y_pred, edges=None, n_bins=12, mask=None):
    """Bias and RMSE as a function of the TARGET's own magnitude.

    This was the P3 check: grad(log d) ~ 1/d in the normal direction, so the
    target's Lipschitz constant is unbounded as d -> 0 while a smooth-activation
    MLP has bounded gradients.  The prediction was heteroscedastic residuals
    with a systematic positive bias in the smallest-d bins.

    WEEK 11 OUTCOME.  P3b held and P3a was FALSIFIED, in the informative
    direction: near-field residual SD (0.343) is SMALLER than far-field (0.627),
    while near-field RMSE is larger (0.944 vs 0.668).  The near-field error is
    therefore almost entirely BIAS, not scatter -- the model emits a near
    constant rather than tracking noisily.  The bias-versus-target slope of
    -0.976 to -0.999 confirms it: a constant output gives exactly -1.  The
    constant is ~6.5 m and the dead zone runs to ~125 m, stable across n_bins
    12, 18 and 24 (2.90 decades in all three).  The n_bins=36 fit reports a
    boundary of 3 m BELOW its own constant of 6.2 m, which is impossible for a
    monotone predictor, so that row is the run-detector truncating on bin noise.

    Binning is on the target, not on phase-space location.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape}")
    y, p = _apply_mask(y, p, mask)
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


def dead_zone(y_true, y_pred, n_bins=12, tol=0.10, mask=None):
    """Does the model emit a CONSTANT over the smallest-d bins?

    Lifted from nb11 §9 so the Week 11 and Week 12 notebooks share one
    implementation; the two had already drifted once by being pasted.

    The statistic is `implied = bias + bin centre`, which is the model's mean
    output in that bin.  Under mean reversion pred = mu + lam*(y - mu) the
    implied output is NOT flat unless lam = 0, so a flat prefix means the
    network is ignoring its input over those bins -- a stronger failure than
    smoothing, and one RMSE cannot see, which is why residual SD is reported
    beside it.  A constant output gives bias = c0 - y, hence slope exactly -1
    against the target; Week 11 measured -0.976 to -0.999.

    `self_consistent` is the addition.  The detector walks the flat prefix
    under a FIXED tolerance that does not scale with bin noise, so at fine
    binning the run truncates early on scatter rather than on structure.  When
    that happens the reported constant falls OUTSIDE the range it was supposedly
    constant over, which is incoherent -- Week 11's n_bins=36 row put a 6.2 m
    constant above a 3 m boundary while 12/18/24 all agreed on 6.5 m and 125 m.
    The flag makes that self-check automatic instead of an eyeball.
    """
    rb = residual_bins(y_true, y_pred, n_bins=n_bins, mask=mask)
    ctr = np.array([(r['bin_lo'] + r['bin_hi']) / 2 for r in rb])
    bias = np.array([r['bias'] for r in rb])
    implied = bias + ctr

    k = 1
    while k < len(implied) and np.ptp(implied[:k + 1]) < tol:
        k += 1

    out = dict(n_bins=int(n_bins), tol=float(tol), k=int(k),
               bin_width_dec=float(rb[0]['bin_hi'] - rb[0]['bin_lo']),
               n_in_flat=int(sum(r['n'] for r in rb[:k])),
               implied=[float(v) for v in implied])
    if k < 3:
        out['flat'] = False
        return out

    lo, hi = float(rb[0]['bin_lo']), float(rb[k - 1]['bin_hi'])
    c0 = float(implied[:k].mean())
    out.update(
        flat=True,
        slope=float(np.polyfit(ctr[:k], bias[:k], 1)[0]),
        constant_log10_km=c0, constant_m=float(10 ** c0 * 1e3),
        boundary_log10_km=hi, boundary_m=float(10 ** hi * 1e3),
        span_dec=hi - lo, ptp=float(np.ptp(implied[:k])),
        self_consistent=bool(lo <= c0 <= hi),
    )
    return out


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


def retention(y_true, pred_transfer, pred_control, mask=None,
              min_control_r2=0.05, se_multiple=2.0):
    """Week 15's primary metric: transfer R^2 as a fraction of control R^2.

    WHY A RATIO AND NOT A DIFFERENCE OF R^2.  Raw test R^2 is not comparable
    across Week 15's folds.  Converting the Week 13 per-cell far-field MI at
    rho^2 = 1 - exp(-2I), the Gaussian-equivalent ceilings are L1 ~ 0.19 / 0.24
    / 0.28 and L2 ~ 0.39 / 0.53 / 0.58: testing on L2 outscores testing on L1 by
    about a factor of two before any model is fit.  Dividing by a control
    scored on the SAME rows removes the ceiling.  (The MI ceilings motivate the
    choice; they are never used as the denominator -- that is empirical.)

    WHY PREDICTIONS AND NOT TWO R^2 VALUES.  The invariant that makes retention
    mean anything is that numerator and denominator are computed over identical
    rows.  A signature taking two scalars cannot enforce it; this one applies a
    single mask to both.

    WHY THE DENOMINATOR IS GUARDED.  R^2 is unbounded below, so a ratio of two
    negatives is positive: a transfer R^2 of -0.5 over a control of -0.2 is
    2.5, which prints as 250% retention when both models have failed.  The
    ratio is only reported when the control clears `min_control_r2` AND exceeds
    `se_multiple` standard errors of its own seed spread.  Otherwise `defined`
    is False and `reason` says which test failed.  Both thresholds were fixed
    before any Week 15 fit.

    Parameters
    ----------
    y_true : (n,) targets in log10 km.
    pred_transfer, pred_control : (n,) or (n_seeds, n) predictions, same rows.
    mask : boolean over n, selecting rows by LABEL only -- see the module note.
    min_control_r2 : floor on the denominator.
    se_multiple : the control must exceed this many standard errors of zero.

    Returns
    -------
    dict with `retention`, `se`, both R^2 values and their per-seed spreads,
    `defined`, and `reason`.  `se` is the delta-method propagation
    |A/B| * sqrt((se_A/A)^2 + (se_B/B)^2), which assumes the two fits are
    independent -- they train on different rows with different seeds, so that
    is approximately true and is stated rather than proved.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    A = np.atleast_2d(np.asarray(pred_transfer, dtype=float))
    B = np.atleast_2d(np.asarray(pred_control, dtype=float))
    if A.shape[1] != y.size or B.shape[1] != y.size:
        raise ValueError(f"predictions must be over the same {y.size} rows; "
                         f"got {A.shape} and {B.shape}")

    def _per_seed(P):
        r2s = []
        for s in range(P.shape[0]):
            yy, pp = _apply_mask(y, P[s], mask)
            r2s.append(regression_metrics(yy, pp)['r2'])
        return np.asarray(r2s, dtype=float)

    a, b = _per_seed(A), _per_seed(B)
    # Mean of per-seed R^2, never the R^2 of the seed-averaged prediction.
    # Ensembling flatters: Week 11's 0.6324 became 0.6576 that way.
    a_bar, b_bar = float(a.mean()), float(b.mean())
    se_a = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float('nan')
    se_b = float(b.std(ddof=1) / np.sqrt(b.size)) if b.size > 1 else float('nan')

    n_rows = int(np.asarray(mask).sum()) if mask is not None else int(y.size)
    out = dict(r2_transfer=a_bar, r2_control=b_bar,
               r2_transfer_per_seed=a.tolist(), r2_control_per_seed=b.tolist(),
               se_transfer=se_a, se_control=se_b,
               n_seeds_transfer=int(a.size), n_seeds_control=int(b.size),
               n_rows=n_rows, retention=float('nan'), se=float('nan'),
               defined=False, reason=None)

    if b_bar < min_control_r2:
        out['reason'] = (f"control R2 {b_bar:.4f} below min_control_r2 "
                         f"{min_control_r2}; the ratio is not interpretable "
                         f"and a negative-over-negative ratio would be "
                         f"positive")
        return out
    if b.size > 1 and np.isfinite(se_b) and se_b > 0 \
            and b_bar < se_multiple * se_b:
        out['reason'] = (f"control R2 {b_bar:.4f} is within {se_multiple} SE "
                         f"({se_b:.4f}) of zero; the denominator is not "
                         f"distinguishable from no signal")
        return out

    ratio = a_bar / b_bar
    se = float('nan')
    if np.isfinite(se_a) and np.isfinite(se_b) and a_bar != 0.0:
        se = abs(ratio) * float(np.sqrt((se_a / a_bar) ** 2
                                        + (se_b / b_bar) ** 2))
    elif np.isfinite(se_a):
        se = abs(se_a / b_bar)          # numerator at zero: ratio is 0 +/- this
    out.update(retention=float(ratio), se=se, defined=True)
    return out


def retention_difference(ret_a, ret_b):
    """Resolve a pre-registered ordering between two retentions.

    P15.2 and P15.3 both compare two folds, and the pre-registration resolves
    each to confirmed, falsified, or UNRESOLVED: a difference smaller than the
    pooled spread is not a direction.  Far-field test n runs 569-923 across the
    five folds, so unresolved is a live outcome and encoding it here keeps it
    from being read as a weak confirmation.

    Returns the signed difference `ret_a - ret_b`, the pooled SE, and a verdict
    in {'a_greater', 'b_greater', 'unresolved', 'undefined'}.
    """
    if not (ret_a.get('defined') and ret_b.get('defined')):
        return dict(verdict='undefined', difference=float('nan'),
                    pooled_se=float('nan'),
                    reason=ret_a.get('reason') or ret_b.get('reason'))
    d = ret_a['retention'] - ret_b['retention']
    sa, sb = ret_a.get('se', np.nan), ret_b.get('se', np.nan)
    pooled = float(np.sqrt(np.nansum([sa ** 2, sb ** 2])))
    if not np.isfinite(pooled) or pooled == 0 or abs(d) <= pooled:
        verdict = 'unresolved'
    else:
        verdict = 'a_greater' if d > 0 else 'b_greater'
    return dict(verdict=verdict, difference=float(d), pooled_se=pooled,
                n_sigma=float(abs(d) / pooled) if pooled else float('inf'))
