# P15 — Week 15 pre-registration

**LOCKED 2026-09-05.** Nothing has been fitted and no Week 15 statistic has
been computed. Both open items from the draft were ruled on before lock and the
rulings are recorded in §6. Amendments from here follow §5.

---

## 0. What Week 15 measures

Week 11's grouped split held out unseen patches of *seen* sheets. Week 15 holds
out a whole level of a categorical key — one family (L1, L2) or one energy (one
of three `C` values) — and asks whether the learned field transfers to a
dynamical regime absent from training.

Evaluated **in the far field only**. Week 13 established that the near field
carries no detectable information in distribution (MI −0.0046 nats, inside its
own permutation null). A near-field transfer measurement would measure the
transfer of nothing. The near field is reported as vacuous, not as a transfer
failure. See P15.5.

---

## 1. Design — settled, not under test

### 1.1 Feature set

`build_features` output **minus the family one-hot** (`family_0`, `family_1`).

Rationale: with a family held out, its one-hot column is constant-zero across
every training row, so its first-layer weights receive exactly zero gradient and
remain at initialization, while the other column is constant-one and trains as
an extra bias. Both flip at test time. The resulting shift is a seed-dependent
constant unrelated to the dynamics.

The **same reduced set is used for every Week 15 fit**, including the energy
folds where the dead-column defect does not arise, so that a family fold and an
energy fold are the same model and their difference is not confounded with the
feature set.

### 1.2 Consequence for the comparator

Changing the feature set changes the model. **No Week 11 or Week 12 R² is
admissible as the Week 15 in-distribution reference.** `week11_m1_r2` = 0.6576,
Week 11's 0.632 and Week 12's 0.6288 are all inadmissible, and which of them is
correct does not bear on Week 15. The reference is B0, refit below. (Settling
0.6576 remains archive hygiene for the paper; it is not on this week's path.)

Recorded as a free measurement: **B0 minus the Week 12 headline is the share of
in-distribution R² that was carried by the family label rather than the
dynamics.** Reported whatever it is.

### 1.3 The eight fits, five seeds each

Energies are indexed by sorted `C` value as `E_a < E_b < E_c`, not by
`energy_id` (see G15.2).

| id | trains on | n_train | n_val | evaluated on |
|----|-----------|---------|-------|--------------|
| B0 | grouped split, all six cells | 14,280 | 2,040 | own 4,080 test rows, and per cell |
| C_fam | mixed subsample, both family test sets excluded from pool | 9,180 | 1,020 | T1 and T2 test rows |
| C_eng | mixed subsample, all three energy test sets excluded from pool | 12,240 | 1,360 | T3, T4, T5 test rows |
| T1 | all L1 | 9,180 | 1,020 | 2,040 L2 rows |
| T2 | all L2 | 9,180 | 1,020 | 2,040 L1 rows |
| T3 | E_b, E_c | 12,240 | 1,360 | 1,360 E_a rows |
| T4 | E_a, E_c | 12,240 | 1,360 | 1,360 E_b rows |
| T5 | E_a, E_b | 12,240 | 1,360 | 1,360 E_c rows |

All counts nominal. Groups are atomic and median size is 2, so achieved counts
will differ; achieved values are recorded, not assumed.

One control per axis serves both or all three folds on that axis, because its
training pool excludes every test set on that axis and its `n` matches all folds
on that axis by symmetry. Control pool is 16,320 rows in both cases
(20,400 − 2 × 2,040 = 20,400 − 3 × 1,360). This makes the control pool a
non-uniform subsample of the full data; logged as a stated asymmetry.

**Training is on all rows. Only evaluation is restricted to the far field.**

Achieved R² is the **mean of per-seed R²**, never the R² of the seed-averaged
prediction.

### 1.4 Split contract

`leave_one_level_out(key, held_out, group_id, frac_test=0.2, frac_val=0.1,
seed=0)` in `src/splits.py`. Returns four boolean row masks plus a metadata
record of achieved counts.

- `test` — grouped subset of rows where `key == held_out`, targeting `frac_test`
  **of the held-out level** (not of the whole dataset), so test size matches
  B0's per-cell test size
- `train` / `val` — grouped carve of all rows where `key != held_out`,
  `frac_val` taken **from that pool**
- `excluded` — remainder of the held-out level; unused by the transfer run,
  available as control training stock

`key` is generic, so `family_id` and `energy_id` both pass through it. Level
lookup by set membership against `np.unique(key)`, never positional indexing.

---

## 2. Metrics

**Raw far-field R²** — reported, but not comparable across folds. Converting the
Week 13 per-cell far-field MI at ρ² = 1 − exp(−2I) (which reproduces the
handoff's 0.39 from 0.2476 exactly), the Gaussian-equivalent ceilings are
L1 ≈ 0.19 / 0.24 / 0.28 and L2 ≈ 0.39 / 0.53 / 0.58. Testing on L2 scores higher
than testing on L1 before any model is fit. The Gaussian equivalence is an
assumption stated and not proved, and is used **only to motivate P15.1** — never
as a denominator.

**Retention (primary)** — R²_transfer / R²_control on the *same rows*, control
being C_fam or C_eng. Isolates distribution shift from training-set size.

**Total degradation (secondary)** — R²_transfer / R²_B0 on the same rows.
Includes the sample-size cost. The gap between the two is the sample-size share.

**Near field** — descriptive only. No transfer claim.

---

## 3. Gates — must pass before any fit

**G15.1 — group atomicity.** No `group_id` straddles a level boundary on either
key. If a group contains both L1 and L2 rows, a family holdout leaks and every
number downstream is contaminated. `leave_one_level_out` asserts `key` is
constant within every group, before any selection, and raises with the
offenders named. Status on the real archive is unknown and not assumed; the
gate is the first thing Week 15 runs.

Recorded during construction, because it changes what the gate is worth: a
straddling group is caught by `groups_disjoint` on every seed **when the
straddle involves the held-out level**, but a group straddling two *other*
levels sits wholly in the training pool, lands in one part, and passes every
seed of that fold while breaking a different one. Week 15 runs five folds off
one dataset, so the post-hoc check is not a substitute for the upfront one.

**G15.2 — energy ordering.** `C` is one-to-one with `energy_id` but not
necessarily monotone in it. The middle energy is identified by sorted `C` value.
The interpolation/extrapolation reading of P15.3 is void if this is skipped.

**G15.3 — matched-n tolerance.** Control `n_train` within 3% of the transfer
run's achieved `n_train`. Achieved values logged. Exact matching is impossible
with atomic groups.

**G15.4 — far-field test n.** Recorded per fold before any prediction is read.
Expected near 1,000 for a family fold and near 700 for an energy fold, which
sets the noise floor on retention. If far-field test n falls below 500 on any
fold, that fold's result is reported as unresolved by construction.

**G15.5 — `split_report` on every fold.** A level holdout will show a large KS
statistic by construction; the number is recorded to quantify how different the
held-out region is, not as a pass/fail.

---

## 4. Predictions

Each prediction resolves to **confirmed**, **falsified**, or **unresolved**. A
difference smaller than the pooled per-seed sd of the two quantities compared is
**unresolved** and reported as such. Falsifications are reported, not smoothed.

**P15.1 — raw R² orders by test cell, not by training cell.**
Far-field R²(T1, test L2) > far-field R²(T2, test L1), and
R²(C_fam on L2 rows) > R²(C_fam on L1 rows).
*Falsified* if either inequality reverses. This is a check on the ceiling
reading; if it fails, the Gaussian-equivalent interpretation of the per-cell MI
is wrong and P15.2's framing must be re-examined before P15.2 is read.

**P15.2 — retention orders opposite to raw R².**
retention(T2: train L2 → test L1) > retention(T1: train L1 → test L2).
Direction: training on the information-rich family transfers better. Stated as a
prediction from an observation. **The per-cell MI split by family is an
observation with no mechanism attached and none is attached here.**
*Confirmed* if the difference exceeds the pooled per-seed sd; *falsified* if it
exceeds it in the opposite direction; *unresolved* otherwise.
The conjunction of P15.1 and P15.2 — the two metrics ordering oppositely — is
the substantive claim and is wrong in an obvious way if the reasoning is off.

**P15.3 — interpolation beats extrapolation across energy.**
retention(T4, holding out E_b) > max(retention(T3), retention(T5)).
Same resolution rule. Void if G15.2 is not run.

**P15.4 — CUT before lock, not skipped.** Predicted that the family asymmetry
would exceed the energy spread. Cut because the comparison is already carried by
P15.2 and P15.3 read side by side, and at these test sizes it was the prediction
most likely to return unresolved. Numbering is preserved rather than closed up,
so P15.5 and P15.6 keep the identifiers they were drafted under.

**P15.5 — the near field is vacuous.**
Near-field transfer R² on every fold lies within the band of the
in-distribution near-field within-window R² (−0.133 in Week 12). Reported as
vacuous rather than as a transfer failure whatever the value. A near-field
transfer number materially *above* that band would be a surprise requiring
explanation before it is reported, not after.

**P15.6 — Risk 19: gap structure does not explain family transfer.**
`gap`, `target_gap_km` and `family_max_gap_km` are already in the archive; no
new compute. Two parts:
1. *Census.* Fraction of far-field rows per cell with `d_km` above half the
   local gap, where the true nearest torus is plausibly an uncomputed one.
2. *Sensitivity.* Recompute P15.2 on the subset with `d_km` below half the local
   gap.
*Arbiter:* if the P15.2 direction survives the sensitivity re-analysis, Risk 19
is closed and reported closed. If it does not, the family transfer result is
reported with the confound stated and Risk 19 goes to Limitations. The census
alone does not close it; the sensitivity re-analysis does.

---

## 5. Amendment rule

Any amendment to this document is decided **from synthetic data, before any real
Week 15 statistic is computed**, and logged with its reason at the point of
decision. This is the Week 12 and Week 13 rule, unchanged.

---

## 6. Rulings applied at lock

**(a) `nearest_point_overlap` is declared NOT APPLICABLE to a level holdout.**
The Week 15 handoff lists it as an instrument built for exactly this. It is not,
in either configuration. With `keys = (family_id, energy_id)` the train and test
key sets are disjoint by construction on the held-out axis, so the overlap is
exactly zero and the number is vacuous. Dropping the held-out key — passing
`energy_id` only for a family fold — makes it non-zero, but it then bins
`alpha_found` and `theta_found` across two different families, and those are
parameterization coordinates on a torus, so the same (α, θ) bin on an L1 sheet
and an L2 sheet is not the same place in phase space. The number would be
parameter coincidence, not geometric proximity.

Week 15 therefore does not quote a residual-overlap figure for any transfer
fold, and says why. The honest replacement is a nearest-neighbour distance
between train and test rows in feature or state space. That instrument does not
exist, is not a day's work, and is **not** in Week 15 scope. It goes to
Limitations as: cross-level folds carry no measured residual-overlap bound.

This rests on a reading of the α/θ convention rather than on a measurement, and
is logged as such.

**(b) P15.4 cut.** See §4.

---

## 7. Not in Week 15

Start-of-week items 5 and 6 (timing `converge_torus` for `t_solve_sec`, logging
the formal cuts) are housekeeping and neither blocks a fit. Both outstanding.

**Item 4 — `week11_m1_r2` = 0.6576 — the research log does not settle it.**
The log carries the three values (0.632 ± 0.020 at line 1188 after the epoch
budget moved to 3000/150/50; 0.6288 ± 0.0214 reproduced at line 1323; 0.6269 on
all held-out rows) and then restates the puzzle unresolved at line 1483 under
*Stuck on*. No entry describes how `results_wk12.json` aggregates seeds.

Two mechanisms both land in the right place and the log cannot separate them:

1. *Seed-averaged prediction (ensembling).* R²(mean prediction) − mean per-seed
   R² equals the mean pointwise across-seed prediction variance over Var(y).
   With Var(y) = 1.046 / 0.3712 = 2.818, which independently matches a uniform
   over 5.810 decades (5.810² / 12 = 2.813), the observed gap of 0.0288 implies
   an across-seed prediction sd of 0.285 decades, about 28% of the residual sd.
   High but not implausible.
2. *Best of five seeds.* 0.6576 sits 1.35 `r2_sd` above 0.6288. The expected max
   of five draws is 1.16 sd above the mean, and P(max ≥ 1.35 sd) ≈ 0.37.
   Entirely unremarkable.

Neither is preferred here. The decisive test needs the archive: from
`predictions_wk12.npz` key M1 (5, 4035), compute the mean of per-seed R², the R²
of the column-mean prediction, and the max of per-seed R², and see which equals
0.6576. Rows are reconstructed with `grouped_split(group_id, frac_test=0.2,
frac_val=0.1, seed=0)` and the reconstruction verified by reproducing 0.6288
within 0.02 first. M1 is δ-independent across the four keys — verify rather than
assume.

Downgraded by §1.2 either way: whichever value is correct, none of the three is
the Week 15 comparator. This is archive hygiene for the paper.
