# Week 15 results

Pre-registration: `W15_PREREG.md` (locked before any fit). Eight fits, five
seeds, one feature set. Gates G15.1, G15.2, G15.3, G15.4, G15.5 all passed.

**THE FITS IN THIS DOCUMENT WERE RE-RUN 2026-09-12.** The originals used
`patience=50` / `lr_patience=25` against the frozen configuration's 150 / 50 —
a kwarg mis-mapping in `mknb15b.py`, found in the 2026-09-11 audit and
confirmed in the archive by an `epochs_run − best_epoch` of exactly 51 on all
forty fits. The corrected run then hit `max_epochs=3000` on 13 of 40 fits, so
the budget went to 8000; 1 of 40 is budget-limited now (T5 seed 1, best_epoch
7978). Every seed that had already converged reproduced its `best_epoch`
exactly across the two corrected runs, so the numbers below are
schedule-independent rather than merely self-consistent.

**Two published conclusions did not survive the correction** — the family
one-hot's R² share (§1) and T4's factor-of-10 match to its control (§2b) — and
the two pre-registered directions swapped robustness (§2a). The superseded
numbers are kept beside the current ones throughout rather than removed.

---

## 1. What was measured

**In-distribution baseline, and the §1.2 free measurement.** B0 on the reduced
feature set gives R² = **0.6294** on all 4,035 held-out rows and **0.6366** on
the 3,496 measured rows, against the archive's 0.6269 and 0.6280 for the Week
11/12 feature set on the same split.

**The family one-hot carries nothing measurable: −0.0025 and −0.0086** — the
reduced set is a hair *higher*, inside the seed spread, in the opposite
direction to the effect that was claimed. The published figure was "≈ 0.04 of
R², about 6% of the total", and it was the epoch budget in its entirety: under
the mis-specified schedule B0 scored 0.5863 / 0.5912, and the whole of that
0.04 closed when the schedule was fixed. Same rows, same split, same
architecture, and now the same budget; the only difference is the two dropped
columns, and it does not show up.

**Transfer.** Far-field RMSE in decades of log₁₀ km, on identical rows, mean of
per-seed on both sides of every ratio.

| fold | held out | n_far | RMSE transfer | RMSE control | ratio | B0 same rows | rows > 3 dex |
|------|----------|-------|---------------|--------------|-------|--------------|--------------|
| T1 | L2 (train L1) | 923 | 98.416 | 1.346 | **73.13** | 1.221 | 89.4% |
| T2 | L1 (train L2) | 849 | 9.954 | 1.076 | **9.25** | 0.964 | 66.1% |
| T3 | E_a (extrap. low) | 625 | 2.215 | 1.194 | **1.85** | 1.112 | 17.2% |
| T4 | E_b (interpolation) | 574 | 2.383 | 1.088 | **2.19** | 1.072 | 14.6% |
| T5 | E_c (extrap. high) | 569 | 5.082 | 1.206 | **4.22** | 1.142 | 33.6% |

Published ratios, for comparison: 14.12 / 8.04 / 2.64 / 1.18 / 5.67. Three of
five moved materially — T1 up 5×, T4 up nearly 2×, T3 down — so the schedule
was not cancelling between a fold and its control, which is what "every fit ran
the same configuration" had been taken to imply. The folds gained different
amounts of validation R² when the budget was fixed (T1 +0.083, T2 +0.069,
T4 +0.060 against C_energy +0.025), and the ratios moved with them.

B0's RMSE on the same rows runs 0.964–1.221, so the matched-n controls sit
within 0.02–0.12 decades of it. **Sample size is excluded as the explanation**,
by the matched-n controls themselves. That is what the two extra fits bought,
and it is the one part of §1 that neither correction touched.

> Corrected 2026-11-12 (external review of the paper draft): this paragraph said
> "within 0.1–0.4 decades of the full-data baseline". The differences are
> 0.125 / 0.112 / 0.082 / 0.016 / 0.064 (T1–T5). B0 is also not "full-data": it
> trained on the 14,409-row grouped split, and 61–72% of each fold's far-field
> test rows are in that training set, so its figure is mostly in-sample.

**The failure is divergence, not degradation.** Against a target spanning −3.81
to 2.00, T1 predicts between **−967.35 and +62.17** on its held-out far-field
rows, a worst single error of 967 decades, with a residual sd of 107. 89.4% of
its rows are wrong by more than three decades. Far-field R² is −21,049 on T1
and −161 on T2.

> Superseded, and why: earlier drafts gave −113.55 to +78.01 with a residual sd
> of 12.37. Those came from `diagnose_wk15.py` under the old fits AND used the
> wrong rows — the range was over all 20,400 archive rows including T1's own
> training rows, and the sd was seed 0 measured on B0's test rows. The script
> now labels both scopes. Separately, `rows > 3 dex` was published as seed 0
> (69.3 / 51.9 / 26.1 / 1.6 / 42.0) while every other column in the same table
> pooled five seeds; it is all-seed above.

---

## 2. Disposition of every pre-registered prediction

| id | disposition | why |
|----|-------------|-----|
| P15.1 | **VOID** | Compares two raw R² that are both far below zero (−21,049 and −161). An ordering between two failures is not the ceiling comparison this predicted. |
| P15.2 | **VOID as locked; direction holds post-hoc, including under the arbiter** | Retention uncomputable. The direction now survives all six metrics, and the arbiter puts it at 80% of seed combinations — above the band §2a calls "not a direction". Upgraded from unresolved on the corrected fits. Still post-hoc. |
| P15.3 | **VOID as locked; direction holds under the arbiter only** | T4 beats T5 under all six metrics, but beats T3 under only four: RMSE ratio puts it at 32% of seed combinations and MAE at 40%. Ruling 1 gives the arbiter the tie-break, and the arbiter holds at 88 / 76 / 64%. Downgraded from "survives everything, robustly". |
| P15.4 | CUT before lock | Recorded in `W15_PREREG.md` §4. |
| P15.5 | **FALSIFIED** | Predicted near-field R² inside the −0.133 band and it is not. The metric computed and came back wrong, so this is a falsification, not a void — §6 ruling 2. What cannot be concluded from a diverged model is the *positive* claim "the near field is vacuous". |
| P15.6 | VOID on discovery | `gap` is an interval index, not a width; no per-row gap width in the archive. Recorded before the fits ran. |

**Retention is void, and the reason was knowable in advance.** Within-window R²
is negative by construction — Week 12 measured −0.887 far and −0.133 near
against 0.628 overall, restricted range, not a bug. Retention was specified as a
ratio of R² on far-field-only rows, so both numerator and denominator are
negative for a *well-fitting* model. The five controls came back at −0.65 to
−1.68, squarely in the Week 12 range. This is a design error in the
pre-registration, it was avoidable from the handoff alone, and it is recorded as
such rather than repaired quietly.

The denominator guard is the only reason it surfaced as a refusal rather than a
number: T1 would otherwise have printed retention = (−21,049)/(−1.68) = 12,529.

**RMSE ratio is post-hoc.** The pre-registered *quantity* is void; the
pre-registered *question* is answerable. Every sentence written from §1 must say
"the pre-registered direction survives under a post-hoc metric," never "the
prediction was confirmed."

---

## 2a. Metric robustness — the post-hoc choice IS load-bearing

Retention is void, so the reported comparison is post-hoc. The risk is not that
it is post-hoc but that a metric was *selected* from candidates and the one
reproducing the pre-registered direction was kept. `notebooks/metric_robustness.py`
tests that directly: six metrics, direction normalized to a common goodness, and
every ordering evaluated over all 5 × 5 seed combinations rather than a point
estimate.

| metric | P15.2 (T2 > T1) | 5×5 | P15.3 vs T3 | 5×5 | P15.3 vs T5 | 5×5 |
|--------|-----------------|-----|-------------|-----|-------------|-----|
| RMSE ratio | holds | 96% | **FAILS** | 32% | holds | 56% |
| MAE ratio | holds | 92% | **FAILS** | 40% | holds | 80% |
| median &#124;err&#124; ratio | holds | 84% | holds | 76% | holds | 88% |
| within factor 3 | holds | 80% | holds | 88% | holds | 100% |
| within factor 10 | holds | 80% | holds | 76% | holds | 100% |
| within factor 100 | holds | 80% | holds | 64% | holds | 96% |

**The two predictions swapped places.** Under the original fits P15.2 held under
5 of 6 metrics and failed the arbiter at 56%, and P15.3 held under 6 of 6 at
96–100%. Under the corrected fits it is the reverse: P15.2 is 6 of 6 with the
arbiter at a uniform 80%, and P15.3 fails the two tail-sensitive metrics against
T3. Neither verdict was stable to a training-schedule change, which is the most
important methodological fact in this section.

**P15.2 — "training on the information-rich family transfers better" — now has
support, and it is still post-hoc.** 80% under the arbiter clears the 56–68%
band that this section previously called "not a direction". It must be written
as a direction that survives a post-hoc metric, never as a confirmation, and the
caveat below applies.

**P15.3 — "interpolation beats extrapolation" — survives only on the arbiter.**
Ruling 1 says the arbiter wins where metrics disagree and it does hold there
(88 / 76 / 64% against T3), but "survives everything" is no longer true and must
not be repeated. The RMSE and MAE failures are against T3 specifically; T4 beats
T5 comfortably on everything.

**The caveat on both.** T5 keeps one budget-limited seed and its RMSE ratio
moved 7.68 → 4.22 between the two corrected runs. Fold *orderings* were stable
across that move; T5 *magnitudes* were not. Any sentence about how badly T5
fails is soft; sentences about where T5 sits relative to T4 are not.

---

## 2b. The arbiter table — and the positive result is withdrawn

Fraction of far-field rows predicted within a factor of N, physical km, mean of
per-seed, transfer over its matched-n control on identical rows.

| fold | ×3 transfer | ×3 control | ×10 transfer | ×10 control | ×100 transfer | ×100 control |
|------|------------|-----------|-------------|------------|--------------|-------------|
| T1 | 0.017 | 0.319 | 0.040 | 0.485 | 0.074 | 0.867 |
| T2 | 0.062 | 0.469 | 0.119 | 0.620 | 0.236 | 0.936 |
| T3 | 0.190 | 0.473 | 0.361 | 0.604 | 0.650 | 0.888 |
| T4 | 0.296 | 0.494 | 0.482 | 0.639 | 0.743 | 0.920 |
| T5 | 0.063 | 0.431 | 0.185 | 0.570 | 0.428 | 0.895 |

**"T4 matches its control at the factor-of-10 level" is WITHDRAWN.** It was
0.603 against 0.592 — a ratio of 1.019, which the previous version of this
document called "the one place in Week 15 where transfer works" and said
"belongs in the paper as prominently as the failures". Under the corrected fits
it is 0.482 against 0.639, a ratio of 0.754.

It should not have been written even then, and the audit flagged it before the
re-run on evidence already in the old archive: the claim was **15 of 25 seed
pairs** — a coin flip by the standard §2a uses to call 56–68% "not a direction"
— and ×10 was the **only one of seven factors** at which T4 exceeded its control
(×1.3 0.673, ×2 0.740, ×3 0.827, ×5 0.901, ×10 1.019, ×30 0.982, ×100 0.947).
It was a crossing point in a monotone row, selected.

Under the corrected fits every fold loses to its control at **every** factor,
and T4 is **0 of 25 seed pairs at all seven** (0.375, 0.504, 0.599, 0.680,
0.754, 0.782, 0.807). There is no cell left to quote.

The controls restate the Week 12 finding in physical units: even in
distribution, a far-field prediction lands within a factor of 10 about 58% of
the time (0.485–0.639 across the five control fits). M1 resolves which decade,
not where inside it.

---

## 3. Mechanism: nothing is attributable

Feature-substitution probe, **five seeds**, each input replaced with its
training median and the model re-scored, `recovered` measured against each
fold's own control RMSE on identical rows. Sensitivity analysis, not a physical
manipulation — a substituted state component is not a valid state.

The headline is not the percentage. It is whether the dominant direction is the
same on every seed.

| fold | excess over control | best feature | recovered | argmax per seed | stable |
|------|--------------------|--------------|-----------|-----------------|--------|
| T1 (family) | 97.070 dec | `x` | 67% ± 34% | x, **vy**, x, x, x | **no** |
| T3 (energy, E_a) | 1.021 dec | `C` | 16% ± 17% | C, y, C, x, vy | no |
| T5 (energy, E_c) | 3.875 dec | `C` | 10% ± 55% | C, C, x, C, vz | no |
| T4 (energy, E_b) | 1.294 dec | `vy` | −18% ± 60% | y, C, vy, x, C | no |

**T1's attribution is withdrawn.** The previous version reported `x` stable on
5/5 seeds and called T1 "the one attributable fold". On the corrected fits it is
4/5 — seed 1 picks `vy` — and the probe's own pre-specified rule is all-seeds:
*"a direction that moves between seeds is not a mechanism; do not name one in
the write-up."* This project has already withdrawn one attribution on exactly
this basis (T5's `C`, from a single seed), and the same standard has to apply
here.

That said, T1 is not the same kind of negative as the others and the difference
should be stated rather than flattened: `x` leads on four of five seeds and
recovers 67%, with nothing else above 7%, and it matches the support
measurement — T1's worst feature is `x` at |z| = 45.3 with `max|z|` equal to
`p99|z|`, the signature of a rigid covariate shift, L1 and L2 sitting at
different `x` in the rotating frame. The honest sentence is that `x` is the
leading candidate and the probe does not establish it. Not that there is no
signal, and not that there is a mechanism.

**T3 and T5 have no dominant direction.** Nothing exceeds 16% on T3 or 10% on
T5, and both argmaxes move across three features. T3's leading feature also
changed identity between runs — `vz` under the old fits, `C` now — which is
itself evidence that nothing there is stable. T5's probe additionally carries
one non-clean seed and the script flags it unreliable.

**The two-training-levels account of T5 stays withdrawn**, and so does the
deflationary reading of P15.3 that rested on it. `C` is nominally T5's leading
feature again, at 10% ± 55 points; a number whose standard deviation is five
times its value is not a revival.

**THE NULL CONTROL IS GONE, and this is the largest hole in §3.** T4's excess
over its control was 0.197 decades under the old fits — below the 0.30 floor —
so the probe refused to report a direction there, and the previous version
correctly said that mattered: *"Had it named one there, nothing else in the run
would be trustworthy."* T4's excess is now **1.294 decades**, 6.6× larger and
well above the floor, so **no fold in this run is below the floor and the probe
has no negative control.** What partially replaces it is that T4's recoveries
are all negative — substituting any feature makes it worse, best −18% — and its
argmax is unstable, so the probe still did not name a spurious direction. That
is weaker than a pre-specified null and must be reported as weaker.

**Failure magnitudes are unstable where the fits are.** T1 is 98.416 ± 75.289
across seeds, T5 5.082 ± 5.615, T3 2.215 ± 0.522, T4 2.383 ± 0.695. Any
single-seed statement about T1 or T5 is unreliable.

**What survives.** Transfer fails on both axes, it fails by divergence rather
than degradation, and no single input direction accounts for it on any fold. The
published contrast — "family transfer fails through one identifiable input
direction, stably; energy transfer fails diffusely" — does not survive: both
axes now fail diffusely, and the family fold differs in magnitude, not in
attributability.

---

## 4. What the paper can claim

The model does not degrade gracefully outside its training distribution. It
diverges — up to 967 decades on a single row, with 89% of T1's held-out rows
wrong by more than three — with sample size excluded by matched-n controls on
identical rows. A screening tool that can return a 10⁹⁶⁷ km distance estimate
when handed an unfamiliar orbit family is not deployable whatever its
in-distribution accuracy.

This strengthens the two-sided central claim rather than complicating it. Week
13 established that supervised learning screens usefully in distribution and
cannot seed a solver. Week 15 adds that the screening does not survive a change
of family or an extrapolation in energy, and fails loudly enough to be
detectable.

**There is no exception, and the previous version's exception was wrong.**
"Energy interpolation is nearly free" and "the model interpolates and does not
extrapolate" both rested on T4 matching its control at one tolerance, and that
is withdrawn (§2b). Every fold loses to its control at every factor. What the
numbers support is an ordering — **energy interpolation degrades least** (ratio
2.19, ×10 goodness 0.754), then low-energy extrapolation (1.85 / 0.597), then
high-energy (4.22 / 0.324), then the two family folds (9.25 / 0.192 and 73.13 /
0.082) — not a regime where transfer is free.

**What Week 15 does NOT establish.** Any mechanism (§3). And the family
asymmetry, P15.2, is now supported under every metric including the arbiter, but
it remains a post-hoc comparison standing in for a void pre-registered one, and
it must be written that way.

---

## 5. Limitations additions

- Retention, the pre-registered primary metric, was uncomputable; the reported
  comparison is post-hoc RMSE ratio
- **The archived fits were re-run twice.** The first archive used a
  mis-specified early-stopping patience; the second hit the epoch budget on 13
  of 40 fits. The numbers here are from the third run, where 1 of 40 is
  budget-limited. Two published conclusions did not survive, and both
  pre-registered directions changed robustness status — so Week 15's verdicts
  are demonstrably sensitive to the training schedule, and that sensitivity is a
  finding about the instrument, not only about this run
- Energy transfer is tested through a feature with two training levels; the
  interpolation/extrapolation contrast is partly a property of that
- Cross-level folds carry no measured residual-overlap bound
  (`nearest_point_overlap` not applicable, `alpha_found`/`theta_found` absent)
- Risk 19 untested: no per-row gap width in the archive
- L2_E2 max gap 53.66 km against 11.06–12.26 elsewhere, 4.4× the 12.3 km design
  target — reported post-hoc
- Two families, three energies only
- **No fold is mechanistically attributable**, and the probe has no negative
  control in this run because T4's excess rose above the 0.30-decade floor
- T1's and T5's failure magnitudes vary across seeds by as much as their own
  size; T5 additionally keeps one budget-limited seed

---

## 6. Rulings

Settled 2026-09-06:

1. **The arbiter is `factor_table` in physical km** — the pre-established Week 12
   arbiter, not a metric chosen this week. RMSE ratio is reported beside it as
   the compact summary, and where they disagree the arbiter wins. Under the
   corrected fits it now disagrees on **P15.3**, not P15.2, and the arbiter
   carries P15.3.
2. **P15.5 is falsified, not void.** The metric computed and came back wrong.
   The earlier void call was mine and ran in the flattering direction.
3. ~~P15.2 downgraded to unresolved~~ — **superseded 2026-09-12.** The
   downgrade was correct on the fits then in hand; on the corrected fits P15.2
   holds under all six metrics with the arbiter at 80%. Recorded rather than
   silently reversed: the ruling did not change, the fits did.
4. `W15_PREREG.md`, `W15_RESULTS.md` and `CUTS_AND_DISPOSITIONS.md` committed to
   `docs/`; log entry appended; tracker at v2_4. Items 5 and 6 closed.

Settled 2026-09-11 (audit):

5. §1 recomputed under mean-of-per-seed on both sides. `rows > 3 dex` was seed 0
   in a table that otherwise pooled five seeds, and two figures in the
   divergence sentence came from the wrong row set; all corrected, with the old
   values and their provenance shown.
6. The `patience` mis-mapping found and fixed in `mknb15b.py`, and the same
   mis-mapping found and fixed in `probe_divergence.py`, which refits models and
   would otherwise have compared differently-trained models against the
   archive's denominator. Those were the only two `MLPConfig(...)` call sites in
   the repo with the defect.
7. The saved history now records `config` and `seed`. The defect survived a week
   because the archive recorded what the fits did and not what they were asked
   to do.

Settled 2026-09-12 (re-run):

8. **"Every fold ran the same schedule, so within-week comparisons are
   unaffected" was wrong**, and it was my reasoning on 2026-09-11. A uniform
   schedule is not an equally converged one: the folds gained different amounts
   of validation R² when the budget was fixed, so the ratios moved with them.
   Assume nothing in this week is schedule-independent without a run that shows
   it.
9. **Two published conclusions withdrawn**: the family one-hot's ≈0.04 of R²
   (§1) and T4's factor-of-10 match (§2b). Both were artifacts of the
   early-stopping schedule; the T4 one was additionally a selected cell and was
   flagged as such before the re-run.
10. **T1's `x` attribution withdrawn** (§3), on the probe's own all-seeds rule.
    The null control is gone with it — no fold is below the 0.30-decade floor
    any more.

**Open:** nothing blocking. T5 keeps one budget-limited seed at
`max_epochs=8000`; raising it again would buy a more stable T5 *magnitude* and
would not change any ordering in this document. Worth one line in the log if it
is done, not a silent bump.
