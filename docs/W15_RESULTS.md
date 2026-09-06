# Week 15 results

Pre-registration: `W15_PREREG.md` (locked before any fit). Eight fits, five
seeds, one feature set. Gates G15.1, G15.2, G15.3, G15.4, G15.5 all passed.

---

## 1. What was measured

**In-distribution baseline, and the §1.2 free measurement.** B0 on the reduced
feature set gives R² = 0.5863 on all 4,035 held-out rows and 0.5912 on the 3,496
measured rows, against the archive's 0.6269 and 0.6280 for the Week 11/12
feature set on the same split. **The family one-hot was carrying ≈ 0.04 of R²,
about 6% of the total.** Same rows, same split, same architecture, same epoch
budget; the only difference is the two dropped columns.

**Transfer.** Far-field RMSE in decades of log₁₀ km, on identical rows.

| fold | held out | n_far | RMSE transfer | RMSE control | ratio | rows > 3 dex |
|------|----------|-------|---------------|--------------|-------|--------------|
| T1 | L2 (train L1) | 923 | 20.193 | 1.430 | 14.12 | 69.3% |
| T2 | L1 (train L2) | 849 | 9.575 | 1.191 | 8.04 | 51.9% |
| T3 | E_a (extrap. low) | 625 | 3.325 | 1.258 | 2.64 | 26.1% |
| T4 | E_b (interpolation) | 574 | 1.348 | 1.139 | 1.18 | 1.6% |
| T5 | E_c (extrap. high) | 569 | 7.226 | 1.274 | 5.67 | 42.0% |

B0's RMSE on the same rows runs 1.055–1.337, so the matched-n controls sit
within 0.1–0.3 decades of the full-data baseline. **Sample size is excluded as
the explanation.** That is what the two extra fits bought.

**The failure is divergence, not degradation.** Against a target spanning
−3.81 to 2.00, T1 predicts between −113.55 and +78.01 with a residual sd of
12.37 decades. Trimming the worst 1% of rows leaves R² at −420, so this is most
of the data rather than a tail.

---

## 2. Disposition of every pre-registered prediction

| id | disposition | why |
|----|-------------|-----|
| P15.1 | **VOID** | Compares two raw R² that are both far below zero (−602, −130). An ordering between two failures is not the ceiling comparison this predicted. |
| P15.2 | **VOID as locked; direction survives post-hoc** | Retention uncomputable. Under RMSE ratio, T2 (8.04) beats T1 (14.12): training on the information-rich family transfers better, as predicted. |
| P15.3 | **VOID as locked; direction survives post-hoc** | Same. T4 (1.18) beats T3 (2.64) and T5 (5.67): interpolation over extrapolation, as predicted. See §3 for a deflationary reading. |
| P15.4 | CUT before lock | Recorded in `W15_PREREG.md` §4. |
| P15.5 | **VOID** | Predicted near-field R² inside the −0.133 band; got −598. Strictly outside, but produced by a model emitting hundred-decade predictions everywhere. "The near field is vacuous" cannot be concluded from a diverged model. |
| P15.6 | VOID on discovery | `gap` is an interval index, not a width; no per-row gap width in the archive. Recorded before the fits ran. |

**Retention is void, and the reason was knowable in advance.** Within-window R²
is negative by construction — Week 12 measured −0.887 far and −0.133 near
against 0.628 overall, restricted range, not a bug. Retention was specified as a
ratio of R² on far-field-only rows, so both numerator and denominator are
negative for a *well-fitting* model. The five controls came back at −0.87 to
−2.03, squarely in the Week 12 range. This is a design error in the
pre-registration, it was avoidable from the handoff alone, and it is recorded as
such rather than repaired quietly.

The denominator guard is the only reason it surfaced as a refusal rather than a
number: T1 would otherwise have printed retention = (−602.58)/(−2.03) = 297.

**RMSE ratio is post-hoc.** The pre-registered *quantity* is void; the
pre-registered *question* is answerable. Every sentence written from §1 must say
"the pre-registered direction survives under a post-hoc metric," never "the
prediction was confirmed."

---

## 3. Mechanism: one fold attributable, two not

Feature-substitution probe, **five seeds**, each input replaced with its
training median and the model re-scored, `recovered` measured against each
fold's own control RMSE on identical rows. Sensitivity analysis, not a physical
manipulation — a substituted state component is not a valid state.

The headline is not the percentage. It is whether the dominant direction is the
same on every seed.

| fold | excess over control | best feature | recovered | argmax stable |
|------|--------------------|--------------|-----------|---------------|
| T1 (family) | 18.429 dec | `x` | 65% ± 23% | **yes**, 5/5 seeds |
| T3 (energy, E_a) | 1.964 dec | `vz` | 21% ± 9% | no — vz, vz, y, vz, vy |
| T5 (energy, E_c) | 4.159 dec | `C` | 15% ± 96% | no — C, C, x, C, x |
| T4 (energy, E_b) | 0.197 dec | — | not reportable | null control refused |

**T1 is the one attributable fold.** `x` dominates on all five seeds and no other
feature exceeds 5%. That matches the support measurement, where T1's worst
feature is `x` at |z| = 45.3 with `max|z|` equal to `p99|z|` — a rigid covariate
shift, every held-out row displaced by nearly the same amount, L1 and L2 sitting
at different `x` in the rotating frame. It recovers 65% of the excess, not all
of it.

**The single-seed C attribution for T5 is FALSIFIED.** The one-seed run put `C`
at 73% and I wrote a mechanism from it. Over five seeds `C` recovers 15% with a
standard deviation of 96 points, and the dominant direction moves between `C`
and `x` across seeds. There is no stable direction. **The two-training-levels
account of T5 has no support and is withdrawn.**

**T3 has no dominant direction either.** Nothing exceeds 21%, and the argmax
moves across three different features. Substituting `vx` makes T3 substantially
*worse* (−107% ± 9%), consistently across seeds; I have no account of that and
am not offering one.

**Consequently the deflationary reading of P15.3 is retracted.** The claim that
interpolation beats extrapolation because `C` has two training levels rested
entirely on the falsified T5 attribution. P15.3's direction still holds in the
numbers; there is now no measured mechanism behind it, and none should be
asserted.

**What survives is a contrast between the two axes.** Family transfer fails
through one identifiable input direction, stably. Energy transfer fails
diffusely, with no single input carrying it. That is reportable as an
observation. It is not an explanation.

**T5's failure magnitude is itself unstable**: RMSE 5.433 ± 5.326 across seeds,
a spread as large as the mean. T1 is 19.859 ± 4.086 and T3 is 3.221 ± 0.921.
Any single-seed statement about T5 is unreliable.

**The null control worked.** T4's excess over control is 0.197 decades, below
the 0.30 floor, and the probe refused to report a direction. Had it named one
there, nothing else in the run would be trustworthy.

**An RMSE convention discrepancy to fix before writing.** §1's table uses pooled
RMSE across seeds and rows; the probe uses the mean of per-seed RMSE, matching
the project's rule that achieved R² is the mean of per-seed values. The two
agree closely except on T5, where pooled gives 7.226 and mean-of-per-seed gives
5.433 — the gap is the seed spread. §1 should be recomputed under the
mean-of-per-seed convention on both sides of every ratio. The fold orderings
that P15.2 and P15.3 turn on are unaffected.

## 4. What the paper can claim

The model does not degrade gracefully outside its training distribution. It
diverges, by up to 45 decades, across most of the held-out rows, with sample
size excluded by matched-n controls on identical rows. A screening tool that can
return a 10⁴⁵ km distance estimate when handed an unfamiliar orbit family is not
deployable whatever its in-distribution accuracy.

This strengthens the two-sided central claim rather than complicating it. Week
13 established that supervised learning screens usefully in distribution and
cannot seed a solver. Week 15 adds that the screening itself does not survive a
change of family or energy, and fails loudly enough to be detectable.

---

## 5. Limitations additions

- Retention, the pre-registered primary metric, was uncomputable; the reported
  comparison is post-hoc RMSE ratio
- Energy transfer is tested through a feature with two training levels; the
  interpolation/extrapolation contrast is partly a property of that
- Cross-level folds carry no measured residual-overlap bound
  (`nearest_point_overlap` not applicable, `alpha_found`/`theta_found` absent)
- Risk 19 untested: no per-row gap width in the archive
- L2_E2 max gap 53.66 km against 11.06–12.26 elsewhere, 4.4× the 12.3 km design
  target — reported post-hoc
- Two families, three energies only
- Mechanism is attributable for the family fold only; energy-fold divergence
  has no single-input account, and the two-training-levels explanation was
  tested and withdrawn
- T5's failure magnitude varies across seeds by as much as its own size

---

## 6. Open — your ruling

1. **Does the RMSE ratio become the reported comparison?** I have seen the
   results, so this choice is not mine to make cleanly.
2. **P15.5's disposition.** Void is my reading; falsified is defensible on the
   letter of it.
3. **`W15_PREREG.md` to `docs/`**, committed now, with both dates logged — drafted
   and locked before any fit, committed after.
4. Start-of-week items 5 and 6 (`t_solve_sec`, logging the formal cuts) remain
   outstanding.
