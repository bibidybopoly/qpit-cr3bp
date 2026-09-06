# Cuts and dispositions

Start-of-week item 6. Every item that was considered and is not in the paper,
with the reason it is not.

**Why this file exists.** A parked item reads to a reviewer exactly like an
unfinished one. The difference between "we chose not to", "we tried and it
failed", and "the instrument does not fit this question" is invisible unless it
is written down, and all three are defensible while silence is not.

**Dates.** Where a date is marked `[log]` it needs to come from
`research_log.md` or the mentor-session record; I have the decision but not the
day. Do not invent them.

---

## The distinctions this register keeps

Six different reasons, and they are not interchangeable:

| term | meaning | reads in the paper as |
|------|---------|----------------------|
| **cut** | deliberate scope decision, could have been done | Methods, one line |
| **retired** | superseded by a stronger result | Methods, with what replaced it |
| **moot** | another result made the question empty | usually nowhere |
| **void** | could not be computed from the data as specified | Limitations |
| **not applicable** | the instrument does not fit this question | Limitations |
| **falsified** | tested and failed | **Results**, as prominently as a confirmation |
| **undisposed** | still needs a decision | must not survive to submission |

A falsification is a result, not a cut, and belongs in Results. It is listed in
§6 only so nothing is lost from the register.

---

## 1. Scope cuts

| item | decided by | when | reason |
|------|-----------|------|--------|
| **Week 14** | Pierce + Dr. Lang | 2026-08-30 | Compress to one remaining research week; freed time went to the writing block |
| **Week 16** | Pierce + Dr. Lang | 2026-08-30 | Same |
| **The regeneration batch** | Pierce + Dr. Lang | 2026-08-30 — **first logged mentor session** | Settled by Pierce's own audit: dropping L2_E2 entirely moves near-field bias 6.2% and flips no verdict. The 562 rows below their error bound go in Limitations |
| **Heteroscedastic σ(x)** | Pierce | `[log]` | Refinement of a likelihood that lost |
| **Interval censoring** | Pierce | `[log]` | Same |
| **P15.4** (family asymmetry > energy asymmetry) | Pierce | 2026-09-05, **before P15.x locked** | Already carried by P15.2 and P15.3 read together; most likely of the six to return unresolved at these test sizes. Numbering preserved so P15.5 and P15.6 keep their identifiers |

## 2. Retired and moot

| item | disposition | reason |
|------|------------|--------|
| **P13.0** — the eikonal / benchmark selector | **retired, not skipped** | The MI result settles the benchmark reading more strongly than the gradient check could have. Say what replaced it |
| **Titan sweep** | **moot** | MI sits at the permutation null, so there is nothing for a sweep to resolve; Week 16 is also gone |

## 3. Void — specified but not computable

| item | when found | why |
|------|-----------|-----|
| **P13.3 leg B** (the Shannon ceiling arm) | Week 13 | The bound has an information-free floor of 1 − N/Var, zero only for a Gaussian target. `log_d_km` is near log-uniform (N/Var 0.6967 against 0.7026 uniform), so the floor sits at 0.303 and the ceiling cannot discriminate below it whatever the MI is |
| **P15.6** (Risk 19 gap confound) | 2026-09-05, **before the fits ran** | `gap` is an integer interval index 0–62, not a width. The archive carries no per-row gap width — only scalar `target_gap_km` = 12.3 and per-cell `family_max_gap_km`. Recovering one means returning to the torus surfaces, which is new compute |
| **Retention** — Week 15's pre-registered primary metric | 2026-09-06, after the fits | Specified as a ratio of R² on far-field-only rows. Within-window R² is negative by construction — Week 12 measured −0.887 far, restricted range, not a bug — so both numerator and denominator are negative for a *well-fitting* model. All five controls returned −0.87 to −2.03. **Knowable in advance from the Week 13 handoff; this was a design error, not a discovery** |
| **P15.1** | 2026-09-06 | Compares two raw R² that are both far below zero (−602, −130). An ordering between two failures is not the ceiling comparison it predicted |
| **P15.5** | 2026-09-06 | Predicted near-field R² inside the −0.133 band, got −598. Strictly outside, but produced by a model emitting hundred-decade predictions everywhere. "The near field is vacuous" cannot be concluded from a diverged model |
| **P15.2, P15.3** — *as locked* | 2026-09-06 | Retention uncomputable. Both **directions survive under post-hoc RMSE ratio** and must be reported with that wording, never as confirmations |

## 4. Not applicable — instrument does not fit

| item | why |
|------|-----|
| **`nearest_point_overlap` on a level holdout** | With the full key set `(family_id, energy_id)` the train and test key sets are disjoint by construction on the held-out axis, so overlap is exactly zero and vacuous. Dropping the held-out key makes it non-zero but bins `alpha_found`/`theta_found` across two families, and those are parameterisation coordinates on a torus — the same (α, θ) bin on an L1 and an L2 sheet is not the same place in phase space. **Separately and harder: the recovered coordinates it documents as its inputs are absent from the shipped archive entirely.** Consequence: cross-level folds carry no measured residual-overlap bound |

## 5. UNDISPOSED — decide before writing

These were deferred to "ride with the regeneration decision". **Regeneration is
cut, so they ride nowhere.** They are permanent properties of the shipped
dataset.

| item | why it still matters |
|------|---------------------|
| **`mknb10.py` missing** | **The one with teeth.** If `qpit_dataset_w10.npz` cannot be regenerated from a script, that is an artifact-reproducibility problem for arXiv and JEI, and it will not get easier during the writing phase |
| **The four `*_KNOWN_DEFECT` pins** | Each pins a known-wrong value. Disclose individually or explain the pin |
| **Notebook constant copies** | Constants duplicated rather than read from `constants.py`; a drift check exists only where the archive carries the value itself |
| **The `normal_directions` fix** | Residual already slated for Limitations, but the fix itself has no disposition |
| **`first_orbit` / `amplitude_direction`** | Now in `halo_families.ipynb`, `floquet_analysis.ipynb` **and** `time_solve.py`, and the two notebook copies have already drifted (`T / 2.0` against `T / 2`). Belongs in `src/orbits.py`. Same fix `metrics.dead_zone` records for nb11 §9 |
| **`week11_m1_r2` residual** | 0.6576 is settled — the R² of Week 11's seed-averaged three-seed prediction. But Week 12's reported 0.6288 ± 0.0214 does not reproduce from the archive, which gives 0.6280 ± 0.0226. M2b reproduces exactly from the same file; only M1 does not. Inside the 0.02 verification tolerance, unexplained at the fourth digit |
| **Censored-miss base rate** | 16 L2 / 7 L1 with no base rate to read it against. One line of code |

## 6. Falsified — these are RESULTS, not cuts

Listed for completeness only. Each belongs in Results, reported as prominently
as a confirmation.

- **P3a** (Week 11) — near-field residual SD is *smaller* than far-field, not larger; the error is bias, not scatter
- **P3's mechanism** (Week 12) — flat output and slope −1 stand; attribution to bounded network gradients does not
- **P13.4** (Week 13) — predicted S > 0.40, measured 0.342 ± 0.008
- **The `kl_entropy` failure mechanism** — observation held, mechanism did not
- **The T5 `C` attribution** (Week 15, 2026-09-06) — a one-seed probe put `C` at 73% and a two-training-levels account was written from it. Over five seeds `C` recovers 15% ± 96 points and the dominant direction moves between `C` and `x`. **Withdrawn, and the deflationary reading of P15.3 that rested on it is retracted**

Four of the five are mechanisms that failed while their observations held. That
pattern is worth one sentence in Discussion.

---

## 7. Limitations, reconciled against the Week 13 handoff

The handoff's Limitations list, updated:

| handoff item | status |
|--------------|--------|
| 562 rows below their error bound | stands |
| `n_below_floor` | **measured as 0** on this archive — `make_windows` reports zero measured rows below floor. Drop or restate |
| Dead zone and its 4.3% on-surface skip rate | stands |
| `normal_directions` residual | stands; see §5 |
| Risk 19 if untested | **untested, and now with a reason** — §3 |
| Void ceiling arm and the 1 − N/Var floor | stands |
| 16/7 censored-miss skew | stands; base rate still one line away |
| Residual sd ≈ 1 decade | stands |
| Two families only | stands, and Week 15 gives it teeth |

**New from Week 15:**

- Retention, the pre-registered primary metric, was uncomputable; the reported
  comparison is post-hoc RMSE ratio
- Energy transfer is tested through a feature with two training levels
- Cross-level folds carry no measured residual-overlap bound
- L2_E2 max gap 53.66 km against 11.06–12.26 elsewhere, 4.4× the design target —
  and it is also the smallest cell at 25 tori, which is the same fact
- Mechanism is attributable for the family fold only; energy-fold divergence has
  no single-input account
- T5's failure magnitude varies across seeds by as much as its own size
- `t_solve_sec` measured on halos offset from the dataset energies by up to
  0.09% of C — immaterial for timing, stated for completeness
