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

## 5. Disposed 2026-09-06 — six of seven closed

These were deferred to "ride with the regeneration decision". Regeneration is
cut, so they rode nowhere and were settled directly.

| item | disposition |
|------|-------------|
| **`mknb10.py` missing** | **CLOSED — the item was overstated.** The dataset IS regenerable: `nb10_dataset_generation.ipynb` builds it end to end and the sampling seed is recorded (`default_rng(20260810)`, created once before the family loop, so draw order is fixed). What is missing is a *builder*, a convention gap, not a reproducibility one. `notebooks/verify_w10.py` added: every column derivable from other stored columns re-derives exactly under the shipped `sampling.py`, `group_id` included. All checks pass. The ten columns needing a full rebuild are named, not passed over. **Reproducer's trap recorded:** `build()` resumes from `../data/families/{tag}.npz` when present, so a from-scratch run must clear that directory or it silently reuses cached families |
| **The `*_KNOWN_DEFECT` pins** | **CLOSED — there is ONE, not four.** `tests/test_sampling.py:143`, `test_normal_directions_still_carry_a_theta_tangent_component_KNOWN_DEFECT`. The handoff's count of four is wrong; either three were removed after 2026-08-30 or the number was approximate. The pin is correct, well-documented, and states its own fix and why it was not applied. **Keep it.** It asserts the residual is present (`resid.max() > 1e-3`) and that the tangents are non-orthogonal (`|cos| > 0.1`), so it fails loudly the day someone fixes `normal_directions`, which is what a pin is for |
| **Notebook constant copies** | **CLOSED — not reproducible.** Neither `halo_families.ipynb` nor `floquet_analysis.ipynb` contains a literal copy of any `constants.py` value |
| **The `normal_directions` fix** | **DISCLOSE, DO NOT FIX — and my first closure of this was wrong.** See §5a |
| **`first_orbit` / `amplitude_direction`** | **CLOSED.** Promoted to `src/orbits.py` with `halo_family()`, deferred `cr3bp_3d` import to match `compute_halo_family`. `time_solve.py` imports them; both notebooks still need their local copies replaced |
| **`week11_m1_r2` residual** | **CLOSED as a disposition, not an explanation.** The reported pair is internally inconsistent: 0.6288 matches the test-mean convention (archive gives 0.6280), 0.0214 matches a fixed-reference convention (archive gives 0.0215 with mean 0.6451). No single convention produces both, and `score_C` is ruled out (identical to `log10(max(d_km, floor))` to 4.4e-16). **Quote 0.6280 ± 0.0226**, which is what a reviewer rerunning the artifact gets |
| **Censored-miss base rate** | **CLOSED, and it survives.** Held-out censored rows are L1 = 277, L2 = 262, so p(L2) = 0.4861. Against that, 16/23 gives expected 11.18 ± 2.40, z = +2.01, one-sided binomial P = 0.035 (≈0.07 two-sided). Marginal at n = 23, and it runs **opposite** to the far-field per-cell MI where L2 carried about twice L1. Limitations, as a stated tension rather than a finding |

### 5a. The `normal_directions` defect, reopened

**I closed this on a pooled median and the structure was in the line below it.**
I reported "median ratio 0.9999 … and 0.9919–1.0000 across the six cells" and
read the spread as noise. It is not noise. It is one cell.

`normal_directions` projects the two surface tangents out sequentially. Removing
the `t_al` component re-introduces a `t_th` component because the tangents are
not orthogonal, so only the tangent projected out **last** is actually cleared.
The pin at `tests/test_sampling.py:143` documents this, names the fix
(orthonormalise before projecting), and says it was not applied because it
changes what `mknb10` generates and so had to ride with the regeneration
decision — which is cut.

Measured on the archive, per cell, normal-direction rows only, ratio of measured
to drawn distance:

| cell | n | median | p05 | p01 | min |
|------|---|--------|-----|-----|-----|
| L1_E0 | 1985 | 0.9995 | 0.9878 | 0.9635 | 0.8159 |
| L1_E1 | 1960 | 0.9998 | 0.9956 | 0.9888 | 0.9053 |
| L1_E2 | 1973 | 1.0000 | 0.9998 | 0.9995 | 0.9918 |
| **L2_E0** | **1922** | **0.9919** | **0.8510** | **0.6934** | **0.2522** |
| L2_E1 | 1984 | 0.9999 | 0.9986 | 0.9951 | 0.9179 |
| L2_E2 | 1945 | 1.0000 | 0.9995 | 0.9984 | 0.9906 |

L2_E0 is 0.81% short in the median and 14.9% at p05. **The pin documents 0.9%
and 14%, measured independently on `data/families/L2_E0.npz`.** It is right to
two significant figures and my pooled number was wrong. Within L2_E0 the effect
is uniform across gap index (0.991–0.995), so it is a property of that cell's
tangent geometry rather than of particular gaps. One sample landed 75% short.

The pin's second claim also holds: **`random_dir` no longer cleanly separates
the two direction regimes.** 4.8% of L2_E0's normal-direction rows fall below
the random-direction median, against 0.0–0.1% in every other cell.

**Bearing on results.** Labels are unaffected — `d_km` comes from
`surface_distance`, measured after the fact. The distortion is in coverage: the
realised distance distribution in L2_E0 is pulled toward the surface relative to
the log-uniform draw. Its far-field share is 0.4895, the lowest of the six
(range 0.4895–0.5094), which is a 2% relative deficit and the direction the
defect predicts. L2_E0 sits in the **test** set of T1 and T3, so both carry
slightly fewer far-field rows than intended. Too small to move an RMSE ratio of
2.64 or 14.12, and it must still be stated.

**Disposition: disclose, do not fix.** The fix requires regeneration.

### Corrections this pass forced on earlier documents

- **The archive does not carry a per-state Jacobi.** `label_and_pack` writes
  `jacobi=np.full(n, C)`, so `jacobi` and `C` are byte-identical. The dangerous
  column in the `features.py` audit is the *recomputed* per-state Jacobi, which
  is not in the archive at all. `assert_sheet_energy` still earns its place, but
  the comment in `mknb15b.py` claiming the archive carries both is wrong
- **My closure of the `normal_directions` item was retracted the same day it
  was written**, on evidence I had already printed. A pooled median over six
  cells hid a defect confined to one. Same failure as the single-seed probe
  earlier in Week 15: a summary statistic standing in for a distribution
- **`alpha_found`/`theta_found` are computed and then discarded**, not
  never-produced. `label_and_pack` returns them; cell 13's `savez_compressed`
  writes an explicit key allowlist and they are not on it. `nearest_point_overlap`
  documents exactly those columns as its inputs, so the dataset's writer and that
  instrument disagree about what the archive contains. Two names on that list
  would have fixed it, and regeneration is cut — an independent second reason
  the instrument is not applicable in Week 15

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
