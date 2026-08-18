# Research log — correction and backfill worksheet

**How to use this.** Every number below is a measurement from your own runs, recoverable
from your notebooks. What I have supplied is the facts, organised into the right dates.
What I have deliberately *not* supplied is your interpretation, your reflections, or your
"stuck on" fields — those have to be yours, and not because of a rule but because they are
the only part of the log that carries evidence of your thinking.

**Rewrite the prose.** The skeleton paragraphs are marked `[SKELETON]`. Treat them as a
list of facts in sentence order, not as text to paste. If a judge reads twenty entries in
your voice and three in someone else's, the three stand out.

**Date your amendments; do not silently rewrite history.** A log is a contemporaneous
record, and backdating reconstructed content into it is the thing that makes a log
worthless as evidence. Use this convention:

```
## 2026-08-06

[original entry, untouched]

**Amended 2026-08-14** — backfilled from notebooks/nb09_continuation.ipynb;
numbers were recorded in the notebook at the time but not carried into this log.
- ...
```

That is honest, it is normal scientific practice, and it converts a gap into a
documented correction. Your Risk 12 ("AI policy violation, real or perceived") is served
by this too: an amendment note dated after the fact, in your voice, with the source
notebook named, is unambiguous.

**Verify before you commit.** Items marked ⚠️ are ones I could not confirm and you need to
check against the actual file or run. 

# Part 2 — Corrections (entries that state something incorrect)

## 2026-06-19

**What's wrong.** Three conservation failures are logged on 06-19 (near L1, tangential
velocity near Earth, near the Moon) and never resolved anywhere in the log. On 06-23 the
L1 deviation is attributed to "L1 being the most chaotic." That explanation is wrong as
stated: the Jacobi constant is an exact integral of motion in the CR3BP and is conserved
along every trajectory regardless of chaos. Your tracker's Week 2 deliverable was
conservation to ~1e-10 and no number is recorded anywhere.

**[SKELETON — amendment]**

> Amended [date]. The 06-19 conservation failures were never closed out in this log.
> Re-running the same four tests at rtol = atol = [___] gives max |ΔC/C| of [___] (L4),
> [___] (L1), [___] (near-Earth tangential), [___] (near-Moon). The Week 2 gate required
> ~1e-10; [these pass / these fail and here is what I did]. 

⚠️ **You need to actually re-run this.** Do not write a number you have not measured. This
is the foundation under every subsequent result, and it is the one gap in the log where an
unrecorded failure sits under everything built on top of it.

---


## 2026-08-13 — the Week 11 entry (largest correction)

**What's wrong.** Three separate problems.

**(a) The run this entry describes was invalidated.** `src/features.py` contained an audit
step that noticed the stored `C` column was constant-per-sheet, treated that as a defect,
and recomputed each state's own Jacobi constant to replace it. Because the Week 10 offsets
were never reprojected onto the energy manifold, a state's own Jacobi constant differs from
its sheet's in proportion to the offset — which is the regression target. That column scored
R² = +0.7976 alone; the stored sheet energy scores −0.0001. B1's 0.816 and M1's 0.567 were
both measured through the leak. The bug was diagnosed by `notebooks/nb11_diag_shortcut.py`,
`features.py` was corrected, and the notebook was rerun.

**(b) "Below around 10² meters, the network stops closing in" conflates two different
floors** — and 10² m = 100 m matches neither.

| | what it is | cause | number |
|---|---|---|---|
| Label floor | property of the **dataset** | σ₃ ≈ 2.0e-4 limits how well the surface itself is determined | 155 mm; 2,502 rows (12%) flagged `at_floor` |
| Near-field degradation | property of the **model** | ∇(log d) ~ 1/d is unbounded; a network's gradient is bounded by the product of its weight norms | P3a: near-field RMSE 1.2285 vs far-field 1.1019. P3b: near-field bias +1.1819 |

⚠️ Work out which one you meant and split them into two statements.

**(c) "Median error of 6.3×" is quoted without naming the statistic, and it is the
flattering one.** M1's RMSE ≈ 1.26 in log₁₀ implies a typical error of about a **factor of
18** in the distance handed to a solver. A median of 6.3× is consistent with that (RMSE is
tail-dominated), but reporting only the median when the RMSE is three times worse is what a
reviewer catches. Report both, or report the accuracy-vs-tolerance curve.

**[SKELETON — replacement entry]**

> **Worked on:** First regressor — invalidated run, diagnosis, correction, clean rerun.
>
> The first run of `nb11_first_regressor.ipynb` produced an anomaly: the B1 shortcut baseline
> scored R² = +0.816 where it was predicted to score near zero, and M1 — with strictly more
> information — scored *worse* at 0.567. A model that underperforms its own restriction is
> not a modelling failure, it is a signal that a feature carries the answer.
>
> Diagnosed with `nb11_diag_shortcut.py` (ablations, grouped split, same learner throughout).
> My stated hypothesis was that the stored `C` column encoded the offset. **That was
> falsified.** The stored `C` has exactly three distinct values — 3.029398206, 3.064703461,
> 3.130606605 — with zero spread inside each sheet, identical across L1 and L2. It is the
> sheet energy, and it is clean: R² = −0.0001 alone. The leak was the **recomputed per-state
> Jacobi constant**, which `features.py` was substituting in: R² = +0.7976 alone.
>
> Why the per-state Jacobi is an artifact rather than legitimate information: at inference
> you *can* compute a query state's C, so it looks fair. It is not, because only three energy
> sheets were sampled. With three, "distance in energy from the nearest sheet" is a
> near-deterministic function of the offset. Real QPIT families fill a continuous energy
> range, so every state sits close in energy to *some* torus and the proxy evaporates. This is
> a **coverage** artifact — the constraint the tracker named when it retired Risk 7.
>
> Corrected ablation, grouped split:
>
> | features | R² |
> |---|---|
> | positions only (dist L1, dist L2) | +0.3606 |
> | positions + sheet C | +0.3615 |
> | full state, no C | +0.2253 |
> | full state + sheet C | +0.2372 |
>
> Note that the full state scores *below* engineered distances — axis-aligned tree splits
> cannot form a Euclidean radius. Legitimate feature-engineering effect, belongs in Methods.
>
> **Scorecard, clean rerun** (predictions unchanged from before the first run):
>
> | | outcome | |
> |---|---|---|
> | P1 shortcut baseline fails (R² < 0.5) | HELD | B1 = +0.3615 |
> | P2 1-NN gap random > grouped by >0.1 | FALSIFIED | random −0.0336, grouped −0.0305, gap −0.0031 |
> | P3a near-field RMSE exceeds far-field | HELD | 1.2285 vs 1.1019 |
> | P3b near-field bias positive | HELD | +1.1819 |
> | P3c overall R² > 0.9 | FALSIFIED | M1 = +0.5391 |
>
> **Week 11 result: M1 = 0.539, B1 = 0.362, B0 ≈ 0.** The regressor beats the shortcut
> baseline by ~0.18 R² and beats the mean predictor decisively.
>
> P2's falsification is uninformative rather than meaningful: both B2 scores are negative, and
> a baseline at zero cannot resolve a gap. The evidence about row dependence comes from the
> group census instead — 6,564 groups over 20,400 rows, median group size 2, 1,762 singletons,
> groups/rows = 0.32. Base points were drawn continuously, so rows are nearly independent by
> construction and there was little duplication for grouping to protect against. Good news
> about the data; it also means `group_id` is doing almost no work.
>
> **Retraction to record.** The scorecard as first written printed "P3c falsified means Risk
> 19 was right and the Cantor-set framing stands." That inference is wrong and must not reach
> the paper. The labeller measures distance to a densified *interpolated continuum* at ~12.3 km
> spacing, so it cannot resolve the gaps between tori — that structure is not present in the
> target as generated. **Risk 19 stays untested; this experiment has no power over it.**
>
> **Risk 20, fourth entry.** Not a unit slip this time but the same species — a plausible
> wrong number from a derived quantity, produced by code that looked like a safety check. The
> pattern worth logging is that the *remedy* was the defect.

**Also record from this week:** two bugs caught, both silent. `nearest_point_overlap`
computed its mixed-radix packing from each side's own key array (train all-zero → radix 2,
test all-one → radix 3, arbitrary collisions). And the MLP's `patience=30` default stopped a
fit after ~95 gradient steps and returned a mean-predictor while reporting
`early_stopped=True` as though converged — patience counts epochs, and an epoch is not a
fixed amount of work.

---

# Part 3 — Backfill (things that happened but were never recorded)

## 2026-07-02 → 2026-08-03 — camp bridge entry

A month with no entry. You did project work there. Add a dated bridge entry.

**[SKELETON]**

> **2026-08-02 — National Youth Science Camp, retrospective.** No daily entries were possible.
> Completed: [reading list], selected GMOS as the computational method. Not completed as
> planned: the invariance-equation derivation and the solver architecture spec, and the
> Olikara/Baresi/Howell emails — all carried into the week of 08-03. Recording the slip
> because the emails were scheduled for Week 2 in the risk plan and went out in Week 7.

Also record the **GMOS decision reasoning**, which appears on 08-03 but should be complete —
four reasons: prevalence in the relevant literature; superior accuracy per Baresi-Olikara-
Scheeres 2018; reuse of Phase 2 infrastructure (DOP853, STM propagation, differential
corrector, pseudo-arclength continuation); and stability information nearly free via the
rotated monodromy matrix `A = R_{-ρ}[∂φ_T/∂u]`, assembled from STMs already being propagated.
Also worth recording what you gave up: an integrator inside the Newton loop, and ω₁/ω₂ as
derived rather than explicit unknowns.

---

## 2026-08-05 — first QPIT convergence (Week 8 content)

Currently six lines for the single most important computational milestone in the project.
Missing:

- **Rank deficiency of exactly 2**, measured by SVD post-convergence: gap of 8×10⁸ between
  σ[−2] and σ[−3], both nulls at machine zero. The two symmetry tangents (latitudinal,
  longitudinal) lie in the nullspace to 12-digit precision.
- **Why the pre-convergence measurement doesn't count**: at the seed the gap was only ~10×,
  polluted at O(ε²) by seed error. The symmetries are exactly flat only *at* a solution, so a
  rank measured off-solution reports your distance from the solution rather than the rank.
- **The correct closure**: two phase constraints plus mean Jacobi fixed, with ρ and T both
  free. (Note the Week 9 correction: an earlier claim that the closure fixed both ρ and C was
  wrong — it fixed only C. Pinning both is inconsistent because the seed's linear ρ differs
  from the true torus ρ at O(ε²).)
- **Amplitude envelope**: converges cleanly from ε = 1e-5 through 2e-3 (344 km); collapses
  at 3e-3.
- Olikara Ch. 5 (conditioning and regularisation) read before the coding began.

---

## 2026-08-06 — family continuation (Week 9 content)

The amplitudes and member counts are recorded; the validation is not. Missing:

- **The κ three-way consistency check.** κ = c · dρ_halo/dC predicts **−11.080**; constant-C
  continuation measures **−11.081**. Four significant figures from two independent routes.
  This is the strongest validation result in the project and it is not in the log.
- **The κ sign correction.** The Week 8 table had the wrong sign because `C_target` came from
  each seed, mixing energy and amplitude effects. The true fixed-energy coefficient is
  negative: ρ *decreases* with amplitude at fixed energy.
- **The constant-ρ family drifts toward lower C**, making the nearest resonance 2/7 ≈ 0.285714
  (below ρ₀/2π = 0.2932), reachable near A ≈ 10,000 km.
- **Two silent-wrong-answer bugs fixed.** `cr3bp_3d.halo_corrector`'s y = 0 event fired at
  t = 0, producing T = 0 with a false "converged" status. `converge_torus` defaulted
  `C_target` to the seed's energy rather than the halo's without saying so, which had an
  ε-sweep comparing tori at different energies.
- **`curve_amplitude` is resolution-dependent** — max over nodes understates by ~0.096% at
  N = 31 (17.1644 km node-max vs 17.1815 km dense). Prefer dense evaluation.
- **The accuracy floor.** Re-deriving a torus by a different route lands 2–155 mm from the
  surface. This comes from the small singular value σ₃ ≈ 2.0e-4 amplifying residuals, and it
  is structural — not fixable by tightening integration tolerances. This is the number that
  becomes the label floor in Week 10.

---

## 2026-08-07 → 2026-08-09 — the label reframe, with its actual argument

**What's wrong with 08-09 as written.** "Impossible to generate usable negatives" is a
symptom, not the argument, and not one number appears. Also, the 08-07 plan (test candidate
negatives by seeing whether they seed a QPIT) was subsequently *falsified* and the log never
records that it was abandoned or why.

The real chain, each link forced by a measurement:

1. **Measure-zero geometry.** Phase space is 6-D; Jacobi conservation confines motion to a
   5-D energy manifold; tori are 2-D and the family sweeps them into a 3-D sheet at fixed C.
   Codimension two, hence Lebesgue measure zero. "Always answer no" scores 100%. Sharpening:
   KAM's positive-measure guarantee concerns *maximal-dimension* tori, which in three degrees
   of freedom are 3-tori. Ours are 2-tori, so the guarantee does not apply.
2. **The tolerance patch fails.** With a tolerance the label becomes a property of the
   tolerance. Measured: the binary label flipped between **1.6 m and 16 m** of displacement on
   a **202 km** torus, and at a 10× margin four points all sitting on genuine tori read
   False/True/True/False depending only on phase, with underlying distances differing by 3×.
3. **All forward-time indicators are dead.** R_u = 146.32 on L2 halo #29 → at rtol 1e-12,
   integrator error reaches O(1) after about **four** stroboscopic periods. NAFF, weighted
   Birkhoff / Das-Yorke, FLI, and MEGNO each need one to two orders of magnitude more. The
   precision argument: each decimal digit buys ~0.46 periods, so even 30-digit arithmetic at
   ~50× the cost reaches ~12 periods. Symplectic methods control secular energy drift, which
   is not the failure here. **These tori are computable but not observable.** This is what
   killed the 08-07 plan.
4. **The gap is full of tori.** 80/80 candidates interpolated between computed members
   converged to genuine tori, with fitted exponent **2.003** — the signature of pure
   second-order interpolation error, i.e. a smooth continuous surface with linear
   interpolation as the only source of error. So the positive class is not 17 thin sheets; it
   is a nearly solid 3-D region with exponentially thin holes at resonances, and the naive
   labelling was wrong essentially everywhere.
5. **The probe score is not a distance.** It scales as **gap²** — chord-versus-arc again —
   so it reports local family curvature and your discretisation choice, not a property of the
   query point. Densify by 2× and the same state's score falls by 4× with nothing physical
   having moved.
6. **Therefore a geometric labeller.** `torus_field.surface_distance` measures distance to the
   interpolated family surface directly, with no integration: **320 states/s** vs ~0.4/s for
   Newton-per-sample, an 800× speedup. This is why the Titan port turned out to be unnecessary
   and Risk 7 was retired — dataset size stopped being the constraint and coverage became it.
7. **Regression on log distance, classification recovered by thresholding.** "Generate C,
   report B": store the continuous distance, train on it, report every classification metric
   as an accuracy-vs-tolerance *curve*. Strictly more information than a single accuracy
   number, and the arbitrary threshold becomes a reported axis instead of a hidden choice.

Also record the practical argument, which is now the primary claim in the tracker: a
distance field has a gradient a solver can descend, so it can warm-start a GMOS or
collocation solve. A classifier cannot — a bit carries no gradient toward the object.

---

## Resonance investigation — currently absent from the log entirely

A fully documented falsified hypothesis with a diagnosed cause. This is the best kind of
material a log can contain and there is no entry for it. ⚠️ Date it to when you actually ran
it.

**[SKELETON]**

> **Resonance control.** Predicted that continuing the family into the 2/7 resonance would
> terminate it. **Falsified** — the family crossed 2/7 without terminating. Diagnosis: I had
> selected the target by smallest ρ-distance rather than smallest denominator q, and q = 7 is
> not low-order, so the resonance zone is too thin to detect.
>
> Retargeted to 1/4, verified reachable and parameterised. **Second failure, structural**: as
> the family approaches 1/4 the extrapolation-error budget collapses from 75 km to 0, so
> non-convergence cannot be separated from a poor seed. The family curvature that signals the
> resonance is the same thing that destroys linear extrapolation. **Risk 18 realised.**
>
> Secondary Question 2 reframed from "ML accuracy degrades near resonances" to
> **characterisation**: detectability threshold in q (invisible at q = 7, strong at q = 4),
> κA² breakdown at 90% of predicted resonance amplitude, power-law repulsion d ~ A^−2.57,
> Fourier resolution growth 31 → 41 → 51. Do not spend further Phase 4 time on this.

---

## 2026-08-11 — dataset generation

"The generation took a while on my computer but not too long" is the entire record of the
Phase 3 exit gate. Missing:

- **20,400 rows**, L1 + L2 × 3 energies (C = 3.029398206, 3.064703461, 3.130606605).
- **Three-arm sampling**: on-surface, offset along normals, offset along random directions,
  with distance drawn log-uniform by design.
- **Floor measured, not guessed**: 155 mm, from the σ₃ result. Below it the surface is not
  determined however finely you densify, so `log_d_km = log10(max(d_km, 1.55e-4))` and rows
  below are flagged `at_floor`. Densifying past g = 12.3 km buys nothing.
- **Honesty note for the paper**: R² will be measured against a marginal distribution *you
  chose*, since d was drawn log-uniform by design. It is not a property of phase space. The
  baselines are the meaningful comparison.
- **Bug caught before generating at scale**: one of 200 on-surface points came back at 448 m
  instead of ~0, at a gap boundary. Would have corrupted the dataset silently.
- ⚠️ **Reconcile the date.** Your tracker records the Phase 3 exit gate as met Aug 9–10; this
  entry dates dataset completion to Aug 11.

---

# Part 4 — Habits to start now

**Mark gates.** Not one gate is marked as met in 1,200 lines — not Phase 2 exit, not the
Week 8 QPIT theory gate, not the make-or-break Phase 3 exit gate. Add a line to the relevant
entries: `**GATE: Phase 3 exit — MET.** [criteria] [evidence].` Your tracker has the gate
list.

**Log the mentor sessions.** Zero appear anywhere in the log, against a plan that mandates a
weekly 90-minute structured session with a written agenda. You are filing a COI disclosure
for a family mentor and writing a "What was your role in the research?" essay. The absence
of any documented boundary between your contribution and your father's is the most exposed
thing in this file. Going forward, per session: date, duration, agenda items you brought,
what he corrected or challenged, what you decided and who decided it. Backfill what you can
remember as a dated amendment — approximate and honest beats absent.

**Fill in "Stuck on / questions" — and it has to be you.** Right now it reads "None yet —
setup phase" in roughly twenty entries (still saying "setup phase" on 07-02) and is blank
from 08-04 onward. The log therefore claims you were never stuck across the entire project,
which is false — the degenerate calibration floor, the A_res formula error, the y = 0 event,
the silent `C_target` default, the nbformat newline bug, the mixed-radix packing bug, the
patience-in-epochs bug, and the feature leak all say otherwise. This is the field judges
read most closely, and it is the one I am not going to draft for you. Write what you did not
understand that day, in your own words, including the things you still do not understand.

**One line per number.** The recurring pattern in this audit is a claim recorded without its
measurement: "converged in 3 iterations" with no residual, "rho (torus) is slightly higher"
with no value, "took a while" with no runtime, "resolution floor" with no floor. If a
sentence asserts a quantity, the next characters should be that quantity and its units. Put
the units in the string, not the comment — that is your own Risk 20 mitigation.

---

# Priority order

If you only do part of this:

1. **08-13** — it is factually wrong and it omits the best material you have.
2. **The Risk 19 retraction** — this one is time-critical, because it is a claim that must
   not reach the paper.
3. **"Tori are near stable orbits"** — a self-contradiction a judge can find in one reading.
4. **Mentor sessions** — start today; backfill approximately.
5. **The 06-19 Jacobi failures** — requires an actual re-run, so start it early.
6. Everything else.
