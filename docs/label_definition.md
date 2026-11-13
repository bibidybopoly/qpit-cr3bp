# Label Definition: Why the Target Is a Distance, Not a Class

**Status:** methods draft, Week 10 (Phase 3).
**Supersedes:** the classification framing in the Phase 1 research question.
**Code:** `src/torus_field.py`, `src/densify.py`, `src/sampling.py`,
`src/torus_probe.py`. Tests: `tests/test_torus_field.py`,
`tests/test_torus_probe.py`. Experiments: `notebooks/nb10_gap_experiment.ipynb`,
`notebooks/nb10_resonance_control.ipynb`, `notebooks/nb10_dataset_generation.ipynb`.

---

## Summary

The project set out to train a network to classify whether a phase-space state
lies on a quasi-periodic invariant torus (QPIT). It now trains a network to
predict the **distance from a state to the nearest QPIT**, with classification
recovered as a thresholded readout reported against tolerance.

The change was not a preference. Each step below was forced by a measurement,
and the measurements are recorded here with the numbers that produced them.

Throughout: L2 halo #29, state
`[1.043805201996, 0, -0.075586294085, 0, 0.392206924669, 0]`,
period `T = 3.131511429611`, `C = 3.060334462`, `rho/2pi = 0.29322914`,
`LU = 384400 km`, `mu = 0.01215058560962404`, DOP853 at `rtol = atol = 1e-12`.

---

## 1. The original question is degenerate as posed

Phase space is six-dimensional. Conservation of the Jacobi constant confines
motion to a five-dimensional energy manifold. The tori are two-dimensional, and
the halo family is a one-parameter family, so the union of tori is
four-dimensional in phase space and **three-dimensional at fixed `C`** — codimension
two, and therefore Lebesgue measure zero.

A sharpening matters here, because it is easy to wave at KAM theory and assume
the problem is well posed. KAM's positive-measure guarantee concerns
**maximal-dimension** tori, which in a three-degree-of-freedom system are
3-tori. The objects in this project are lower-dimensional, partially hyperbolic
2-tori living in the center manifold of a periodic orbit whose monodromy also
carries a hyperbolic pair. These are different objects with different measure.
The positive-measure KAM tori do exist somewhere in the energy manifold; they
are not what is computed here and not what the labels denote.

**Consequence.** "Classify torus membership for arbitrary initial conditions,"
read literally, admits the solution *always answer no*, at 100% accuracy. Draw an
initial condition uniformly from any reasonable region and the probability it
lies on a QPIT is exactly zero. The question requires an operational amendment,
and that amendment **is** the label definition.

Three candidates were identified:

| | definition | status |
|---|---|---|
| **A** | positive if within `eps` of a computed torus | rejected, §2 |
| **B** | positive if the trajectory *is* quasi-periodic | rejected, §3 |
| **C** | regression on distance to the nearest torus | adopted, §4–5 |

Definition A was set aside early on two grounds. `eps` is arbitrary; and because
these tori are partially hyperbolic, a point `eps` away along the unstable
direction is `eps`-close in state space while behaving nothing like the torus, so
the distance being thresholded is not the quantity that matters. B was adopted
first: a behavioral test, thresholded against a noise floor measured rather than
chosen.

---

## 2. The gap experiment eliminates the naive negative class

**Motivation.** Any labeling scheme of the form "on my computed surfaces,
positive; nearby and off them, negative" assumes the space between adjacent
computed tori is empty. At fixed `C` the center-manifold dynamics reduce to a
two-dimensional area-preserving map whose invariant circles form a Cantor set,
interleaved with resonant island chains. The gaps should be *mostly full* of tori
that were simply never computed. Labeling those negative is systematic, not
random, error: the network would be trained to reject exactly the objects it is
meant to find.

**Method.** For each of the 16 adjacent pairs in a 17-member constant-`C` family,
five interpolants `U_alpha = (1-alpha) U_a + alpha U_b` were formed at
`alpha = 0.2, 0.4, 0.5, 0.6, 0.8`, each node re-projected exactly onto the
`C` level set, then seeded into `converge_torus` with `C` pinned and
`U_ref = U_cand`.

Energy re-projection is closed-form and reuses tested code rather than
duplicating a convention: since `C = 2*Omega - |v|^2` and `Omega` depends only on
position, holding position fixed gives `|v_new| = sqrt(C(x) + |v|^2 - C_target)`.
Interpolating two states of a level set leaves it by `O(delta^2)`; sampling a gap
at an energy that drifts with `alpha` would reproduce a known Phase 2
contamination bug in new clothes.

**Result: 80 of 80 candidates converged to genuine tori**, across gaps from
1.08 km to 62.50 km.

**Internal validation.** Midpoint displacement against gap width fits a power law
with exponent **2.003** — pure second-order interpolation error, which is what
linear interpolation across a curved family must produce and what nothing else in
the construction would. Independently, pairs 10–11 and 15–16 have nearly
identical absolute gaps (62.50 vs 62.47 km) at very different amplitudes
(140 vs 452 km) and return nearly identical *raw* displacement (2.6e-3 vs
2.9e-3 km), so the score tracks the gap and not the amplitude.

**Interpretation.** The probe was measuring the family's own curvature, meaning
the interpolants lie *on* tori. The positive class is not a sparse set of surfaces
with empty space between them; it is a nearly solid three-dimensional sheet with
exponentially thin holes at resonances. "Off my seventeen computed surfaces,
therefore negative" is wrong essentially everywhere in the gap — not imperfect,
wrong.

---

## 3. The shadowing bound eliminates behavioral labeling

**The bound.** `stability_index` on the mid-family member (A = 202.5 km) gives
`nu = 73.178054`, so `R_u = 146.318672`, confirmed two ways in the same cell
(power iteration and `max|eig(DG)|` agreeing to five digits, with
`R_u * R_s = 0.99921`). The torus reproduces the halo's own
`|lambda_u| = 146.350278` to four decimals, so the bound can be read off the
periodic orbit before any torus is computed.

These tori are partially hyperbolic, so the unstable bundle amplifies **any**
displacement by `R_u` per stroboscopic period — including integrator error, which
is such a displacement. It reaches `O(1)` after

```
n_max = log(1/eps) / log(R_u) = log(10^12) / log(146.32) = 5.54 periods
```

about four of them clean. Past that the trajectory is no longer near the torus it
started on, so any long-time diagnostic measures the escape rather than the object.

**Confirmed empirically, not merely inferred.** At a 60-period window, all three
arms of the experiment escaped — *including the positive control*. States sampled
exactly on a converged torus left the region before the window ended.

**Every forward-time indicator fails, and for the same reason.**

- **NAFF frequency drift** requires ~200 periods for a usable noise floor. Short
  by a factor of 72.
- **Weighted Birkhoff averaging (Das–Yorke)**, the standard tool for exactly the
  short-orbit regime, was benchmarked on the Chirikov standard map against
  ground truth from an independently computed Lyapunov exponent. It needs
  roughly **32 iterates** for >95% classification; at 8 it reaches only an
  optimistically-thresholded 85%. At the four available the half-length is
  `N = 2`, the weight sum degenerates, and the estimator returns **NaN** — not
  inaccurate, undefined.
- FLI, MEGNO, and any other trajectory-based indicator fail identically.

**This is not a tolerance problem.** Escape time scales as
`log(1/eps)/log(R_u)`, so recovering a factor of 72 in window length would
require `eps ~ 10^-865`. Double precision is not the binding constraint; the
hyperbolicity is.

**This is the reason GMOS exists.** These tori are **computable but not
observable**. Olikara and Baresi solve a boundary-value problem rather than
integrating forward, and the shadowing bound is why that is necessary rather than
merely convenient.

*Method-development note.* The weighted-Birkhoff implementation and its rejection
measurement are retained in `src/labeling.py` rather than deleted. An
evaluated-and-rejected method with numbers attached is worth more in a methods
section than a method that was never tried.

---

## 4. The surviving test returns a distance, not a class

`converge_torus` never integrates longer than one stroboscopic period, so the
shadowing bound does not bind on it. It is therefore the only available probe.

But it is insensitive to a small displacement along the unstable bundle, because
a torus genuinely *does* pass within `1e-6` of such a point. Its honest output was
never binary.

**The bundle control makes this decisive.** Displacing a converged member along
its own unstable bundle by a controlled amount:

| imposed `d` | measured | ratio |
|---|---|---|
| 1e-8 | 7.914e-6 | — |
| 1e-7 | 7.915e-5 | linear to 4 figures |

and the **binary label flips across that step**: a displacement of **1.6 m reads
positive, 16 m reads negative, on a 202 km torus**. The underlying physics is
smooth and monotone over four decades; the bit is an arbitrary cut placed through
it, and where the cut falls does more work than the dynamics.

Beyond `d ~ 1.78e-7` the ratio jumps from 1.14 to 160 in a single grid step of
1.78x — Newton leaving the basin of the nearby torus for a different one. That
break was located by a fine sweep and is the origin of the censoring flag.

**B is also uninformative where it remains defined.** Given §2, the interesting
variation under a binary label lives in exactly two places: off the center
manifold, which is nearly trivial to separate, and inside resonance gaps, where
the distance field goes non-smooth. Both are questions about a distance. Only one
of them survives being collapsed to a bit.

---

## 5. Definition C, and what it bought

**The target.** `d(x) = min over S of ||x - s||`, where
`S = { u_alpha(theta) }` is the family treated as a **continuous surface** at fixed
`C`, in six dimensions, reported as `log10 d` in km.

Six dimensions, not position-only: the network receives a full state, so a
position-only label would assign identical targets to two states at the same point
with different velocities, one of which may be on a torus and the other not.
Nondimensional CR3BP units already fix the scaling, and `converge_torus`'s own
convergence test is a mixed 6-D infinity norm.

**Surface, not finite set.** A point midway between members spaced `g` apart sits
`~g/2` from both, while its true nearest torus is at essentially zero distance
(§2). Enumerating members would overestimate by `g/2` and require thousands of
them. Interpolating instead puts the error at exactly the quantity §2 measured,
`c*g^2` with `c = 2.6e-3 km / (62.5 km)^2 = 6.656e-7 /km`. The `alpha`
minimization is closed-form, since the interpolant is linear in `alpha`, leaving
only a one-dimensional minimization over `theta`.

**Accuracy floor is measured, not assumed.** Re-deriving a torus by a different
route and asking how far it lands from the surface gives **2–155 mm** (mean 58 mm)
across five members. The surface is therefore not determined below ~0.155 m
however finely the family is densified — this is the `sigma_3 ~ 2.0e-4` result
from Phase 2, where the small singular value amplifies residuals so that the
`8e-13` invariance residual overstates real curve determination by roughly seven
orders of magnitude. **Densifying past `g = 12.3 km` buys nothing**: the curves
themselves are the limit, not the spacing.

**What the reframing unlocked.**

| | Newton per sample (B) | surface distance (C) |
|---|---|---|
| throughput | ~0.4 states/s | **321 states/s** |
| dynamic range | capped at 34.2 m | unlimited |
| absolute accuracy | ~10% (calibration 1.1433) | **0.04%** |
| calibration constant | required, direction-dependent | none |
| sensitive to `R_u` | yes (fatally) | no |

The last row matters for scope: L1 halos are more unstable than L2, which would
have been fatal to any behavioral scheme and is now irrelevant, because nothing
integrates forward.

Cross-validated against the probe as an independent measurement across four
decades: the surface recovers imposed displacements to **0.04%**
(9.9993e-10 against 1.0e-09; 9.9961e-09 against 1.0e-08), while the probe sits
~10% low even after its 1.1433 calibration. The residual disagreement is the
probe's — its calibration was measured on unstable-bundle displacements and does
not transfer exactly to arbitrary directions. This inverts the roles: the surface
is the reference and the probe is the cross-check.

**Classification is recovered for free** by thresholding, and is reported as an
**accuracy-versus-tolerance curve** rather than a single number. This makes the
threshold a stated dimension of the result instead of a hidden choice.

---

## 6. What was given up

**The resonance-degradation claim in its classifier-accuracy form.** Two
constructions were attempted and neither yielded usable negatives.

- **2/7** (`dC = +1.888e-3`, halo landing at `C = 3.062235`, predicted crossing
  matched to three parts per million). The family crossed cleanly between members
  19 and 20 and continued to 968 km. `q = 7` is not low order: the island chain is
  far thinner than the 56 km continuation step. Selecting the target by *nearest
  in rho* rather than *smallest q* was the error.
- **1/4** (`delta = 1.900e-5`, halo at `C = 3.071865`, predicted
  `A_res = 506.2 km`). Here the resonance is unmistakable — the `kappa*A^2` model
  breaks at `A = 453 km`, **90% of predicted**, after holding to 0.3% over a 23x
  amplitude range; the family is then repelled as `d ~ A^-2.57`, a power law that
  never reaches zero at finite amplitude; and the Fourier resolution adapts
  **31 -> 41 -> 51** as the curve develops structure. But the extrapolation-error
  budget, measured against targets where a torus is *known* to exist, collapses
  from 75 km far from resonance to **zero** at the nearest pair. Non-convergence
  there cannot be separated from a poor seed.

The obstruction is structural rather than incidental: the family curvature that
signals the resonance is the same thing that destroys linear extrapolation, so
this construction can never separate them.

**Q2 survives as characterization.** Resonance detectability has a sharp threshold
in `q` — invisible at `q = 7`, strong at `q = 4` — and near a detectable resonance
the frequency map departs from `kappa*A^2` while the invariant curve's Fourier
content grows. All measured. None of it is an ML result, and the paper should not
imply otherwise.

**Honest costs of C.** Regression metrics read less immediately than "94%
accurate." And the distance field is genuinely non-smooth, because the tori form a
Cantor set, so `R^2` will look mediocre for reasons that are physics rather than
modeling failure. That needs framing in the results section rather than apology.

---

## 7. Why this is the version that serves the stated purpose

The project's purpose is a model useful to future researchers. The concrete use is
warm-starting a GMOS or collocation solve: telling it where to look.

A network predicting **distance to the nearest torus** is a natural initializer
for that. A classifier is not — a bit carries no gradient toward the object.
Definition C is therefore not only the version the physics permits, it is the
version that matches what the model is for.

---

## Appendix: measurements referenced

| quantity | value | source |
|---|---|---|
| `nu`, `R_u` (A = 202.5 km) | 73.178054, 146.318672 | `qpit_family.ipynb` cell 22 |
| halo unstable multiplier | 146.350278 | `halo_stability` on #29 |
| shadowing bound `n_max` | 5.54 periods | `log(1e12)/log(R_u)` |
| gap experiment | 80/80 converged | `nb10_gap_experiment` |
| gap displacement exponent | 2.003 | ibid., log-log fit |
| interpolation coefficient `c` | 6.656e-7 /km | ibid. |
| weighted Birkhoff requirement | ~32 iterates | standard map vs MLE |
| probe linear regime break | 1.78e-7 to 3.16e-7 | censor sweep |
| probe Newton overshoot | 1.1433 | ibid., 4 decades |
| censor level | 8.891e-8 = 34.2 m | ibid., amplitude-independent |
| surface reproducibility floor | 2–155 mm (mean 58) | 5 members, re-derived |
| surface absolute accuracy | 0.04% | vs imposed displacement |
| surface throughput | 321 states/s | 400 states |
| `kappa` (C = 3.060334) | -11.080552 | 4 independent validations |
| `kappa` (C = 3.071865) | -10.9535 | local fit, 1.1% drift |
| `drho_halo/dC` | -3.981297 | crossing predicted to 3e-6 in C |
