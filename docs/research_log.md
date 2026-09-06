# Research Log

## 2026-05-27

**Worked on:** 

- Project planning and environment setup.

- 3Blue1Brown linear algebra series (math prep)



**Setup completed:**

- Python 3.12 installed

- Virtual environment created

- Core scientific Python stack installed (numpy, scipy, matplotlib, jupyter, pytorch)

- VSCode + Git configured

- Project folder structure created

- Git repository initialized



**Progress:**

- Identified research direction: ML for QPIT identification in CR3BP



**Next session:**

- Begin reading Koon et al. Chapter 1



## 2026-05-28

**Worked on:** 

- Further planning, math learning, and environment setup


**Setup completed:**

- Got the titan environment setup


**Reading completed:**

- Completed learning a basic understanding of linear algebra



**Stuck on / questions:**

- Is there any other ways of going about eigenvectors and eigendecomposition



**Next session:**

- Learn multivariable calc and read Koon and Olikara

-*Important notes
- - on titan the are three partition that are useful: bw (the main one), sb (larger one), and gpu (contains gpus)
- - sb and gpu could be used later for greater processing
- - on bw, PreemptMode=REQUEUE, this means jobs should be checkpointable or idempotent (smaller separate jobs)



## 2026-05-29

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Gained a basic understanding of multivarible equations



**Next session:**

- Learn multivariable calc and read Koon and Olihara



## 2026-05-31

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about partial derivatives of multivarible equations



**Next session:**

- Learn multivariable calc and read Koon and Olihara



## 2026-06-01

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about gradient and directional derivatives



**Next session:**

- Learn multivariable calc and read Koon and Olikara



## 2026-06-03

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about the chain rule for multivar and about curvature



**Next session:**

- Learn multivariable calc and read Koon and Olikara



## 2026-06-04

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about divergence and curl for vector-valued functions



**Next session:**

- Learn multivariable calc, specifically Laplacian and Jacobian, and read Koon and Olihara



## 2026-06-05

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about Jacobian, Laplacian, Lagrangian, tangent planes, and quadratic approximations.



**Next session:**

- Learn multivariable calc, specifically line integrals, and read Koon and Olikara



## 2026-06-06

**Worked on:** 

- Multivarible calc and reading



**Reading completed:**

- Learned about line integrals and read the first chapters of Koon and Olikara



**Next session:**

- Learn multivariable calc, specifically double integrals



## 2026-06-07

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about double integrals and triple integrals



**Next session:**

- Going to a summer camp and will have very little time to study Multivar Calc, will not be able to add entries into this log



## 2026-06-14

**Worked on:** 

- Came back from summer camp and worked on multivarible calc



**Reading completed:**

- Learned about surface integrals



**Next session:**

- Multivar calc, specifically flux and different theorems



## 2026-06-15

**Worked on:** 

- Multivarible calc



**Reading completed:**

- Learned about Stokes' theorem and variations of it



**Next session:**

- Final confirmation of comprehension of multivar calc and beginning ordinary diff eq



## 2026-06-16

**Worked on:** 

- Multivarible calc and ODE



**Reading completed:**

- Confirmed fluency of multivar and began learning ODE



**Next session:**

- Continuing ODE



## 2026-06-18

**Worked on:** 

- ODE



**Reading completed:**

- Watched the first few MIT OCW videos for ODE and began the first problem set.



**Next session:**

- Continuing ODE



## 2026-06-19

**Worked on:** 

- ODE and the cr3bp integrator



**Reading completed:**

- Finished the first ODE problem set

- Read the second chapter of Koon



**Progress:**
- Confirmed that the cr3bp integrator worked

-Tested Jacobi conservation when starting with small velocity near L4 - passed
-Tested Jacobi conservation when starting near L1 - failed
-Tested Jacobi conservation when starting with tangential velocity near Earth - failed
-Tested Jacobi conservation when starting near the Moon - failed

-Plotted the zero-velocity curves for several Jacobi constants



**Amended 8-14**

- The 06-19 conservation failures were never closed out in this log
- Re-running the same four tests at rtol = atol = 1e-12 gives max |ΔC/C| of 
- 1.55e-11 (L4)
- 1.65e-10 (L1)
- 3.95e-08 (near-Earth tangential)
- 1.50e-11 (near-Moon)
- The Week 2 gate required ~1e-10; near-Earth and L4 fail, while near-moon and L1 succeed



**Next session:**

- Continuing ODE



## 2026-06-20

**Worked on:** 

- ODE



**Reading completed:**

- Learned about systems of ODE



**Next session:**

- Continuing ODE



## 2026-06-22

**Worked on:** 

- ODE



**Reading completed:**

- Learned how to classify fixed points of system of ODE

- Read Pathak and learned about reservoir computing



**Next session:**

- Building a Lorenz Reservoir Computer



## 2026-06-23

**Worked on:** 

- Lorenz Reservoir Computer



**Reading completed:**

- Pytorch 60 Minute Blitz



**Progress:**

- Tested the cr3bp integrator on the Titan
- It worked but produced the greatest deviation around L1
- This is most likely due to L1 being the most chaotic

- Began the Reservoir Computer with the Lorenz Equations  
- The Lorenz attractor is just so beautiful
![Graph of the lorenz attractor.](fig_lorenz_attractor.png)

- When building the reservoir, it did not work unless it was fed normalized values
- It would be smart to normalize values down the line for the actual QPIT identifier too
- Its important to note that a model that scores well on training data may not score well on data it has never seen before
- The finished reservoir computer was able to match real values for around 2 Lyapunov times
- This result is worse than that of Pathak but the bug fixing and process were enough to learn the mechanics of a reservoir computer



**Amended 8-14**

- My 06-23 attribution of the L1 deviation to chaos was incomplete
- C is an exact integral of motion, so chaos cannot break conservation
- What degrades near L1 is the 1/r² and 1/r³ terms making the mechanics of the cr3bp locally stiff, which raises the integrator's local error per step
- Chaos amplifies the resulting state error but is not the source



**Next session:**

- Finishing ODE



## 2026-06-24

**Worked on:** 

- ODE 



**Reading completed:**

- Learned about nonlinear systems of ODE
- Learned you can use the local linearization to gain information about the phase portrait



## 2026-06-25

**Worked on:** 

- ODE and dynamical systems



**Reading completed:**

- Read Ch. 6 and 7 of Strogatz



**Amended 8-18**

**GATE: Phase 1 exit — MET.** CR3BP integrator + energy conservation; RC reproduces Lorenz; multivar+ODE fluency



**Next session:**

- Continuing dynamical Systems



## 2026-06-26

**Worked on:** 

- Reading



**Reading completed:**

- Read Koon Ch. 3



**Next session:**

- Continuing Reading




## 2026-06-27

**Worked on:** 

- Reading



**Reading completed:**

- Read Olikara Ch. 2 and Howell



**Next session:**

- Developing the single-shooting differential corrector



## 2026-06-28

**Worked on:** 

- Jacobian and STM integration



**Progress:**

- Derived the Jacobian for the CR3BP system

- Validated the STM against small differences and it worked greatly



**Next session:**

- Developing the single-shooting differential corrector



## 2026-06-29

**Worked on:** 

- Single-Shooting Differential Corrector



**Progress:**

- Built the single-shooting differential corrector and used it to find the first periodic orbit of the project

- Used the eigenvectors of the Jacobian Matrix around L1 to get a good initial guess then used the differential corrrector to quadratically converge closer and closer to real orbit

- The final QPIT identifier could be used to generate initial guesses for a similar process

- Generated a families of Lyapunov orbits around L1 and L2
- Ran into errors and diffulcties with the differential corrector jumping to different branches of families or zeroing out
- This error was caused by using a fix distance between consecutive orbits
- Fixed by using a pseudo arclength
![Graphed L1 and L2 lyapunov famililies in the planar Earth-moon cr3bp.](fig_lyapunov_families.png)



**Next session:**

- Floquet Theory



## 2026-06-29 (part 2)
### Did a lot today so decided to split it up

**Worked on:** 

- Floquet theory



**Reading completed:**

- What I learned today connected everything of the project so far
- I now have a much better understanding of the mechanics

- I previously thought that QPITs were an entirely different thing from periodic orbits but now I understand that they revolve around a certain type of them
- They are orbits with two frequencies, the first one is the same as a normal periodic orbit
- What makes it a torus is its second frequency which makes a circle perpendicular to the path of the first orbit
- What makes it quasi-periodic is that the ratio of the frequencies is irrational, they path of the orbit never lines up with itself when it comes back around

- Also, floquet theory is an amazing way of understanding orbits

- Used it to calculate that the families of orbits I produced earlier were all hyperbolic
- This fact means that they are likely no tori around them
- Tori are near stable orbits, so in the future it would be best to investigate halo, or other orbits



**Amended 8-17**
- Two corrections to this entry:
- First, "tori are near stable orbits" is wrong
- Tori require a center (elliptic) direction, not stability
- The halo orbits I moved to in Week 6 have a saddle pair at λ ≈ 2024.6 alongside a center pair at 0.97301 ± 0.23077j
- This orbit is strongly unstable and elliptic at the same time, and the center pair is the reason tori exist there
- Second, "the families I produced were all hyperbolic" is true only of my planar Lyapunov families where there is no out-of-plane direction to be elliptic in
- In the spatial problem a planar Lyapunov orbit does carry an out-of-plane center pair, so the statement needs that qualifier



**Next session:**

- Halo Orbits



## 2026-06-30

**Worked on:** 

- Manifolds



**Reading completed:**

- Read Koon Ch. 4, Gomez, and the first half of Burden & Faires Ch. 5

- Hartman-Grobman legitimizes the method of findind manifolds from a periodic orbit'
- At the point where the two over lap they are tangent, thus the linearization of the periodic orbit at a point can be used to find the start or end of a manifold



**Progress:**

- Generated stable and unstable manifolds for a L1 Lyapunov orbit
- Unstable comes from and stable goes into the orbit
- The manifolds assymptotically approaches and comes from the orbit

- In preperation for halo orbits, I expanded the existing system into 3d (because halos are 3d) and created a new, more complex jacobian and STM to accompany it
- This STM was validated and is ready to produce halos



**Amended 8-17**

- "Hartman-Grobman" is incorrect, the Stable Manifold Theorem is instead
- 3D CR3BP equations of motion, 6×6 Jacobian, and 42-dimensional STM all validated to 1e-10 against finite differences.



**Next session:**

- Halo Orbits



## 2026-07-01

**Worked on:** 

- Halo Orbits



**Progress:**

- Calculated families of Halo Orbits around L1 and L2 using all of the preexisting infrastructure of the project

- These are where I suspect QPITs to live around and the monodromy eigenvalues already have center pairs
- I am right on the cusp of calculating QPITs for this project



**Amended 8-18**

**GATE: Phase 2 exit — MET.** Reproduced L1+L2 Lyapunov + halo families; manifolds match published figures



**Next session:**

- QPIT reading



## 2026-07-02

**Worked on:** 

- Halo Orbits Floquet Analysis



**Progress:**

- Analysed the floquet multipliers of the Halo Families I generated to look for where QPITs could be near

- The two best candidates:
- L1 orbit 16 — C = 3.077, ρ/2π = 0.272
- L2 orbit 29 — C = 3.060, ρ/2π = 0.295

- This is my last real session before going to the National Youth Science Camp
- At camp I will:  
- -Read all relevent material about generating QPITs
- -Choose the method I will implement to actually do so
- -And reach out to Olikara and Baresi asking for their help

- All of this will be in preparation for when I come back from camp in a month
- When I come back, I will be jumping straight into the most difficult part of the project: Computing QPITs
- Hopefully all of the preparation of done before camp and will do during camp will set me up to succeed after it
- So until I return, Goodbye



**Next session:**

- QPIT Computation



## 2026-08-02

**Amended 8-17**

- National Youth Science Camp, retrospective; no daily entries were possible

- Read Olikara's thesis, Baresi-Olikara-Scheeres 2018, and Baresi's thesis
- Selected GMOS as the computational method



## 2026-08-03

**Worked on:** 

- Reading and Emails

- I have come back from summer camp and decided to ease into my research



**Reading completed:**

- Began reading McCarthy & Howell, AAS 19-329



**Progress:**

- Decided on GMOS for this project for a couple reasons:
- The literature shows that it offers the most accuracy compared to computational costs
- GMOS gives stability information for the QPITs it generates from the rotated monodromy matrix
- GMOS builds off the current infrastucture I have built

- Began drafting emails to Olikara, Baresi, and Howell asking for clarification on a few things regarding GMOS computation
- Having others help me with QPIT computation will be very helpful



**Amended 8-17**

- Here I selected L2 halo #29 over L1 #16 as the first target under the reasoning L2 #29 sits farther from the low-order resonances (0.038 from 1/3, 0.045 from 1/4 in ρ/2π) and maintains a clean center pair across the full tracked family range, per McCarthy & Howell
- Also, constant frequency-ratio continuation avoids the resonance-termination problems affecting constant-energy and constant-mapping-time families



**Stuck on / questions:**

- Does GMOS have any quirks for 3d Tori
- How do you easily converge on the first torus in a family
- Has anyone already done what Im aiming to do



**Next session:**

- QPIT Computation
- Emails Sent



## 2026-08-04

**Worked on:** 

- Emails and GMOS implementation



**Progress:**

- Sent the emails

- Derived the invariance equation
- Began to implement GMOS



**Next session:**

- QPIT Computation



## 2026-08-05

**Worked on:** 

- First QPIT



**Progress:**

- Converged on the first QPIT of the project
- Orbit that this QPIT came from was L2 orbit 29 — C=3.060, ρ_linear = 1.842 rad (ρ/2π = 0.2932)
- This QPIT was found using GMOS that I for the most part already implemented
- It converged in 3 iterations
- The QPIT amplitude is 17.2 km, T is 3.13, C is 3.06, rho (linear) is ρ = 1.842 rad and rho (torus) is ρ/2π = 0.2932
- This is really big and was one of the hardest and most anticipated parts of this project for me
- The full dataset generation is getting closer and closer



**Amended 8-18**

- Rank deficiency of exactly 2, measured by SVD post-convergence: gap of 8×10⁸ between σ[−2] and σ[−3], both nulls at machine zero
- The two symmetry tangents (latitudinal, longitudinal) lie in the nullspace to 12-digit precision

- Why the pre-convergence measurement doesn't count: at the seed the gap was only ~10×, polluted at O(ε²) by seed error
- The symmetries are exactly flat only at a solution, so a rank measured off-solution reports your distance from the solution rather than the rank

- The correct closure: two phase constraints plus mean Jacobi fixed, with ρ and T both free 

- Amplitude envelope: converges cleanly from ε = 1e-5 through 2e-3 (344 km); collapses at 3e-3



**Next session:**

- QPIT Continuation



## 2026-08-06

**Worked on:** 

- QPIT Continuation



**Progress:**

- Generated the first QPIT family of this project
- Contiuning off a found QPIT is easier than generating one from a guess

- Generated 23 members of a constant rotation number family
- Amplitudes ranged from 17 to 890 km

- Generated 17 members of a constant energy family
- Amplitudes ranged from 17 to 515 km

- Also generated the stability and manifolds of one torus
- This was mainly trivial given that GMOS gives the information needed to find this



**Amended 8-18**

- The κ three-way consistency check:
- κ = c · dρ_halo/dC predicts −11.080 while constant-C continuation measures −11.081
- Four significant figures from two independent routes

- The Week 8 table had the wrong sign because `C_target` came from each seed, mixing energy and amplitude effects
- The true fixed-energy coefficient is negative because ρ *decreases* with amplitude at fixed energy

- The constant-ρ family drifts toward lower C, making the nearest resonance 2/7 ≈ 0.285714 (below ρ₀/2π = 0.2932), reachable near A ≈ 10,000 km

- `cr3bp_3d.halo_corrector`'s y = 0 event fired at t = 0, producing T = 0 with a false "converged" status

- `converge_torus` defaulted `C_target` to the seed's energy rather than the halo's without saying so, which had an ε-sweep comparing tori at different energies

- `curve_amplitude` is resolution-dependent  
- Max over nodes understates by ~0.096% at N = 31 (17.1644 km node-max vs 17.1815 km dense)
- Prefer dense evaluation

- Re-deriving a torus by a different route lands 2–155 mm from the surface
- This comes from the small singular value σ₃ ≈ 2.0e-4 amplifying residuals, and it is structural — not fixable by tightening integration tolerances
- This is the number that becomes the label floor in Week 10.



**Next session:**

- Data-set generation



## 2026-08-07

**Worked on:** 

- QPIT Data-set generation



**Progress:**

- Today, I spent my time preparing for the data-set generation
- While preparing I arrived at a few conclusions

- If I just continue through a family of QPITs and label that as positives and then generate a box of points around L2 as negatives, then the dataset would be horrible
- The neural network might just learn what points are near the L2 halo
- Or there might be false negatives: the random points generated in between generated QPITs might be able to seed a new QPIT but if immediately labeled as a negative it introduces error into the dataset
- To solve these problems, negatives will be random samples from conditions similar to those of the QPITs and they will be tested to see if they do correctly seed a QPIT
- Also, I will generate QPITs around both L2 and L1 and at different energy levels
- This will ensure the network actually learns what conditions give a QPIT or not



**Amended 8-18**
- The plan to test sample condititions for being able to seed a QPIT turned out to not work at all
- Please look to the next days entry for greater detail



**Next session:**

- QPIT Data-set generation



## 2026-08-09

**Worked on:** 

- QPIT Data-set generation



**Progress:**

- Today, was a big reframe

- I decided that the model that is being developed by this project will not identify tori but instead predict the distance to the nearest tori
- This decision was largely made on the grounds that it was impossible to generate usable negatives for the data set
- It was simply too difficult to say that certain starting conditions were guaranteed to not to give a torus
- The tori live on a 3d sheet where anywhere on the sheet gives a torus and anywhere that doesn't doesn't
- So a model that predicts how close you are to the nearest torus or how close to the sheet would be so much more effective

- Now that the final model will predict the distance to the nearest tori it might actually be more useful for seeding a torus solver



**Amended 8-18**

- The reframe was influenced by more than what was covered in this days log

- The phase space is 6-D; Jacobi conservation confines motion to a 5-D energy manifold; tori are 2-D and the family sweeps them into a 3-D sheet at fixed C
- Codimension two, hence Lebesgue measure zero "Always answer no" scores 100% 
- Sharpening: KAM's positive-measure guarantee concerns maximal-dimension tori, which in three degrees of freedom are 3-tori
- Mine are 2-tori, so the guarantee does not apply

- The tolerance patch fails
- With a tolerance the label becomes a property of the tolerance
- The binary label flipped between 1.6 m and 16 m of displacement on a 202 km torus
- At a 10× margin four points all sitting on genuine tori read False/True/True/False depending only on phase, with underlying distances differing by 3×

- All forward-time indicators are dead
- R_u = 146.32 on L2 halo #29 → at rtol 1e-12, integrator error reaches O(1) after about four stroboscopic periods
- NAFF, weighted Birkhoff / Das-Yorke, FLI, and MEGNO each need one to two orders of magnitude more
- Each decimal digit buys ~0.46 periods, so even 30-digit arithmetic at ~50× the cost reaches ~12 periods
- Symplectic methods control secular energy drift, which is not the failure here
- These tori are computable but not observable
- This is what killed the 08-07 plan.
 
- The gap is full of tori
- 80/80 candidates interpolated between computed members converged to genuine tori, with fitted exponent 2.003 — the signature of pure second-order interpolation error
- i.e. a smooth continuous surface with linear interpolation as the only source of error
- So the positive class is not 17 thin sheets; it is a nearly solid 3-D region with exponentially thin holes at resonances, and the naive labelling was wrong essentially everywhere

- The probe score is not a distance
- It scales as gap² so it reports local family curvature and my discretisation choice
- Not a property of the query point
- Densify by 2× and the same state's score falls by 4× with nothing physical having moved

- `torus_field.surface_distance` measures distance to the interpolated family surface directly, with no integration
- 320 states/s vs ~0.4/s for Newton-per-sample
- Made the Titan port unnecessary and Risk 7 was retired 
- Dataset size stopped being the constraint and coverage became it

- Regression on log distance, classification recovered by thresholding
- Store the continuous distance, train on it, report every classification metric as an accuracy-vs-tolerance curve
- Strictly more information than a single accuracy number, and the arbitrary threshold becomes a reported axis instead of a hidden choice

- A distance field has a gradient a solver can descend, so it can warm-start a GMOS or collocation solve, a classifier cannot


**Next session:**

- QPIT Data-set generation



## 2026-08-11

**Worked on:** 

- QPIT Data-set generation



**Progress:**

- I completed the QPIT data-set
- The generation took a while on my computer but not too long
- This was the most difficult part of the project done
- QPITs are really high level and I've learned a lot up until this point
- Im now essentially in the home stretch and the primary part of the research: training of the model



**Amended 8-18**

**GATE: Phase 3 exit — MET.** Family of QPITs computed and labeled

- In total the data-set had 20,400 rows, L1 + L2 × 3 energies (C = 3.029398206, 3.064703461, 3.130606605)
- Three-arm sampling: on-surface, offset along normals, offset along random directions, with distance drawn log-uniform by design
- The floor was measured at 155 mm, from the σ₃ result 
- Below it the surface is not determined however finely you densify, so `log_d_km = log10(max(d_km, 1.55e-4))` and rows below are flagged `at_floor`
- Densifying past g = 12.3 km buys nothing
- Note: R² will be measured against a arbritary marginal distribution, since d was drawn log-uniform by design
- It is not a property of phase space The baselines are the meaningful comparison
- Bug caught before generating at scale: one of 200 on-surface points came back at 448 m instead of ~0, at a gap boundary Would have corrupted the dataset silently



**Next session:**

- Neural network design and training



## 2026-08-12

**Worked on:** 

- Neural Network design



**Reading completed:**

- Watched episodes 1-4 of 3Blue1Brown's seires over neural networks



**Amended 8-18**

- While watching these videos I ran some tests in the background
- Predicted that continuing the family into the 2/7 resonance would terminate it
- Falsified — the family crossed 2/7 without terminating
- I think this happened because I had selected the target by smallest ρ-distance rather than smallest denominator q, and q = 7 is not low-order, so the resonance zone is too thin to detect
- Retargeted to 1/4, verified reachable and parameterised

- Second failure: 
- As the family approaches 1/4 the extrapolation-error budget collapses from 75 km to 0, so non-convergence cannot be separated from a poor seed
- The family curvature that signals the resonance is the same thing that destroys linear extrapolation 
- Risk 18 realised

- Secondary Question 2 reframed from "ML accuracy degrades near resonances" to characterisation: 
- Detectability threshold in q (invisible at q = 7, strong at q = 4), κA² breakdown at 90% of predicted resonance amplitude, power-law repulsion d ~ A^−2.57, Fourier resolution growth 31 → 41 → 51
- Do not spend further Phase 4 time on this



**Next session:**

- Neural network design and training



## 2026-08-13

**Worked on:** 

- First Regressor



**Progress:**

- Developed, trained, and tested the first regressor
- Learned that there is a resolution floor for the regressor
- Below around 10^2 meters, the network stops closing in
- This regressor had a median error of 6.3x
- This model shows that the data-set is learnable but the model itself is not useful and has real limits



**Amended 8-22**

- The first run of `nb11_first_regressor.ipynb` produced an anomaly: 
- The B1 shortcut baseline scored R² = +0.816 where it was predicted to score near zero
- M1, with strictly more information, scored worse at 0.567
- This is not a modelling failure, it is a signal that a feature carries the answer

- Diagnosed with `nb11_diag_shortcut.py` (ablations, grouped split, same learner throughout)
- My stated hypothesis was that the stored `C` column encoded the offset
- That was falsified
- The stored `C` has exactly three distinct values — 3.029398206, 3.064703461, 3.130606605
- Zero spread inside each sheet, identical across L1 and L2
- It is the sheet energy, and it is clean: R² = −0.0001 alone
- The leak was the recomputed per-state Jacobi constant, which `features.py` was substituting in: R² = +0.7976 alone

- Why the per-state Jacobi is an artifact rather than legitimate information: 
- At inference you can compute a query state's C, so it looks fair
- It is not, because only three energy sheets were sampled
- With three, "distance in energy from the nearest sheet" is a near-deterministic function of the offset
- Real QPIT families fill a continuous energy range, so every state sits close in energy to some torus and the proxy evaporates
- This is a coverage artifact, the constraint the tracker named when it retired Risk 7.

- Corrected ablation, grouped split:
- | features | R² |
- | positions only (dist L1, dist L2) | +0.3606 |
- | positions + sheet C | +0.3615 |
- | full state, no C | +0.2253 |
- | full state + sheet C | +0.2372 |

- Note that the full state scores below engineered distances 
- Axis-aligned tree splits cannot form a Euclidean radius
- Legitimate feature-engineering effect

- Scorecard, clean rerun 
- predictions unchanged from before the first run:

- | | outcome | |
- | P1 shortcut baseline fails (R² < 0.5) | HELD | B1 = +0.3615 |
- | P2 1-NN gap random > grouped by >0.1 | FALSIFIED | random −0.0336, grouped −0.0305, gap −0.0031 |
- | P3a near-field RMSE exceeds far-field | HELD | 1.2285 vs 1.1019 |
- | P3b near-field bias positive | HELD | +1.1819 |
- | P3c overall R² > 0.9 | FALSIFIED | M1 = +0.5391 |

- Week 11 result: M1 = 0.539, B1 = 0.362, B0 ≈ 0
- The regressor beats the shortcut baseline by ~0.18 R² and beats the mean predictor decisively 

- P2's falsification is uninformative rather than meaningful
- Both B2 scores are negative, and a baseline at zero cannot resolve a gap
- The evidence about row dependence comes from the group census instead 
- 6,564 groups over 20,400 rows, median group size 2, 1,762 singletons, groups/rows = 0.32
- Base points were drawn continuously, so rows are nearly independent by construction and there was little duplication for grouping to protect against
- This also means `group_id` is doing almost no work

- The scorecard as first written printed "P3c falsified means Risk 19 was right and the Cantor-set framing stands" 
- That inference is wrong and must not reach the paper
- The labeller measures distance to a densified interpolated continuum at ~12.3 km spacing
- So, it cannot resolve the gaps between tori 
- That structure is not present in the target as generated 
- Risk 19 stays untested; this experiment has no power over it

- Two bugs caught:
- `nearest_point_overlap` computed its mixed-radix packing from each side's own key array 
- The MLP's `patience=30` default stopped a fit after ~95 gradient steps and returned a mean-predictor while reporting `early_stopped=True` as though converged



**Amended 8-23.**

- The P3a/P3b line above is the pre-fix scoring and does not match what Week 12 works from
- P3a was rescored on residual SD instead of RMSE, because RMSE contains the bias and cannot independently show heteroscedasticity
- On SD it is FALSIFIED: near 0.3429 against far 0.6267, while near RMSE stays higher at 0.9440 vs 0.6682
- So the near-field error is almost all bias with tighter scatter, not noisier fitting
- P3b holds at +0.8795 on measured rows only, censored rows excluded
- M1 also moved, 0.539 -> 0.632 +/- 0.020, after the epoch budget went to 3000/150/50
- The 600-epoch default had ended all three seeds on the ceiling with val loss still falling



**Next session:**

- Making network more accurate



## 2026-08-14

**Worked on:** 

- Making network more accurate



**Reading completed:**
- Statquest videos over neural networks and statistical methods
- videos over bias-variance, overfitting, bootstrapping, confidence intervals, hypothesis testing, gradient-descent variants, and regularization



**Next session:**

- Making network more accurate



## 2026-08-17

**Worked on:** 

- Auditing repo



**Progress:**
- Today I primarily audited and fixed small holes, errors, and inconsistencies across the repo



**Next session:**

- Making network more accurate



## 2026-08-18

**Worked on:** 

- Auditing repo



**Progress:**
- Almost finished auditing and fixing up the repo
- I have added a lot of missing tests to insure my code is not broken



**Stuck on / questions:**

- There are a few bugs that have been in the codebase unfound for a while, but none of them are substantial enough to cause any concern or need of the data to be regenerated



**Amended 8-23.** 

- The line above is right about the bugs, but it needs one exception on the record: 
- L2_E2 did need regenerating, for a reason unrelated to any of them

- nb10 recorded it at generation time as max gap 53.66 km, error bound 1.916 m, 9 failed probes
- That is 12.4x the 0.155 m reproducibility floor, where the other five families sat near 0.09 m
- It left 562 measured rows (3.14%) sitting below their own error bound
- I had assumed that was a real gap in the family,but it's not
- `probe_gap` returns `max_iter` at the default `itmax=15`
- At `itmax=40` the same midpoint converges to `res_inf = 6.3e-11`, and so does every other alpha I tried (0.2, 0.35, 0.65, 0.8)
- Resonance is ruled out, the bracketing members sit at `d_res = 0.0787`
- It was an iteration budget shortfall
- L2_E2 needs more Newton iterations than the others because its members sit at `rho/2pi ~ 0.0787` against HALO29's 0.293, which is a harder Newton problem
- Re-densified at `itmax=40` with a finishing pass at 80:
- 25 -> 54 members, 53.66 -> 11.92 km max gap, 1.916 m -> 0.095 m bound (0.61x floor), 9 -> 0 failed probes, pure infill over the same 23.9 - 395.3 km span
- Saved as `data/families/L2_E2_dense.npz`
- `L2_E2.npz` is deliberately untouched, because `qpit_dataset_w10.npz` was built from it.
- `data/famC_dense.npz` was missing entirely (data/ is gitignored), so all 10 tests in `test_torus_field.py` had never actually run
- Rebuilt 17 -> 67 members, 0.072 m bound
- It reproduces the recorded `PROBE_CAL = 1.1433` and those 9 skips are now green.

- What it changes going forward:

- I am not re-running Weeks 11/12 over it
- Re-scoring P3a/P3b from the frozen `predictions_wk11.npz` with L2_E2 dropped entirely (17% of test rows) moves the near-field bias +0.8795 -> +0.8247, a 6.2% shift
- It flips no verdict
- P3a stays falsified and, P3b stays held
- Swapping the dense family in means re-running nb10, which regenerates `qpit_dataset_w10.npz` and invalidates Weeks 11-12 until they are re-run
- Batching it for after Week 12 closes, together with the `normal_directions` fix and an `mknb10.py` written first, so I invalidate those results once instead of three times
- Payoff when I do: 562 -> 0 measured rows below their bound
- Every row still under the new bound is already `at_floor`, so its target is clamped and the error cannot propagate
- All six families would then have a labelling limit set by the reproducibility floor rather than by family spacing, which is what the whole densification argument was aiming at
- Rule I am taking from this: when a probe fails with `max_iter`, raise `itmax` before concluding the family has a gap
- The default 15 is not enough at low rho.



**Next session:**

- Making network more accurate



## 2026-08-23

**Worked on:** 

- Tobit loss and asinh target, run as a 2x2 against the frozen Week 11 baseline



**Progress:**

- Step 0 cleared all three gates
- Censored 2502/20400 = 12.26%, and it sits where the sampler puts it: all 2400 `surface` rows, 1.67% of `bundle`, 0.40% of `offset`
- Drawn vs measured came back med_ratio 0.9999, IQR ~0.001, in every one of twelve bins from 0.2 m to 100 km
- I had set that up to check FLOOR_KM independently and it cannot
- It measures precision against the same interpolated surface; the floor is an accuracy limit, how far apart two derivations of one torus land
- Censoring is still right, but the reason is surface determination, not measurement resolution
- It did independently reproduce the `normal_directions` bug size: 0.9999 puts the residual tangential component at ~1.4% of the step
- Dead zone reproduces, constant 6.54-6.58 m, boundary 124.62 m, slope -0.976 to -0.979, identical at n_bins 12/18/24
- n_bins=36 reported a 6.23 m constant sitting above a 3.04 m boundary and got flagged automatically
- A constant outside the range it was constant over is incoherent, so that row is the fixed 0.10 tolerance truncating on bin noise
- delta anchored to the boundary, which flattens 2.905 of 5.810 decades, half the range and exactly the half the model already fails in
- M1 reproduced at 0.6288 +/- 0.0214 on measured rows against 0.632 +/- 0.020

- 32 fits, 2h05
- Both changes lost, at every delta, on every metric in physical km
- At delta = 124.6 m, near-field fraction within a factor of 10: M1 0.664, M2a 0.434, M2b 0.637, M2c 0.476
- asinh's cost is monotone in how much it flattens: 0.638 / 0.489 / 0.434 / 0.368 across the four deltas
- Near-field bias went the wrong way under every change, 0.184 -> 1.177 for M2a

- The thing worth keeping from this week: M2a scored val MSE 0.148-0.162 against M1's 0.362-0.396
- In its own target space that is R2 0.85 vs 0.62, a large clean-looking win
- In physical km it is 43% within a factor of 10 against M1's 66%
- Scored in target space I would have reported asinh as a success and recommended it
- R2 is only scale-invariant under an affine change of target, and asinh is not affine in log10 d
- The near-field factor-of-N table in km caught it, and only because it was fixed as the arbiter before the run

- Scorecard: P12a HELD, P12b HELD, P12c FALSIFIED (dR2 -0.0375), P12d FALSIFIED (-0.1875 vs 2sd 0.0429), P12e FALSIFIED on its first clause, P12f held vacuously (0.000 -> 0.001)
- P12e's second clause is supported: Tobit alone hurts, Tobit on top of asinh helps at three of four deltas on both metrics
- The operating point shows why, asinh moves the prediction from 1.43 sigma above the censoring threshold to 0.43
- That is where the censoring statement starts carrying information instead of just pushing harder
- I predicted t = -1.44 and lambda/|t| = 1.311 from the fitted sigma, measured -1.43 and 1.313

- P3's mechanism does not survive this
- I attributed the dead zone to grad(log d) ~ 1/d being unbounded against a bounded network gradient
- asinh is exactly the fix for that, and applying it made the dead zone worse, monotonically in delta
- The observation stands, flat output and slope -1, The attribution does not
- Competing reading is an information limit: if the features do not determine d below 125 m then a constant is the right answer there and no target change can help
- That also explains why near-field SD is smaller than far-field, which never sat right under the gradient story

- M2b's number is contaminated
- Seed 2 inflated log_sigma from -0.4876 to +0.1073, never left the loss plateau at 1.4808, stopped at epoch 172 and reported `early_stopped=True`
- Same silent failure as the `patience=30` bug from 08-13, reached a different way, and `losses.py` had asserted in its own docstring that it would not happen
- It got averaged into a five-seed band as though it were a fit
- 0.4271 +/- 0.2865 is not a measurement
- Backing it out puts the four that trained near 0.53, still below M1, so the verdict holds but the number needs an asterisk wherever it is quoted
- Fixed by detection rather than clamping, since clamping log_sigma zeroes its gradient and freezes it at the bound
- `train_mlp` now records `val_r2` and flags a mean predictor, and the scorecard counts dead seeds beside budget-limited ones

- Mutation tested the five new modules, 21 injected bugs, 18 caught and 3 survived
- All three were real gaps I would not have found by reading the tests
- Green suite proves nothing on its own, confirmed the hard way



**Stuck on / questions:**

- The gate fired `regenerate, grid refinement required` on d_near = -0.1875
- That is the registered rule satisfied, but on different evidence than it was written for
- I wrote it for "no gain, therefore data-limited", and what happened is that both levers actively hurt
- Stronger evidence, but it does not prove modelling is exhausted, only that my mechanism was wrong
- Want to settle that before committing compute to a regeneration



**Next session:**

- Mutual information between features and d, restricted to rows below 125 m, on the frozen dataset with no fitting



## 2026-08-30

**Worked on:** 

- Mutual information between features and d, near vs far field, on the frozen dataset with no fitting
- Triage benchmark

**GATE: Phase 4 exit — MET.** Train QPIT model, characterize phase-space-resolved performance, benchmark speedup. After this, paper is writeable.



**Progress:**

- Regeneration cut
- My own audit is what settled it: dropping L2_E2 entirely moves near-field bias 6.2% and flips no verdict
- The 562 rows below their bound go in Limitations
- Weeks 14 and 16 also cut, so the calendar is 13, 15, then writing

- `src/mi.py` and `tests/test_mi.py`, 42 tests, mutation 28/28
- Two real bugs, both found by tests rather than by reading
- `_knn_radius` checked the k-th neighbour radius for zero instead of the nearest
- With k=5 and one twin per point the k-th radius is positive while the tie-free assumption is already broken, so the estimate comes back finite, plausible and wrong
- Bootstrap resampling produces exactly that case on almost every draw
- `make_windows` documented the near window as [floor, boundary] and only enforced the upper bound

- The `nextafter` strictness mutant survived the first pass
- On continuous data it shifts the estimate 0.009 nats at n=6000, below the seed-to-seed spread, so no tolerance on the Gaussian oracle separates it
- Killed with an integer lattice, where ties at the k-NN radius are the rule instead of an accident: 2.16 nats separation on input with no randomness

- `build_features` returns 9 columns and THREE of them are categorical, not one
- C has 3 distinct values (three energy levels), family_0/family_1 are binary
- `copula_transform` breaks ties by row order, so a categorical column makes the estimate depend on archive storage order
- Handled by the exact chain rule I([X,G];Y) = I(G;Y) + I(X;Y|G), conditioning on G = (family_id, energy_id), 6 cells at exactly 3400 rows each
- Verified C constant within energy_id and the map one-to-one, so the decomposition is an identity, not an approximation
- Continuous block is the 6 state components only

- The pre-registered calibration gate fired, and the bootstrap turned out to be the cause
- It inflated MI by a fixed +0.19 nats at every true value, an offset rather than a spread
- Bootstrap is invalid for k-NN estimators: resampling with replacement duplicates ~26% of rows, the 1e-12 jitter separates twins by far less than the ~1e-4 spacing of real neighbours, and the radii collapse
- Removed from the arbiter, significance now from the permutation null which creates no duplicates
- The point-estimate under-read is local: -0.003 at true R2 0.05, -0.101 at 0.75
- A global worst-case correction left +0.0007 of usable slack, so the map is inverted locally instead, 7 grid points x 5 reps, interior LOO 0.0002 to 0.021, refusing to extrapolate
- Amendment logged 08-28, all of it decided from synthetic data before any window statistic existed

- One tied target pair in 17898 in-window rows, `offset` sampler, different groups, multiplicity 2
- Chance float collision, expected at order one for ~9000 values over ~3 decades in float32
- Dropped one row chosen at random; keeping the first would correlate the survivor with archive order
- `n_below_floor` = 0, so my diagnosis of the tie as the log10(max(d, FLOOR)) clamp manufacturing an atom was wrong
- There were no sub-floor measured rows at all. Observation right, mechanism wrong, same pattern as P3

- near I_total -0.0046 nats, inside its own permutation null (p95 0.0132), k sweep -0.0073 / -0.0046 / -0.0068
- far 0.2476 nats, above its null (p95 0.0281), all six cells positive, k sweep 0.2934 / 0.2476 / 0.1915
- Matched at n = 8908 per window, same 6 dimensions, same cells
- P13.1 PASS, P13.2 CONFIRMED, P13.3 INFORMATIONAL on leg A
- Thinned to 31%: near -0.0174 still inside its null, far 0.1565 still above
- Far magnitude drops 0.2476 -> 0.1565 under thinning, real inflation from dependent rows, conclusion unmoved
- Gaussian-equivalent R2, stated as an assumption not a proof: near <= 0.026, far ~0.39 (0.44 to 0.32 across k)

- Leg B, the ceiling, is VOID and the reason is worth keeping
- The Shannon bound has an information-free floor of 1 - N/Var, which is zero only for a Gaussian target
- log_d_km is near log-uniform here, N/Var 0.6967 against 0.7026 for a uniform, so the floor sits at 0.303 and the bound cannot discriminate below it whatever the MI is
- My calibration matched n, dimension and cell structure and used a Gaussian target, where the floor is ~0
- The one axis it did not match is the one the entropy-power term depends on

- Tier-1 needed a null before it could be read at all
- Max-statistic permutation null for family-wise control, since 12 comparisons at a per-test 95th expects about one spurious hit
- Family-wise 0.0254 near / 0.0235 far, per-test 0.0186 / 0.0187
- Near: nothing clears either threshold, `vz` 0.0099 and `vx` 0.0092 were never significant on their own terms
- Far: only `x`, at 0.1210. The y/z/vx cluster at 0.0116-0.0129 is below both
- Categorical threshold differs by window, 0.0052 near vs 0.0251 far, because it is a KL difference on raw log10 d and not distribution-free like the ranked ones
- Data processing check: near +0.0165 against a joint null of 0.0190, both consistent with zero
- Far -0.2362, so the joint exceeds its largest marginal by 0.24 bits and the far-field information sits in combinations, not one coordinate

- Within-window R2 is negative, -0.133 near and -0.887 far, against 0.628 overall
- Restricted range, not a bug, and the arithmetic closes: residual variances 0.799 and 1.295 average 1.047 against the overall 1.046
- Residual sd ~1.02 decades, so M1 resolves WHICH DECADE a state sits in and not where inside it
- That is the same fact as 66% within a factor of 10, seen from another side

- P13.0 retired rather than skipped: the MI result settles the benchmark reading more strongly than the eikonal check could have
- A warm start has to place a state accurately in exactly the regime where nothing is detectable
- The raw inference-to-solve ratio is NOT the speedup and must not be reported as one
- Triage decides whether to attempt a solve, it does not replace one, so the cap is 1/(1-S)
- t_infer 0.854 us/row, so screening every candidate costs under 0.001% of one solve and the cap is binding
- At 95% recall, D_OP = boundary: S = 0.342 +/- 0.008, 115 missed of 2289, cap 1.52x
- P13.4 FALSIFIED (0.342 against 0.40), P13.5 CONFIRMED, P13.6 triage useful
- 2289 of 4035 held-out rows are worth solving, 56.7%, so a perfect predictor caps at 1.76x
- M1 reaches 79% of the achievable solves-avoided, and the cap is set by class balance, not by the model
- Specificity is the number that survives a change of sampling design: 1265 of 1746 not-worth-solving rows correctly skipped, 72%
- 539 censored + 1750 measured near = 2289, third independent confirmation of the split reconstruction

- On-surface rows are NOT protected by the dead zone, which I had predicted they would be
- 23 of 539 skipped at 95% recall, a 4.3% miss rate against 5.3% for measured near rows
- Marginally better than the class average, but not immune, and my prediction was 1.000
- The dead zone says where the model LANDS when it fires, not how reliably it fires
- Consistent with the MI result: nothing distinguishes an on-surface point from a nearby off-surface one, so d = 0 earns no protection
- Misses split 22 surface / 1 offset, which is the 95.9% surface base rate exactly



**Stuck on / questions:**

- `week11_m1_r2` = 0.6576 in results_wk12.json matches neither 0.632 nor 0.6288
- All held-out rows gives 0.6269, so it is not a censored-rows population difference
- The archive now offers three M1 R2 values and the paper will quote one

- Censored misses split 16 L2 / 7 L1 and I have no base rate to read that against
- If censored rows are near-even by family it is z ~ 1.9, marginal at n = 23
- It also runs opposite to the far-field per-cell MI, where L2 carried about twice L1

- `scorecard` has no budget_ltd or dead_seeds column, so that check could not be run as written
- r2_sd settles it anyway: M1 0.0214 against the 0.35 one dead seed would force
- M2b's 0.2865 matches the one-dead-seed prediction to three decimals, which is the seed 2 collapse



**Next session:**

- Week 15, cross-family transfer, with only 2 families in the archive



## 2026-09-06

**Worked on:**

- Week 15, the last research week: cross-family and cross-energy transfer
- P15.x pre-registration, `leave_one_level_out`, the reduced feature set, eight fits
- Start-of-week items 4, 5, 6



**Progress:**

- P15.x drafted and locked 09-05, before any fit
- Six predictions, five gates, one feature set across all eight fits
- P15.4 cut before lock, numbering preserved so P15.5 and P15.6 keep their identifiers
- `nearest_point_overlap` declared not applicable to a level holdout: with the full key set the overlap is zero by construction, without it the binning compares parameterisation coordinates across families
- Harder reason found later: `alpha_found`/`theta_found` are absent from the shipped archive entirely, so it could never have run as documented

- `leave_one_level_out` in `src/splits.py`, four masks plus achieved-count metadata
- `frac_test` is a fraction of the HELD-OUT LEVEL, not the dataset, or the control has no training stock
- Val is carved from the training pool only; early stopping on held-out-level rows would pick the epoch that best fits the regime the test set exists to measure
- Group atomicity asserted upfront, seed-independently, not inferred from a split
- Measured why: `groups_disjoint` catches a straddler on every seed when the straddle involves the held-out level, but a group straddling two OTHER levels passed 12/12 on the third fold while being live on the other two
- `_assert_partition` and `groups_disjoint` generalised to any key count so `excluded` is covered by the same checks
- 42 tests, mutation 27/27 with 3 verified equivalent under the atomicity guard

- Family one-hot dropped, `build_features(..., include_family=False)`, `FEATURE_SET_W15` seven columns
- `assert_feature_set` added because `include_family` defaults True and a fit that forgets it gets nine columns silently
- `assert_sheet_energy` added: the energy column must be constant within `energy_id`, which is the condition, not a distinct-value count
- 24 tests in `test_features.py`, 39 in `test_metrics.py`

- G15.1 PASS, zero straddling groups on either key
- G15.2 PASS, `energy_id` is monotone in C, so holding out 1 really is interpolation
- G15.4 PASS, far-field test n 569-923 against the 500 floor
- G15.5: KS(train,test) 0.019-0.029 on every fold. I had written that a level holdout would show a large KS by construction. It does not, and that is better than predicted — the folds are not confounded by target-distribution shift
- Far-field count 8908, matching Week 13's MI window exactly

- B0 on the reduced set: 0.5863 all held-out, 0.5912 measured, against 0.6269 and 0.6280
- So the family one-hot was carrying ~0.04 of R2, about 6% of the total. That is the §1.2 free measurement

- Transfer does not degrade, it DIVERGES. T1 predicts -113 to +78 on a target spanning -3.81 to 2.00
- RMSE ratios transfer/control on identical rows: T1 14.12, T2 8.04, T3 2.64, T4 1.18, T5 5.67
- 69% of T1 rows wrong by over 3 decades; trimming the worst 1% leaves R2 at -420, so it is most of the data
- Controls at 1.19-1.43 against B0 at 1.06-1.34, so sample size is excluded

- RETENTION IS VOID and it was knowable in advance. Specified as a ratio of R2 on far-field-only rows, and within-window R2 is negative by construction — Week 12 recorded -0.887 far, restricted range, not a bug
- All five controls returned -0.87 to -2.03, squarely in that range
- The denominator guard is the only reason it surfaced as a refusal: T1 would have printed (-602.58)/(-2.03) = 297
- My design error, not a discovery, and logged as one

- P15.1, P15.5 void. P15.2, P15.3 void as locked, both directions survive under post-hoc RMSE ratio
- The readout initially printed P15.2 and P15.3 as FALSIFIED. A prediction whose metric could not be computed is VOID; the bug ran in the flattering direction and is fixed

- Feature-substitution probe, 5 seeds: T1 rides on `x`, 65% +/- 23%, argmax stable 5/5
- T5's single-seed `C` attribution at 73% is FALSIFIED: over 5 seeds `C` recovers 15% +/- 96 points and the argmax moves between `C` and `x`
- The two-training-levels account is withdrawn and the deflationary reading of P15.3 that rested on it is retracted
- T3 has no dominant direction either, nothing above 21%, argmax across three features
- T4 null control refused to report, excess 0.197 decades below the 0.30 floor, which is the probe working
- Family transfer fails through one identifiable direction, energy transfer diffusely. Observation, not explanation

- `week11_m1_r2` = 0.6576 is the R2 of the seed-averaged prediction of Week 11's THREE seeds, all held-out rows
- 0.632 at log line 1188 is the mean of those same three per-seed values, 0.6324
- Week 12's five M1 seeds begin with Week 11's three, identical to four decimals, so they are one run extended
- M1 is delta-independent across all four keys, verified not assumed
- Split reconstruction verified directly against the stored `test_mask`, `y_test` to 4.4e-16, and `at_floor_test`

- `t_solve_sec` = 2.416 median, 2.594 mean, range 1.882-3.862 across the six cells, continuation, N=31
- Corrector and tangent account for 92% of wall time; the tangent is 23% of a step and must not be dropped
- Seconds saved per candidate = S * mean = 0.887 s. The mean, not the median, because rows are 3400 per cell so cells weight equally and this is an expectation
- t_infer 39.2 us, ratio 1.6e-05, realised speedup 1.5197 against a cap of 1.5198, so the cap is attained to four decimals
- L1_E0 is the outlier at 3.862 s/member with the only corrector failure and 6 extra calls, consistent with node-count raises as the curve outgrows N=31

- `docs/CUTS_AND_DISPOSITIONS.md` written
- Six dispositions kept distinct — cut, retired, moot, void, not applicable, falsified — because they read differently to a reviewer
- Four of five falsifications are mechanisms that failed while their observations held. Worth a sentence in Discussion



**Stuck on / questions:**

- Week 12's 0.6288 +/- 0.0214 does not reproduce from the archive, which gives 0.6280 +/- 0.0226
- M2b reproduces exactly from the same file, so it is not the row population, the R2 convention, or float32
- Inside the 0.02 verification tolerance, unexplained at the fourth digit. Points at the stored M1 predictions not being the run that produced the logged pair

- `vx` substitution makes T3 substantially WORSE, -107% +/- 9%, consistently across seeds. No account, and not offering one

- §1 of the results uses pooled RMSE, the probe uses mean of per-seed. They agree except on T5, 7.226 against 5.433. Recompute §1 under mean-of-per-seed on both sides before writing; the orderings do not move

- Handoff Limitations lists `n_below_floor`, but `make_windows` reports zero measured rows below floor on this archive. Drop or restate



**Next session:**

- Then writing