# Distance to Quasi-Periodic Invariant Tori in the Earth–Moon CR3BP

Code and data for the paper *Supervised Learning of Distance to Quasi-Periodic
Invariant Tori in the Earth–Moon Circular Restricted Three-Body Problem: Triage,
a Near-Field Information Limit, and Divergence Under Transfer* (Pierce Lang,
2026). Archived on Zenodo: <https://doi.org/10.5281/zenodo.24328874>.

## What this project does

Quasi-periodic invariant tori (QPITs) in the Earth–Moon circular restricted
three-body problem (CR3BP) are expensive to compute. This project computes
families of QPITs around L1 and L2 halo orbits with GMOS, builds a dataset of
20,400 states labeled with their distance to the nearest torus, and trains a
neural network to predict that distance. Classifying states as on or off a
torus turned out to be ill-posed (see `docs/label_definition.md`), so the model
predicts log distance instead.

The main findings:

- **Triage.** At 95% recall the network skips 34.2% of candidate seeds before
  they reach the solver, a 1.52× speedup and 79% of what a perfect screen
  could achieve.
- **A near-field information limit.** Below about 125 m the inputs carry no
  detectable information about distance, so the network's prediction there is
  flat. Changing the target or the loss does not fix it.
- **Divergence under transfer.** On families or energy levels it was not
  trained on, the network does not degrade gracefully; it diverges.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | CR3BP dynamics, periodic-orbit and torus solvers (`cr3bp_3d.py`, `qpit.py`), dataset construction (`torus_field.py`, `densify.py`, `sampling.py`), models and losses, splits, metrics, mutual information (`mi.py`) |
| `tests/` | The test suite |
| `notebooks/` | The experiments, roughly by week: dataset generation (`nb10_*`), the first regressor (`nb11_*`), targets and losses (`nb12_*`), mutual information and triage (`wk13_*`), transfer (`nb15_*`); `mknb*.py` build the later notebooks |
| `reproduce/` | Recomputes the results of Sections 3.1–3.6 and the final-run numbers of Section 3.7, and draws the figures, from `data/` |
| `data/` | The published data; see `data/README.md` |
| `docs/` | Research log, label definition, Week 15 pre-registration and results, and the register of cut and void items |
| `scripts/` | Test scripts for the Titan supercomputer; not used for any reported result |

## Setup

Python 3.12 with the packages in `requirements.txt`. In PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch` and `torchvision` are CPU builds and need the PyTorch index; see the
header of `requirements.txt`.

## Reproducing the paper

```
python reproduce/reproduce_paper.py
```

This prints the numbers behind the Results, section by section, and writes
Figures 1 and 2 to `figures/`. Two kinds of number are not recomputed: the
asinh network's validation R² in Section 3.3 (validation predictions were not
saved) and the comparisons with the first, overwritten run in Section 3.7.
Both are quoted from `docs/research_log.md`. Nothing is retrained, so it
runs in seconds. To retrain the models, run the notebooks; everything runs on a
CPU.

To regenerate the dataset from scratch, run
`notebooks/nb10_dataset_generation.ipynb` (sampling seed `20260810`) and check
it with `notebooks/verify_w10.py`. Note that `build()` reuses any family it
finds in `data/families/`, so clear that folder first for a true from-scratch
run.

## Known issues

These are described in the paper's Limitations section.

- The L2 family at C = 3.1306 (`L2_E2`) has a largest gap of 53.66 km against a
  12.3 km target. A re-densified version is saved as
  `data/families/L2_E2_dense.npz`, but the dataset was built from the original.
- In the L2 family at C = 3.0294, offsets meant to point straight out from the
  surface keep a small sideways component. This is pinned by a test in
  `tests/test_sampling.py`.
- The Week 15 transfer fits were re-run after a configuration error; see
  `docs/W15_RESULTS.md`.

## Tests

```
pytest tests/
```

Every test file also runs standalone (`python tests/test_qpit_spectral.py`) and
imports no pytest. The two slow files integrate real tori and train real models;
deselect them for a run of about a minute:

```
pytest tests/ --deselect tests/test_qpit_continuation.py \
              --deselect tests/test_slow_paths.py
```

## License

The code is released under the MIT License (`LICENSE`). The data in `data/`
are released under the Creative Commons Attribution 4.0 International License
(`LICENSE-DATA`).

## Citation

See `CITATION.cff`. The archived code and data have the DOI
[10.5281/zenodo.24328874](https://doi.org/10.5281/zenodo.24328874).

## Acknowledgments

Oral Roberts University provided access to its Titan supercomputer, which was
not used for the results reported here. This project was developed with
substantial assistance from Claude, an AI model by Anthropic; the paper's
Disclosure of Assistance describes its role.
