# QPIT Identification with Neural Networks

Supervised learning for quasi-periodic invariant torus identification in the Earth-Moon Circular Restricted Three-Body Problem.

## Project Overview

This project investigates whether a neural network, trained on classically-computed quasi-periodic invariant torus (QPIT) data, can accurately identify torus membership for arbitrary initial conditions in the planar Earth-Moon CR3BP. The mission-relevant context is current cislunar operations such as the NASA Lunar Gateway, which operates in a quasi-periodic regime around halo orbits.

## Status

In progress. Research project for the 2026–2027 academic year.

## Repository Structure

- `src/` — Python modules (CR3BP dynamics, differential corrector, QPIT computation, ML)
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `scripts/` — SLURM job scripts and one-off utilities
- `tests/` — Automated tests
- `docs/` — Research log and notes
- `data/` — Generated datasets (not tracked by git)
- `results/` — Experimental outputs (not tracked by git)
- `figures/` — Generated figures

## Environment

Python 3.12 with packages listed in `requirements.txt`. Setup (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch`/`torchvision` are CPU builds and need the PyTorch index; see the header
of `requirements.txt`.

## Tests

```
pytest tests/
```

Every test file also runs standalone (`python tests/test_qpit_spectral.py`) and
imports no pytest. The two slow files integrate real tori and train real models;
deselect them for a ~1 minute run:

```
pytest tests/ --deselect tests/test_qpit_continuation.py \
              --deselect tests/test_slow_paths.py
```

## Acknowledgments

Computational resources provided by Oral Roberts University's Titan supercomputer.

## License

To be determined.