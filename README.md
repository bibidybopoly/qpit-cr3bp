\# QPIT Identification with Neural Networks



Supervised learning for quasi-periodic invariant torus identification in the Earth-Moon Circular Restricted Three-Body Problem.



\## Project Overview



This project investigates whether a neural network, trained on classically-computed quasi-periodic invariant torus (QPIT) data, can accurately identify torus membership for arbitrary initial conditions in the planar Earth-Moon CR3BP. The mission-relevant context is current cislunar operations such as the NASA Lunar Gateway, which operates in a quasi-periodic regime around halo orbits.



\## Status



In progress. Research project for the 2026–2027 academic year.



\## Repository Structure



\- `src/` — Python modules (CR3BP dynamics, differential corrector, QPIT computation, ML)

\- `notebooks/` — Jupyter notebooks for exploration and analysis

\- `scripts/` — SLURM job scripts and one-off utilities

\- `tests/` — Automated tests

\- `docs/` — Research log and notes

\- `data/` — Generated datasets (not tracked by git)

\- `results/` — Experimental outputs (not tracked by git)

\- `figures/` — Generated figures



\## Environment



Python 3.12 with packages listed in `requirements.txt`. Setup:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
## Acknowledgments

Mentored by Dr. Andrew Lang (Oral Roberts University). Computational resources provided by ORU's Titan supercomputer.

## License

To be determined.