# Data

All files are released under CC BY 4.0 (see `LICENSE-DATA` in the repository
root). States are in the nondimensional rotating frame of the Earth–Moon CR3BP,
with mu = 0.01215058560962404 and a length unit of 384,400 km.

## The dataset

`qpit_dataset_w10.npz` holds 20,400 labeled states, 3,400 for each of six
cells (two families × three energy levels). The main arrays:

| Key | Meaning |
|---|---|
| `states` | The 6-D state (x, y, z, vx, vy, vz) |
| `C` | Jacobi constant of the energy level the state was drawn at |
| `family_id` | 0 = L1, 1 = L2 |
| `energy_id` | 0, 1, 2 in order of increasing C (3.0294, 3.0647, 3.1306) |
| `d_km` | Distance to the nearest torus surface in the family, in km |
| `floor_km` | The 0.155 m reproducibility floor |
| `at_floor` | True where `d_km` is below the floor |
| `group_id` | Surface patch the state was drawn from; splits keep groups whole |
| `source` | Sampling arm: on the surface, offset along a direction, or along the unstable bundle |

The regression target is `log10(max(d_km, floor_km))`.

## Torus families

| File | Contents |
|---|---|
| `families/L1_E0.npz` … `families/L2_E2.npz` | The six torus families the dataset was built from |
| `families/L2_E2_dense.npz` | `L2_E2` re-densified after the dataset was built; not used by it |
| `qpit_family_L2h29.npz` | The first QPIT family, around L2 halo orbit 29 |
| `qpit_family_L2_quarter_resonance.npz` | A family continued toward the 1/4 resonance |
| `famC_dense.npz` | A densified constant-energy family used by `tests/test_torus_field.py` |

## Model outputs

| Folder | Contents |
|---|---|
| `wk11/` | The first regressor: predictions (3 seeds) and results, including the baselines |
| `wk12/` | Targets and losses. Keys are `delta__cell`: `M1` is the paper's baseline network (5 seeds), `M2a` the asinh target, `M2b` the Tobit loss, `M2c` both. `M2a` and `M2c` predictions are in asinh units; convert them with `src/targets.py: to_log10_km`. |
| `wk13/` | Mutual-information and triage results |
| `wk15/` | Transfer: predictions over all 20,400 rows for `B0`, the controls, and folds `T1` to `T5` (5 seeds each), plus training histories |

`reproduce/reproduce_paper.py` shows how each file is used.
