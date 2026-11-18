"""Figure: the near-field dead zone of the baseline network.

Usage: python3 make_fig_deadzone.py DATASET.npz PREDICTIONS_WK12.npz OUT.png
Measured test rows; predictions from the five-seed baseline network
(predictions_wk12.npz key 'boundary__M1'); split reconstructed as in
src/splits.py grouped_split(seed=0, frac_test=0.2).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds = np.load(sys.argv[1]); pr = np.load(sys.argv[2]); out = sys.argv[3]
g = ds["group_id"]; uniq = np.unique(g)
order = np.random.default_rng(0).permutation(uniq)
tg = set(order[: int(round(0.2 * len(uniq)))].tolist())
test = np.fromiter((v in tg for v in g), bool, len(g))
floor = float(ds["floor_km"])
dk = ds["d_km"][test]; af = ds["at_floor"][test]
y = np.log10(np.maximum(dk, floor))[~af]            # measured rows only
P = pr["boundary__M1"].astype(float)[:, ~af]         # (5 seeds, n)

edges = np.linspace(y.min(), y.max(), 25)
idx = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
ctr, mean, lo, hi = [], [], [], []
for b in range(len(edges) - 1):
    m = idx == b
    if not m.any():
        continue
    r = P[:, m]
    ctr.append(np.median(y[m])); mean.append(r.mean())
    lo.append(np.percentile(r, 10)); hi.append(np.percentile(r, 90))
to_m = lambda v: 1e3 * 10 ** np.asarray(v)
ctr, mean, lo, hi = map(to_m, (ctr, mean, lo, hi))

BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=300)
edge_m = 124.6
ax.axvspan(0.1, edge_m, color="#f0efec", zorder=0, lw=0)
ax.plot([0.1, 2e5], [0.1, 2e5], ls=(0, (4, 3)), color=INK2, lw=1.2, zorder=1, label="Perfect prediction")
ax.fill_between(ctr, lo, hi, color=BLUE, alpha=0.18, lw=0, zorder=2, label="10th–90th percentile of predictions")
ax.plot(ctr, mean, "o-", color=BLUE, lw=2, ms=4.5, mec="white", mew=1.2, zorder=4, label="Mean prediction")
ax.axvline(edge_m, color=INK2, lw=1, zorder=1)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.12, 1.3e5); ax.set_ylim(0.12, 1.3e5)
ax.set_xlabel("True distance to the torus family (m)", color=INK)
ax.set_ylabel("Predicted distance (m)", color=INK)
ticks = [1, 10, 100, 1e3, 1e4, 1e5]
labels = ["1", "10", "100", "1,000", "10,000", "100,000"]
for setter in (ax.set_xticks, ax.set_yticks):
    setter(ticks)
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.minorticks_off()
ax.grid(True, color=GRID, lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
ax.tick_params(colors=INK2)

ax.text(0.17, 2.0e4, "Near-field dead zone\n(below 124.6 m)", color=INK, fontsize=9.5, va="top")
ax.annotate("Prediction stays near 7 m", xy=(1.0, 6.5), xytext=(0.17, 50),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
h, l = ax.get_legend_handles_labels()
order = [l.index(k) for k in ("Mean prediction", "10th–90th percentile of predictions", "Perfect prediction")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out)
