"""Figure: mutual information between the inputs and distance, near vs far.

Usage: python3 make_fig_mi.py RESULTS_WK13_MI.json OUT.png
Bars: KSG estimate at k = 5. Whiskers: range over k = 3, 5, 10.
Dashed ticks: 95th percentile of the 200-shuffle permutation null.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1])); out = sys.argv[2]
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=300)
labels = ["Near field\n(0.155 m to 124.6 m)", "Far field\n(124.6 m to 100 km)"]
for i, w in enumerate(("near", "far")):
    est = r[w]["I_total"]; ks = list(r[w]["by_k"].values()); p95 = r[w]["null_p95"]
    ax.bar(i, est, width=0.5, color=BLUE, zorder=2)
    ax.plot([i, i], [min(ks), max(ks)], color=INK, lw=1.2, zorder=3)
    for v in (min(ks), max(ks)):
        ax.plot([i - 0.06, i + 0.06], [v, v], color=INK, lw=1.2, zorder=3)
    ax.plot([i - 0.38, i + 0.38], [p95, p95], ls=(0, (4, 2)), color=INK2, lw=1.4, zorder=4)
    ypos = est if est > 0 else -0.019
    ax.text(i + 0.30, ypos, f"{est:+.4f}".replace("-", "\u2212"), ha="left", va="center", fontsize=9, color=INK)
ax.axhline(0, color=INK2, lw=0.8, zorder=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_xlim(-0.6, 1.75)
ax.set_ylim(-0.03, 0.33)
ax.set_ylabel("Mutual information (nats)", color=INK)
ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
ax.tick_params(colors=INK2)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend([Patch(color=BLUE), Line2D([0], [0], color=INK, lw=1.2),
           Line2D([0], [0], color=INK2, lw=1.4, ls=(0, (4, 2)))],
          ["Estimate (k = 5)", "Range over k = 3, 5, 10", "Shuffled-data 95th percentile"],
          loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out)
