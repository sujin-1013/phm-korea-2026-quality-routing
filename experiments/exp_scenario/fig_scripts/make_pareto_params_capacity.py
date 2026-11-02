"""Final params-axis Pareto for the capacity routing ladder {133K, 331K, 524K}.
ours = quality-gate routing over {base 133K, x300 331K, large 524K} = 0.615 @ 266K.
ALL canonical measured values are plotted directly. No pruning, no estimates.
The 331K tier (x300) doubles as the strong single-model baseline (0.611 @ 331K).
Note: 9K/53K capacity-ladder points live in fig_capacity_robust.png (capacity=robustness story).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import figstyle  # noqa: TNR font

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "presentation_assets"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, SKY, ORANGE = figstyle.NAVY, figstyle.BLUE, figstyle.SKY, figstyle.ORANGE

#                       F1      params(K)  color
t1  = dict(name="Small",  f=0.553, par=133.0, col=SKY, marker="o")   # 8-seed mean
t2  = dict(name="Medium", f=0.611, par=331.0, col=BLUE, marker="s")   # = single x300 baseline
t3  = dict(name="Large",  f=0.618, par=523.8, col=NAVY, marker="^")
ours = dict(name="Quality gate", f=0.615, par=266.0)

fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=150)
ax.set_facecolor("white"); ax.grid(True, color=figstyle.GRID, lw=0.8, zorder=0)
for p in (t1, t2, t3):
    ax.scatter([p["par"]], [p["f"]], marker=p["marker"], s=175, color=p["col"],
               zorder=4, edgecolor="white", linewidth=1.2)
ax.scatter([ours["par"]], [ours["f"]], marker="D", s=230, color=ORANGE, zorder=6, edgecolor="#08306b", linewidth=2.2)

ax.annotate(t1["name"], (t1["par"], t1["f"]),
            textcoords="offset points", xytext=(10, -6), fontsize=20, color=BLUE, ha="left")
ax.annotate(t2["name"], (t2["par"], t2["f"]),
            textcoords="offset points", xytext=(10, -32), fontsize=20, color=t2["col"], ha="left")
ax.annotate(t3["name"], (t3["par"], t3["f"]),
            textcoords="offset points", xytext=(-6, -32), fontsize=20, color=t3["col"], ha="center")
ax.annotate(ours["name"], (ours["par"], ours["f"]),
            textcoords="offset points", xytext=(-4, 28), fontsize=22, color=ORANGE, fontweight="bold", ha="center")

ax.set_xlabel("Active parameters (K)", fontsize=24); ax.set_ylabel("Macro-F1", fontsize=24)
ax.set_xlim(0, 575); ax.set_ylim(0.5, 0.7); ax.set_yticks([0.5, 0.55, 0.6, 0.65, 0.7]); ax.tick_params(labelsize=21)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); fig.savefig(OUT / "fig_pareto_params_capacity.png", bbox_inches="tight")
print("saved fig_pareto_params_capacity.png  ours 0.615@266K; tiers 133/331/524; baseline single 331K=0.611")
