"""Capacity = robustness per-SNR figures (from-scratch capacity ladder small/mid/large).
Shows: larger model capacity -> more robust to noise (large holds at low SNR, small collapses).
Uses sweepfig_s*.csv (capacity-ladder single models + their own routing 'ours' = small/mid/large gate).
This is a DIFFERENT family from fig_main_gate.png (qw-tier pruning); keep both.
Produces:
  fig_capacity_robust.png          -> small/mid/large only (pure capacity story)
  fig_capacity_robust_routing.png  -> small/mid/large + ours routing overlay (same family)
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import figstyle  # noqa: TNR font
EXP = "/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
OUT = "/home/sujin/phm-korea-2026-snrgate/presentation_assets"
S = ["inf", "20", "10", "5", "0", "-5", "-10"]; x = np.arange(7)
NAVY, BLUE, SKY, ORANGE = figstyle.NAVY, figstyle.BLUE, figstyle.SKY, figstyle.ORANGE


def mean(col, fname):
    def one(s):
        d = {str(r["snr"]): float(r[col]) for r in csv.DictReader(open(f"{EXP}/{fname}_s{s}.csv"))}
        return [d[k] for k in S]
    return np.array([one(s) for s in [42, 1, 7]]).mean(0)


small, mid, large = mean("small", "sweepfig"), mean("mid", "sweepfig"), mean("large", "sweepfig")
# UNIFIED routing = qw-tier method (0.634 @ 5.26M MACs), same 'ours' as fig_main_gate / fig_pareto.
ours = mean("ours", "qwtier_sweep")


def base_ax():
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
    ax.set_facecolor("white"); ax.grid(True, color=figstyle.GRID, lw=0.8, zorder=0)
    ax.plot(x, small, "-o", color=SKY, lw=2.6, ms=7, markeredgecolor="white",
            markeredgewidth=1.0, label="Small", zorder=3)
    ax.plot(x, mid, "--s", color=BLUE, lw=2.8, ms=7, markeredgecolor="white",
            markeredgewidth=1.0, label="Medium", zorder=3)
    ax.plot(x, large, "-^", color=NAVY, lw=3.1, ms=8, markeredgecolor="white",
            markeredgewidth=1.0, label="Large", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(["clean", "20", "10", "5", "0", "-5", "-10"], fontsize=17)
    ax.set_xlabel("SNR (dB)", fontsize=19); ax.set_ylabel("Macro-F1", fontsize=19)
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.1)); ax.tick_params(labelsize=17)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#9aa7b4")
    return fig, ax


# 1) capacity ladder only
fig, ax = base_ax()
ax.legend(loc="upper right", fontsize=17, frameon=True, framealpha=0.95, edgecolor="#cdd6df", borderpad=0.6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_capacity_robust.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_capacity_robust.png  large@-10=%.3f small@-10=%.3f" % (large[-1], small[-1]))

# 2) capacity ladder + ours routing overlay (same family)
fig, ax = base_ax()
ax.plot(x, ours, "--D", color=ORANGE, lw=3.2, ms=8, markerfacecolor="white",
        markeredgewidth=2.2, label="Quality gate, ours", zorder=5)
ax.legend(loc="upper right", fontsize=17, frameon=True, framealpha=0.95, edgecolor="#cdd6df", borderpad=0.6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_capacity_robust_routing.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_capacity_robust_routing.png  ours@-10=%.3f" % ours[-1])
