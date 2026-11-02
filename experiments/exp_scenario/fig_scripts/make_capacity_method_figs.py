"""Per-SNR robustness figures for the CONFIRMED method tiers.
Reads tier_sweep_s*.csv (base 133K / x300 331K / large 524K) and sweepfig_s*.csv (small 9K / mid 53K).
Produces:
  fig_capacity_robust_method.png  -> Small(133K)/Medium(331K)/Large(524K)  [matches pareto labels]
  fig_capacity_robust_5tier.png   -> 9K/53K/133K/331K/524K full size spectrum
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
from pathlib import Path

import figstyle  # noqa: TNR font

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "exp_scenario"
OUT = ROOT / "presentation_assets"
OUT.mkdir(parents=True, exist_ok=True)
S = ["inf", "20", "10", "5", "0", "-5", "-10"]; x = np.arange(7)


def mean(fname, col):
    def one(s):
        with (EXP / f"{fname}_s{s}.csv").open(newline="") as fp:
            d = {str(r["snr"]): float(r[col]) for r in csv.DictReader(fp)}
        return [d[k] for k in S]
    return np.array([one(s) for s in [42, 1, 7]]).mean(0)


k9  = mean("sweepfig", "small")
k53 = mean("sweepfig", "mid")
k133 = mean("tier_sweep", "base")
k331 = mean("tier_sweep", "x300")
k524 = mean("tier_sweep", "large")


def setup():
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
    ax.set_facecolor("white"); ax.grid(True, color="#e3e9ef", lw=0.9, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(["clean", "20", "10", "5", "0", "-5", "-10"], fontsize=15)
    ax.set_xlabel("SNR (dB)", fontsize=17); ax.set_ylabel("Macro-F1", fontsize=17)
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.1)); ax.tick_params(labelsize=15)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#9aa7b4")
    return fig, ax


# 1) confirmed-method tiers: Small=133K, Medium=331K, Large=524K
fig, ax = setup()
ax.plot(x, k133, "-o", color="#9ecae1", lw=2.4, ms=6, label="Small (133K)", zorder=3)
ax.plot(x, k331, "-o", color="#3182bd", lw=2.6, ms=6, label="Medium (331K)", zorder=3)
ax.plot(x, k524, "-o", color="#08306b", lw=3.0, ms=7, label="Large (524K)", zorder=4)
ax.legend(loc="upper right", fontsize=15, frameon=True, framealpha=0.95, edgecolor="#cdd6df", borderpad=0.6)
fig.tight_layout(); fig.savefig(OUT / "fig_capacity_robust_method.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_capacity_robust_method.png")

# 2) full 5-size spectrum
fig, ax = setup()
ax.plot(x, k9,   "-o", color="#c6dbef", lw=2.2, ms=5, label="9K", zorder=3)
ax.plot(x, k53,  "-o", color="#9ecae1", lw=2.2, ms=5, label="53K", zorder=3)
ax.plot(x, k133, "-o", color="#6baed6", lw=2.4, ms=6, label="133K", zorder=3)
ax.plot(x, k331, "-o", color="#3182bd", lw=2.6, ms=6, label="331K", zorder=4)
ax.plot(x, k524, "-o", color="#08306b", lw=3.0, ms=7, label="524K", zorder=5)
ax.legend(loc="upper right", fontsize=15, frameon=True, framealpha=0.95, edgecolor="#cdd6df", borderpad=0.6, title="Model size")
fig.tight_layout(); fig.savefig(OUT / "fig_capacity_robust_5tier.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_capacity_robust_5tier.png")
print(f"  @-10dB: 9K={k9[-1]:.3f} 53K={k53[-1]:.3f} 133K={k133[-1]:.3f} 331K={k331[-1]:.3f} 524K={k524[-1]:.3f}")
