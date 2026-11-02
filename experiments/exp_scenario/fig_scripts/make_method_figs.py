"""New method/illustration figures (Times New Roman via figstyle):
  fig_estimator_q.png  - measured quality q vs SNR (estimator tracks noise)
  fig_estimator_method.png - estimator self-supervised training schematic
  fig_waveform.png     - one window: clean / 0 dB / -10 dB (fault buried by noise)
  fig_gate.png         - gate rule schematic: quality axis -> 3 tiers
  fig_pipeline.png     - end-to-end pipeline diagram
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv, math
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import figstyle  # noqa
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
EXP = "/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
OUT = "/home/sujin/phm-korea-2026-snrgate/presentation_assets"
S = ["clean", "20", "10", "5", "0", "-5", "-10"]; x = np.arange(7)
NAVY, BLUE, SKY, ORANGE = figstyle.NAVY, figstyle.BLUE, figstyle.SKY, figstyle.ORANGE
GREY, GRID, TEXT = figstyle.GREY, figstyle.GRID, figstyle.TEXT

# ---------- 1) quality vs SNR ----------
d = {r["snr"]: r for r in csv.DictReader(open(f"{EXP}/qsnr.csv"))}
qm = np.array([float(d[s]["q_mean"]) for s in S])
q25 = np.array([float(d[s]["q_p25"]) for s in S]); q75 = np.array([float(d[s]["q_p75"]) for s in S])
tgt = [1.0] + [1 - 1/(1+math.exp(v/10)) for v in [20, 10, 5, 0, -5, -10]]
fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.fill_between(x, q25, q75, color=ORANGE, alpha=0.14, zorder=2, label="measured q (IQR)")
ax.plot(x, qm, "-D", color=ORANGE, lw=figstyle.LW_MAIN, ms=9, markerfacecolor="white",
        markeredgewidth=2.2, zorder=4, label="measured q (mean)")
ax.plot(x, tgt, "--", color=NAVY, lw=figstyle.LW_SECONDARY, zorder=3, label="target (from SNR)")
ax.set_xticks(x); ax.set_xticklabels(S, fontsize=17)
ax.set_xlabel("SNR (dB)", fontsize=19); ax.set_ylabel("Quality score  q", fontsize=19)
ax.set_ylim(-0.02, 1.04); ax.tick_params(labelsize=17)
ax.legend(fontsize=17, loc="lower left", frameon=True, framealpha=0.95, edgecolor="#cdd6df")
ax.annotate("higher q = cleaner", (6.0, 0.95), fontsize=17, color="#555", ha="right")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_estimator_q.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_estimator_q.png")

# ---------- 2) estimator training schematic ----------
fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=150)
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

def method_box(cx, cy, w, h, txt, fc, tc="white", fs=15):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        fc=fc, ec="white", lw=1.2
    ))
    ax.text(cx, cy, txt, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold")

def method_arrow(x0, y0, x1, y1, color="#444", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        lw=1.9, color=color, connectionstyle=f"arc3,rad={rad}"
    ))

method_box(1.1, 2.35, 1.7, 0.78, "Clean\nwindow", GREY, fs=14)
method_arrow(1.95, 2.35, 2.75, 2.35)
method_box(3.55, 2.35, 1.75, 0.78, "Inject\nAWGN", ORANGE, fs=14)
ax.text(3.55, 3.05, "known SNR", ha="center", fontsize=14, color=TEXT)
method_arrow(4.43, 2.35, 5.25, 2.35)
method_box(6.25, 2.35, 1.95, 0.90, "0.13K\n1D-CNN", NAVY, fs=15)
method_arrow(7.25, 2.35, 8.12, 2.35)
method_box(8.85, 2.35, 1.10, 0.74, "q", BLUE, fs=18)

method_box(3.55, 0.95, 2.05, 0.72, "SNR-derived\ntarget q*", SKY, fs=14)
method_arrow(4.55, 0.95, 7.95, 1.78, color=BLUE, rad=-0.12)
method_arrow(8.55, 2.00, 8.10, 1.55, color=ORANGE, rad=0.12)
ax.text(7.95, 1.32, "MSE(q, q*)", ha="center", fontsize=16, color=TEXT, fontweight="bold")
ax.text(5.0, 0.20, "fault labels are not used for routing", ha="center",
        fontsize=15, color="#666666")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_estimator_method.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_estimator_method.png")

# ---------- 3) waveform clean / 0 / -10 ----------
wf = np.load(f"{EXP}/waveform.npz"); N = 2048
fig, axs = plt.subplots(3, 1, figsize=(8.6, 6.0), dpi=150, sharex=True)
ylim = math.ceil(max(np.max(np.abs(wf[key][:N])) for key in ["clean", "db0", "dbm10"]) * 10) / 10
t_ms = np.arange(N) / 64000.0 * 1000.0  # 64 kHz -> time in ms (0 .. 32 ms)
for ax, key, ttl, col in zip(axs, ["clean", "db0", "dbm10"], ["Clean", "0 dB", "-10 dB"], [NAVY, BLUE, ORANGE]):
    ax.plot(t_ms, wf[key][:N], color=col, lw=1.0)
    ax.set_ylabel(ttl, fontsize=17, rotation=0, ha="right", va="center", labelpad=22)
    ax.set_ylim(-ylim, ylim); ax.set_yticks([-1, 0, 1])
    ax.set_xlim(0, 32); ax.set_xticks([0, 8, 16, 24, 32])
    ax.grid(True, color="#edf2f7", lw=0.8); ax.tick_params(labelsize=17)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axs[-1].set_xlabel("Time (ms)", fontsize=19)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_waveform.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_waveform.png")

# ---------- 4) gate rule schematic ----------
fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=150)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
t_lo, t_hi = 0.2, 0.35
regions = [(0.0, t_lo, NAVY, "Large\n524K"), (t_lo, t_hi, BLUE, "Medium\n331K"), (t_hi, 1.0, SKY, "Small\n133K")]
for x0, x1, col, lbl in regions:
    ax.add_patch(plt.Rectangle((x0, 0.45), x1-x0, 0.32, color=col, alpha=0.9, ec="white", lw=1.2))
    ax.text((x0+x1)/2, 0.61, lbl, ha="center", va="center", color="white", fontsize=17, fontweight="bold")
ax.add_patch(FancyArrowPatch((0.02, 0.30), (0.98, 0.30), arrowstyle="-|>", mutation_scale=16,
                             lw=2.0, color=TEXT, clip_on=False))
ax.text(0.02, 0.20, "0", ha="center", va="center", fontsize=15, color=TEXT)
ax.text(0.98, 0.20, "1", ha="center", va="center", fontsize=15, color=TEXT)
ax.text(0.5, 0.12, "quality score  q  (low = noisy  →  high = clean)", ha="center", fontsize=17)
for tv, label in ((t_lo, r"$\tau_{low}$"), (t_hi, r"$\tau_{high}$")):
    ax.plot([tv, tv], [0.30, 0.82], color=TEXT, lw=1.8, ls=(0, (1.2, 2.0)))
    ax.text(tv, 0.86, label, ha="center", fontsize=17, color=TEXT)
ax.text(0.10, 0.93, "noisy input", ha="center", fontsize=15, color=NAVY)
ax.text(0.90, 0.93, "clean input", ha="center", fontsize=15, color=BLUE)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gate.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_gate.png")

# ---------- 5) pipeline diagram ----------
fig, ax = plt.subplots(figsize=(10.5, 3.2), dpi=150)
ax.set_xlim(0, 12.2); ax.set_ylim(0, 3.2); ax.axis("off")
def box(cx, cy, w, h, txt, fc, tc="white", fs=14):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec="white", lw=1.1))
    ax.text(cx, cy, txt, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold")
def arrow(x0, x1, y=1.6):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#444"))
def route_arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.8,
                                 color="#444", connectionstyle="arc3,rad=0.0"))
box(1.1, 1.6, 1.9, 1.0, "Vibration\nwindow\n2048", GREY, "white", 15)
arrow(2.1, 2.9)
box(3.9, 1.6, 2.0, 1.0, "Quality\nestimator\n(0.13K) → q", ORANGE, "white", 15)
arrow(4.95, 5.7)
box(6.7, 1.6, 1.9, 1.0, "Gate\n(2 thresholds)", NAVY, "white", 15)
# three tier boxes stacked at right
for cy, lbl, col in [(2.55, "Small 133K", SKY), (1.6, "Medium 331K", BLUE), (0.65, "Large 524K", NAVY)]:
    route_arrow(7.75, 1.6, 8.25, cy)
    box(9.3, cy, 2.0, 0.74, lbl, col, "white", 15)
arrow(10.35, 10.95, 1.6)
ax.text(11.55, 1.6, "N / IR / OR", ha="center", va="center", fontsize=15, color=TEXT)
ax.text(8.2, 0.06, "only one tier runs per window", ha="center", fontsize=15, color="#666")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_pipeline.png")

# ---------- 6) motivation: parameter-robustness trade-off ----------
fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_facecolor("white"); ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("Inference cost", fontsize=24); ax.set_ylabel("Expected performance", fontsize=24)
ax.set_xticks([0.2, 0.8]); ax.set_xticklabels(["low", "high"], fontsize=21)
ax.set_yticks([0.2, 0.8]); ax.set_yticklabels(["low", "high"], fontsize=21)

target = plt.Rectangle((0.36, 0.68), 0.24, 0.14, facecolor=ORANGE, alpha=0.10,
                       edgecolor=ORANGE, linewidth=1.5, linestyle="--", zorder=1)
ax.add_patch(target)
ax.text(0.48, 0.83, "Target region", ha="center", va="bottom", fontsize=20, color=ORANGE)

ax.scatter([0.24], [0.34], s=230, marker="o", color=SKY, edgecolor="white",
           linewidth=1.3, zorder=4, label="Small model")
ax.scatter([0.78], [0.78], s=260, marker="^", color=NAVY, edgecolor="white",
           linewidth=1.3, zorder=4, label="Large model")
ax.scatter([0.48], [0.76], s=260, marker="D", color=ORANGE, edgecolor=NAVY,
           linewidth=1.4, zorder=5, label="Target")

ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cdd6df", fontsize=20)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_motivation.png", bbox_inches="tight"); plt.close(fig)
print("saved fig_motivation.png")
