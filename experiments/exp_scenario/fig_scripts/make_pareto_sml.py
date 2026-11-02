"""Cost-accuracy frontier: quality-weighted-pruned tier ladder (qw40/qw20/dense) + qw-tier
routing (ours). All points are the SAME family (Full model pruned 40%/20%/0%), so the
single-model reference tiers are exactly what ours routes over. Canonical 3-seed mixed-SNR.
Produces:
  fig_pareto.png         -> x = avg MACs/input (M)    [canonical]
  fig_pareto_params.png  -> x = model cost (K params)  [qw40/ours params are estimates ~]
ours = qw-tier routing 0.634 @ 5.26M MACs. See 최종_메소드_및_실험_전체기록.md (3-B, 3-D).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import figstyle  # noqa: TNR font

OUT = "/home/sujin/phm-korea-2026-snrgate/presentation_assets"

# Full model pruned by structured channel pruning. F1 + MACs canonical (record 3-B);
# nonzero params: qw20=417K, dense=524K canonical; qw40 & ours weighted = estimates (~).
#                       label          F1      MACs(M) params(K)  param_est
qw40  = dict(name="Small tier",  f=0.611, macs=3.72, par=237.0, pest=True,  col="#6baed6")
qw20  = dict(name="Medium tier", f=0.631, macs=6.45, par=417.0, pest=False, col="#3182bd")
dense = dict(name="Full tier",   f=0.624, macs=9.96, par=523.8, pest=False, col="#08306b")
ours  = dict(name="Quality gate, ours (routing)", f=0.634, macs=5.26, par=330.0, pest=True)

ORANGE = "#E8722C"
TIERS = [qw40, qw20, dense]


def draw(xkey, xlabel, xmax, fname, params=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=150)
    ax.set_facecolor("white"); ax.grid(True, color="#e3e9ef", lw=0.9, zorder=0)
    for p in TIERS:
        ax.scatter([p[xkey]], [p["f"]], s=150, color=p["col"], zorder=4, edgecolor="white", linewidth=1.2)
    ax.scatter([ours[xkey]], [ours["f"]], marker="D", s=230, color=ORANGE, zorder=6, edgecolor="#08306b", linewidth=2.2)

    unit = "M" if xkey == "macs" else "K"
    def tag(p):
        e = "≈" if (params and p.get("pest")) else ""
        return f"{p['f']:.3f}@{e}{p[xkey]:g}{unit}"

    ax.annotate(f"{qw40['name']}\n{tag(qw40)}", (qw40[xkey], qw40["f"]),
                textcoords="offset points", xytext=(0, -34), fontsize=10, color=qw40["col"], ha="center")
    ax.annotate(f"{qw20['name']}\n{tag(qw20)}", (qw20[xkey], qw20["f"]),
                textcoords="offset points", xytext=(14, -30), fontsize=10, color=qw20["col"], ha="left")
    ax.annotate(f"{dense['name']}\n{tag(dense)}", (dense[xkey], dense["f"]),
                textcoords="offset points", xytext=(-6, -32), fontsize=10, color=dense["col"], ha="center")
    ax.annotate(f"{ours['name']}\n{tag(ours)}", (ours[xkey], ours["f"]),
                textcoords="offset points", xytext=(0, 24), fontsize=11.5, color=ORANGE, fontweight="bold", ha="center")

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_xlim(0, xmax); ax.set_ylim(0.595, 0.648); ax.tick_params(labelsize=11)
    ax.set_yticks([0.60, 0.61, 0.62, 0.63, 0.64])
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#9aa7b4")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", bbox_inches="tight"); plt.close(fig)
    print("saved", fname)


draw("macs", "Average MACs per input (M)", 11, "fig_pareto.png")
draw("par",  "Model cost (K params)",      570, "fig_pareto_params.png", params=True)
