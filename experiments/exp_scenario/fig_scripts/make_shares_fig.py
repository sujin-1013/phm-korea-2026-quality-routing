"""Per-SNR routing tier-selection share (stacked bars) for confirmed {133,331,524}.
Reads shares_confirmed.csv (5-seed iso avg). TNR style."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
import os as _o, sys as _s; _s.path.insert(0,_o.path.dirname(_o.path.abspath(__file__))); import figstyle  # noqa
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"; OUT="/home/sujin/phm-korea-2026-snrgate/presentation_assets"
NAVY,BLUE,SKY=figstyle.NAVY,figstyle.BLUE,figstyle.SKY
rows=list(csv.DictReader(open(f"{EXP}/shares_confirmed.csv")))
labels=[r["snr"] for r in rows]; labels=["clean" if l=="clean" else l for l in labels]
sm=np.array([float(r["small_133K"]) for r in rows])*100
md=np.array([float(r["medium_331K"]) for r in rows])*100
lg=np.array([float(r["large_524K"]) for r in rows])*100
x=np.arange(len(rows))
fig,ax=plt.subplots(figsize=(9.0,5.6),dpi=150)
ax.bar(x,sm,color=SKY,edgecolor="white",lw=1.0,label="Small (133K)",zorder=3)
ax.bar(x,md,bottom=sm,color=BLUE,edgecolor="white",lw=1.0,label="Medium (331K)",zorder=3)
ax.bar(x,lg,bottom=sm+md,color=NAVY,edgecolor="white",lw=1.0,label="Large (524K)",zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=17)
ax.set_xlabel("SNR (dB)",fontsize=19); ax.set_ylabel("Tier selection share (%)",fontsize=19)
ax.set_ylim(0,100); ax.set_yticks(np.arange(0,101,20)); ax.tick_params(labelsize=17)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=3,fontsize=15,frameon=False,handlelength=1.3,columnspacing=1.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shares_per_snr.png",bbox_inches="tight"); plt.close(fig)
print("saved fig_shares_per_snr.png")
