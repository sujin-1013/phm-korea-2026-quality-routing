import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
R=list(csv.DictReader(open("/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/imp_prune.csv")))
def col(k): return np.array([float(r[k]) for r in R])
sp=[0,30,50,70]
dense=col("dense"); p30=col("sp30"); p50=col("sp50"); p70=col("sp70")
M=[dense.mean(),p30.mean(),p50.mean(),p70.mean()]; S=[dense.std(),p30.std(),p50.std(),p70.std()]
nz=[524,365,260,156]
fig,ax=plt.subplots(figsize=(8.2,5.2),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
ax.errorbar(sp,M,yerr=S,fmt="-o",color="#08306b",lw=2.6,ms=8,capsize=5,zorder=4,label="pruned large + fine-tune")
ax.axhline(0.624,ls="--",color="#9aa7b4",lw=1.6,zorder=2,label="dense large (0.624)")
for x,m,n in zip(sp,M,nz):
    ax.annotate(f"{n}K nz",(x,m),textcoords="offset points",xytext=(0,14),ha="center",fontsize=9.5,color="#555")
ax.set_xticks(sp); ax.set_xticklabels([f"{s}%" for s in sp],fontsize=11)
ax.set_xlabel("Pruning sparsity (global magnitude)",fontsize=12.5)
ax.set_ylabel("Macro-F1 (mixed-SNR, 3-seed)",fontsize=12.5)
ax.set_ylim(0.0,1.0); ax.set_yticks(np.arange(0,1.01,0.2)); ax.tick_params(labelsize=10)
ax.text(0.5,0.12,"Performance flat up to 70% sparsity → large has redundant capacity\n(prunable to 156K), yet a 9K model trained from scratch collapses.",
        transform=ax.transAxes,fontsize=10.5,color="#08306b",ha="center",
        bbox=dict(boxstyle="round,pad=0.4",fc="#f0f4f8",ec="#cdd6df"))
ax.legend(loc="upper right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_pruning.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out,"means",[f"{m:.3f}" for m in M])
