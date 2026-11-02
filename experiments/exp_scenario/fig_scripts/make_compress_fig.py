import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
def rd(seed): return {k:float(v) for k,v in next(iter(csv.DictReader(open(f"{EXP}/imp_prune_hi_s{seed}.csv")))).items()}
R=[rd(s) for s in [42,1,7]]
def m(k): return np.mean([r[k] for r in R])
# pruned-large frontier (nonzero K, F1)
pruned=[(524,0.624,0.028),(156,m("sp70_f1"),np.std([r["sp70_f1"] for r in R])),
        (104,m("sp80_f1"),np.std([r["sp80_f1"] for r in R])),
        (52,m("sp90_f1"),np.std([r["sp90_f1"] for r in R])),
        (26,m("sp95_f1"),np.std([r["sp95_f1"] for r in R]))]
# from-scratch tiers (canonical mixed-SNR)
scratch=[(9,0.491),(53,0.547),(524,0.624)]
fig,ax=plt.subplots(figsize=(8.8,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
px=[p[0] for p in pruned]; py=[p[1] for p in pruned]; pe=[p[2] for p in pruned]
ax.errorbar(px,py,yerr=pe,fmt="-o",color="#E8722C",lw=2.8,ms=8,capsize=4,zorder=5,label="Large trained → pruned + fine-tune")
sx=[s[0] for s in scratch]; sy=[s[1] for s in scratch]
ax.plot(sx,sy,"--s",color="#3182bd",lw=2.2,ms=8,zorder=4,label="Trained from scratch (small/mid/large)")
ax.axhline(0.624,ls=":",color="#9aa7b4",lw=1.4,zorder=2)
# annotate the key gap at ~52K
ax.annotate("",xy=(52,0.606),xytext=(52,0.547),arrowprops=dict(arrowstyle="<->",color="#222",lw=1.6))
ax.text(60,0.576,"+0.059\nat equal\n~52K params",fontsize=10,color="#222",fontweight="bold",va="center")
ax.annotate("pruned 52K (10×)\n0.606",(52,0.606),textcoords="offset points",xytext=(0,16),fontsize=9.5,color="#E8722C",ha="center",fontweight="bold")
ax.annotate("mid scratch 53K\n0.547",(53,0.547),textcoords="offset points",xytext=(-2,-30),fontsize=9.5,color="#3182bd",ha="center")
ax.annotate("small 9K · 0.491",(9,0.491),textcoords="offset points",xytext=(14,-6),fontsize=9.5,color="#3182bd")
ax.annotate("large 524K · 0.624",(524,0.624),textcoords="offset points",xytext=(-10,10),fontsize=9.5,color="#08306b",ha="right")
ax.set_xscale("log"); ax.set_xlabel("Nonzero parameters (K, log scale)",fontsize=12.5)
ax.set_ylabel("Macro-F1 (mixed-SNR, 3-seed)",fontsize=12.5)
ax.set_xticks([9,26,52,104,156,524]); ax.set_xticklabels(["9","26","52","104","156","524"],fontsize=10)
ax.set_ylim(0.40,0.72); ax.tick_params(labelsize=10)
ax.set_title("Train big, deploy small: pruned large keeps robustness where\nfrom-scratch small models collapse",fontsize=12.5,fontweight="bold",color="#08306b",pad=10)
ax.legend(loc="lower right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_compress.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
print("pruned frontier:",[(p[0],round(p[1],3)) for p in pruned])
