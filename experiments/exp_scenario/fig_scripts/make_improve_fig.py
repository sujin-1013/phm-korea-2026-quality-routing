import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
def col(path,k): return np.array([float(r[k]) for r in csv.DictReader(open(path))])
# clean-large canonical from mixedfig
large=np.array([0.594,0.616,0.661])
ens=col(f"{EXP}/imp_ensemble.csv","ens_sml")
p70=col(f"{EXP}/imp_prune.csv","sp70")
na_cv=col(f"{EXP}/imp_noiseaug.csv","aug_large")
na_nv=col(f"{EXP}/imp_noiseaug_nv.csv","aug_nv")
items=[("Single large\n(clean-train)",large,"#08306b"),
       ("Ensemble\n(s+m+l)",ens,"#9ecae1"),
       ("Pruning 70%\n+ fine-tune",p70,"#6baed6"),
       ("Noise-aug\n(clean-val)",na_cv,"#b0a160"),
       ("Noise-aug\n(noisy-val)",na_nv,"#E8722C")]
fig,ax=plt.subplots(figsize=(9.4,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,axis="y",color="#e3e9ef",lw=0.9,zorder=0)
xs=np.arange(len(items))
for i,(lab,v,c) in enumerate(items):
    mu,sd=v.mean(),v.std()
    ax.bar(i,mu,0.6,color=c,zorder=3,edgecolor="white",yerr=sd,capsize=4,error_kw=dict(ecolor="#555",lw=1.3))
    ax.text(i,mu+sd+0.012,f"{mu:.3f}",ha="center",fontsize=11.5,fontweight="bold",color=c)
ax.axhline(large.mean(),ls="--",color="#9aa7b4",lw=1.5,zorder=2)
ax.set_xticks(xs); ax.set_xticklabels([it[0] for it in items],fontsize=10.5)
ax.set_ylabel("Macro-F1 (mixed-SNR, 3-seed)",fontsize=12.5); ax.set_ylim(0,0.8); ax.set_yticks(np.arange(0,0.81,0.1)); ax.tick_params(labelsize=10)
ax.set_title("Can we beat the single-large ceiling? Only training-time noise helps.",fontsize=12.5,fontweight="bold",color="#08306b",pad=10)
ax.text(0.5,0.045,"inference-time methods (ensemble/pruning) do not exceed large; noise-aug training (deployment-matched val) does: +0.027 & lower variance",
        transform=ax.transAxes,fontsize=9.3,color="#555",ha="center",style="italic")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_improve.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out,[f"{it[0].splitlines()[0]}={it[1].mean():.3f}" for it in items])
