import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
def load(seed):
    return {r["method"]:(float(r["f1"]),float(r["cost_K"])) for r in csv.DictReader(open(f"/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/mixedfig_s{seed}.csv"))}
D=[load(s) for s in [42,1,7]]
def agg(m): a=np.array([d[m][0] for d in D]); c=np.mean([d[m][1] for d in D]); return a.mean(),a.std(),c
order=[("single_small","Single small","#9ecae1"),("single_mid","Single mid","#3182bd"),
       ("single_large","Single large","#08306b"),("hardgate","Hard gate","#8a929b"),
       ("ours_costmatch","Quality gate, ours","#E8722C")]
fig,ax=plt.subplots(figsize=(8.6,5.6),dpi=150); ax.set_facecolor("white")
ax.grid(True,axis="y",color="#e3e9ef",lw=0.9,zorder=0)
xs=np.arange(len(order))
for i,(k,lab,col) in enumerate(order):
    mu,sd,c=agg(k)
    ax.bar(i,mu,0.62,color=col,zorder=3,edgecolor="white",linewidth=1.0,
           yerr=sd,capsize=4,error_kw=dict(ecolor="#555",lw=1.3))
    ax.text(i,mu+sd+0.012,f"{mu:.3f}",ha="center",va="bottom",fontsize=12,fontweight="bold",color=col)
    ax.text(i,-0.035,f"{c:.0f}K",ha="center",va="top",fontsize=10,color="#666")
ax.set_xticks(xs); ax.set_xticklabels([o[1] for o in order],fontsize=11.5)
ax.set_ylabel("Macro-F1  (mixed-SNR test, 3-seed)",fontsize=13)
ax.set_ylim(0,0.72); ax.set_yticks(np.arange(0,0.72,0.1)); ax.tick_params(labelsize=11)
ax.text(0.012,-0.105,"avg active params:",transform=ax.get_xaxis_transform(),fontsize=9.5,color="#666",ha="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_mixed_bar.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
for k,lab,_ in order: mu,sd,c=agg(k); print(f"  {lab:20s} {mu:.3f}±{sd:.3f} @{c:.0f}K")
