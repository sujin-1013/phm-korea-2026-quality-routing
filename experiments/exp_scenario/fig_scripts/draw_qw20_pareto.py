import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
# singles mixed-SNR (canonical), params K
def mixmean(m):
    a=[ {r["method"]:float(r["f1"]) for r in csv.DictReader(open(f"{EXP}/mixedfig_s{s}.csv"))} for s in [42,1,7]]
    return np.mean([d[m] for d in a])
sm=(9,mixmean("single_small")); mi=(53,mixmean("single_mid")); lg=(524,mixmean("single_large"))
# qw20: params 417K, mixed F1 from qw20single
def qw20mix():
    v=[]
    for s in [42,1,7]:
        for r in csv.reader(open(f"{EXP}/qw20single_s{s}.csv")):
            if r[0]=="mixed_f1": v.append(float(r[1]))
    return np.mean(v)
ou=(417,qw20mix())
fig,ax=plt.subplots(figsize=(8.6,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
def pt(a,col,s=150): ax.scatter([a[0]],[a[1]],s=s,color=col,zorder=4,edgecolor="white",linewidth=1.2)
pt(sm,"#9ecae1"); pt(mi,"#3182bd"); pt(lg,"#08306b")
ax.scatter([ou[0]],[ou[1]],s=240,color="#E8722C",zorder=6,edgecolor="#08306b",linewidth=2.2)
ax.annotate(f"Single small\n(9K)",(sm[0],sm[1]),textcoords="offset points",xytext=(8,-30),fontsize=11,color="#3182bd")
ax.annotate(f"Single mid (53K)",(mi[0],mi[1]),textcoords="offset points",xytext=(10,8),fontsize=11,color="#3182bd")
ax.annotate(f"Single large\n(524K)",(lg[0],lg[1]),textcoords="offset points",xytext=(6,-34),fontsize=11,color="#08306b",ha="center")
ax.annotate(f"Quality-weighted pruned, ours\n(417K · 1.5x fewer params)",(ou[0],ou[1]),textcoords="offset points",xytext=(-20,24),fontsize=12,color="#E8722C",fontweight="bold",ha="center")
ax.set_xlabel("Parameters (K)",fontsize=13); ax.set_ylabel("Macro-F1  (mixed-SNR, 3-seed)",fontsize=13)
ax.set_xlim(0,575); ax.set_ylim(0,1.0); ax.set_yticks(np.arange(0,1.01,0.1)); ax.tick_params(labelsize=11)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_pareto.png"; fig.savefig(out,bbox_inches="tight")
print("saved",out,"qw20=%.3f@417K large=%.3f@524K"%(ou[1],lg[1]))
