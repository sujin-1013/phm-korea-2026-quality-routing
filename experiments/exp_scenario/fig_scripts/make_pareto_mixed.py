import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
def load(seed):
    return {r["method"]:(float(r["f1"]),float(r["cost_K"])) for r in csv.DictReader(open(f"/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/mixedfig_s{seed}.csv"))}
D=[load(s) for s in [42,1,7]]
def agg(m): a=np.array([d[m][0] for d in D]); c=np.mean([d[m][1] for d in D]); return a.mean(),c
sm=agg("single_small"); mi=agg("single_mid"); lg=agg("single_large")
hg=agg("hardgate"); ou=agg("ours_costmatch")
fig,ax=plt.subplots(figsize=(8.6,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
def pt(a,col,s=150): ax.scatter([a[1]],[a[0]],s=s,color=col,zorder=4,edgecolor="white",linewidth=1.2)
pt(sm,"#9ecae1"); pt(mi,"#3182bd"); pt(lg,"#08306b")
pt(hg,"#8a929b")
# ours: orange fill + navy ring, larger
ax.scatter([ou[1]],[ou[0]],s=230,color="#E8722C",zorder=6,edgecolor="#08306b",linewidth=2.2)
ax.annotate(f"Single small\n({sm[1]:.0f}K)",(sm[1],sm[0]),textcoords="offset points",xytext=(6,-30),fontsize=11,color="#3182bd")
ax.annotate(f"Single mid ({mi[1]:.0f}K)",(mi[1],mi[0]),textcoords="offset points",xytext=(10,8),fontsize=11,color="#3182bd")
ax.annotate(f"Single large\n({lg[1]:.0f}K)",(lg[1],lg[0]),textcoords="offset points",xytext=(6,-34),fontsize=11,color="#08306b",ha="center")
ax.annotate(f"Hard gate ({hg[1]:.0f}K)",(hg[1],hg[0]),textcoords="offset points",xytext=(8,-18),fontsize=11,color="#8a929b")
ax.annotate(f"Quality gate, ours\n({ou[1]:.0f}K)",(ou[1],ou[0]),textcoords="offset points",xytext=(-20,26),fontsize=12,color="#E8722C",fontweight="bold",ha="center")
ax.set_xlabel("Effective parameters per input (K)",fontsize=13)
ax.set_ylabel("Macro-F1  (mixed-SNR test, 3-seed)",fontsize=13)
ax.set_xlim(0,575); ax.set_ylim(0,1.0); ax.set_yticks(np.arange(0,1.01,0.1)); ax.tick_params(labelsize=11)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_pareto.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
for nm,a in [("small",sm),("mid",mi),("large",lg),("hard",hg),("ours",ou)]: print(f"  {nm:6s} F1={a[0]:.3f} @{a[1]:.0f}K")
