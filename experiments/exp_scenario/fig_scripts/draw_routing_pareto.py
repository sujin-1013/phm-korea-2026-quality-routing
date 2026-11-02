import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
# large-derived compression family + routing, MACs axis, mixed-SNR 3-seed
def qw20mix():
    v=[]
    for s in [42,1,7]:
        for r in csv.reader(open(f"{EXP}/qw20single_s{s}.csv")):
            if r[0]=="mixed_f1": v.append(float(r[1]))
    return np.mean(v)
def routeiso():
    f=[]; c=[]
    for s in [42,1,7]:
        d={r[0]:r[1] for r in csv.reader(open(f"{EXP}/exp_qw20_route_s{s}.csv"))}
        # header row then data row -> reparse properly
    # simpler: read DictReader
    fs=[]; cs=[]
    for s in [42,1,7]:
        row=next(iter(csv.DictReader(open(f"{EXP}/exp_qw20_route_s{s}.csv"))))
        fs.append(float(row["route_iso_f1"])); cs.append(float(row["route_iso_macs"])/1e6)
    return np.mean(fs),np.mean(cs)
large=(9.96,0.624); qw20=(6.45,qw20mix()); qw40=(3.72,0.611); rf,rc=routeiso(); rout=(rc,rf)
fig,ax=plt.subplots(figsize=(8.8,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
# frontier line through single compressed models
fr=[qw40,qw20,large]
ax.plot([p[0] for p in fr],[p[1] for p in fr],"-",color="#9aa7b4",lw=1.6,zorder=2)
def pt(a,col,s=150): ax.scatter([a[0]],[a[1]],s=s,color=col,zorder=4,edgecolor="white",linewidth=1.2)
pt(qw40,"#6baed6"); pt(qw20,"#3182bd"); pt(large,"#08306b")
ax.scatter([rout[0]],[rout[1]],marker="D",s=230,color="#E8722C",zorder=6,edgecolor="#08306b",linewidth=2.2)
ax.annotate(f"qw40 prune\n0.611@3.7M",(qw40[0],qw40[1]),textcoords="offset points",xytext=(2,-32),fontsize=10,color="#3182bd",ha="center")
ax.annotate(f"qw20 prune\n{qw20[1]:.3f}@6.5M",(qw20[0],qw20[1]),textcoords="offset points",xytext=(2,12),fontsize=10,color="#08306b",ha="center")
ax.annotate(f"Single large\n0.624@9.96M",(large[0],large[1]),textcoords="offset points",xytext=(-6,-30),fontsize=10,color="#08306b",ha="center")
ax.annotate(f"Quality gate, ours (routing)\n{rout[1]:.3f}@{rout[0]:.2f}M",(rout[0],rout[1]),textcoords="offset points",xytext=(0,22),fontsize=11.5,color="#E8722C",fontweight="bold",ha="center")
ax.set_xlabel("Average MACs per input (M)",fontsize=13); ax.set_ylabel("Macro-F1  (mixed-SNR, 3-seed)",fontsize=13)
ax.set_xlim(0,11); ax.set_ylim(0.45,0.72); ax.tick_params(labelsize=11)
ax.set_title("Cost-accuracy frontier: quality-gate routing maintains accuracy at lower avg MACs",fontsize=11,fontweight="bold",color="#08306b",pad=8)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_pareto.png"; fig.savefig(out,bbox_inches="tight")
print("saved",out,"routing %.3f@%.2fM"%(rout[1],rout[0]))
