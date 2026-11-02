import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
def rd(seed): return {k:float(v) for k,v in next(iter(csv.DictReader(open(f"{EXP}/exp_route_compressed_s{seed}.csv")))).items()}
R=[rd(s) for s in [42,1,7]]
def m(k): return np.mean([r[k] for r in R])
# single-model frontier (structured-pruned tiers)
single=[(m("cheap_macs")/1e6,m("cheap_f1"),"struct70%\n(cheap)"),
        (m("mid_macs")/1e6,m("mid_f1"),"struct30%\n(mid)"),
        (m("big_macs")/1e6,m("big_f1"),"dense large")]
route=[(m("route_costm_macs")/1e6,m("route_costm_f1"),"routing\ncost-match"),
       (m("route_accmax_macs")/1e6,m("route_accmax_f1"),"routing\nacc-max")]
fig,ax=plt.subplots(figsize=(8.6,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
sx=[p[0] for p in single]; sy=[p[1] for p in single]
ax.plot(sx,sy,"-o",color="#08306b",lw=2.8,ms=10,zorder=4,label="Single structured-pruned model")
soff=[(8,-22),(-66,-24),(10,4)]  # cheap, mid, dense label offsets
for (x,y,lab),off in zip(single,soff): ax.annotate(lab,(x,y),textcoords="offset points",xytext=off,fontsize=9.5,color="#08306b")
roff=[(2,-30),(8,12)]  # cost-match (below), acc-max (above-right)
for (x,y,lab),off in zip(route,roff):
    ax.scatter([x],[y],marker="D",s=150,color="#E8722C",zorder=5,edgecolor="#08306b",linewidth=1.6)
    ax.annotate(lab,(x,y),textcoords="offset points",xytext=off,fontsize=9.5,color="#E8722C",fontweight="bold")
# dominance arrow: routing acc-max vs single struct30 at same F1
ax.annotate("",xy=(m("mid_macs")/1e6,m("mid_f1")),xytext=(m("route_accmax_macs")/1e6,m("route_accmax_f1")),
            arrowprops=dict(arrowstyle="->",color="#777",lw=1.5,ls="--"))
ax.text(2.6,0.665,"same F1, fewer MACs\n→ single model wins",fontsize=9.5,color="#444",ha="left")
ax.set_xlabel("Average MACs per window (M)",fontsize=12.5); ax.set_ylabel("Macro-F1 (mixed-SNR, 3-seed)",fontsize=12.5)
ax.set_xlim(0,11); ax.set_ylim(0.45,0.70); ax.tick_params(labelsize=10)
ax.set_title("Routing does not beat a single compressed model\n(single structured-pruned model dominates the accuracy–compute frontier)",fontsize=12,fontweight="bold",color="#08306b",pad=10)
ax.legend(loc="lower right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_route_vs_compress.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
print("single:",[(round(p[0],2),round(p[1],3)) for p in single],"route:",[(round(p[0],2),round(p[1],3)) for p in route])
