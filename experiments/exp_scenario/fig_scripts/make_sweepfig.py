import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
S=["inf","20","10","5","0","-5","-10"]; x=np.arange(7)
def load(seed,col):
    d={(str(r["snr"])):float(r[col]) for r in csv.DictReader(open(f"/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/sweepfig_s{seed}.csv"))}
    return [d[s] for s in S]
def mean(col): return np.array([load(s,col) for s in [42,1,7]]).mean(0)
small,mid,large,hard,ours=mean("small"),mean("mid"),mean("large"),mean("hard"),mean("ours")
fig,ax=plt.subplots(figsize=(9.0,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
ax.plot(x,small,"-o",color="#9ecae1",lw=2.4,ms=6,label="Single small",zorder=3)
ax.plot(x,mid,"-o",color="#3182bd",lw=2.6,ms=6,label="Single mid",zorder=3)
ax.plot(x,large,"-o",color="#08306b",lw=3.0,ms=7,label="Single large",zorder=4)
ax.plot(x,hard,"-o",color="#8a929b",lw=2.4,ms=6,label="Hard gate",zorder=3)
ax.plot(x,ours,"--D",color="#E8722C",lw=3.2,ms=8,markerfacecolor="white",markeredgewidth=2.2,label="Quality gate, ours",zorder=5)
ax.set_xticks(x); ax.set_xticklabels(["clean","20","10","5","0","-5","-10"],fontsize=12)
ax.set_xlabel("SNR (dB)",fontsize=14); ax.set_ylabel("Macro-F1",fontsize=14)
ax.set_ylim(0,1.0); ax.set_yticks(np.arange(0,1.01,0.1)); ax.tick_params(labelsize=11)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
ax.legend(loc="upper right",fontsize=11.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df",borderpad=0.7)
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_main_gate.png"; fig.savefig(out,bbox_inches="tight")
print("saved",out)
for nm,a in [("small",small),("mid",mid),("large",large),("hard",hard),("ours",ours)]: print(f"  {nm:6s} avg={a.mean():.3f}")
