import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
S=["inf","20","10","5","0","-5","-10"]; x=np.arange(7)
def sweepmean(col):
    def one(s): d={str(r["snr"]):float(r[col]) for r in csv.DictReader(open(f"{EXP}/sweepfig_s{s}.csv"))}; return [d[k] for k in S]
    return np.array([one(s) for s in [42,1,7]]).mean(0)
def qw20mean():
    def one(s):
        d={}
        for r in csv.reader(open(f"{EXP}/qw20single_s{s}.csv")):
            if r[0] in S: d[r[0]]=float(r[1])
        return [d[k] for k in S]
    return np.array([one(s) for s in [42,1,7]]).mean(0)
small,mid,large=sweepmean("small"),sweepmean("mid"),sweepmean("large"); ours=qw20mean()
fig,ax=plt.subplots(figsize=(9.0,5.6),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
ax.plot(x,small,"-o",color="#9ecae1",lw=2.4,ms=6,label="Single small (9K)",zorder=3)
ax.plot(x,mid,"-o",color="#3182bd",lw=2.6,ms=6,label="Single mid (53K)",zorder=3)
ax.plot(x,large,"-o",color="#08306b",lw=3.0,ms=7,label="Single large (524K · 9.96M MACs)",zorder=4)
ax.plot(x,ours,"--D",color="#E8722C",lw=3.2,ms=8,markerfacecolor="white",markeredgewidth=2.2,label="Quality-weighted pruned, ours (417K · 6.45M MACs)",zorder=5)
ax.set_xticks(x); ax.set_xticklabels(["clean","20","10","5","0","-5","-10"],fontsize=12)
ax.set_xlabel("SNR (dB)",fontsize=14); ax.set_ylabel("Macro-F1",fontsize=14)
ax.set_ylim(0,1.0); ax.set_yticks(np.arange(0,1.01,0.1)); ax.tick_params(labelsize=11)
ax.set_title("Quality-weighted pruned model ≈ single large across SNR at 1.5× fewer params",fontsize=11.5,fontweight="bold",color="#08306b",pad=8)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
ax.legend(loc="upper right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df",borderpad=0.6)
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_main_gate.png"; fig.savefig(out,bbox_inches="tight")
print("saved",out,"ours avg=%.3f large avg=%.3f"%(ours.mean(),large.mean()))
