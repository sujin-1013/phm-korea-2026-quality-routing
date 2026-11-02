import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
EXP="/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario"
S=["inf","20","10","5","0","-5","-10"]; x=np.arange(7)
def sweep(col):
    def one(s):
        d={str(r["snr"]):float(r[col]) for r in csv.DictReader(open(f"{EXP}/sweepfig_s{s}.csv"))}; return [d[k] for k in S]
    return np.array([one(s) for s in [42,1,7]]).mean(0)
def mix(m):
    a=[ {r["method"]:(float(r["f1"]),float(r["cost_K"])) for r in csv.DictReader(open(f"{EXP}/mixedfig_s{s}.csv"))} for s in [42,1,7]]
    return np.mean([d[m][0] for d in a]), np.mean([d[m][1] for d in a])

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.4,4.9),dpi=150)
# ---- LEFT: capacity = robustness ----
small,mid,large=sweep("small"),sweep("mid"),sweep("large")
axL.set_facecolor("white"); axL.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
axL.plot(x,small,"-o",color="#9ecae1",lw=2.6,ms=6,label="small (9K)",zorder=3)
axL.plot(x,mid,"-o",color="#3182bd",lw=2.6,ms=6,label="mid (53K)",zorder=3)
axL.plot(x,large,"-o",color="#08306b",lw=3.2,ms=7,label="large (524K)",zorder=4)
axL.set_xticks(x); axL.set_xticklabels(["clean","20","10","5","0","-5","-10"],fontsize=11)
axL.set_xlabel("SNR (dB)",fontsize=12); axL.set_ylabel("Macro-F1",fontsize=12)
axL.set_ylim(0,1.0); axL.set_yticks(np.arange(0,1.01,0.2)); axL.tick_params(labelsize=10)
axL.legend(loc="upper right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df")
axL.annotate("large alone\nsurvives noise",(6,large[6]),textcoords="offset points",xytext=(-78,30),
             fontsize=10.5,color="#08306b",fontweight="bold",ha="center",
             arrowprops=dict(arrowstyle="->",color="#08306b",lw=1.4))
axL.set_title("Capacity matters\nLarger CNNs are more robust to noise",fontsize=12.5,fontweight="bold",color="#08306b",pad=8)
for s in ["top","right"]: axL.spines[s].set_visible(False)
for s in ["left","bottom"]: axL.spines[s].set_color("#9aa7b4")
# ---- RIGHT: routing efficiency (pareto) ----
sm,mi,lg=mix("single_small"),mix("single_mid"),mix("single_large")
hg,ou=mix("hardgate"),mix("ours_costmatch")
axR.set_facecolor("white"); axR.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
def pt(a,col,s=140): axR.scatter([a[1]],[a[0]],s=s,color=col,zorder=4,edgecolor="white",linewidth=1.2)
pt(sm,"#9ecae1"); pt(mi,"#3182bd"); pt(lg,"#08306b"); pt(hg,"#8a929b")
axR.scatter([ou[1]],[ou[0]],s=240,color="#E8722C",zorder=6,edgecolor="#08306b",linewidth=2.2)
axR.annotate("small",(sm[1],sm[0]),textcoords="offset points",xytext=(8,-4),fontsize=10,color="#3182bd")
axR.annotate("mid",(mi[1],mi[0]),textcoords="offset points",xytext=(8,-4),fontsize=10,color="#3182bd")
axR.annotate("large (524K)",(lg[1],lg[0]),textcoords="offset points",xytext=(-8,-18),fontsize=10,color="#08306b",ha="right")
axR.annotate("Hard gate",(hg[1],hg[0]),textcoords="offset points",xytext=(6,-16),fontsize=10,color="#8a929b")
axR.annotate("Quality gate, ours\n0.611 @ 398K (1.32× cheaper)",(ou[1],ou[0]),textcoords="offset points",xytext=(-12,22),
             fontsize=11,color="#E8722C",fontweight="bold",ha="center")
axR.set_xlim(0,575); axR.set_ylim(0,1.0); axR.set_yticks(np.arange(0,1.01,0.2)); axR.tick_params(labelsize=10)
axR.set_xlabel("Effective parameters per input (K)",fontsize=12); axR.set_ylabel("Macro-F1 (mixed-SNR)",fontsize=12)
axR.set_title("Routing helps efficiency\nQuality gate approaches large-model F1 at lower cost",fontsize=12.5,fontweight="bold",color="#E8722C",pad=8)
for s in ["top","right"]: axR.spines[s].set_visible(False)
for s in ["left","bottom"]: axR.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_conclusion.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
