"""Oracle headroom figure (S9): per-SNR large_acc vs oracle(any-tier-correct) from saved
small/mid/large. Shows complementarity EXISTS (oracle>>large) but estimator can't capture it.
Reuses saved models (no training). System python3 has no torch -> compute via sg311? No:
this needs torch for inference. Run with /tmp/sg311/bin/python; plotting via matplotlib in
system python3. So: stage1 (torch) dumps per-SNR arrays to npz; stage2 (mpl) plots.
This file is stage2 (plot) reading /tmp/oracle_persnr.npz.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
D=np.load("/tmp/oracle_persnr.npz")
S=["clean","20","10","5","0","-5","-10"]; x=np.arange(7)
la=D["large"]; oa=D["oracle"]
fig,ax=plt.subplots(figsize=(8.8,5.4),dpi=150); ax.set_facecolor("white"); ax.grid(True,color="#e3e9ef",lw=0.9,zorder=0)
ax.plot(x,la,"-o",color="#08306b",lw=2.8,ms=7,zorder=4,label="Single large (accuracy)")
ax.plot(x,oa,"--o",color="#E8722C",lw=2.8,ms=7,zorder=4,label="Oracle: best tier per window (upper bound)")
ax.fill_between(x,la,oa,color="#E8722C",alpha=0.12,zorder=2)
for i in range(7):
    if oa[i]-la[i]>0.04: ax.annotate(f"+{oa[i]-la[i]:.2f}",(x[i],(la[i]+oa[i])/2),fontsize=9,color="#b5531a",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(S,fontsize=11); ax.set_xlabel("SNR (dB)",fontsize=12.5)
ax.set_ylabel("Accuracy (mixed-SNR test, 3-seed)",fontsize=12.5); ax.set_ylim(0,1.0); ax.tick_params(labelsize=10)
ax.set_title("Complementarity exists but is unreachable: oracle (label-aware) beats large,\nyet no cheap router (SNR/confidence) identifies those windows",fontsize=11.5,fontweight="bold",color="#08306b",pad=10)
ax.text(0.5,0.07,"headroom = 8% of windows (large wrong, small/mid right), concentrated at 0–5 dB",transform=ax.transAxes,fontsize=9.5,color="#555",ha="center",style="italic")
ax.legend(loc="upper right",fontsize=10.5,frameon=True,framealpha=0.95,edgecolor="#cdd6df")
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#9aa7b4")
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_oracle.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
