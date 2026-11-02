import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
D=np.load("/tmp/wave_raw.npz"); fs=int(D["fs"])
N,OR,ORn=D["N"],D["OR"],D["ORnoisy"]; t=np.arange(len(N))/fs*1000.0
fig,ax=plt.subplots(3,1,figsize=(9.2,6.2),dpi=150,sharex=True)
panels=[(N,"Normal (clean)","#3182bd"),
        (OR,"Outer-race fault (clean)","#08306b"),
        (ORn,"Outer-race fault + AWGN (−10 dB)","#E8722C")]
for a,(sig,title,col) in zip(ax,panels):
    a.plot(t,sig,color=col,lw=0.8); a.set_facecolor("white"); a.set_ylabel("Acc. (a.u.)",fontsize=11)
    a.text(0.012,0.90,title,transform=a.transAxes,fontsize=12,fontweight="bold",color=col,va="top",
           bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.7))
    a.grid(True,color="#eef2f6",lw=0.8); a.margins(x=0)
    for s in ["top","right"]: a.spines[s].set_visible(False)
    for s in ["left","bottom"]: a.spines[s].set_color("#9aa7b4")
# rows 1&2 share scale (real, comparable amplitude); row 3 its own (noise dominates)
yc=max(np.abs(N).max(),np.abs(OR).max())*1.08
ax[0].set_ylim(-yc,yc); ax[1].set_ylim(-yc,yc)
yn=np.abs(ORn).max()*1.05; ax[2].set_ylim(-yn,yn)
ax[2].set_xlabel("Time (ms)",fontsize=12)
fig.suptitle("Paderborn-PU vibration · 64 kHz · 40 ms segment",fontsize=11,color="#666",y=0.997)
fig.tight_layout(); out="/home/sujin/phm-korea-2026-snrgate/presentation_assets/fig_waveform.png"
fig.savefig(out,bbox_inches="tight"); print("saved",out)
