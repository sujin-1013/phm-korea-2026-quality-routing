"""Per-SNR routing selection shares (mixed test, grouped by true injected SNR).
Loads SAVED final models (no retrain). Outputs per-SNR small/mid/large share for
ours / hardgate / oracle. 3-seed averaged.
"""
import os, sys, itertools
import numpy as np, torch, csv
_ROOT="/home/sujin/phm-korea-2026-snrgate"
sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"
TIERS=["small","mid","large"]; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]
SNR_ORDER=["inf","20","10","5","0","-5","-10"]; dev="cuda"

def mixed_noise(X, seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(X,dtype=torch.float32).clone(); N=X.shape[0]; assign=rng.integers(0,len(LEVELS),N); tsnr=np.empty(N,dtype=object)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]; tsnr[m]=("inf" if not np.isfinite(s) else str(int(s)))
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn,tsnr

def load_models(seed):
    M={}; pc={}
    for sz in TIERS:
        bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); st=torch.load(f"{CKPT}/cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)
        m.load_state_dict(st["model"]); M[sz]=m.eval(); pc[sz]=st["npar"]
    est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev)
    est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
    return M,pc,est

def infer(M,est,Xn):
    N=Xn.shape[0]
    with torch.no_grad():
        lg={}
        for sz in TIERS:
            out=[];
            for i in range(0,N,256): out.append(M[sz](Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
            lg[sz]=np.concatenate(out)
        q=[];
        for i in range(0,N,256): q.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
        q=np.concatenate(q)
    return lg,q
route_idx=lambda q,t1,t2:(q>=t1).astype(int)+(q>=t2).astype(int)
pred_stack=lambda lg,idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])

acc={r:{m:np.zeros((7,3)) for m in ["ours","hard","oracle"]} for r in [0]}  # placeholder
agg={m:np.zeros((7,3)) for m in ["ours","hard","oracle"]}
for seed in [42,1,7]:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    M,pc,est=load_models(seed)
    # operating point: argmax val acc on mixed-val (same as canonical ours)
    Xvm,_=mixed_noise(Xv,111); lgv,qv=infer(M,est,Xvm)
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(a,b) for a,b in itertools.product(grid,grid) if a<b]
    t1o,t2o=max(pairs,key=lambda p:macro_f1(yv,pred_stack(lgv,route_idx(qv,*p))))
    t1h,t2h=np.quantile(qv,0.33),np.quantile(qv,0.66)
    # test
    Xn,tsnr=mixed_noise(Xte,222); lgt,qt=infer(M,est,Xn)
    idx={"ours":route_idx(qt,t1o,t2o),"hard":route_idx(qt,t1h,t2h),
         "oracle":np.array([0 if (t=="inf" or int(t)>=10) else (1 if int(t)>=0 else 2) for t in tsnr])}
    for si,sn in enumerate(SNR_ORDER):
        mask=tsnr==sn; n=mask.sum()
        for m in idx:
            sh=[ (idx[m][mask]==k).mean() for k in range(3)] if n>0 else [0,0,0]
            agg[m][si]+=np.array(sh)
for m in agg: agg[m]/=3.0
# print table
hdr="SNR   | ours s/m/l        | hard s/m/l        | oracle s/m/l"
print(hdr); print("-"*len(hdr))
lab={"inf":"clean"}
for si,sn in enumerate(SNR_ORDER):
    def f(m): a=agg[m][si]*100; return f"{a[0]:4.0f}/{a[1]:4.0f}/{a[2]:4.0f}"
    print(f"{lab.get(sn,sn):>5s} | {f('ours')} | {f('hard')} | {f('oracle')}")
# save csv
with open(f"{_ROOT}/experiments/exp_scenario/shares_per_snr.csv","w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["snr","ours_small","ours_mid","ours_large","hard_small","hard_mid","hard_large","oracle_small","oracle_mid","oracle_large"])
    for si,sn in enumerate(SNR_ORDER):
        row=[lab.get(sn,sn)]
        for m in ["ours","hard","oracle"]: row+= [f"{x:.4f}" for x in agg[m][si]]
        w.writerow(row)
print("saved experiments/exp_scenario/shares_per_snr.csv")
