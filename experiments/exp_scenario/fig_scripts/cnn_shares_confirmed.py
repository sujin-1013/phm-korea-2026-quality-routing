"""Per-SNR routing tier-selection share for the CONFIRMED method {base,x300,large}=133/331/524.
iso operating point (min val cost s.t. val F1 >= val_large-0.01), matching the 266K headline.
Multi-seed; trains the slide-size estimator per seed, groups test windows by injected SNR.
Outputs per-SNR small/mid/large share averaged over seeds + per-seed cost (shows bimodality).
"""
import argparse, itertools, sys
from pathlib import Path

import numpy as np, torch, torch.nn as nn, csv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
SIZES["x300"]=(32,5)
from phm_routing.models.noise_estimator import DEPLOYED_ESTIMATOR_CHANNELS, NoiseEstimator1D, snr_to_target
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT = ROOT / "experiments" / "exp_scenario" / "final_models"
dev = "cuda" if torch.cuda.is_available() else "cpu"
TIERS=["base","x300","large"]; LEVELS=[20.,10.,5.,0.,-5.,-10.,float("inf")]
SNR_ORDER=["inf","20","10","5","0","-5","-10"]
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=str,default="42,1,7,2,3"); ap.add_argument("--epochs",type=int,default=30)
a=ap.parse_args(); SEEDS=[int(s) for s in a.seeds.split(",")]

def mixed_noise(X, seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=X.shape[0]; assign=rng.integers(0,len(LEVELS),N); tsnr=np.empty(N,dtype=object)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]; tsnr[m]=("inf" if not np.isfinite(s) else str(int(s)))
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn,tsnr

def tier_preds(m,Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)

def train_est(Xtr,channels,epochs,seed):
    est=NoiseEstimator1D(channels=tuple(channels),kernel_size=7).to(dev)
    npar=sum(p.numel() for p in est.parameters())
    opt=torch.optim.AdamW(est.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.MSELoss()
    Xt=torch.tensor(np.asarray(Xtr),dtype=torch.float32); rng=np.random.default_rng(seed); g=torch.Generator().manual_seed(seed)
    lut={s:float(snr_to_target(s)) for s in LEVELS}
    for ep in range(epochs):
        est.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; xb=Xt[idx].clone(); n=xb.shape[0]
            assign=rng.integers(0,len(LEVELS),n); tgt=np.empty(n,dtype=np.float32)
            for li,s in enumerate(LEVELS):
                msk=assign==li; tgt[msk]=lut[s]; ii=np.where(msk)[0]
                if np.isfinite(s) and len(ii): xb[ii]=add_awgn(xb[ii],snr_db=float(s),generator=g)
            pred=est(xb.unsqueeze(1).to(dev)); loss=lossf(pred,torch.tensor(tgt,device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    est.eval(); return est,npar

def est_out(est,Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return 1.0-np.concatenate(o)
route_k=lambda q,taus: sum((q<t).astype(int) for t in taus)
pred_stack=lambda lg,idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])

agg=np.zeros((7,3)); costs=[]; f1s=[]
print(f"[shares confirmed {TIERS}] seeds={SEEDS}", flush=True)
for seed in SEEDS:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,16000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte)
    M={}; pc={}
    for sz in TIERS:
        bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); st=torch.load(CKPT / f"cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)
        m.load_state_dict(st["model"]); M[sz]=m.eval(); pc[sz]=st["npar"]
    est,enp=train_est(Xtr,DEPLOYED_ESTIMATOR_CHANNELS,a.epochs,seed)
    Xvn,_=mixed_noise(Xv,111); Xtn,tsnr=mixed_noise(Xte,222)
    lgv={t:tier_preds(M[t],Xvn) for t in TIERS}; lgt={t:tier_preds(M[t],Xtn) for t in TIERS}
    qv=est_out(est,Xvn); qt=est_out(est,Xtn); val_large=macro_f1(yv,lgv["large"])
    # iso thresholds
    grid=[round(x,2) for x in np.arange(0.10,0.96,0.05)]; EST=enp/1e3
    def cost(idx): sh=[(idx==k).mean() for k in range(3)]; return sum(sh[k]*pc[TIERS[k]] for k in range(3))/1e3+EST
    best=None
    for taus in itertools.combinations(grid,2):
        iv=route_k(qv,taus); fv=macro_f1(yv,pred_stack(lgv,iv)); c=cost(iv)
        if fv>=val_large-0.01 and (best is None or c<best[1]): best=(taus,c)
    if best is None:
        bf=max(itertools.combinations(grid,2),key=lambda t:macro_f1(yv,pred_stack(lgv,route_k(qv,t)))); best=(bf,cost(route_k(qv,bf)))
    taus,val_cost=best; it=route_k(qt,taus); c=cost(it); f1=macro_f1(yte,pred_stack(lgt,it)); costs.append(c); f1s.append(f1)
    sh=np.zeros((7,3))
    for si,sn in enumerate(SNR_ORDER):
        mask=tsnr==sn; n=mask.sum()
        sh[si]=[(it[mask]==k).mean() for k in range(3)] if n>0 else [0,0,0]
    agg+=sh
    print(f"  seed{seed}: tau={taus} F1={f1:.4f} val_cost={val_cost:.0f}K test_cost={c:.0f}K", flush=True)
agg/=len(SEEDS)
print(f"\nmean F1={np.mean(f1s):.4f}  mean cost={np.mean(costs):.0f}K  per-seed cost={[round(x) for x in costs]}")
print("\nSNR별 tier 선택 (%, base133 / x300 331 / large524):")
lab={"inf":"clean"}
for si,sn in enumerate(SNR_ORDER):
    a3=agg[si]*100; print(f"  {lab.get(sn,sn):>5s} : {a3[0]:5.1f} / {a3[1]:5.1f} / {a3[2]:5.1f}")
with (ROOT / "experiments" / "exp_scenario" / "shares_confirmed.csv").open("w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["snr","small_133K","medium_331K","large_524K"])
    for si,sn in enumerate(SNR_ORDER): w.writerow([lab.get(sn,sn)]+[f"{x:.4f}" for x in agg[si]])
print("saved shares_confirmed.csv")
