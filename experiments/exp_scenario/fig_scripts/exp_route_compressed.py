"""Routing over STRUCTURED-PRUNED MACs-tiers: build {dense, struct30%(5.0M), struct70%(1.0M)}
from large, route per-window by noise estimator. Eval F1 vs AVG MACs on mixed-SNR.
Tests if routing over compressed tiers beats single struct30% (0.621@5.0M) on the
accuracy-MACs frontier. Includes oracle (any-tier-correct) upper bound. 3 seeds via --seeds.
"""
import os, sys, csv, argparse, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
CONV_OUTLEN=[128,64,32,16,8]; KS=[64,3,3,3,3]; FT_EP=12
EST_MACS=0.30e6  # estimator overhead (small, approx); reported separately
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def convs(m): return [mod for mod in m.modules() if isinstance(mod,nn.Conv1d)]
def surviving(m): return [int((c.weight.abs().sum(dim=(1,2))>0).sum().item()) for c in convs(m)]
def macs(surv):
    tot=0; prev=1
    for l,(ol,k) in enumerate(zip(CONV_OUTLEN,KS)): tot+=ol*surv[l]*prev*k; prev=surv[l]
    return tot
def predict(m,Xn):
    out=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): out.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(out)
def make_pruned(seed,frac,Xtr,ytr,Xv,yv,cw):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"])
    if frac>0:
        for c in convs(m): prune.ln_structured(c,name="weight",amount=frac,n=2,dim=0)
        opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3); lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
        Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0)
        Xvt=torch.tensor(Xv,dtype=torch.float32); best=-1; bs=None
        for ep in range(FT_EP):
            m.train(); perm=torch.randperm(Xt.shape[0])
            for i in range(0,Xt.shape[0],64):
                idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev)
                opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
            m.eval()
            from numpy import concatenate as cc
            vf=macro_f1(yv,predict(m,Xvt))
            if vf>best: best=vf; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(bs)
    m.eval(); return m, macs(surviving(m))
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev); est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
def scores(Xn):
    q=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): q.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(q)
rows=[]
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3); cnn_run.AUG=(float("inf"),)
    # tiers: cheap=struct70, mid=struct30, big=dense
    big,mB=make_pruned(seed,0.0,Xtr,ytr,Xv,yv,cw); mid,mM=make_pruned(seed,0.30,Xtr,ytr,Xv,yv,cw); chp,mC=make_pruned(seed,0.70,Xtr,ytr,Xv,yv,cw)
    TM=[chp,mid,big]  # cheap,mid,big models
    tier_macs=[mC,mM,mB]  # cheap,mid,big in MACs
    Xvn=mixed_noise(Xv,333); Xtn=mixed_noise(Xte,222)
    Pv=[predict(t,Xvn) for t in TM]; Pt=[predict(t,Xtn) for t in TM]
    qv=scores(Xvn); qt=scores(Xtn)
    route=lambda q,t1,t2:(q>=t1).astype(int)+(q>=t2).astype(int)  # 0=cheap,1=mid,2=big (higher noise->bigger)
    def stack(P,idx): return np.array([P[idx[i]][i] for i in range(len(idx))])
    def cost(idx): sh=[(idx==k).mean() for k in range(3)]; return sum(sh[k]*tier_macs[k] for k in range(3))
    f_big=macro_f1(yte,Pt[2]); f_mid=macro_f1(yte,Pt[1]); f_chp=macro_f1(yte,Pt[0])
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    vlf=macro_f1(yv,Pv[2])  # big val F1
    accmax=None; costm=None
    for t1,t2 in pairs:
        fv=macro_f1(yv,stack(Pv,route(qv,t1,t2)))
        if accmax is None or fv>accmax[0]: accmax=(fv,t1,t2)
        ct=cost(route(qt,t1,t2))
        if fv>=vlf-0.01 and (costm is None or ct<costm[3]): costm=(fv,t1,t2,ct)
    ia=route(qt,accmax[1],accmax[2]); fa=macro_f1(yte,stack(Pt,ia)); ca=cost(ia)
    if costm: ic=route(qt,costm[1],costm[2]); fc=macro_f1(yte,stack(Pt,ic)); cc_=cost(ic)
    else: fc,cc_=fa,ca
    # oracle any-correct upper bound over the 3 compressed tiers
    cor=[(Pt[k]==yte) for k in range(3)]; anyc=cor[0]|cor[1]|cor[2]
    rows.append({"seed":seed,"cheap_f1":f_chp,"cheap_macs":mC,"mid_f1":f_mid,"mid_macs":mM,"big_f1":f_big,"big_macs":mB,
                 "route_accmax_f1":fa,"route_accmax_macs":ca,"route_costm_f1":fc,"route_costm_macs":cc_,"oracle_acc":anyc.mean()})
    print(f"seed{seed}: tiers cheap={f_chp:.3f}@{mC/1e6:.2f}M mid={f_mid:.3f}@{mM/1e6:.2f}M big={f_big:.3f}@{mB/1e6:.2f}M | route accmax={fa:.3f}@{ca/1e6:.2f}M costm={fc:.3f}@{cc_/1e6:.2f}M | oracle_acc={anyc.mean():.3f}",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/exp_route_compressed_s{seed}.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[-1].keys())); w.writeheader(); w.writerow(rows[-1])
print("done seeds",a.seeds,flush=True)
