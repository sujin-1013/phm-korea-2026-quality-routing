"""Quality-weighted pruning: use the quality estimator's noise score to decide WHICH
channels to keep (vs plain magnitude). Channel importance schemes at SAME per-layer
sparsity (=same MACs), so only WHICH channels differ:
  mag    : L2 weight norm (standard magnitude pruning)        [baseline]
  act    : mean|activation| over mixed calib (unweighted)     [control]
  qweight: sum_w q_w * mean|activation|, q=quality(noise) score -> keep channels active
           on NOISY windows (the proposed quality-driven pruning)
Structured channel prune + clean fine-tune, eval mixed-SNR. 3 seeds via --seeds.
"""
import os, sys, csv, argparse
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
CONV_OUTLEN=[128,64,32,16,8]; KS=[64,3,3,3,3]; FT_EP=12; FRACS=[0.30,0.50]
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        msk=assign==li; idx=np.where(msk)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def evalf(m,Xn,y):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return macro_f1(y,np.concatenate(o))
def convs(m): return [mod for mod in m.modules() if isinstance(mod,nn.Conv1d)]
def surv(m): return [int((c.weight.abs().sum(dim=(1,2))>0).sum().item()) for c in convs(m)]
def macs(s):
    t=0; prev=1
    for l,(ol,k) in enumerate(zip(CONV_OUTLEN,KS)): t+=ol*s[l]*prev*k; prev=s[l]
    return t
def fresh(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
def channel_importance(m,Xcal,q,scheme):
    cs=convs(m); imp=[None]*len(cs)
    if scheme=="mag":
        for i,c in enumerate(cs): imp[i]=c.weight.detach().pow(2).sum(dim=(1,2)).sqrt().cpu().numpy()
        return imp
    acc=[torch.zeros(c.out_channels,device=dev) for c in cs]; wsum=0.0; hooks=[]
    def mk(i):
        def h(mod,inp,out):
            a=out.detach().abs().mean(dim=2)  # (B,C)
            w=qb.view(-1,1) if scheme=="qweight" else torch.ones(a.shape[0],1,device=dev)
            acc[i]+=(a*w).sum(dim=0)
        return h
    for i,c in enumerate(cs): hooks.append(c.register_forward_hook(mk(i)))
    m.eval()
    with torch.no_grad():
        for j in range(0,Xcal.shape[0],256):
            xb=Xcal[j:j+256].to(dev); qb=torch.tensor(q[j:j+256],dtype=torch.float32,device=dev)
            wsum+= (qb.sum().item() if scheme=="qweight" else xb.shape[0]); m(xb)
    for hk in hooks: hk.remove()
    for i in range(len(cs)): imp[i]=(acc[i]/max(wsum,1e-6)).cpu().numpy()
    return imp
def prune_by_importance(m,imp,fr):
    for c,iv in zip(convs(m),imp):
        n=len(iv); k=int(round(fr*n));
        if k<=0: continue
        kill=np.argsort(iv)[:k]  # lowest importance channels
        mask=torch.ones_like(c.weight); mask[kill]=0
        prune.custom_from_mask(c,name="weight",mask=mask)
def finetune(m,Xtr,ytr,yv,valX,cw):
    opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3); lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0); best=-1; bs=None
    for ep in range(FT_EP):
        m.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev)
            opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
        m.eval(); vf=evalf(m,valX,yv)
        if vf>best: best=vf; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); return m
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev); est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
def qscore(Xn):
    q=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): q.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(q)
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xcal=mixed_noise(Xv,333); q=qscore(Xcal); Xcal_t=torch.tensor(Xcal,dtype=torch.float32) if not torch.is_tensor(Xcal) else Xcal
    valX=mixed_noise(Xv,334); Xte_n=mixed_noise(Xte,222)
    res={"seed":seed}
    for fr in FRACS:
        for scheme in ["mag","act","qweight"]:
            m=fresh(seed); imp=channel_importance(m,Xcal,q,scheme); prune_by_importance(m,imp,fr)
            m=finetune(m,Xtr,ytr,yv,valX,cw); f=evalf(m,Xte_n,yte); mm=macs(surv(m))
            res[f"{scheme}{int(fr*100)}_f1"]=f; res[f"{scheme}{int(fr*100)}_macs"]=mm
            print(f"seed{seed} {scheme}-struct{int(fr*100)}%: F1={f:.3f} MACs={mm/1e6:.2f}M",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/exp_quality_prune_s{seed}.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
print("done",a.seeds,flush=True)
