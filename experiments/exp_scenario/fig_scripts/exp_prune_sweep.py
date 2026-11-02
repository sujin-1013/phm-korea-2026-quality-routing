"""Dynamic pruning sweep for train-big-deploy-small. Variants (--variant):
  saug  : structured channel L2 prune {30,50%} + NOISE-AUG fine-tune (combine compression
          with the +0.027 robust-training win) -> best deployment model? eval mixed.
  rand  : unstructured {70,90%} RANDOM vs MAGNITUDE(L1) pruning, clean FT -> control that
          the magnitude criterion actually matters (not just fine-tuning).
  sfine : structured channel L2 {10,20,40%} clean FT -> fill the accuracy-MACs frontier.
All reuse saved large + fine-tune, 3 seeds via --seeds. Report F1 + MACs/nonzero.
"""
import os, sys, csv, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
CONV_OUTLEN=[128,64,32,16,8]; KS=[64,3,3,3,3]; FT_EP=12; NAUG=[float("inf"),20.,10.,5.,0.,-5.,-10.]
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
def pall(m): return [(mod,"weight") for mod in m.modules() if isinstance(mod,(nn.Conv1d,nn.Linear))]
def surv(m): return [int((c.weight.abs().sum(dim=(1,2))>0).sum().item()) for c in convs(m)]
def macs(s):
    t=0; prev=1
    for l,(ol,k) in enumerate(zip(CONV_OUTLEN,KS)): t+=ol*s[l]*prev*k; prev=s[l]
    return t
def nonzero(m):
    tot=z=0
    for mod,_ in pall(m):
        w=mod.weight; tot+=w.numel(); z+=int((w==0).sum())
    return tot-z
def fresh(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
def finetune(m,Xtr,ytr,Xv,yv,cw,aug,valX):
    opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3); lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0); best=-1; bs=None
    for ep in range(FT_EP):
        m.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; snr=float(aug[torch.randint(len(aug),(1,)).item()])
            xb=add_awgn(Xt[idx].clone(),snr_db=snr,generator=g).to(dev)
            opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
        m.eval(); vf=evalf(m,valX,yv)
        if vf>best: best=vf; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); return m
ap=argparse.ArgumentParser(); ap.add_argument("--variant",required=True); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xte_n=mixed_noise(Xte,222); Xvt_clean=torch.tensor(Xv,dtype=torch.float32); Xv_n=mixed_noise(Xv,333)
    res={"seed":seed}
    if a.variant=="saug":
        for fr in [0.30,0.50]:
            m=fresh(seed)
            for c in convs(m): prune.ln_structured(c,name="weight",amount=fr,n=2,dim=0)
            m=finetune(m,Xtr,ytr,Xv,yv,cw,NAUG,Xv_n)  # noise-aug FT, select on noisy val
            f=evalf(m,Xte_n,yte); mm=macs(surv(m)); res[f"saug{int(fr*100)}_f1"]=f; res[f"saug{int(fr*100)}_macs"]=mm
            print(f"seed{seed} STRUCT{int(fr*100)}%+noiseaugFT: F1={f:.3f} MACs={mm/1e6:.2f}M",flush=True)
    elif a.variant=="rand":
        for sp in [0.70,0.90]:
            for crit,name in [(prune.L1Unstructured,"mag"),(prune.RandomUnstructured,"rand")]:
                m=fresh(seed); prune.global_unstructured(pall(m),pruning_method=crit,amount=sp)
                m=finetune(m,Xtr,ytr,Xv,yv,cw,[float("inf")],Xvt_clean)
                f=evalf(m,Xte_n,yte); nz=nonzero(m); res[f"{name}{int(sp*100)}_f1"]=f; res[f"{name}{int(sp*100)}_nz"]=nz
                print(f"seed{seed} {name}-unstr{int(sp*100)}%: F1={f:.3f} nonzero={nz/1e3:.0f}K",flush=True)
    elif a.variant=="sfine":
        for fr in [0.10,0.20,0.40]:
            m=fresh(seed)
            for c in convs(m): prune.ln_structured(c,name="weight",amount=fr,n=2,dim=0)
            m=finetune(m,Xtr,ytr,Xv,yv,cw,[float("inf")],Xvt_clean)
            f=evalf(m,Xte_n,yte); mm=macs(surv(m)); res[f"sf{int(fr*100)}_f1"]=f; res[f"sf{int(fr*100)}_macs"]=mm
            print(f"seed{seed} STRUCT{int(fr*100)}%cleanFT: F1={f:.3f} MACs={mm/1e6:.2f}M",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/exp_sweep_{a.variant}_s{seed}.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
print("done",a.variant,a.seeds,flush=True)
