"""P3 deployment positive: push pruning sparsity 80/90/95% (+70 ref) to find the
compression frontier — how few nonzero params keep large-level noise robustness.
Reuse saved large, magnitude prune + clean fine-tune, eval mixed-SNR. Per-seed via arg
so two GPUs can split seeds. Honest: maintain (not improve) large accuracy at low params.
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
SPARS=[0.70,0.80,0.90,0.95]; FT_EP=12
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def evalf(m,Xn,y):
    out=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): out.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return macro_f1(y,np.concatenate(out))
def pparams(m): return [(mod,"weight") for mod in m.modules() if isinstance(mod,(nn.Conv1d,nn.Linear))]
def nonzero(m):
    tot=z=0
    for mod,_ in pparams(m):
        w=mod.weight; tot+=w.numel(); z+=int((w==0).sum())
    return tot-z
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    cw=inverse_freq_class_weights(ytr,num_class=3); cnn_run.AUG=(float("inf"),); Xn=mixed_noise(Xte,222)
    bc,nb=SIZES["large"]
    dense=CNN1D(bc,nb).to(dev); dense.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); dense.eval()
    fd=evalf(dense,Xn,yte); res={"seed":seed,"dense":fd}
    for sp in SPARS:
        m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"])
        prune.global_unstructured(pparams(m),pruning_method=prune.L1Unstructured,amount=sp)
        opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3); lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
        Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0)
        Xvt=torch.tensor(Xv,dtype=torch.float32); best=-1; bs=None
        for ep in range(FT_EP):
            m.train(); perm=torch.randperm(Xt.shape[0])
            for i in range(0,Xt.shape[0],64):
                idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev)
                opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
            m.eval(); vf=evalf(m,Xvt,yv)
            if vf>best: best=vf; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(bs); m.eval(); f=evalf(m,Xn,yte); nz=nonzero(m)
        res[f"sp{int(sp*100)}_f1"]=f; res[f"sp{int(sp*100)}_nz"]=nz
        print(f"seed{seed} sp{int(sp*100)}%: F1={f:.3f} (dense {fd:.3f}, d{f-fd:+.3f}) nonzero={nz/1e3:.0f}K",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/imp_prune_hi_s{seed}.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
print("done seeds",a.seeds,flush=True)
