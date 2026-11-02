"""US-2: prune large (global L1 magnitude 30/50/70%) + clean fine-tune, eval mixed-SNR.
Tests user's hypothesis: does pruning remove redundancy and IMPROVE over dense large?
Honest: 3 seeds, report win/no-win vs dense large 0.624. train->prune->finetune (standard).
"""
import os, sys, csv
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
SPARS=[0.30,0.50,0.70]; FT_EP=12
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
def prune_params(m): return [(mod,"weight") for mod in m.modules() if isinstance(mod,(nn.Conv1d,nn.Linear))]
def nonzero(m):
    tot=z=0
    for mod,_ in prune_params(m):
        w=mod.weight; tot+=w.numel(); z+=int((w==0).sum())
    return tot-z,tot
rows=[]
for seed in [42,1,7]:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    cw=inverse_freq_class_weights(ytr,num_class=3); cnn_run.AUG=(float("inf"),)
    Xn=mixed_noise(Xte,222)
    bc,nb=SIZES["large"]
    # dense ref (from saved)
    dense=CNN1D(bc,nb).to(dev); dense.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); dense.eval()
    f_dense=evalf(dense,Xn,yte)
    res={"seed":seed,"dense":f_dense}
    for sp in SPARS:
        m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"])
        prune.global_unstructured(prune_params(m),pruning_method=prune.L1Unstructured,amount=sp)
        # fine-tune clean (mask stays zero)
        opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3)
        lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
        Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0)
        Xvt=torch.tensor(Xv,dtype=torch.float32); best=-1; best_state=None
        for ep in range(FT_EP):
            m.train(); perm=torch.randperm(Xt.shape[0])
            for i in range(0,Xt.shape[0],64):
                idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev)
                opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
            m.eval(); vf=evalf(m,Xvt,yv)
            if vf>best: best=vf; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best_state); m.eval()
        f_pr=evalf(m,Xn,yte); nz,tot=nonzero(m)
        res[f"sp{int(sp*100)}"]=f_pr; res[f"nz{int(sp*100)}"]=nz
        print(f"seed{seed} sparsity{int(sp*100)}%: F1={f_pr:.3f} (dense {f_dense:.3f}, delta {f_pr-f_dense:+.3f}) nonzero={nz/1e3:.0f}K/{tot/1e3:.0f}K",flush=True)
    rows.append(res)
print("\n=== 3-seed mean (vs dense large 0.624 canonical) ===")
ds=np.array([r["dense"] for r in rows]); print(f"  dense large (rerun) {ds.mean():.3f}+-{ds.std():.3f}")
for sp in SPARS:
    v=np.array([r[f"sp{int(sp*100)}"] for r in rows]); nz=np.mean([r[f"nz{int(sp*100)}"] for r in rows])/1e3
    print(f"  pruned {int(sp*100)}% +ft  {v.mean():.3f}+-{v.std():.3f}  nonzero~{nz:.0f}K  delta_vs_dense={v.mean()-ds.mean():+.3f}")
with open(f"{_ROOT}/experiments/exp_scenario/imp_prune.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
print("saved imp_prune.csv")
