"""US-2 CONTROL: dense large + SAME 12-epoch clean fine-tune, NO pruning.
Isolates whether the pruning gain is from pruning itself or just extra training epochs.
3 seeds, eval mixed-SNR. If dense+ft ~= pruned+ft -> gain is extra training, not pruning.
"""
import os, sys, csv
import numpy as np, torch, torch.nn as nn
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"; FT_EP=12
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
rows=[]
for seed in [42,1,7]:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    cw=inverse_freq_class_weights(ytr,num_class=3); Xn=mixed_noise(Xte,222)
    bc,nb=SIZES["large"]
    m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"])
    f_dense=evalf(m.eval(),Xn,yte)
    # SAME fine-tune as pruning run but NO pruning
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
    m.load_state_dict(best_state); f_ft=evalf(m.eval(),Xn,yte)
    rows.append({"seed":seed,"dense":f_dense,"dense_ft":f_ft})
    print(f"seed{seed}: dense={f_dense:.3f}  dense+{FT_EP}ep-ft(NO prune)={f_ft:.3f}  delta={f_ft-f_dense:+.3f}",flush=True)
d=np.array([r["dense"] for r in rows]); df=np.array([r["dense_ft"] for r in rows])
print(f"\n=== 3-seed mean ===\n  dense large         {d.mean():.3f}+-{d.std():.3f}\n  dense + 12ep ft     {df.mean():.3f}+-{df.std():.3f}  delta_vs_dense={df.mean()-d.mean():+.3f}")
print("  -> compare this delta to the pruned+ft delta to attribute the gain (pruning vs extra-training).")
with open(f"{_ROOT}/experiments/exp_scenario/imp_prune_control.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["seed","dense","dense_ft"]); w.writeheader()
    for r in rows: w.writerow(r)
print("saved imp_prune_control.csv")
