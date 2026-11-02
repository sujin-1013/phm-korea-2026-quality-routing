"""US-3b: noise-aug training of large with NOISY-val checkpoint selection (textbook-correct
protocol: val matches deployment noise). Fixes the clean-val mismatch handicap of US-3.
Pre-declared single variant (not fishing). 3 seeds, eval same mixed-SNR test. Report vs
clean-large 0.624 AND vs US-3 clean-val noise-aug 0.633.
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
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
AUG=[float("inf"),20.0,10.0,5.0,0.0,-5.0,-10.0]
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
    cw=inverse_freq_class_weights(ytr,num_class=3)
    Xv_noisy=mixed_noise(Xv,333)  # noisy val for selection (seed != test 222)
    Xte_noisy=mixed_noise(Xte,222)
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
    lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0)
    best=-1; best_state=None
    for ep in range(50):
        m.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; snr=float(AUG[torch.randint(len(AUG),(1,)).item()])
            xb=add_awgn(Xt[idx].clone(),snr_db=snr,generator=g).to(dev)
            opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
        m.eval(); vf=evalf(m,Xv_noisy,yv)  # select on NOISY val
        if vf>best: best=vf; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    f=evalf(m,Xte_noisy,yte)
    torch.save({"model":m.state_dict(),"protocol":"noise_aug_noisyval","best_noisyval":best},f"{CKPT}/cnn_large_noiseaug_nv_s{seed}.pt")
    rows.append({"seed":seed,"aug_nv":f}); print(f"seed{seed}: noise-aug(noisy-val) mixed-F1={f:.3f}",flush=True)
v=np.array([r["aug_nv"] for r in rows])
print(f"\n=== 3-seed mean ===\n  noise-aug large (noisy-val sel) {v.mean():.3f}+-{v.std():.3f} @524K")
print(f"  vs clean-large 0.624: {v.mean()-0.624:+.3f}   vs noise-aug clean-val 0.633: {v.mean()-0.633:+.3f}")
with open(f"{_ROOT}/experiments/exp_scenario/imp_noiseaug_nv.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["seed","aug_nv"]); w.writeheader()
    for r in rows: w.writerow(r)
print("saved imp_noiseaug_nv.csv")
