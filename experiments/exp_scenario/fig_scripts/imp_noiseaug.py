"""US-3: noise-augmented training of large (AWGN aug incl clean), eval mixed-SNR.
Standard noise-robustness booster. Honest: DIFFERENT protocol (not clean-only) -> caveat.
3 seeds, report vs clean-trained large 0.624.
"""
import os, sys, csv
import numpy as np, torch
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"; CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"
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
    # noise-aug training: AUG includes clean + SNR sweep (val selection on clean val keeps it comparable)
    cnn_run.AUG=(float("inf"),20.0,10.0,5.0,0.0,-5.0,-10.0)
    m,bv,npar=cnn_run.train("large",Xtr,ytr,Xv,yv,50,64,dev,cw); m.eval()
    Xn=mixed_noise(Xte,222); f_aug=evalf(m,Xn,yte)
    torch.save({"model":m.state_dict(),"npar":npar,"best_val":bv,"protocol":"noise_aug"},f"{CKPT}/cnn_large_noiseaug_s{seed}.pt")
    rows.append({"seed":seed,"aug_large":f_aug})
    print(f"seed{seed}: noise-aug large mixed-F1={f_aug:.3f} (clean-trained large ref=0.624 canonical)",flush=True)
v=np.array([r["aug_large"] for r in rows])
print(f"\n=== 3-seed mean ===\n  noise-aug large {v.mean():.3f}+-{v.std():.3f} @524K  delta_vs_clean-large(0.624)={v.mean()-0.624:+.3f}")
with open(f"{_ROOT}/experiments/exp_scenario/imp_noiseaug.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["seed","aug_large"]); w.writeheader()
    for r in rows: w.writerow(r)
print("saved imp_noiseaug.csv")
