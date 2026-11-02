"""US-1: Tier ensemble (softmax mean of small+mid+large) on mixed-SNR. Reuse saved models.
Honest: equal-weight soft-vote (pre-declared). 3 seeds. Reports vs canonical large 0.624.
"""
import os, sys, csv
import numpy as np, torch, torch.nn.functional as F
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; TIERS=["small","mid","large"]; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def load(sz,seed):
    bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); st=torch.load(f"{CKPT}/cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)
    m.load_state_dict(st["model"]); return m.eval(),st["npar"]
rows=[]
for seed in [42,1,7]:
    set_seed(seed)
    _,_,_,_,Xte,yte=materialise("indist",None,seed); Xte,yte=cap(Xte,yte,12000,seed+9)
    Xn=mixed_noise(Xte,222); N=Xn.shape[0]
    probs={}; npar={}
    for sz in TIERS:
        m,p=load(sz,seed); npar[sz]=p; out=[]
        with torch.no_grad():
            for i in range(0,N,256): out.append(F.softmax(m(Xn[i:i+256].to(dev)),dim=-1).cpu().numpy())
        probs[sz]=np.concatenate(out)
    f_small=macro_f1(yte,probs["small"].argmax(1)); f_mid=macro_f1(yte,probs["mid"].argmax(1)); f_large=macro_f1(yte,probs["large"].argmax(1))
    ens3=macro_f1(yte,(probs["small"]+probs["mid"]+probs["large"]).argmax(1))
    ens_ml=macro_f1(yte,(probs["mid"]+probs["large"]).argmax(1))
    rows.append((seed,f_small,f_mid,f_large,ens3,ens_ml,npar["small"]+npar["mid"]+npar["large"]))
    print(f"seed{seed}: small={f_small:.3f} mid={f_mid:.3f} large={f_large:.3f} | ens3={ens3:.3f} ens(mid+large)={ens_ml:.3f}",flush=True)
lgv=np.array([r[3] for r in rows]); e3v=np.array([r[4] for r in rows]); emlv=np.array([r[5] for r in rows])
cost=rows[0][6]/1e3
print(f"\n=== 3-seed mean ===")
print(f"  large(ref)     {lgv.mean():.3f}+-{lgv.std():.3f} @524K")
print(f"  ensemble(s+m+l){e3v.mean():.3f}+-{e3v.std():.3f} @{cost:.0f}K  delta_vs_large={e3v.mean()-lgv.mean():+.3f}")
print(f"  ensemble(m+l)  {emlv.mean():.3f}+-{emlv.std():.3f} @577K  delta_vs_large={emlv.mean()-lgv.mean():+.3f}")
with open(f"{_ROOT}/experiments/exp_scenario/imp_ensemble.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["seed","small","mid","large","ens_sml","ens_ml","cost_params"])
    for r in rows: w.writerow(r)
print("saved imp_ensemble.csv")
