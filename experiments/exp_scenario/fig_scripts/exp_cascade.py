"""Experiment 1: confidence/disagreement CASCADE over saved tiers (GPU0, no training).
Run small+mid first; escalate to large when they DISAGREE or min margin < tau.
Sweep tau on mixed-VAL, eval mixed-TEST. 3 seeds. Report cost-F1 Pareto vs
always-large (0.624@524K) and SNR-routing (0.611@398K). Tests if difficulty-based
routing captures the 8% oracle headroom that the noise estimator cannot.
"""
import os, sys, csv
import numpy as np, torch, torch.nn.functional as F
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
PC={"small":9.2,"mid":53.1,"large":523.8}  # K params
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def load(sz,seed):
    bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m.eval()
def probs(m,Xn):
    out=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): out.append(F.softmax(m(Xn[i:i+256].to(dev)),dim=-1).cpu().numpy())
    return np.concatenate(out)
def margin(p): s=np.sort(p,axis=1); return s[:,-1]-s[:,-2]
rows=[]
for seed in [42,1,7]:
    set_seed(seed)
    _,_,Xv,yv,Xte,yte=materialise("indist",None,seed); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte)
    Xvn=mixed_noise(Xv,333); Xtn=mixed_noise(Xte,222)
    M={sz:load(sz,seed) for sz in ["small","mid","large"]}
    pv={sz:probs(M[sz],Xvn) for sz in ["small","mid","large"]}
    pt={sz:probs(M[sz],Xtn) for sz in ["small","mid","large"]}
    def run(p, tau):
        ps,pm,pl=p["small"],p["mid"],p["large"]
        cs,cm=ps.argmax(1),pm.argmax(1)
        agree=cs==cm; conf=np.minimum(margin(ps),margin(pm))>=tau
        accept=agree&conf  # cheap path
        pred=pl.argmax(1).copy(); pred[accept]=cs[accept]
        esc=(~accept).mean()  # fraction to large
        cost=(PC["small"]+PC["mid"])+esc*PC["large"]  # small+mid always run
        return pred,cost,esc
    # sweep tau on val, pick Pareto points + the iso-large-F1 min-cost point
    large_f1_v=macro_f1(yv,pv["large"].argmax(1))
    taus=[round(x,2) for x in np.arange(0.0,0.96,0.05)]
    cand=[]
    for t in taus:
        predv,cv,_=run(pv,t); fv=macro_f1(yv,predv); cand.append((t,fv,cv))
    # operating point A: min-cost s.t. val F1 >= large_f1_v - 0.005 (iso-accuracy)
    iso=[(t,fv,cv) for t,fv,cv in cand if fv>=large_f1_v-0.005]
    tA=min(iso,key=lambda r:r[2])[0] if iso else max(cand,key=lambda r:r[1])[0]
    # operating point B: max val F1 (accuracy-first)
    tB=max(cand,key=lambda r:r[1])[0]
    predA,cA,eA=run(pt,tA); predB,cB,eB=run(pt,tB)
    fA=macro_f1(yte,predA); fB=macro_f1(yte,predB); fL=macro_f1(yte,pt["large"].argmax(1))
    rows.append({"seed":seed,"large":fL,"iso_f1":fA,"iso_cost":cA,"iso_esc":eA,"acc_f1":fB,"acc_cost":cB,"acc_esc":eB,"tA":tA,"tB":tB})
    print(f"seed{seed}: large={fL:.3f} | iso[tau={tA:.2f}]: F1={fA:.3f}@{cA:.0f}K esc={eA*100:.0f}% | accmax[tau={tB:.2f}]: F1={fB:.3f}@{cB:.0f}K esc={eB*100:.0f}%",flush=True)
def ms(k): a=np.array([r[k] for r in rows]); return a.mean(),a.std()
print("\n=== 3-seed mean (cascade, mixed-SNR) ===")
print(f"  always-large       {ms('large')[0]:.3f}+-{ms('large')[1]:.3f} @524K")
print(f"  cascade iso-acc    {ms('iso_f1')[0]:.3f}+-{ms('iso_f1')[1]:.3f} @{ms('iso_cost')[0]:.0f}K  (vs SNR-route 0.611@398K)")
print(f"  cascade acc-max    {ms('acc_f1')[0]:.3f}+-{ms('acc_f1')[1]:.3f} @{ms('acc_cost')[0]:.0f}K  delta_vs_large={ms('acc_f1')[0]-ms('large')[0]:+.3f}")
with open(f"{_ROOT}/experiments/exp_scenario/exp_cascade.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
print("saved exp_cascade.csv")
