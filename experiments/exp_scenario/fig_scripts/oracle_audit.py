"""Consensus experiment #1: oracle 'any-tier-correct' audit on EXISTING saved tiers.
Decides go/no-go for ANY accuracy gain from routing over current small/mid/large.
If oracle-any-correct ~= large -> complementarity impossible -> routing is cost-only.
Reuses saved models, NO training. 3 seeds, mixed-SNR, per-SNR bins. Honest upper bound.
"""
import os, sys, csv
import numpy as np, torch
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; TIERS=["small","mid","large"]; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
SNR_ORDER=["inf","20","10","5","0","-5","-10"]
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N); tsnr=np.empty(N,dtype=object)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]; tsnr[m]=("inf" if not np.isfinite(s) else str(int(s)))
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn,tsnr
def load(sz,seed):
    bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m.eval()
def preds(m,Xn):
    out=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): out.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(out)
agg={"large_acc":[],"oracle_acc":[],"large_f1":[],"oracle_f1":[],"headroom":[]}
per_snr_large={s:[] for s in SNR_ORDER}; per_snr_oracle={s:[] for s in SNR_ORDER}
for seed in [42,1,7]:
    set_seed(seed)
    _,_,_,_,Xte,yte=materialise("indist",None,seed); Xte,yte=cap(Xte,yte,12000,seed+9)
    Xn,tsnr=mixed_noise(Xte,222); y=np.asarray(yte)
    P={sz:preds(load(sz,seed),Xn) for sz in TIERS}
    correct={sz:(P[sz]==y) for sz in TIERS}
    any_correct=correct["small"]|correct["mid"]|correct["large"]
    # oracle prediction: pick a tier that's correct if any exists, else large (upper bound)
    oracle_pred=P["large"].copy()
    for sz in ["small","mid"]:
        fix=(~correct["large"])&correct[sz]; oracle_pred[fix]=P[sz][fix]
    agg["large_acc"].append(correct["large"].mean()); agg["oracle_acc"].append(any_correct.mean())
    agg["large_f1"].append(macro_f1(y,P["large"])); agg["oracle_f1"].append(macro_f1(y,oracle_pred))
    # headroom = windows large WRONG but small or mid RIGHT
    agg["headroom"].append(((~correct["large"])&(correct["small"]|correct["mid"])).mean())
    for s in SNR_ORDER:
        m=tsnr==s
        if m.sum()>0:
            per_snr_large[s].append(correct["large"][m].mean()); per_snr_oracle[s].append(any_correct[m].mean())
    print(f"seed{seed}: large_acc={correct['large'].mean():.3f} oracle_acc={any_correct.mean():.3f} large_f1={macro_f1(y,P['large']):.3f} oracle_f1={macro_f1(y,oracle_pred):.3f} headroom={agg['headroom'][-1]:.3f}",flush=True)
def ms(k): a=np.array(agg[k]); return a.mean(),a.std()
print("\n=== 3-seed mean (mixed-SNR) ===")
print(f"  large accuracy   {ms('large_acc')[0]:.3f}+-{ms('large_acc')[1]:.3f}")
print(f"  ORACLE accuracy  {ms('oracle_acc')[0]:.3f}+-{ms('oracle_acc')[1]:.3f}  (any-tier-correct upper bound)")
print(f"  large macro-F1   {ms('large_f1')[0]:.3f}+-{ms('large_f1')[1]:.3f}")
print(f"  ORACLE macro-F1  {ms('oracle_f1')[0]:.3f}+-{ms('oracle_f1')[1]:.3f}")
print(f"  HEADROOM (large wrong, small/mid right) = {ms('headroom')[0]*100:.1f}%")
print(f"  => routing accuracy gain possible only if headroom large; oracle_acc - large_acc = {ms('oracle_acc')[0]-ms('large_acc')[0]:+.3f}")
print("\n=== per-SNR (large_acc -> oracle_acc) ===")
for s in SNR_ORDER:
    la=np.mean(per_snr_large[s]); oa=np.mean(per_snr_oracle[s]); print(f"  {('clean' if s=='inf' else s):>5s}: {la:.3f} -> {oa:.3f}  (+{oa-la:.3f})")
with open(f"{_ROOT}/experiments/exp_scenario/oracle_audit.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["metric","mean","std"])
    for k in agg: w.writerow([k,f"{ms(k)[0]:.4f}",f"{ms(k)[1]:.4f}"])
print("saved oracle_audit.csv")
