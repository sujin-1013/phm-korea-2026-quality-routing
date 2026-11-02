"""Per-SNR sweep with 50-epoch tiers, SAME routing operating point as the mixed-SNR
bar/pareto (tau picked on mixed-val). Outputs per-SNR CSV: snr,small,mid,large,hard,ours.
Honest: ours uses the val-selected tau (no test tuning); reports as-is.
"""
import os, sys, argparse, itertools
import numpy as np, torch, csv
_ROOT="/home/sujin/phm-korea-2026-snrgate"
sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; os.makedirs(CKPT,exist_ok=True)
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed

TIERS=["small","mid","large"]; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; EST_OVH=7.6
SWEEP=[float("inf"),20.0,10.0,5.0,0.0,-5.0,-10.0]

def mixed_noise(X, seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(X,dtype=torch.float32).clone(); N=X.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn

def fixed_noise(X, snr, seed):
    if not np.isfinite(snr): return torch.tensor(X,dtype=torch.float32).clone()
    gh=torch.Generator().manual_seed(seed); Xn=torch.tensor(X,dtype=torch.float32).clone()
    for j in range(0,Xn.shape[0],512): Xn[j:j+512]=add_awgn(Xn[j:j+512],snr_db=float(snr),generator=gh)
    return Xn

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--retrain",action="store_true")
    a=ap.parse_args(); set_seed(a.seed); dev="cuda"
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,a.seed)
    Xtr,ytr=cap(Xtr,ytr,20000,a.seed+7); Xv,yv=cap(Xv,yv,4000,a.seed+8); Xte,yte=cap(Xte,yte,12000,a.seed+9)
    cw=inverse_freq_class_weights(ytr,num_class=3); cnn_run.AUG=(float("inf"),)
    print(f"[sweepfig seed{a.seed}] clean-only train={Xtr.shape[0]} test={Xte.shape[0]}",flush=True)
    models={}; pc={}
    for sz in TIERS:
        ck=f"{CKPT}/cnn_{sz}_s{a.seed}.pt"
        if os.path.exists(ck) and not a.retrain:
            bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); st=torch.load(ck,map_location=dev,weights_only=False)
            m.load_state_dict(st["model"]); models[sz]=m.eval(); pc[sz]=st["npar"]
            print(f"  loaded {sz} ({pc[sz]/1e3:.0f}K) best_val={st['best_val']:.3f} from ckpt",flush=True)
        else:
            m,bv,npar=cnn_run.train(sz,Xtr,ytr,Xv,yv,a.epochs,64,dev,cw); models[sz]=m.eval(); pc[sz]=npar
            torch.save({"model":m.state_dict(),"npar":npar,"best_val":bv,"base_ch":SIZES[sz][0],"n_blocks":SIZES[sz][1]},ck)
            print(f"  trained+saved {sz} ({npar/1e3:.0f}K) best_val={bv:.3f}",flush=True)
    est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev)
    ck=torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False); est.load_state_dict(ck["model"]); est.eval()

    def preds_scores(Xn):
        N=Xn.shape[0]
        with torch.no_grad():
            lg={}
            for sz in TIERS:
                out=[]
                for i in range(0,N,256): out.append(models[sz](Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
                lg[sz]=np.concatenate(out)
            q=[]
            for i in range(0,N,256): q.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
            q=np.concatenate(q)
        return lg,q
    pred_stack=lambda lg,idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])
    route_idx=lambda q,t1,t2:(q>=t1).astype(int)+(q>=t2).astype(int)

    # ---- calibrate tau on MIXED val (same operating point as bar/pareto) ----
    Xvm=mixed_noise(Xv,111); lgv,qv=preds_scores(Xvm)
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    t1o,t2o=max(pairs,key=lambda p:macro_f1(yv,pred_stack(lgv,route_idx(qv,*p))))
    t1h,t2h=np.quantile(qv,0.33),np.quantile(qv,0.66)
    print(f"  ours tau=({t1o:.2f},{t2o:.2f}) hard tau=({t1h:.2f},{t2h:.2f})",flush=True)

    # ---- per-SNR sweep on test ----
    rows=[]
    for snr in SWEEP:
        Xn=fixed_noise(Xte,snr,222); lg,q=preds_scores(Xn)
        r={"snr":("inf" if not np.isfinite(snr) else int(snr))}
        for sz in TIERS: r[sz]=macro_f1(yte,lg[sz])
        r["hard"]=macro_f1(yte,pred_stack(lg,route_idx(q,t1h,t2h)))
        r["ours"]=macro_f1(yte,pred_stack(lg,route_idx(q,t1o,t2o)))
        rows.append(r); print(f"  snr={str(r['snr']):>4s} small={r['small']:.3f} mid={r['mid']:.3f} large={r['large']:.3f} hard={r['hard']:.3f} ours={r['ours']:.3f}",flush=True)
    out=f"/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/sweepfig_s{a.seed}.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["snr","small","mid","large","hard","ours"]); w.writeheader()
        for r in rows: w.writerow(r)
    avg=lambda k: np.mean([r[k] for r in rows]); print(f"  AVG small={avg('small'):.3f} mid={avg('mid'):.3f} large={avg('large'):.3f} hard={avg('hard'):.3f} ours={avg('ours'):.3f}  saved {out}",flush=True)

if __name__=="__main__": main()
