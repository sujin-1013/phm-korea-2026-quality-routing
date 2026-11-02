"""Mixed-SNR (per-window random SNR) aggregate comparison for the routing figure.
Tiers small/mid/large, clean-only trained. Methods: single tiers, oracle(SNR-band),
HardGate(uninformed .33/.66 val-quantile τ), estimator routing (val-acc-calibrated).
Writes per-seed CSV: method,f1,cost_K. Honest: reports tie/loss as-is, no cherry-pick.
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

def mixed_noise(X, seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(X,dtype=torch.float32).clone(); N=X.shape[0]; assign=rng.integers(0,len(LEVELS),N); tsnr=np.empty(N)
    for li,s in enumerate(LEVELS):
        m=assign==li; tsnr[m]=(99 if not np.isfinite(s) else s); idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn, tsnr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--retrain",action="store_true")
    a=ap.parse_args(); set_seed(a.seed); dev="cuda"
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,a.seed)
    Xtr,ytr=cap(Xtr,ytr,20000,a.seed+7); Xv,yv=cap(Xv,yv,4000,a.seed+8); Xte,yte=cap(Xte,yte,12000,a.seed+9)
    cw=inverse_freq_class_weights(ytr,num_class=3); cnn_run.AUG=(float("inf"),)
    print(f"[mixedfig seed{a.seed}] clean-only train={Xtr.shape[0]} test={Xte.shape[0]}",flush=True)
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

    def eval_set(X,y,seed):
        Xn,tsnr=mixed_noise(X,seed); N=Xn.shape[0]
        with torch.no_grad():
            lg={}
            for sz in TIERS:
                out=[]
                for i in range(0,N,256): out.append(models[sz](Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
                lg[sz]=np.concatenate(out)
            q=[]
            for i in range(0,N,256): q.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
            q=np.concatenate(q)
        return y,tsnr,lg,q,N
    yv2,tsv,lgv,qv,Nv=eval_set(Xv,yv,111); yt2,tst,lgt,qt,Nt=eval_set(Xte,yte,222)
    pred_stack=lambda lg,idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])
    route_idx=lambda q,t1,t2:(q>=t1).astype(int)+(q>=t2).astype(int)
    def cost(idx): sh=[(idx==k).mean() for k in range(3)]; return sum(sh[k]*pc[TIERS[k]] for k in range(3))/1e3+EST_OVH

    rows=[]
    for sz in TIERS: rows.append((f"single_{sz}", macro_f1(yt2,lgt[sz]), pc[sz]/1e3))
    # oracle SNR-band: clean/20/10->small, 5/0->mid, -5/-10->large
    o=np.array([0 if t>=10 else (1 if t>=0 else 2) for t in tst]); rows.append(("oracle", macro_f1(yt2,pred_stack(lgt,o)), cost(o)))
    # HardGate: uninformed fixed tau at val score 33/66 quantiles (no accuracy tuning)
    t1h,t2h=np.quantile(qv,0.33),np.quantile(qv,0.66); idxh=route_idx(qt,t1h,t2h)
    rows.append(("hardgate", macro_f1(yt2,pred_stack(lgt,idxh)), cost(idxh)))
    # ===== routing frontier over tau grid (val-selected, NEVER test-selected) =====
    val_large=macro_f1(yv2,lgv["large"])
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    frontier=[]  # (val_f1, test_f1, cost, t1, t2)
    for t1,t2 in pairs:
        fv=macro_f1(yv2,pred_stack(lgv,route_idx(qv,t1,t2)))
        ft=macro_f1(yt2,pred_stack(lgt,route_idx(qt,t1,t2))); c=cost(route_idx(qt,t1,t2))
        frontier.append((fv,ft,c,t1,t2))
    # ours_accmax: argmax VAL accuracy
    fv,ft,c,t1o,t2o=max(frontier,key=lambda r:r[0]); rows.append(("ours_accmax",ft,c)); print(f"  [accmax] val={fv:.3f} test={ft:.3f} @{c:.0f}K tau=({t1o:.2f},{t2o:.2f})",flush=True)
    # ours_costmatch: min COST s.t. VAL f1 >= val_large - 0.01 (claim-B rule, val-selected)
    ok=[r for r in frontier if r[0]>=val_large-0.01]
    if ok:
        fv,ft,c,t1c,t2c=min(ok,key=lambda r:r[2]); rows.append(("ours_costmatch",ft,c)); print(f"  [costmatch] val={fv:.3f}(>=val_large {val_large:.3f}-.01) test={ft:.3f} @{c:.0f}K tau=({t1c:.2f},{t2c:.2f})",flush=True)
    else:
        rows.append(("ours_costmatch",ft,c)); print(f"  [costmatch] NONE reach val_large-.01; fallback accmax",flush=True)
    # ---- routing selection shares (mixed test) for ours / hardgate / oracle ----
    def share(idx): return [(idx==k).mean() for k in range(3)]
    idxo=route_idx(qt,t1o,t2o)  # ours operating point (costmatch fell back to accmax)
    sh_ours=share(idxo); sh_hard=share(idxh); sh_or=share(o)
    print(f"  SHARE small/mid/large  ours={sh_ours[0]*100:.1f}/{sh_ours[1]*100:.1f}/{sh_ours[2]*100:.1f}  hard={sh_hard[0]*100:.1f}/{sh_hard[1]*100:.1f}/{sh_hard[2]*100:.1f}  oracle={sh_or[0]*100:.1f}/{sh_or[1]*100:.1f}/{sh_or[2]*100:.1f}",flush=True)
    out=f"/home/sujin/phm-korea-2026-snrgate/experiments/exp_scenario/mixedfig_s{a.seed}.csv"
    with open(out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","f1","cost_K","sh_small","sh_mid","sh_large"])
        shmap={"ours_accmax":sh_ours,"ours_costmatch":sh_ours,"hardgate":sh_hard,"oracle":sh_or}
        for nm,f1,c in rows:
            sh=shmap.get(nm,[None,None,None])
            w.writerow([nm,f"{f1:.4f}",f"{c:.1f}"]+[("" if s is None else f"{s:.4f}") for s in sh])
            print(f"  {nm:14s} F1={f1:.3f} cost={c:.0f}K",flush=True)
    print(f"  ours tau=({t1o:.2f},{t2o:.2f}) hard tau=({t1h:.2f},{t2h:.2f})  saved {out}",flush=True)

if __name__=="__main__": main()
