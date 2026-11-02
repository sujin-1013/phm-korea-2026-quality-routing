"""Per-SNR sweep of the qweight-pruned-tier routing method (ours). Build tiers
{dense, qw20, qw40}, calibrate routing thresholds on MIXED val (accmax), then for each
fixed SNR inject noise to test, route by quality score, eval F1. 3 seeds. Output per-SNR
ours CSV (singles small/mid/large come from existing sweepfig). NOTE: at fixed SNR routing
~collapses to one tier (degenerate); the quantitative win is in mixed-SNR (bar chart).
"""
import os, sys, csv, argparse, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
SWEEP=[float("inf"),20.,10.,5.,0.,-5.,-10.]; FT_EP=12
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def fixed_noise(X,snr,seed):
    if not np.isfinite(snr): return torch.tensor(np.asarray(X),dtype=torch.float32).clone()
    gh=torch.Generator().manual_seed(seed); Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone()
    for j in range(0,Xn.shape[0],512): Xn[j:j+512]=add_awgn(Xn[j:j+512],snr_db=float(snr),generator=gh)
    return Xn
def predf(m,Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)
def evalf(m,Xn,y): return macro_f1(y,predf(m,Xn))
def convs(m): return [mod for mod in m.modules() if isinstance(mod,nn.Conv1d)]
def fresh(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
def channel_importance_q(m,Xcal,q):
    cs=convs(m); acc=[torch.zeros(c.out_channels,device=dev) for c in cs]; wsum=0.0; hooks=[]
    def mk(i):
        def h(mod,inp,out): a=out.detach().abs().mean(dim=2); acc[i]+=(a*qb.view(-1,1)).sum(0)
        return h
    for i,c in enumerate(cs): hooks.append(c.register_forward_hook(mk(i)))
    m.eval()
    with torch.no_grad():
        for j in range(0,Xcal.shape[0],256):
            xb=Xcal[j:j+256].to(dev); qb=torch.tensor(q[j:j+256],dtype=torch.float32,device=dev); wsum+=qb.sum().item(); m(xb)
    for hk in hooks: hk.remove()
    return [(acc[i]/max(wsum,1e-6)).cpu().numpy() for i in range(len(cs))]
def prune_imp(m,imp,fr):
    for c,iv in zip(convs(m),imp):
        k=int(round(fr*len(iv)));
        if k<=0: continue
        kill=np.argsort(iv)[:k]; mask=torch.ones_like(c.weight); mask[kill]=0; prune.custom_from_mask(c,name="weight",mask=mask)
def finetune(m,Xtr,ytr,yv,valX,cw):
    opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-3); lossf=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=dev))
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); g=torch.Generator().manual_seed(0); best=-1; bs=None
    for ep in range(FT_EP):
        m.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev)
            opt.zero_grad(); loss=lossf(m(xb),yt[idx].to(dev)); loss.backward(); opt.step()
        m.eval(); vf=evalf(m,valX,yv)
        if vf>best: best=vf; bs={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); return m
def build_qw(seed,fr,Xcal,q,Xtr,ytr,yv,valX,cw):
    m=fresh(seed); prune_imp(m,channel_importance_q(m,Xcal,q),fr); return finetune(m,Xtr,ytr,yv,valX,cw)
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev); est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
def qscore(Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(o)
route=lambda qs,t1,t2:(qs>=t1).astype(int)+(qs>=t2).astype(int)
stack=lambda P,idx: np.array([P[idx[i]][i] for i in range(len(idx))])
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xcal=mixed_noise(Xv,333); q=qscore(Xcal); valX=mixed_noise(Xv,334)
    dense=fresh(seed); qw20=build_qw(seed,0.20,Xcal,q,Xtr,ytr,yv,valX,cw); qw40=build_qw(seed,0.40,Xcal,q,Xtr,ytr,yv,valX,cw)
    tiers=[qw40,qw20,dense]
    # calibrate accmax thresholds on mixed val
    qv=qscore(valX); Pv=[predf(t,valX) for t in tiers]
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    t1o,t2o=max(pairs,key=lambda p:macro_f1(yv,stack(Pv,route(qv,*p))))
    # per-SNR sweep: route at each fixed SNR
    rows=[]
    for snr in SWEEP:
        Xn=fixed_noise(Xte,snr,222); qt=qscore(Xn); Pt=[predf(t,Xn) for t in tiers]
        f=macro_f1(yte,stack(Pt,route(qt,t1o,t2o)))
        rows.append((("inf" if not np.isfinite(snr) else int(snr)),f))
        print(f"seed{seed} snr={str(rows[-1][0]):>4s} ours={f:.3f}",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/qwsweep_s{seed}.csv","w",newline="") as fp:
        w=csv.writer(fp); w.writerow(["snr","ours"]); [w.writerow(r) for r in rows]
    print(f"seed{seed} tau=({t1o},{t2o}) saved",flush=True)
print("done",a.seeds,flush=True)
