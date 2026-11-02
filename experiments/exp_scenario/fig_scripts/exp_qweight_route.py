"""(1) Fine low-sparsity sweep mag vs qweight {10,20,30,40%} for consistency.
(2) Routing over QUALITY-WEIGHTED-pruned tiers {dense, qw20, qw40} by quality score,
    calibrated to MAINTAIN accuracy (min avg MACs s.t. val F1 >= dense-0.01) -> reduce
    params/MACs only where it doesn't hurt. 3 seeds via --seeds. Reuse saved large + FT.
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
CONV_OUTLEN=[128,64,32,16,8]; KS=[64,3,3,3,3]; FT_EP=12
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        msk=assign==li; idx=np.where(msk)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def predf(m,Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)
def evalf(m,Xn,y): return macro_f1(y,predf(m,Xn))
def convs(m): return [mod for mod in m.modules() if isinstance(mod,nn.Conv1d)]
def surv(m): return [int((c.weight.abs().sum(dim=(1,2))>0).sum().item()) for c in convs(m)]
def macs(s):
    t=0; prev=1
    for l,(ol,k) in enumerate(zip(CONV_OUTLEN,KS)): t+=ol*s[l]*prev*k; prev=s[l]
    return t
def fresh(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
def channel_importance(m,Xcal,q,scheme):
    cs=convs(m); imp=[None]*len(cs)
    if scheme=="mag":
        for i,c in enumerate(cs): imp[i]=c.weight.detach().pow(2).sum(dim=(1,2)).sqrt().cpu().numpy()
        return imp
    acc=[torch.zeros(c.out_channels,device=dev) for c in cs]; wsum=0.0; hooks=[]
    def mk(i):
        def h(mod,inp,out):
            a=out.detach().abs().mean(dim=2); w=qb.view(-1,1) if scheme=="qweight" else torch.ones(a.shape[0],1,device=dev); acc[i]+=(a*w).sum(0)
        return h
    for i,c in enumerate(cs): hooks.append(c.register_forward_hook(mk(i)))
    m.eval()
    with torch.no_grad():
        for j in range(0,Xcal.shape[0],256):
            xb=Xcal[j:j+256].to(dev); qb=torch.tensor(q[j:j+256],dtype=torch.float32,device=dev)
            wsum+=(qb.sum().item() if scheme=="qweight" else xb.shape[0]); m(xb)
    for hk in hooks: hk.remove()
    for i in range(len(cs)): imp[i]=(acc[i]/max(wsum,1e-6)).cpu().numpy()
    return imp
def prune_imp(m,imp,fr):
    for c,iv in zip(convs(m),imp):
        k=int(round(fr*len(iv)))
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
def build(seed,scheme,fr,Xcal,q,Xtr,ytr,yv,valX,cw):
    m=fresh(seed)
    if fr>0: prune_imp(m,channel_importance(m,Xcal,q,scheme),fr)
    return finetune(m,Xtr,ytr,yv,valX,cw)
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev); est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
def qscore(Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(o)
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xcal=mixed_noise(Xv,333); q=qscore(Xcal); valX=mixed_noise(Xv,334); Xte_n=mixed_noise(Xte,222)
    qv=qscore(valX); qt=qscore(Xte_n)  # routing scores
    res={"seed":seed}
    # (1) fine sweep mag vs qweight at 10/20/30/40
    qmodels={}
    for fr in [0.10,0.20,0.30,0.40]:
        for scheme in ["mag","qweight"]:
            m=build(seed,scheme,fr,Xcal,q,Xtr,ytr,yv,valX,cw); f=evalf(m,Xte_n,yte); mm=macs(surv(m))
            res[f"{scheme}{int(fr*100)}_f1"]=f; res[f"{scheme}{int(fr*100)}_macs"]=mm
            if scheme=="qweight" and fr in (0.20,0.40): qmodels[fr]=m
            print(f"seed{seed} {scheme}{int(fr*100)}%: F1={f:.3f} MACs={mm/1e6:.2f}M",flush=True)
    # (2) routing over qweight tiers {dense, qw20, qw40}, maintain accuracy
    dense=fresh(seed)  # dense is already trained; no prune
    tiers=[qmodels[0.40],qmodels[0.20],dense]; tmacs=[macs(surv(qmodels[0.40])),macs(surv(qmodels[0.20])),macs(surv(dense))]
    Pv=[predf(t,valX) for t in tiers]; Pt=[predf(t,Xte_n) for t in tiers]
    route=lambda qs,t1,t2:(qs>=t1).astype(int)+(qs>=t2).astype(int)
    stack=lambda P,idx: np.array([P[idx[i]][i] for i in range(len(idx))])
    cost=lambda idx: sum([(idx==k).mean()*tmacs[k] for k in range(3)])
    # reference = BEST single tier on val (not dense) -> maintain the MAX achievable accuracy
    bestvf=max(macro_f1(yv,Pv[k]) for k in range(3)); bestk=int(np.argmax([macro_f1(yv,Pv[k]) for k in range(3)]))
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    iso=None; accmax=None
    for t1,t2 in pairs:
        fv=macro_f1(yv,stack(Pv,route(qv,t1,t2))); ct=cost(route(qt,t1,t2))
        if accmax is None or fv>accmax[0]: accmax=(fv,t1,t2)
        if fv>=bestvf-0.005 and (iso is None or ct<iso[3]): iso=(fv,t1,t2,ct)
    if iso is None: iso=(accmax[0],accmax[1],accmax[2],cost(route(qt,accmax[1],accmax[2])))
    idt=route(qt,iso[1],iso[2]); fr_=macro_f1(yte,stack(Pt,idt)); cr=cost(idt)
    ida=route(qt,accmax[1],accmax[2]); fa=macro_f1(yte,stack(Pt,ida)); ca=cost(ida)
    accmax_valf1=accmax[0]; accmax_testf1=fa  # val-vs-test consistency check at accmax point
    # val-SELECTED single tier (apples-to-apples: pick tier by val, eval test) vs test-oracle single
    valsel_k=int(np.argmax([macro_f1(yv,Pv[k]) for k in range(3)])); valsel_single_f1=macro_f1(yte,Pt[valsel_k]); valsel_single_macs=tmacs[valsel_k]
    bestsingle_f1=max(macro_f1(yte,Pt[k]) for k in range(3)); bestsingle_macs=tmacs[int(np.argmax([macro_f1(yte,Pt[k]) for k in range(3)]))]
    res.update({"route_iso_f1":fr_,"route_iso_macs":cr,"route_accmax_valf1":accmax_valf1,"route_accmax_testf1":fa,"route_accmax_macs":ca,
                "valsel_single_f1":valsel_single_f1,"valsel_single_macs":valsel_single_macs,"best_single_f1":bestsingle_f1,"best_single_macs":bestsingle_macs,"dense_f1":macro_f1(yte,Pt[2])})
    print(f"seed{seed} iso:F1={fr_:.3f}@{cr/1e6:.2f}M | accmax val={accmax_valf1:.3f} test={fa:.3f}@{ca/1e6:.2f}M (gap {accmax_valf1-fa:+.3f}) | valsel-single test={valsel_single_f1:.3f}@{valsel_single_macs/1e6:.2f}M | test-best-single={bestsingle_f1:.3f}",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/exp_qweight_route_s{seed}.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(res.keys())); w.writeheader(); w.writerow(res)
print("done",a.seeds,flush=True)
