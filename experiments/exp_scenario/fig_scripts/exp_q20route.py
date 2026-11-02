"""2-tier routing {qw20(cheap,6.45M), dense(big,9.96M)} — cheap tier raised to qw20 to
keep clean high. All tiers .eval() (bug-free). Route by quality score, 1 threshold, iso
calibrated on mixed-val (maintain best single -0.005, min MACs). Reports mixed F1 + avg
MACs + per-SNR ours. 3 seeds. Compares to qw20 single and {qw40,qw20} routing.
"""
import os, sys, csv, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.,10.,5.,0.,-5.,-10.,float("inf")]; dev="cuda"
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
    assert not m.training,"eval!"
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)
def evalf(m,Xn,y): return macro_f1(y,predf(m,Xn))
def convs(m): return [mod for mod in m.modules() if isinstance(mod,nn.Conv1d)]
def fresh(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
def chimp_q(m,Xcal,q):
    cs=convs(m); acc=[torch.zeros(c.out_channels,device=dev) for c in cs]; ws=0.0; hk=[]
    def mk(i):
        def h(mod,inp,out): a=out.detach().abs().mean(dim=2); acc[i]+=(a*qb.view(-1,1)).sum(0)
        return h
    for i,c in enumerate(cs): hk.append(c.register_forward_hook(mk(i)))
    m.eval()
    with torch.no_grad():
        for j in range(0,Xcal.shape[0],256):
            xb=Xcal[j:j+256].to(dev); qb=torch.tensor(q[j:j+256],dtype=torch.float32,device=dev); ws+=qb.sum().item(); m(xb)
    for h in hk: h.remove()
    return [(acc[i]/max(ws,1e-6)).cpu().numpy() for i in range(len(cs))]
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
def build_qw(seed,fr,Xcal,q,Xtr,ytr,yv,valX,cw):
    m=fresh(seed); prune_imp(m,chimp_q(m,Xcal,q),fr); return finetune(m,Xtr,ytr,yv,valX,cw)
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
est=NoiseEstimator1D(channels=(8,16,16,32)).to(dev); est.load_state_dict(torch.load("/tmp/small_est.pt",map_location="cpu",weights_only=False)["model"]); est.eval()
def qscore(Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(o)
TM={"qw20":6.45,"dense":9.96}
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xcal=mixed_noise(Xv,333); q=qscore(Xcal); valX=mixed_noise(Xv,334); Xte_n=mixed_noise(Xte,222)
    qw20=build_qw(seed,0.20,Xcal,q,Xtr,ytr,yv,valX,cw); dense=fresh(seed); dense.eval()
    tiers=[qw20,dense]; tmacs=[6.45,9.96]
    # sanity
    fq=evalf(qw20,Xte_n,yte); fd=evalf(dense,Xte_n,yte)
    # 1-threshold routing: q<t -> qw20, else dense. calibrate iso on mixed val
    qv=qscore(valX); Pv=[predf(t,valX) for t in tiers]; qt=qscore(Xte_n); Pt=[predf(t,Xte_n) for t in tiers]
    bestvf=max(macro_f1(yv,Pv[0]),macro_f1(yv,Pv[1]))
    def routed(qs,P,t): idx=(qs>=t).astype(int); return np.array([P[idx[i]][i] for i in range(len(idx))]),idx
    grid=[round(x,2) for x in np.arange(0.05,0.96,0.05)]; iso=None
    for t in grid:
        pv,iv=routed(qv,Pv,t); fv=macro_f1(yv,pv); ct=(iv==0).mean()*tmacs[0]+(iv==1).mean()*tmacs[1]
        if fv>=bestvf-0.005 and (iso is None or ct<iso[2]): iso=(fv,t,ct)
    if iso is None: iso=(0,grid[-1],tmacs[1])
    t_=iso[1]; pmix,imix=routed(qt,Pt,t_); fmix=macro_f1(yte,pmix); cmix=(imix==0).mean()*tmacs[0]+(imix==1).mean()*tmacs[1]
    print(f"seed{seed} SANITY qw20={fq:.3f}@6.45M dense={fd:.3f}@9.96M | ROUTE{{qw20,dense}} mixed F1={fmix:.3f}@{cmix:.2f}M share(qw20/dense)={(imix==0).mean()*100:.0f}/{(imix==1).mean()*100:.0f} t={t_}",flush=True)
    # per-SNR
    rows=[]
    for snr in SWEEP:
        Xn=fixed_noise(Xte,snr,222); qs=qscore(Xn); Ps=[predf(t,Xn) for t in tiers]
        p,_=routed(qs,Ps,t_); f=macro_f1(yte,p); rows.append((("inf" if not np.isfinite(snr) else int(snr)),f))
    print("  per-SNR ours:"+" ".join(f"{r[1]:.3f}" for r in rows),flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/q20route_s{seed}.csv","w",newline="") as fp:
        w=csv.writer(fp); w.writerow(["snr","ours"]); [w.writerow(r) for r in rows]
        w.writerow(["mixed_f1",fmix]); w.writerow(["mixed_macs",cmix])
print("done",a.seeds,flush=True)
