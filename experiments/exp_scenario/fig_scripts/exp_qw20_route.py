"""qw20-anchored routing (BUG-FREE: all tiers forced .eval(), no BN train-mode).
Tiers {qw40(3.72M), qw20(6.45M), dense(9.96M)} all eval. Route by quality score.
Operating points: (a) iso-qw20 = maintain best single (qw20) val F1 -0.005, MIN avg MACs;
(b) accmax. Mixed-SNR, 3 seeds. Sanity: print each tier mixed-F1 (must match canonical
qw20~0.646/qw40~0.611/dense~0.624) to confirm no BN artifact. Reports F1 + avg MACs.
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
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.,10.,5.,0.,-5.,-10.,float("inf")]; dev="cuda"
CONV_OUTLEN=[128,64,32,16,8]; KS=[64,3,3,3,3]; FT_EP=12
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def predf(m,Xn):
    assert not m.training, "tier must be in eval mode!"
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
route=lambda qs,t1,t2:(qs>=t1).astype(int)+(qs>=t2).astype(int)
stack=lambda P,idx: np.array([P[idx[i]][i] for i in range(len(idx))])
for seed in a.seeds:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte); cw=inverse_freq_class_weights(ytr,num_class=3)
    Xcal=mixed_noise(Xv,333); q=qscore(Xcal); valX=mixed_noise(Xv,334); Xte_n=mixed_noise(Xte,222)
    dense=fresh(seed); dense.eval()  # FORCE eval (the bug fix)
    qw20=build_qw(seed,0.20,Xcal,q,Xtr,ytr,yv,valX,cw); qw40=build_qw(seed,0.40,Xcal,q,Xtr,ytr,yv,valX,cw)
    tiers=[qw40,qw20,dense]; tmacs=[macs(surv(qw40)),macs(surv(qw20)),macs(surv(dense))]
    # sanity: each tier mixed-F1
    f_t=[evalf(t,Xte_n,yte) for t in tiers]
    print(f"seed{seed} SANITY tiers: qw40={f_t[0]:.3f}@{tmacs[0]/1e6:.2f}M qw20={f_t[1]:.3f}@{tmacs[1]/1e6:.2f}M dense={f_t[2]:.3f}@{tmacs[2]/1e6:.2f}M",flush=True)
    qv=qscore(valX); qt=qscore(Xte_n); Pv=[predf(t,valX) for t in tiers]; Pt=[predf(t,Xte_n) for t in tiers]
    qw20_valf1=macro_f1(yv,Pv[1])  # anchor = qw20 (best single target)
    grid=[round(x,2) for x in np.arange(0.05,0.91,0.05)]; pairs=[(t1,t2) for t1,t2 in itertools.product(grid,grid) if t1<t2]
    iso=None; accmax=None
    for t1,t2 in pairs:
        fv=macro_f1(yv,stack(Pv,route(qv,t1,t2))); ct=sum((route(qt,t1,t2)==k).mean()*tmacs[k] for k in range(3))
        if accmax is None or fv>accmax[0]: accmax=(fv,t1,t2)
        if fv>=qw20_valf1-0.005 and (iso is None or ct<iso[3]): iso=(fv,t1,t2,ct)
    if iso is None: iso=(accmax[0],accmax[1],accmax[2],sum((route(qt,accmax[1],accmax[2])==k).mean()*tmacs[k] for k in range(3)))
    idt=route(qt,iso[1],iso[2]); fi=macro_f1(yte,stack(Pt,idt)); ci=sum((idt==k).mean()*tmacs[k] for k in range(3)); shi=[(idt==k).mean() for k in range(3)]
    print(f"seed{seed} ROUTE iso-qw20: F1={fi:.3f}@{ci/1e6:.2f}M (vs qw20 {f_t[1]:.3f}@6.45M) share={shi[0]*100:.0f}/{shi[1]*100:.0f}/{shi[2]*100:.0f} tau=({iso[1]},{iso[2]})",flush=True)
    with open(f"{_ROOT}/experiments/exp_scenario/exp_qw20_route_s{seed}.csv","w",newline="") as fp:
        w=csv.writer(fp); w.writerow(["seed","qw40_f1","qw20_f1","dense_f1","route_iso_f1","route_iso_macs","sh_qw40","sh_qw20","sh_dense"])
        w.writerow([seed,f_t[0],f_t[1],f_t[2],fi,ci,shi[0],shi[1],shi[2]])
    # ---- per-SNR sweep (fixed SNR per level) for pruned tiers + routing (fig_main_gate) ----
    def fixed_noise(X,snr,gseed):
        g=torch.Generator().manual_seed(gseed); Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone()
        if np.isfinite(snr):
            for j in range(0,Xn.shape[0],512): Xn[j:j+512]=add_awgn(Xn[j:j+512],snr_db=float(snr),generator=g)
        return Xn
    with open(f"{_ROOT}/experiments/exp_scenario/qwtier_sweep_s{seed}.csv","w",newline="") as fp:
        w=csv.writer(fp); w.writerow(["snr","qw40","qw20","dense","ours"])
        for snr in [float("inf"),20.,10.,5.,0.,-5.,-10.]:
            Xs=fixed_noise(Xte,snr,seed+(999 if not np.isfinite(snr) else int(snr)+100))
            Ps=[predf(t,Xs) for t in tiers]; qs=qscore(Xs)
            row=[("inf" if not np.isfinite(snr) else "%d"%int(snr)),
                 macro_f1(yte,Ps[0]),macro_f1(yte,Ps[1]),macro_f1(yte,Ps[2]),
                 macro_f1(yte,stack(Ps,route(qs,iso[1],iso[2])))]
            w.writerow(row)
    print(f"seed{seed} wrote qwtier_sweep_s{seed}.csv",flush=True)
print("done",a.seeds,flush=True)
