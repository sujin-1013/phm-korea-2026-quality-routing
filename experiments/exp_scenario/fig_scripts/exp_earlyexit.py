"""Experiment 2: shared-backbone EARLY-EXIT (GPU1, trains new model, 3 seeds).
One backbone (64ch x 5 blocks ~ large) with 3 GAP+Linear exit heads after blocks 1/3/5.
Joint multi-exit CE training, clean-only (matches canonical protocol). Inference: exit at
shallowest head whose margin>=tau_k (calibrated on mixed-val). Eval mixed-test.
Tests whether weight-sharing (shallow exits inherit deep features) captures the oracle
headroom that SEPARATE tiers could not. Reports: each exit standalone F1 (forced),
cascade F1 + avg effective params, vs separate-large 0.624@524K.
"""
import os, sys, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap
from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
EXITS=[1,3,5]  # block indices after which heads attach
CH=[64,128,128,256,256]  # block output channels (block i -> CH[i-1])
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
class EarlyExitCNN(nn.Module):
    def __init__(self,num_class=3):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(1,64,64,stride=16,padding=24),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2))
        self.blocks=nn.ModuleList(); prev=64
        for ch in CH:
            self.blocks.append(nn.Sequential(nn.Conv1d(prev,ch,3,padding=1),nn.BatchNorm1d(ch),nn.ReLU(),nn.MaxPool1d(2))); prev=ch
        self.heads=nn.ModuleDict({str(e):nn.Linear(CH[e-1],num_class) for e in EXITS})
        # effective params up to each exit (stem + blocks<=e + head)
        self.eff={}
        base=sum(p.numel() for p in self.stem.parameters())
        for e in EXITS:
            n=base+sum(p.numel() for b in self.blocks[:e] for p in b.parameters())+sum(p.numel() for p in self.heads[str(e)].parameters())
            self.eff[e]=n
    def forward(self,x):
        x=self.stem(x.unsqueeze(1)); outs={}
        for i,b in enumerate(self.blocks,1):
            x=b(x)
            if i in EXITS: outs[i]=self.heads[str(i)](x.mean(-1))
        return outs
def all_logits(m,Xn):
    m.eval(); res={e:[] for e in EXITS}
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256):
            o=m(Xn[i:i+256].to(dev))
            for e in EXITS: res[e].append(F.softmax(o[e],dim=-1).cpu().numpy())
    return {e:np.concatenate(res[e]) for e in EXITS}
def margin(p): s=np.sort(p,axis=1); return s[:,-1]-s[:,-2]
rows=[]
for seed in [42,1,7]:
    set_seed(seed)
    Xtr,ytr,Xv,yv,Xte,yte=materialise("indist",None,seed)
    Xtr,ytr=cap(Xtr,ytr,20000,seed+7); Xv,yv=cap(Xv,yv,4000,seed+8); Xte,yte=cap(Xte,yte,12000,seed+9)
    yv=np.asarray(yv); yte=np.asarray(yte)
    cw=torch.tensor(inverse_freq_class_weights(ytr,num_class=3),dtype=torch.float32,device=dev)
    m=EarlyExitCNN().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
    lossf=nn.CrossEntropyLoss(weight=cw)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long)
    Xvt=torch.tensor(Xv,dtype=torch.float32); g=torch.Generator().manual_seed(0); best=-1; best_state=None
    for ep in range(50):
        m.train(); perm=torch.randperm(Xt.shape[0])
        for i in range(0,Xt.shape[0],64):
            idx=perm[i:i+64]; xb=add_awgn(Xt[idx].clone(),snr_db=float("inf"),generator=g).to(dev); yb=yt[idx].to(dev)
            o=m(xb); loss=sum(lossf(o[e],yb) for e in EXITS)
            opt.zero_grad(); loss.backward(); opt.step()
        # select on clean val by DEEP exit (matches canonical clean-val selection)
        m.eval(); pv=all_logits(m,Xvt)[EXITS[-1]]; vf=macro_f1(yv,pv.argmax(1))
        if vf>best: best=vf; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    torch.save({"model":m.state_dict(),"eff":m.eff},f"{CKPT}/earlyexit_s{seed}.pt")
    # eval mixed
    Xvn=mixed_noise(Xv,333); Xtn=mixed_noise(Xte,222)
    pv=all_logits(m,Xvn); pt=all_logits(m,Xtn)
    # standalone forced-exit F1 (each head alone) on test
    fexit={e:macro_f1(yte,pt[e].argmax(1)) for e in EXITS}
    eff=m.eff
    # cascade: exit at shallowest head whose margin>=tau; calibrate (t1,t2) on mixed-val
    def cascade(p,t1,t2):
        e1,e2,e3=EXITS; pred=p[e3].argmax(1).copy(); cost=np.full(len(pred),eff[e3],dtype=float)
        m1=margin(p[e1])>=t1; pred[m1]=p[e1].argmax(1)[m1]; cost[m1]=eff[e1]
        m2=(~m1)&(margin(p[e2])>=t2); pred[m2]=p[e2].argmax(1)[m2]; cost[m2]=eff[e2]
        return pred,cost.mean()/1e3
    grid=[round(x,2) for x in np.arange(0.0,0.96,0.05)]
    vlf=macro_f1(yv,pv[EXITS[-1]].argmax(1))
    best_iso=None; best_acc=None
    for t1 in grid:
        for t2 in grid:
            predv,cv=cascade(pv,t1,t2); fv=macro_f1(yv,predv)
            if best_acc is None or fv>best_acc[0]: best_acc=(fv,t1,t2)
            if fv>=vlf-0.005:
                if best_iso is None or cv<best_iso[3]: best_iso=(fv,t1,t2,cv)
    # eval chosen ops on test
    ta=best_acc[1:]; predA,cA=cascade(pt,*ta); fA=macro_f1(yte,predA)
    if best_iso: ti=best_iso[1:3]; predI,cI=cascade(pt,*ti); fI=macro_f1(yte,predI)
    else: ti=ta; fI,cI=fA,cA
    rows.append({"seed":seed,"exit1":fexit[EXITS[0]],"exit2":fexit[EXITS[1]],"deep":fexit[EXITS[2]],
                 "casc_acc_f1":fA,"casc_acc_cost":cA,"casc_iso_f1":fI,"casc_iso_cost":cI,
                 "eff1":eff[EXITS[0]]/1e3,"eff3":eff[EXITS[1]]/1e3,"eff5":eff[EXITS[2]]/1e3})
    print(f"seed{seed}: exits F1 {fexit[EXITS[0]]:.3f}/{fexit[EXITS[1]]:.3f}/{fexit[EXITS[2]]:.3f} (eff {eff[EXITS[0]]/1e3:.0f}/{eff[EXITS[1]]/1e3:.0f}/{eff[EXITS[2]]/1e3:.0f}K) | cascade acc={fA:.3f}@{cA:.0f}K iso={fI:.3f}@{cI:.0f}K",flush=True)
def ms(k): a=np.array([r[k] for r in rows]); return a.mean(),a.std()
print("\n=== 3-seed mean (shared-backbone early-exit, mixed-SNR) ===")
print(f"  deep exit (full)   {ms('deep')[0]:.3f}+-{ms('deep')[1]:.3f} @{ms('eff5')[0]:.0f}K   (vs separate-large 0.624@524K)")
print(f"  shallow exit1      {ms('exit1')[0]:.3f}+-{ms('exit1')[1]:.3f} @{ms('eff1')[0]:.0f}K")
print(f"  mid exit2          {ms('exit2')[0]:.3f}+-{ms('exit2')[1]:.3f} @{ms('eff3')[0]:.0f}K")
print(f"  cascade acc-max    {ms('casc_acc_f1')[0]:.3f}+-{ms('casc_acc_f1')[1]:.3f} @{ms('casc_acc_cost')[0]:.0f}K  delta_vs_seplarge={ms('casc_acc_f1')[0]-0.624:+.3f}")
print(f"  cascade iso-acc    {ms('casc_iso_f1')[0]:.3f}+-{ms('casc_iso_f1')[1]:.3f} @{ms('casc_iso_cost')[0]:.0f}K")
with open(f"{_ROOT}/experiments/exp_scenario/exp_earlyexit.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
print("saved exp_earlyexit.csv")
