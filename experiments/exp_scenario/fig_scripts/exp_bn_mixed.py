"""Realistic test: BN-adapt on MIXED-SNR batches (per-window random SNR, batch contains
mixed noise) vs eval. Also bs256/bs32. If gain holds on mixed batches -> deployable
positive; if it collapses -> only works for homogeneous-SNR batches. Single large, 3 seeds.
"""
import os, sys, argparse
import numpy as np, torch
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; dev="cuda"; LEVELS=[20.,10.,5.,0.,-5.,-10.,float("inf")]
def mixed_noise(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; assign=rng.integers(0,len(LEVELS),N)
    for li,s in enumerate(LEVELS):
        m=assign==li; idx=np.where(m)[0]
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn
def predf(m,Xn,bs,shuffle=None):
    o=[]; order=np.arange(Xn.shape[0]) if shuffle is None else shuffle
    pred=np.empty(Xn.shape[0],dtype=np.int64)
    with torch.no_grad():
        for i in range(0,len(order),bs):
            ii=order[i:i+bs]; out=m(Xn[ii].to(dev)).argmax(-1).cpu().numpy(); pred[ii]=out
    return pred
def load(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
for seed in a.seeds:
    set_seed(seed); _,_,_,_,Xte,yte=materialise("indist",None,seed); Xte,yte=cap(Xte,yte,12000,seed+9); yte=np.asarray(yte)
    Xn=mixed_noise(Xte,222)  # per-window random SNR (mixed within every batch)
    m=load(seed); m.eval(); fe=macro_f1(yte,predf(m,Xn,256))
    m=load(seed); m.train(); fb256=macro_f1(yte,predf(m,Xn,256))
    m=load(seed); m.train(); fb32=macro_f1(yte,predf(m,Xn,32))
    print(f"seed{seed} MIXED-SNR: eval={fe:.3f} | BN-adapt(bs256)={fb256:.3f} (d{fb256-fe:+.3f}) | BN-adapt(bs32)={fb32:.3f} (d{fb32-fe:+.3f})",flush=True)
print("done",a.seeds,flush=True)
