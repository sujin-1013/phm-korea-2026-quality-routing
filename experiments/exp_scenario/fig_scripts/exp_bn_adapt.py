"""Verify test-time BatchNorm adaptation: single large, EVAL mode (clean running stats)
vs TRAIN mode (test-batch stats) per-SNR. The routing 'win' was actually this. Honest
check, reuse saved large, 3 seeds. Also vary batch size (BN-adapt needs representative
batches): bs=256 (same-SNR batch) vs bs=1 (single-sample, no adaptation benefit).
"""
import os, sys, csv, argparse
import numpy as np, torch
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; dev="cuda"; SWEEP=[float("inf"),20.,10.,5.,0.,-5.,-10.]
def fixed_noise(X,snr,seed):
    if not np.isfinite(snr): return torch.tensor(np.asarray(X),dtype=torch.float32).clone()
    gh=torch.Generator().manual_seed(seed); Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone()
    for j in range(0,Xn.shape[0],512): Xn[j:j+512]=add_awgn(Xn[j:j+512],snr_db=float(snr),generator=gh)
    return Xn
def predf(m,Xn,bs):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],bs): o.append(m(Xn[i:i+bs].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)
def load(seed):
    bc,nb=SIZES["large"]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_large_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m
ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,required=True); a=ap.parse_args()
for seed in a.seeds:
    set_seed(seed); _,_,_,_,Xte,yte=materialise("indist",None,seed); Xte,yte=cap(Xte,yte,12000,seed+9); yte=np.asarray(yte)
    for snr in SWEEP:
        Xn=fixed_noise(Xte,snr,222)
        m=load(seed); m.eval(); f_eval=macro_f1(yte,predf(m,Xn,256))
        m=load(seed); m.train(); f_bn256=macro_f1(yte,predf(m,Xn,256))   # BN batch-stats, bs256
        m=load(seed); m.train(); f_bn32=macro_f1(yte,predf(m,Xn,32))     # smaller batch
        lab=("inf" if not np.isfinite(snr) else int(snr))
        print(f"seed{seed} snr={str(lab):>4s}: eval={f_eval:.3f} | BN-adapt(bs256)={f_bn256:.3f} (d{f_bn256-f_eval:+.3f}) | BN-adapt(bs32)={f_bn32:.3f}",flush=True)
print("done",a.seeds,flush=True)
