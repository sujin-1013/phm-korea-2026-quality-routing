import os,sys; import numpy as np, torch
_ROOT="/home/sujin/phm-korea-2026-snrgate"; sys.path.insert(0,_ROOT); sys.path.insert(0,_ROOT+"/src"); os.chdir(_ROOT); sys.path.insert(0,"/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.seed import set_seed
CKPT=f"{_ROOT}/experiments/exp_scenario/final_models"; TIERS=["small","mid","large"]; LEVELS=[20.0,10.0,5.0,0.0,-5.0,-10.0,float("inf")]; dev="cuda"
SNR=["inf","20","10","5","0","-5","-10"]
def mixed(X,seed):
    rng=np.random.default_rng(seed); gh=torch.Generator().manual_seed(seed)
    Xn=torch.tensor(np.asarray(X),dtype=torch.float32).clone(); N=Xn.shape[0]; a=rng.integers(0,len(LEVELS),N); ts=np.empty(N,dtype=object)
    for li,s in enumerate(LEVELS):
        msk=a==li; idx=np.where(msk)[0]; ts[msk]=("inf" if not np.isfinite(s) else str(int(s)))
        if np.isfinite(s):
            for j in range(0,len(idx),512): ii=idx[j:j+512]; Xn[ii]=add_awgn(Xn[ii],snr_db=float(s),generator=gh)
    return Xn,ts
def load(sz,seed):
    bc,nb=SIZES[sz]; m=CNN1D(bc,nb).to(dev); m.load_state_dict(torch.load(f"{CKPT}/cnn_{sz}_s{seed}.pt",map_location=dev,weights_only=False)["model"]); return m.eval()
def pred(m,Xn):
    o=[]
    with torch.no_grad():
        for i in range(0,Xn.shape[0],256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)
La={s:[] for s in SNR}; Oa={s:[] for s in SNR}
for seed in [42,1,7]:
    set_seed(seed); _,_,_,_,Xte,yte=materialise("indist",None,seed); Xte,yte=cap(Xte,yte,12000,seed+9)
    Xn,ts=mixed(Xte,222); y=np.asarray(yte); P={sz:pred(load(sz,seed),Xn) for sz in TIERS}
    cor={sz:(P[sz]==y) for sz in TIERS}; anyc=cor["small"]|cor["mid"]|cor["large"]
    for s in SNR:
        msk=ts==s
        if msk.sum()>0: La[s].append(cor["large"][msk].mean()); Oa[s].append(anyc[msk].mean())
large=np.array([np.mean(La[s]) for s in SNR]); oracle=np.array([np.mean(Oa[s]) for s in SNR])
np.savez("/tmp/oracle_persnr.npz",large=large,oracle=oracle)
print("dumped per-SNR. large:",[round(v,3) for v in large],"oracle:",[round(v,3) for v in oracle])
