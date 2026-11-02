"""Shrink the quality estimator and check routing performance is retained.
For each estimator channel config: train it (self-sup MSE on injected SNR), then run the
confirmed {base,x300,large} iso routing and report ours F1 / cost. Tier models are fixed.
Usage: --seed 42 --configs "8,16,16,32;4,8,8,16;..."
"""
import os, sys, argparse, itertools
import numpy as np, torch, torch.nn as nn
_R = "/home/sujin/phm-korea-2026-snrgate"
sys.path.insert(0, _R); sys.path.insert(0, _R + "/src"); os.chdir(_R); sys.path.insert(0, "/tmp")
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
SIZES["x300"] = (32, 5)
from phm_routing.models.noise_estimator import NoiseEstimator1D, snr_to_target
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT = f"{_R}/experiments/exp_scenario/final_models"; dev = "cuda"
TIERS = ["base", "x300", "large"]; LEVELS = [20., 10., 5., 0., -5., -10., float("inf")]; EST_OVH_BASE = 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--configs", type=str, required=True)  # "8,16,16,32;4,8,8,16"
ap.add_argument("--epochs", type=int, default=30)
a = ap.parse_args(); set_seed(a.seed)

Xtr, ytr, Xv, yv, Xte, yte = materialise("indist", None, a.seed)
Xtr, ytr = cap(Xtr, ytr, 16000, a.seed+7); Xv, yv = cap(Xv, yv, 4000, a.seed+8); Xte, yte = cap(Xte, yte, 12000, a.seed+9)
yv = np.asarray(yv); yte = np.asarray(yte)
models = {}; pc = {}
for sz in TIERS:
    bc, nb = SIZES[sz]; m = CNN1D(bc, nb).to(dev); st = torch.load(f"{CKPT}/cnn_{sz}_s{a.seed}.pt", map_location=dev, weights_only=False)
    m.load_state_dict(st["model"]); models[sz] = m.eval(); pc[sz] = st["npar"]

def mixed_noise(X, seed):
    rng = np.random.default_rng(seed); gh = torch.Generator().manual_seed(seed)
    Xn = torch.tensor(np.asarray(X), dtype=torch.float32).clone(); N = X.shape[0]; assign = rng.integers(0, len(LEVELS), N)
    for li, s in enumerate(LEVELS):
        m = assign == li; idx = np.where(m)[0]
        if np.isfinite(s):
            for j in range(0, len(idx), 512): ii = idx[j:j+512]; Xn[ii] = add_awgn(Xn[ii], snr_db=float(s), generator=gh)
    return Xn

def tier_preds(m, Xn):
    o = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)

# tier predictions on val/test mixed (estimator-independent) — compute once
Xvn = mixed_noise(Xv, 111); Xtn = mixed_noise(Xte, 222)
lgv = {t: tier_preds(models[t], Xvn) for t in TIERS}; lgt = {t: tier_preds(models[t], Xtn) for t in TIERS}
val_large = macro_f1(yv, lgv["large"])
pred_stack = lambda lg, idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])
def route_k(q, taus):
    out = np.zeros(len(q), dtype=int)
    for t in taus: out += (q >= t).astype(int)
    return out

def train_est(channels, epochs):
    est = NoiseEstimator1D(channels=tuple(channels), kernel_size=7).to(dev)
    npar = sum(p.numel() for p in est.parameters())
    opt = torch.optim.AdamW(est.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.MSELoss()
    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32); rng = np.random.default_rng(0); g = torch.Generator().manual_seed(0)
    tgt_lut = {s: float(snr_to_target(s)) for s in LEVELS}
    for ep in range(epochs):
        est.train(); perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], 64):
            idx = perm[i:i+64]; xb = Xt[idx].clone(); n = xb.shape[0]
            assign = rng.integers(0, len(LEVELS), n); tgt = np.empty(n, dtype=np.float32)
            for li, s in enumerate(LEVELS):
                msk = assign == li; tgt[msk] = tgt_lut[s]; ii = np.where(msk)[0]
                if np.isfinite(s) and len(ii): xb[ii] = add_awgn(xb[ii], snr_db=float(s), generator=g)
            pred = est(xb.unsqueeze(1).to(dev)); loss = lossf(pred, torch.tensor(tgt, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    est.eval()
    return est, npar

def est_out(est, Xn):
    o = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return np.concatenate(o)  # noise level (0=clean,1=noisy)

def iso_route(qv, qt, est_npar):
    EST = est_npar / 1e3
    def cost(idx): sh = [(idx == k).mean() for k in range(3)]; return sum(sh[k]*pc[TIERS[k]] for k in range(3))/1e3 + EST
    grid = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]
    best = None
    for taus in itertools.combinations(grid, 2):
        fv = macro_f1(yv, pred_stack(lgv, route_k(qv, taus)))
        it = route_k(qt, taus); c = cost(it)
        if fv >= val_large - 0.01 and (best is None or c < best[1]): best = (taus, c, macro_f1(yte, pred_stack(lgt, it)))
    if best is None:  # fallback accmax
        bestf = max(itertools.combinations(grid, 2), key=lambda taus: macro_f1(yv, pred_stack(lgv, route_k(qv, taus))))
        it = route_k(qt, bestf); best = (bestf, cost(it), macro_f1(yte, pred_stack(lgt, it)))
    return best  # (taus, cost, test_f1)

print(f"[estsize seed{a.seed}]  config        params   q-MSE(val)  ours_F1  ours_cost", flush=True)
for cfgs in a.configs.split(";"):
    ch = [int(x) for x in cfgs.split(",")]
    est, npar = train_est(ch, a.epochs)
    # val q-MSE
    qv = est_out(est, Xvn); qt = est_out(est, Xtn)
    taus, cst, f1 = iso_route(qv, qt, npar)
    print(f"  {cfgs:14s} {npar:6d}   ours_F1={f1:.4f}  cost={cst:.0f}K  tau={taus}", flush=True)
