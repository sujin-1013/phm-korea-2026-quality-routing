"""Measure data for new presentation figures (seed42, confirmed {base,x300,large} routing):
  - q distribution per SNR (estimator quality vs SNR)
  - tier-selection shares per SNR (routing behavior)  at iso operating point
  - one waveform window: clean / 0 dB / -10 dB
Writes: qsnr.csv, share_bxl.csv, waveform.npz
"""
import itertools, sys
from pathlib import Path

import numpy as np, torch, csv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
SIZES["x300"] = (32, 5)
from phm_routing.models.noise_estimator import (
    DEPLOYED_ESTIMATOR_CHANNELS,
    DEPLOYED_ESTIMATOR_OVERHEAD_K,
    LEGACY_ESTIMATOR_CHANNELS,
    LEGACY_ESTIMATOR_OVERHEAD_K,
    NoiseEstimator1D,
)
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT = ROOT / "experiments" / "exp_scenario" / "final_models"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TIERS = ["base", "x300", "large"]; LEVELS = [20., 10., 5., 0., -5., -10., float("inf")]
set_seed(SEED)
Xtr, ytr, Xv, yv, Xte, yte = materialise("indist", None, SEED)
Xv, yv = cap(Xv, yv, 4000, SEED+8); Xte, yte = cap(Xte, yte, 12000, SEED+9); yv = np.asarray(yv); yte = np.asarray(yte)
models = {}; pc = {}
for sz in TIERS:
    bc, nb = SIZES[sz]; m = CNN1D(bc, nb).to(dev); st = torch.load(CKPT / f"cnn_{sz}_s{SEED}.pt", map_location=dev, weights_only=False)
    m.load_state_dict(st["model"]); models[sz] = m.eval(); pc[sz] = st["npar"]
def load_estimator():
    preferred = CKPT / "quality_est_0.13k.pt"
    legacy = CKPT / "noise_est_7.6k.pt"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(f"missing estimator checkpoint: {preferred} or {legacy}")
    state = torch.load(path, map_location=dev, weights_only=False)
    weights = state.get("model") or state.get("state_dict") or state if isinstance(state, dict) else state
    if isinstance(state, dict) and "channels" in state:
        channels = tuple(int(v) for v in state["channels"])
    elif path == legacy:
        channels = LEGACY_ESTIMATOR_CHANNELS
    else:
        channels = DEPLOYED_ESTIMATOR_CHANNELS
    overhead_k = LEGACY_ESTIMATOR_OVERHEAD_K if path == legacy else DEPLOYED_ESTIMATOR_OVERHEAD_K
    model = NoiseEstimator1D(channels=channels).to(dev)
    model.load_state_dict(weights)
    model.eval()
    print(f"loaded estimator {path.name} ({overhead_k:.3f}K overhead)", flush=True)
    return model, overhead_k


est, EST_OVH = load_estimator()

def mixed_noise(X, seed):
    rng = np.random.default_rng(seed); gh = torch.Generator().manual_seed(seed)
    Xn = torch.tensor(np.asarray(X), dtype=torch.float32).clone(); N = X.shape[0]; assign = rng.integers(0, len(LEVELS), N)
    for li, s in enumerate(LEVELS):
        m = assign == li; idx = np.where(m)[0]
        if np.isfinite(s):
            for j in range(0, len(idx), 512): ii = idx[j:j+512]; Xn[ii] = add_awgn(Xn[ii], snr_db=float(s), generator=gh)
    return Xn

def noise_q(Xn):
    o = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256): o.append(est(Xn[i:i+256].to(dev).unsqueeze(1)).cpu().numpy())
    return 1.0 - np.concatenate(o)  # quality = 1 - noise

def preds(m, Xn):
    o = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)

def route_k(q, taus):  # q=quality; low quality -> bigger tier. thresholds on quality desc
    out = np.zeros(len(q), dtype=int)
    for t in taus: out += (q < t).astype(int)   # more thresholds crossed (lower q) -> higher tier idx
    return out

def cost(idx): sh = [(idx == k).mean() for k in range(3)]; return sum(sh[k]*pc[TIERS[k]] for k in range(3))/1e3 + EST_OVH

# ---- iso operating point on val (quality thresholds) ----
Xvn = mixed_noise(Xv, 111); qv = noise_q(Xvn); lgv = {t: preds(models[t], Xvn) for t in TIERS}
Xtn = mixed_noise(Xte, 222); qt = noise_q(Xtn); lgt = {t: preds(models[t], Xtn) for t in TIERS}
pred_stack = lambda lg, idx: np.array([lg[TIERS[idx[i]]][i] for i in range(len(idx))])
val_large = macro_f1(yv, lgv["large"])
grid = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
combos = [c for c in itertools.combinations(sorted(grid, reverse=True), 2)]  # (t_hi, t_lo) descending
best = None
for taus in combos:
    iv = route_k(qv, taus); fv = macro_f1(yv, pred_stack(lgv, iv))
    c = cost(iv)
    if fv >= val_large - 0.01 and (best is None or c < best[1]): best = (taus, c)
taus = best[0] if best else (0.5, 0.3)
print("iso quality thresholds:", taus, flush=True)

# ---- per fixed SNR: q stats + routing shares ----
def fixed(snr, gs):
    g = torch.Generator().manual_seed(gs); Xn = torch.tensor(np.asarray(Xte), dtype=torch.float32).clone()
    if np.isfinite(snr):
        for j in range(0, Xn.shape[0], 512): Xn[j:j+512] = add_awgn(Xn[j:j+512], snr_db=float(snr), generator=g)
    return Xn
with (ROOT / "experiments" / "exp_scenario" / "qsnr.csv").open("w", newline="") as f1, \
     (ROOT / "experiments" / "exp_scenario" / "share_bxl.csv").open("w", newline="") as f2:
    w1 = csv.writer(f1); w1.writerow(["snr", "q_mean", "q_p25", "q_p75"])
    w2 = csv.writer(f2); w2.writerow(["snr", "small_133K", "medium_331K", "large_524K"])
    for lbl, snr in [("clean", float("inf")), ("20", 20), ("10", 10), ("5", 5), ("0", 0), ("-5", -5), ("-10", -10)]:
        Xs = fixed(snr, SEED + (999 if not np.isfinite(snr) else int(snr)+100)); q = noise_q(Xs)
        w1.writerow([lbl, f"{q.mean():.4f}", f"{np.percentile(q,25):.4f}", f"{np.percentile(q,75):.4f}"])
        idx = route_k(q, taus); sh = [float((idx == k).mean()) for k in range(3)]
        w2.writerow([lbl] + [f"{s:.4f}" for s in sh])
        print(f"  SNR={lbl:>5}: q={q.mean():.3f}  share s/m/l={[round(s,2) for s in sh]}", flush=True)

# ---- waveform sample (one IR window): clean / 0 / -10 dB ----
ir = np.where(yte == 1)[0][0]; x0 = torch.tensor(Xte[ir:ir+1], dtype=torch.float32)
g = torch.Generator().manual_seed(7)
clean = x0[0].numpy(); db0 = add_awgn(x0.clone(), snr_db=0.0, generator=g)[0].numpy(); dbm10 = add_awgn(x0.clone(), snr_db=-10.0, generator=g)[0].numpy()
np.savez(ROOT / "experiments" / "exp_scenario" / "waveform.npz", clean=clean, db0=db0, dbm10=dbm10)
print("saved qsnr.csv, share_bxl.csv, waveform.npz", flush=True)
