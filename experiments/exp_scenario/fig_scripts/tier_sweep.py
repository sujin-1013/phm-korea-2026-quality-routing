"""Per-SNR sweep for the CONFIRMED method tiers {base 133K, x300 331K, large 524K}.
Clean test windows + fixed AWGN per SNR level. Writes tier_sweep_s{seed}.csv: snr,base,x300,large.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import cnn_run
from cnn_run import materialise, cap, CNN1D, SIZES
SIZES["x300"] = (32, 5)
from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed
CKPT = ROOT / "experiments" / "exp_scenario" / "final_models"
dev = "cuda" if torch.cuda.is_available() else "cpu"
TIERS = ["base", "x300", "large"]

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args(); set_seed(a.seed)
_, _, _, _, Xte, yte = materialise("indist", None, a.seed); Xte, yte = cap(Xte, yte, 12000, a.seed+9); yte = np.asarray(yte)
models = {}
for sz in TIERS:
    bc, nb = SIZES[sz]; m = CNN1D(bc, nb).to(dev)
    m.load_state_dict(torch.load(CKPT / f"cnn_{sz}_s{a.seed}.pt", map_location=dev, weights_only=False)["model"]); models[sz] = m.eval()

def predf(m, Xn):
    o = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 256): o.append(m(Xn[i:i+256].to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(o)

def fixed_noise(X, snr, gs):
    g = torch.Generator().manual_seed(gs); Xn = torch.tensor(np.asarray(X), dtype=torch.float32).clone()
    if np.isfinite(snr):
        for j in range(0, Xn.shape[0], 512): Xn[j:j+512] = add_awgn(Xn[j:j+512], snr_db=float(snr), generator=g)
    return Xn

out = ROOT / "experiments" / "exp_scenario" / f"tier_sweep_s{a.seed}.csv"
with out.open("w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["snr", "base", "x300", "large"])
    for snr in [float("inf"), 20., 10., 5., 0., -5., -10.]:
        Xs = fixed_noise(Xte, snr, a.seed + (999 if not np.isfinite(snr) else int(snr)+100))
        row = [("inf" if not np.isfinite(snr) else "%d" % int(snr))] + [macro_f1(yte, predf(models[t], Xs)) for t in TIERS]
        w.writerow(row)
        print(f"  seed{a.seed} SNR={row[0]:>4}: base={row[1]:.3f} x300={row[2]:.3f} large={row[3]:.3f}", flush=True)
print("saved", out, flush=True)
