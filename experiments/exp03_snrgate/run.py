"""SNRGate joint training — main paper method.

Runs ``train_routing`` (estimator + LearnableSNRGate + 3 LoRA-MOMENT tiers +
per-tier heads) on Paderborn-PU 4-class data (or synthetic fallback), then:
  - dumps the trained classifier checkpoint
  - sweeps SNR ∈ {inf, 20, 10, 5, 0, -5, -10} dB and reports per-SNR macro-F1
  - records routing distribution per SNR (S/B/L share at hard inference)
  - records final learned gate centers

Pair this with exp02_routing (rule-based ``HardGateRouter`` baseline) to
compute the hard-vs-learnable ablation reported in the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.noise_injection import add_awgn, snr_sweep_levels
from phm_routing.data.paderborn import (
    SyntheticConfig,
    discover_files,
    load_signal,
    synthesize,
)
from phm_routing.data.windowing import sliding_windows
from phm_routing.training.train_routing import RoutingTrainConfig, train_routing
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed

WINDOW = 2048
STRIDE = 1024


def materialise(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_tr, y_tr, X_val, y_val)`` of windowed labelled vibration."""
    entries = discover_files(data_root)
    if entries:
        win_list, lab_list = [], []
        for e in entries:
            sig = load_signal(e.path)[: 64_000 * 4].astype(np.float32)
            w = sliding_windows(sig, WINDOW, STRIDE)
            win_list.append(w)
            lab_list.append(np.full(len(w), e.label_idx, dtype=np.int64))
        X = np.concatenate(win_list)
        y = np.concatenate(lab_list)
    else:
        print(f"[warn] no Paderborn .mat under {data_root} — using synthetic", flush=True)
        sigs, labels, _ = synthesize(SyntheticConfig(n_per_class=8))
        win_list, lab_list = [], []
        for s, lab in zip(sigs, labels):
            w = sliding_windows(s, WINDOW, STRIDE)
            win_list.append(w)
            lab_list.append(np.full(len(w), int(lab), dtype=np.int64))
        X = np.concatenate(win_list)
        y = np.concatenate(lab_list)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n_val = max(1, len(X) // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def snr_sweep_eval(
    classifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    snr_levels: list[float],
    batch_size: int,
    device: str,
) -> tuple[dict, dict]:
    """Hard-gate inference at each SNR. Returns (per_snr_f1, per_snr_routing)."""
    aug_gen = torch.Generator(device="cpu").manual_seed(123)
    Xv = torch.from_numpy(X_val).float().unsqueeze(1)
    yv = torch.from_numpy(y_val).long()
    f1: dict[str, float] = {}
    routing: dict[str, dict[str, int]] = {}
    classifier.eval()
    for snr in snr_levels:
        preds, tiers_used = [], []
        for i in range(0, Xv.shape[0], batch_size):
            xb = Xv[i : i + batch_size].clone()
            if np.isfinite(snr):
                xb = add_awgn(xb, snr_db=float(snr), generator=aug_gen)
            xb = xb.to(device)
            with torch.no_grad():
                logits, tier_idx = classifier.inference_forward(xb)
            preds.append(logits.argmax(-1).cpu().numpy())
            tiers_used.append(tier_idx.cpu().numpy())
        y_pred = np.concatenate(preds)
        tier_arr = np.concatenate(tiers_used)
        label = "inf" if not np.isfinite(snr) else f"{snr:g}"
        f1[label] = macro_f1(yv.numpy(), y_pred)
        c = Counter(tier_arr.tolist())
        total = sum(c.values())
        routing[label] = {
            "S": c.get(0, 0) / total,
            "B": c.get(1, 0) / total,
            "L": c.get(2, 0) / total,
        }
    return f1, routing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--estimator-ckpt", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 2

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    Xtr, ytr, Xval, yval = materialise(args.data_root)
    print(f"[info] train={Xtr.shape[0]} val={Xval.shape[0]} window={WINDOW}")

    cfg = RoutingTrainConfig(
        estimator_ckpt=args.estimator_ckpt,
        train_estimator=args.estimator_ckpt is None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )
    classifier, history = train_routing(Xtr, ytr, Xval, yval, cfg)

    ckpt = args.out_dir / "checkpoints" / "snrgate.pt"
    torch.save(
        {
            "estimator": classifier.estimator.state_dict(),
            "gate": classifier.gate.state_dict(),
            "tiers": [m.state_dict() for m in classifier.tiers],
            "history": history,
        },
        ckpt,
    )
    print(f"[done] saved {ckpt}")

    # SNR sweep
    levels = snr_sweep_levels()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    f1, routing = snr_sweep_eval(classifier, Xval, yval, levels, args.batch_size, device)

    sweep_path = args.out_dir / "snr_sweep.csv"
    with sweep_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "macro_f1", "share_S", "share_B", "share_L"])
        for label, score in f1.items():
            r = routing[label]
            w.writerow([label, f"{score:.4f}", f"{r['S']:.3f}", f"{r['B']:.3f}", f"{r['L']:.3f}"])
    print(f"[done] snr sweep -> {sweep_path}")

    summary_path = args.out_dir / "summary.json"
    summary = {
        "final_centers": classifier.gate.centers.detach().cpu().tolist(),
        "final_T": float(classifier.gate.T),
        "val_f1_history": history["val_f1"],
        "T_schedule": history["T"],
        "lora_r": args.lora_r,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
