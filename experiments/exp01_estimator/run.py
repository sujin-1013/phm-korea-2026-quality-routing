"""Phase 2 estimator training + calibration analysis.

Outputs:
  - ``checkpoints/noise_estimator.pt`` — trained estimator weights
  - ``calibration.csv`` — true_snr × predicted_noise_level scatter (for paper Fig.)
  - ``fp_rate.txt`` — fault/noise disentangle false-positive rate on clean windows
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.paderborn import (
    SyntheticConfig,
    discover_files,
    load_signal,
    synthesize,
)
from phm_routing.data.windowing import sliding_windows
from phm_routing.training.train_estimator import (
    EstimatorTrainConfig,
    calibration_scan,
    train_estimator,
)
from phm_routing.utils.seed import set_seed

WINDOW = 2048
STRIDE = 1024


def materialise_windows(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_windows, val_windows)``, each shape (N, WINDOW)."""
    entries = discover_files(data_root)
    if entries:
        signals = []
        for e in entries:
            sig = load_signal(e.path)[: 64_000 * 4]
            signals.append(sig.astype(np.float32))
        all_windows = np.concatenate([sliding_windows(s, WINDOW, STRIDE) for s in signals])
    else:
        print(f"[warn] no Paderborn .mat under {data_root} — using synthetic", flush=True)
        sigs, _, _ = synthesize(SyntheticConfig(n_per_class=8))
        all_windows = np.concatenate([sliding_windows(s, WINDOW, STRIDE) for s in sigs])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(all_windows))
    n_val = len(all_windows) // 5
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return all_windows[tr_idx], all_windows[val_idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 3
        args.batch_size = 32

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    Xtr, Xval = materialise_windows(args.data_root)
    print(f"[info] train_windows={Xtr.shape[0]} val_windows={Xval.shape[0]} window={WINDOW}")

    cfg = EstimatorTrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, history = train_estimator(Xtr, Xval, cfg)

    # Save weights
    ckpt = args.out_dir / "checkpoints" / "noise_estimator.pt"
    torch.save({"state_dict": model.state_dict(), "history": history}, ckpt)
    print(f"[done] saved {ckpt}")

    # Calibration: predict over the 7 SNR levels
    snr_levels = list(cfg.snr_levels)
    cal = calibration_scan(model, Xval, snr_levels)
    cal_path = args.out_dir / "calibration.csv"
    with cal_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "mean_pred", "std_pred", "n"])
        for snr, preds in cal.items():
            label = "inf" if not np.isfinite(snr) else f"{snr:g}"
            w.writerow([label, f"{preds.mean():.4f}", f"{preds.std():.4f}", len(preds)])
    print(f"[done] calibration -> {cal_path}")

    # FP rate on clean windows
    fp_path = args.out_dir / "fp_rate.txt"
    last = history["fp_rate_clean"][-1]
    fp_path.write_text(f"fp_rate_clean (final epoch, threshold=0.33): {last:.4f}\n")
    print(f"[done] fp rate -> {fp_path}: {last:.4f}")


if __name__ == "__main__":
    main()
