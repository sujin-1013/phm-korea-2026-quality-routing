"""Compatibility helpers for older figure scripts.

Final CNN/capacity-ladder implementations now live under ``src/phm_routing``.
This module keeps the historical ``cnn_run`` import surface used by archived
experiment scripts while removing the old hardcoded absolute path and duplicate
model/training implementation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from phm_routing.data.noise_injection import add_awgn, snr_sweep_levels
from phm_routing.data.paderborn import inverse_freq_class_weights
from phm_routing.models.cnn1d import CAPACITY_SIZES, CNN1D
from phm_routing.training.train_cnn import (
    DEFAULT_AUGMENT_SNR,
    CLEAN_ONLY_AUGMENT,
    cap_arrays,
    materialise_scenario,
    predict_labels,
    train_cnn_classifier,
)
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[3]
WINDOW = 2048
STRIDE = 2048
AUG = DEFAULT_AUGMENT_SNR
SIZES = {name: (spec.base_channels, spec.n_blocks) for name, spec in CAPACITY_SIZES.items()}


def materialise(scenario, test_condition, seed):
    return materialise_scenario(
        ROOT / "data" / "raw" / "paderborn_pu",
        scenario=scenario,
        test_condition=test_condition,
        seed=seed,
        window=WINDOW,
        stride=STRIDE,
    )


def cap(X, y, n, sd):
    return cap_arrays(X, y, n, sd)


def train(size, Xtr, ytr, Xv, yv, epochs, bs, dev, cw):
    result = train_cnn_classifier(
        size,
        Xtr,
        ytr,
        Xv,
        yv,
        epochs=epochs,
        batch_size=bs,
        device=dev,
        class_weights=cw,
        augment_snr=AUG,
    )
    return result.model, result.best_val_f1, result.n_parameters


def sweep(m, Xte, yte, dev):
    levels = [level for level in snr_sweep_levels() if level != -20.0]
    Xt = torch.tensor(Xte, dtype=torch.float32)
    gen = torch.Generator().manual_seed(123)
    out = {}
    m.eval()
    for snr in levels:
        preds = []
        with torch.no_grad():
            for start in range(0, Xt.shape[0], 256):
                xb = Xt[start : start + 256].clone()
                if np.isfinite(snr):
                    xb = add_awgn(xb, snr_db=float(snr), generator=gen)
                preds.append(m(xb.to(dev)).argmax(-1).cpu().numpy())
        out["inf" if not np.isfinite(snr) else f"{snr:g}"] = macro_f1(yte, np.concatenate(preds))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["indist", "loco", "a2r", "bdis"])
    parser.add_argument("--test-condition", default="N09_M07_F10")
    parser.add_argument("--sizes", default="small,base,large")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap-train", type=int, default=20000)
    parser.add_argument("--cap-test", type=int, default=10000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--aug-clean", action="store_true", help="include clean(inf) in training aug pool")
    parser.add_argument("--clean-only", action="store_true", help="train on CLEAN only")
    args = parser.parse_args()

    set_seed(args.seed)
    global AUG
    if args.clean_only:
        AUG = CLEAN_ONLY_AUGMENT
        print(f"[aug] CLEAN-ONLY training: {AUG}", flush=True)
    elif args.aug_clean:
        AUG = (float("inf"),) + DEFAULT_AUGMENT_SNR
        print(f"[aug] clean included: {AUG}", flush=True)
    else:
        AUG = DEFAULT_AUGMENT_SNR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.scenario + (f"[{args.test_condition}]" if args.scenario == "loco" else "")
    Xtr, ytr, Xv, yv, Xte, yte = materialise(args.scenario, args.test_condition, args.seed)
    Xtr, ytr = cap(Xtr, ytr, args.cap_train, args.seed + 7)
    Xv, yv = cap(Xv, yv, 4000, args.seed + 8)
    Xte, yte = cap(Xte, yte, args.cap_test, args.seed + 9)
    print(f"[{tag}] train={Xtr.shape[0]} val={Xv.shape[0]} test={Xte.shape[0]}", flush=True)

    class_weights = inverse_freq_class_weights(ytr, num_class=3)
    snr_columns = ["inf", "20", "10", "5", "0", "-5", "-10"]
    rows = []
    for size in [part.strip() for part in args.sizes.split(",") if part.strip()]:
        model, best_val, n_parameters = train(size, Xtr, ytr, Xv, yv, args.epochs, args.batch_size, device, class_weights)
        f1 = sweep(model, Xte, yte, device)
        avg = sum(f1[snr] for snr in snr_columns) / len(snr_columns)
        print(
            f"=== CNN-{size} ({n_parameters / 1e3:.0f}K params): "
            f"best_val={best_val:.3f} test 7-SNR avg={avg:.3f} "
            f"(clean={f1['inf']:.3f}, -10={f1['-10']:.3f}) ===",
            flush=True,
        )
        rows.extend((f"CNN-{size}", n_parameters, snr, value) for snr, value in f1.items())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "params", "snr_db", "macro_f1"])
        writer.writerows(rows)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
