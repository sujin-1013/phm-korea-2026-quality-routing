"""Re-evaluate SNRGate (and HardGate sibling) checkpoint with per-class F1
and per-bearing accuracy breakdown.

Reuses the saved snrgate.pt checkpoint — no re-training. Computes:
  1. Per-class F1 (N/IR/OR) per method (SNRGate, HardGate)
  2. Per-bearing accuracy (10 test bearings) per method

All metrics averaged across the 7-SNR window (clean ~ -10 dB), matching
the paper's reported `평균` column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import (
    CLASS_NAMES_3,
    CLASS_TO_IDX_3,
    discover_files,
    load_signal,
    pu_c1_a2r_split,
)
from phm_routing.data.windowing import sliding_windows
from phm_routing.models.router import HardGateRouter, RouterConfig
from phm_routing.training.train_routing import RoutingTrainConfig, build_classifier

DATA_ROOT = ROOT / "data" / "raw" / "paderborn_pu"
CKPT_PATH = ROOT / "experiments" / "exp03_snrgate_main" / "checkpoints" / "snrgate.pt"
OUT_DIR = ROOT / "experiments" / "exp03_snrgate_main"
SNRS_AVG = [None, 20, 10, 5, 0, -5, -10]  # 7-SNR window for averaging
WINDOW = 2048
STRIDE = 2048


def f1_score_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_class: int = 3) -> np.ndarray:
    """Per-class F1 = 2·TP / (2·TP + FP + FN)."""
    f1 = np.zeros(num_class, dtype=np.float64)
    for c in range(num_class):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        denom = 2 * tp + fp + fn
        f1[c] = (2 * tp / denom) if denom > 0 else 0.0
    return f1


def main() -> None:
    print("[info] loading classifier from ckpt...", flush=True)
    cfg = RoutingTrainConfig(num_class=3, train_estimator=False)
    classifier, _ = build_classifier(cfg)
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    classifier.estimator.load_state_dict(ckpt["estimator"])
    classifier.gate.load_state_dict(ckpt["gate"])
    for i, tier in enumerate(classifier.tiers):
        tier.load_state_dict(ckpt["tiers"][i])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(device).eval()
    print(f"[info] device={device}, gate centers={classifier.gate.centers.detach().cpu().tolist()}", flush=True)

    hard_router = HardGateRouter(RouterConfig())

    print("[info] discovering test split...", flush=True)
    entries = discover_files(DATA_ROOT, classes=CLASS_NAMES_3)
    split = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)
    print(f"[info] test files: {len(split.test)}", flush=True)

    rows = []
    aug_gen = torch.Generator(device="cpu").manual_seed(123)

    for ei, entry in enumerate(split.test):
        try:
            sig = load_signal(entry.path)[: 64_000 * 4].astype(np.float32)
        except Exception as exc:
            print(f"[warn] skip {entry.path.name}: {exc}", flush=True)
            continue
        win = sliding_windows(sig, WINDOW, STRIDE)  # (n_win, 2048)
        y_true = entry.label_idx
        n = win.shape[0]

        for snr in SNRS_AVG:
            x_t = torch.from_numpy(win).float().unsqueeze(1)  # (n, 1, 2048)
            if snr is not None:
                x_t = add_awgn(x_t, snr_db=float(snr), generator=aug_gen)
            x_t = x_t.to(device)

            with torch.no_grad():
                # SNRGate — learned gate
                logits_sg, tier_sg = classifier.inference_forward(x_t)
                pred_sg = logits_sg.argmax(-1).cpu().numpy()
                # HardGate — rule-based gate over the same trained tiers
                noise = classifier.estimator(x_t)
                tier_hg = hard_router(noise)
                logits_hg = torch.empty(x_t.shape[0], 3, device=x_t.device)
                for k, tier_model in enumerate(classifier.tiers):
                    mask = tier_hg == k
                    if mask.any():
                        logits_hg[mask] = tier_model(x_enc=x_t[mask]).logits
                pred_hg = logits_hg.argmax(-1).cpu().numpy()

            snr_label = "inf" if snr is None else str(snr)
            for w_idx in range(n):
                rows.append({
                    "bearing_id": entry.bearing_id,
                    "class": CLASS_NAMES_3[y_true],
                    "trial": entry.trial,
                    "operating_condition": entry.operating_condition,
                    "snr": snr_label,
                    "method": "SNRGate",
                    "true": int(y_true),
                    "pred": int(pred_sg[w_idx]),
                    "tier": int(tier_sg[w_idx].item()),
                })
                rows.append({
                    "bearing_id": entry.bearing_id,
                    "class": CLASS_NAMES_3[y_true],
                    "trial": entry.trial,
                    "operating_condition": entry.operating_condition,
                    "snr": snr_label,
                    "method": "HardGate",
                    "true": int(y_true),
                    "pred": int(pred_hg[w_idx]),
                    "tier": int(tier_hg[w_idx].item()),
                })
        print(f"  [{ei+1}/{len(split.test)}] {entry.bearing_id} {entry.path.name} ({n} windows)", flush=True)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "per_sample_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[done] saved {csv_path} ({len(df):,} rows)", flush=True)

    # ---------- Per-class F1 ----------
    print("\n=== Per-class F1 (averaged across 7 SNRs: clean, 20, 10, 5, 0, -5, -10) ===")
    pc_rows = []
    for method in ["SNRGate", "HardGate"]:
        sub = df[df.method == method]
        per_snr_f1 = []
        for snr in SNRS_AVG:
            snr_label = "inf" if snr is None else str(snr)
            s = sub[sub.snr == snr_label]
            f1 = f1_score_per_class(s.true.values, s.pred.values)
            per_snr_f1.append(f1)
        avg_f1 = np.mean(per_snr_f1, axis=0)
        macro = avg_f1.mean()
        pc_rows.append({
            "method": method,
            "F1_N": avg_f1[0],
            "F1_IR": avg_f1[1],
            "F1_OR": avg_f1[2],
            "Macro-F1": macro,
        })
    pc_df = pd.DataFrame(pc_rows)
    print(pc_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pc_df.to_csv(OUT_DIR / "per_class_f1.csv", index=False)
    print(f"[done] saved {OUT_DIR / 'per_class_f1.csv'}", flush=True)

    # ---------- Per-bearing accuracy ----------
    print("\n=== Per-bearing accuracy (averaged across 7 SNRs) ===")
    pb_rows = []
    for method in ["SNRGate", "HardGate"]:
        sub = df[df.method == method]
        for bearing in sorted(sub.bearing_id.unique()):
            sb = sub[sub.bearing_id == bearing]
            cls = sb["class"].iloc[0]
            per_snr_acc = []
            for snr in SNRS_AVG:
                snr_label = "inf" if snr is None else str(snr)
                ss = sb[sb.snr == snr_label]
                acc = (ss.true == ss.pred).mean() if len(ss) else 0.0
                per_snr_acc.append(acc)
            avg_acc = float(np.mean(per_snr_acc))
            n_per_snr = int(len(sb) / len(SNRS_AVG))
            pb_rows.append({
                "method": method,
                "bearing_id": bearing,
                "class": cls,
                "n_windows": n_per_snr,
                "accuracy": avg_acc,
            })
    pb_df = pd.DataFrame(pb_rows)
    print(pb_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pb_df.to_csv(OUT_DIR / "per_bearing_accuracy.csv", index=False)
    print(f"[done] saved {OUT_DIR / 'per_bearing_accuracy.csv'}", flush=True)


if __name__ == "__main__":
    main()
