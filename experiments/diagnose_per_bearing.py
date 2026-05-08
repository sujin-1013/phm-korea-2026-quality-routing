"""Diagnostic analysis on existing per_sample_predictions.csv.

Produces:
  A. Confusion matrix per failing bearing (KA15, KA22) and contrast bearings (KA16, KA30)
  B. Per-bearing routing distribution (SNRGate)
  C. Per-SNR per-bearing accuracy matrix (heatmap-ready)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "experiments" / "exp03_snrgate_main" / "per_sample_predictions.csv"
OUT_DIR = ROOT / "experiments" / "exp03_snrgate_main"

CLASS_NAMES = ("N", "IR", "OR")
TIER_NAMES = ("S", "B", "L")
SNR_ORDER = ["inf", "20", "10", "5", "0", "-5", "-10", "-20"]
SNR_AVG_7 = ["inf", "20", "10", "5", "0", "-5", "-10"]


def main() -> None:
    print(f"[info] loading {CSV}", flush=True)
    df = pd.read_csv(CSV, dtype={"snr": str})
    print(f"[info] {len(df):,} rows", flush=True)

    sg = df[df.method == "SNRGate"].copy()
    hg = df[df.method == "HardGate"].copy()

    # ===========================================================
    # A. Confusion matrix per bearing of interest
    # ===========================================================
    print("\n=== A. Confusion matrix per bearing (SNRGate, 7-SNR aggregated) ===")
    bearings_of_interest = ["KA15", "KA22", "KA16", "KA30", "K001"]
    rows_a = []
    for bearing in bearings_of_interest:
        sub = sg[(sg.bearing_id == bearing) & (sg.snr.isin(SNR_AVG_7))]
        if len(sub) == 0:
            continue
        y_true = sub["true"].values
        y_pred = sub["pred"].values
        true_class = sub["class"].iloc[0]
        n = len(sub)
        # fraction predicted as each class
        pred_dist = {CLASS_NAMES[c]: float((y_pred == c).mean()) for c in range(3)}
        rows_a.append({
            "bearing_id": bearing,
            "true_class": true_class,
            "n_predictions": n,
            "pred_N": pred_dist["N"],
            "pred_IR": pred_dist["IR"],
            "pred_OR": pred_dist["OR"],
            "accuracy (correct = pred == true)": float((y_pred == y_true).mean()),
        })
    confmat_df = pd.DataFrame(rows_a)
    print(confmat_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    confmat_df.to_csv(OUT_DIR / "diag_A_confusion_per_bearing.csv", index=False)
    print(f"[done] {OUT_DIR / 'diag_A_confusion_per_bearing.csv'}")

    # ===========================================================
    # B. Per-bearing routing distribution (SNRGate)
    # ===========================================================
    print("\n=== B. Per-bearing routing distribution (SNRGate, 7-SNR) ===")
    rows_b = []
    for bearing in sorted(sg.bearing_id.unique()):
        sub = sg[(sg.bearing_id == bearing) & (sg.snr.isin(SNR_AVG_7))]
        if len(sub) == 0:
            continue
        true_class = sub["class"].iloc[0]
        tier_dist = {TIER_NAMES[k]: float((sub.tier == k).mean()) for k in range(3)}
        accuracy = float((sub.pred == sub.true).mean())
        rows_b.append({
            "bearing_id": bearing,
            "true_class": true_class,
            "share_S": tier_dist["S"],
            "share_B": tier_dist["B"],
            "share_L": tier_dist["L"],
            "accuracy": accuracy,
        })
    routing_df = pd.DataFrame(rows_b)
    print(routing_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    routing_df.to_csv(OUT_DIR / "diag_B_routing_per_bearing.csv", index=False)
    print(f"[done] {OUT_DIR / 'diag_B_routing_per_bearing.csv'}")

    # ===========================================================
    # C. Per-SNR per-bearing accuracy matrix
    # ===========================================================
    print("\n=== C. Per-SNR per-bearing accuracy (SNRGate) ===")
    rows_c = []
    for bearing in sorted(sg.bearing_id.unique()):
        sub = sg[sg.bearing_id == bearing]
        if len(sub) == 0:
            continue
        true_class = sub["class"].iloc[0]
        row = {"bearing_id": bearing, "true_class": true_class}
        for snr in SNR_ORDER:
            ss = sub[sub.snr == snr]
            row[snr] = float((ss.pred == ss.true).mean()) if len(ss) else float("nan")
        rows_c.append(row)
    matrix_df = pd.DataFrame(rows_c)
    print(matrix_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    matrix_df.to_csv(OUT_DIR / "diag_C_per_snr_per_bearing.csv", index=False)
    print(f"[done] {OUT_DIR / 'diag_C_per_snr_per_bearing.csv'}")

    # ===========================================================
    # Summary takeaways
    # ===========================================================
    print("\n=== Diagnostic summary ===")
    # KA15/KA22 misclassification mode
    for bearing in ("KA15", "KA22"):
        sub = sg[(sg.bearing_id == bearing) & (sg.snr.isin(SNR_AVG_7))]
        pred_n = (sub.pred == 0).mean()
        pred_ir = (sub.pred == 1).mean()
        pred_or = (sub.pred == 2).mean()
        tier_dist = {TIER_NAMES[k]: (sub.tier == k).mean() for k in range(3)}
        print(f"  {bearing} (true=OR): predicted as N={pred_n:.1%}, IR={pred_ir:.1%}, OR={pred_or:.1%}; "
              f"routing S={tier_dist['S']:.1%} B={tier_dist['B']:.1%} L={tier_dist['L']:.1%}")
    # Clean-only check (does it fail even at clean?)
    print()
    for bearing in ("KA15", "KA22"):
        sub_clean = sg[(sg.bearing_id == bearing) & (sg.snr == "inf")]
        if len(sub_clean):
            acc_clean = (sub_clean.pred == sub_clean.true).mean()
            print(f"  {bearing} clean-only accuracy: {acc_clean:.3f}  (n={len(sub_clean)})")


if __name__ == "__main__":
    main()
