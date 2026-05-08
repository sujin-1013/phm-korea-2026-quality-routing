"""Per-class F1 + per-bearing accuracy diagnostic for baseline classifiers.

Inference runs over raw test files (one bearing at a time) so we can attribute
predictions to bearing IDs. The standard SNR list covers the 7-SNR average
(clean ~ -10 dB) plus -20 dB for the full sweep.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import CLASS_NAMES_3, load_signal
from phm_routing.data.windowing import sliding_windows
from phm_routing.utils.metrics import macro_f1

SNRS_AVG: tuple = (None, 20, 10, 5, 0, -5, -10)
SNRS_FULL: tuple = (None, 20, 10, 5, 0, -5, -10, -20)
SAMPLE_RATE = 64_000


def per_class_per_bearing_eval(
    model,
    test_entries: Iterable,
    batch_size: int,
    device: str,
    *,
    window: int,
    stride: int,
    snrs_avg: Iterable = SNRS_AVG,
    snrs_full: Iterable = SNRS_FULL,
    seconds: float = 4.0,
):
    """Run per-bearing inference, return (per_class_f1, per_bearing_acc, sweep, df).

    `df` is the full per-window/per-SNR prediction frame, useful for downstream
    diagnostic scripts.
    """
    snrs_avg = list(snrs_avg)
    snrs_full = list(snrs_full)
    aug_gen = torch.Generator(device="cpu").manual_seed(123)
    rows = []
    n_samples = int(SAMPLE_RATE * seconds)

    for entry in test_entries:
        sig = load_signal(entry.path)[:n_samples].astype(np.float32)
        win = sliding_windows(sig, window, stride)
        y_true = entry.label_idx
        n = win.shape[0]
        for snr in snrs_full:
            x_t = torch.from_numpy(win).float().unsqueeze(1)
            if snr is not None:
                x_t = add_awgn(x_t, snr_db=float(snr), generator=aug_gen)
            x_t = x_t.to(device)
            with torch.no_grad():
                preds = model(x_enc=x_t).logits.argmax(-1).cpu().numpy()
            snr_label = "inf" if snr is None else str(snr)
            for w_idx in range(n):
                rows.append({
                    "bearing_id": entry.bearing_id,
                    "class": CLASS_NAMES_3[y_true],
                    "snr": snr_label,
                    "true": int(y_true),
                    "pred": int(preds[w_idx]),
                })

    df = pd.DataFrame(rows)

    # Per-class F1 averaged across snrs_avg (clean ~ -10 dB)
    f1_per_snr = []
    for snr in snrs_avg:
        snr_label = "inf" if snr is None else str(snr)
        sub = df[df.snr == snr_label]
        f1s = []
        for c in range(3):
            tp = ((sub["true"] == c) & (sub["pred"] == c)).sum()
            fp = ((sub["true"] != c) & (sub["pred"] == c)).sum()
            fn = ((sub["true"] == c) & (sub["pred"] != c)).sum()
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        f1_per_snr.append(f1s)
    avg_f1 = np.mean(f1_per_snr, axis=0)
    per_class_dict = {
        "F1_N": float(avg_f1[0]),
        "F1_IR": float(avg_f1[1]),
        "F1_OR": float(avg_f1[2]),
        "Macro-F1": float(avg_f1.mean()),
    }

    # Per-bearing accuracy averaged across snrs_avg
    per_bearing = []
    for bearing in sorted(df.bearing_id.unique()):
        sb = df[df.bearing_id == bearing]
        cls = sb["class"].iloc[0]
        accs = []
        for snr in snrs_avg:
            snr_label = "inf" if snr is None else str(snr)
            ss = sb[sb.snr == snr_label]
            acc = (ss["true"] == ss["pred"]).mean() if len(ss) else 0.0
            accs.append(acc)
        per_bearing.append({
            "bearing_id": bearing,
            "class": cls,
            "n_per_snr": int(len(sb) / max(len(snrs_avg), 1)),
            "accuracy": float(np.mean(accs)),
        })

    # Full SNR sweep (Macro-F1 per SNR)
    sweep = {}
    for snr in snrs_full:
        snr_label = "inf" if snr is None else str(snr)
        sub = df[df.snr == snr_label]
        sweep[snr_label] = float(macro_f1(sub["true"].values, sub["pred"].values))

    return per_class_dict, per_bearing, sweep, df
