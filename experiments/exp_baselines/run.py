"""Single-tier baselines (MOMENT-S/B/L) with LoRA + head + mixed-SNR aug.

Fair comparison against SNRGate joint training: same data, same aug, same
LoRA budget per tier, but no routing — each tier is trained independently
and evaluated at the 7-SNR sweep (clean..-10 dB) plus -20 dB.

With `--diag`, also runs per-class F1 + per-bearing accuracy on the test set
and saves checkpoints (replaces former `baselines_with_diag.py`).

Pairs with experiments/exp03_snrgate_main/ for the paper's main table:
  | Method            | clean | 20 | 10 | 5 | 0 | -5 | -10 |
  | MOMENT-S only     | ...                                     |
  | MOMENT-B only     | ...                                     |
  | MOMENT-L only     | ...                                     |
  | HardGate routing  | ...                                     |
  | SNRGate (ours)    | ...                                     |
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.noise_injection import snr_sweep_levels
from phm_routing.data.paderborn import (
    CLASS_NAMES_3,
    discover_files,
    inverse_freq_class_weights,
    pu_c1_a2r_split,
)
from phm_routing.data.protocol import NUM_CLASS, STRIDE, WINDOW, materialise
from phm_routing.eval.baseline_diagnostic import per_class_per_bearing_eval
from phm_routing.training.train_baseline import snr_sweep_eval, train_single_tier
from phm_routing.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-bearing", type=int, default=8)
    p.add_argument(
        "--sizes",
        type=str,
        default="small,base,large",
        help="comma-separated tier sizes to train",
    )
    p.add_argument(
        "--snr-levels",
        type=str,
        default=None,
        help="Override sweep SNRs (CSV; e.g. 'inf,0,-20'). Default: full 8-level sweep.",
    )
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument(
        "--diag",
        action="store_true",
        help="Save checkpoints + run per-class F1 / per-bearing accuracy on the test set.",
    )
    args = p.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.diag:
        (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    Xtr, ytr, Xval, yval, Xtest, ytest = materialise(
        args.data_root,
        window=WINDOW,
        stride=STRIDE,
        max_files_per_bearing=args.max_files_per_bearing,
        seed=args.seed,
    )
    print(
        f"[info] train={Xtr.shape[0]} val={Xval.shape[0]} test={Xtest.shape[0]}  "
        f"protocol=PU-C1 A2R 3-class (paper/DATA_PROTOCOL_LOCKED.md)",
        flush=True,
    )

    class_weights = inverse_freq_class_weights(ytr, num_class=NUM_CLASS)
    print(f"[info] class_weights (inverse_freq, classes={CLASS_NAMES_3}): {class_weights}", flush=True)

    # Test entries needed for diagnostic per-bearing analysis
    test_entries = None
    if args.diag:
        entries = discover_files(args.data_root, classes=CLASS_NAMES_3)
        split = pu_c1_a2r_split(
            entries,
            max_files_per_bearing=args.max_files_per_bearing,
            seed=args.seed,
        )
        test_entries = split.test
        print(f"[info] test files: {len(test_entries)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.snr_levels:
        levels = []
        for tok in args.snr_levels.split(","):
            tok = tok.strip().lower()
            levels.append(float("inf") if tok in ("inf", "clean") else float(tok))
        print(f"[info] SNR sweep override: {levels}", flush=True)
    else:
        levels = snr_sweep_levels()  # 8 levels: inf,20,10,5,0,-5,-10,-20 (locked)
    sizes = [s.strip() for s in args.sizes.split(",")]
    augment_snr = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)  # PU-C1 train aug pool

    rows = []
    histories = {}
    diag_summary: dict = {"per_class": {}, "per_bearing": {}, "sweep": {}}
    for size in sizes:
        print(f"\n=== Training MOMENT-{size} only ===", flush=True)
        model, hist = train_single_tier(
            size, Xtr, ytr, Xval, yval,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lora_r=args.lora_r,
            augment_snr=augment_snr,
            device=device,
            num_class=NUM_CLASS,
            class_weights=class_weights,
            lr_lora=args.lr_lora,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
        )
        histories[size] = hist

        if args.diag:
            ckpt_path = args.out_dir / "checkpoints" / f"baseline_{size}.pt"
            torch.save({
                "model": model.state_dict(),
                "history": hist,
                "class_weights": class_weights,
                "num_class": NUM_CLASS,
                "size": size,
            }, ckpt_path)
            print(f"[done] saved {ckpt_path}", flush=True)

            print("  Running diagnostic eval on test set...", flush=True)
            per_class, per_bearing, sweep, df = per_class_per_bearing_eval(
                model, test_entries, args.batch_size, device,
                window=WINDOW, stride=STRIDE,
            )
            diag_summary["per_class"][size] = per_class
            diag_summary["per_bearing"][size] = per_bearing
            diag_summary["sweep"][size] = sweep
            df.to_csv(args.out_dir / f"diag_baseline_{size}_predictions.csv", index=False)
            print(f"  per_class: {per_class}", flush=True)
            for label, score in sweep.items():
                rows.append({"method": f"MOMENT-{size}-only", "snr_db": label, "macro_f1": score})
        else:
            # SNR sweep on TEST set (PU-C1 real-damage bearings — A2R generalization).
            f1 = snr_sweep_eval(model, Xtest, ytest, levels, args.batch_size, device)
            print(f"[{size}] test sweep: {f1}", flush=True)
            for label, score in f1.items():
                rows.append({"method": f"MOMENT-{size}-only", "snr_db": label, "macro_f1": score})

        # free GPU memory before next tier
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.out_dir / "baselines_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "snr_db", "macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] {csv_path}", flush=True)

    summary = {
        "protocol": "PU-C1 A2R 3-class (paper/DATA_PROTOCOL_LOCKED.md)",
        "num_class": NUM_CLASS,
        "class_names": list(CLASS_NAMES_3),
        "class_weights": class_weights,
        "n_train_windows": int(Xtr.shape[0]),
        "n_val_windows": int(Xval.shape[0]),
        "n_test_windows": int(Xtest.shape[0]),
        "sweep_eval_set": "test (PU-C1 real-damage bearings)",
        "histories": {s: h["val_f1"] for s, h in histories.items()},
        "best_epoch": {s: h.get("best_epoch", -1) for s, h in histories.items()},
        "best_val_f1": {s: h.get("best_val_f1", float("nan")) for s, h in histories.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.diag:
        (args.out_dir / "diag_baselines_summary.json").write_text(
            json.dumps(diag_summary, indent=2)
        )
        print(f"[done] diag summary -> {args.out_dir / 'diag_baselines_summary.json'}", flush=True)

        print("\n=== Per-class F1 (7-SNR avg) ===")
        print(f"{'size':<8} {'F1_N':<8} {'F1_IR':<8} {'F1_OR':<8} {'Macro':<8}")
        for size in sizes:
            pc = diag_summary["per_class"][size]
            print(f"{size:<8} {pc['F1_N']:<8.3f} {pc['F1_IR']:<8.3f} {pc['F1_OR']:<8.3f} {pc['Macro-F1']:<8.3f}")

        print("\n=== Per-bearing accuracy (7-SNR avg) ===")
        bearings = [b["bearing_id"] for b in diag_summary["per_bearing"][sizes[0]]]
        print(f"{'bearing':<10} {'class':<5} " + " ".join(f"{s:<8}" for s in sizes))
        for i, bearing in enumerate(bearings):
            cls = diag_summary["per_bearing"][sizes[0]][i]["class"]
            accs = [f"{diag_summary['per_bearing'][s][i]['accuracy']:.3f}" for s in sizes]
            print(f"{bearing:<10} {cls:<5} " + " ".join(f"{a:<8}" for a in accs))


if __name__ == "__main__":
    main()
