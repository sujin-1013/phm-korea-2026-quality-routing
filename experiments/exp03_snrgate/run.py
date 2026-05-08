"""SNRGate joint training — main paper method.

Runs ``train_routing`` (estimator + LearnableSNRGate + 3 LoRA-MOMENT tiers +
per-tier heads) on **Paderborn-PU PU-C1 A2R 3-class data** (locked protocol
defined in ``paper/DATA_PROTOCOL_LOCKED.md``):

  - 17 train bearings (artificial damage) + within-train val (KA09/KI09 + K002 80/20 split)
  - 10 test bearings (real damage, A2R generalization)

Then:
  - dumps the (best-on-val) trained classifier checkpoint
  - sweeps SNR ∈ {inf, 20, 10, 5, 0, -5, -10, -20} dB **on the test set**
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
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.noise_injection import snr_sweep_levels
from phm_routing.data.paderborn import CLASS_NAMES_3, inverse_freq_class_weights
from phm_routing.data.protocol import NUM_CLASS, STRIDE, WINDOW, materialise
from phm_routing.eval.routing_sweep import routing_snr_sweep
from phm_routing.models.router import GateConfig
from phm_routing.training.train_routing import RoutingTrainConfig, train_routing
from phm_routing.utils.seed import set_seed


def snr_sweep_eval(
    classifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    snr_levels: list[float],
    batch_size: int,
    device: str,
) -> tuple[dict, dict]:
    """Hard-gate inference at each SNR. Returns (per_snr_f1, per_snr_routing)."""
    classifier.eval()

    @torch.no_grad()
    def _infer(xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return classifier.inference_forward(xb)

    return routing_snr_sweep(_infer, X_val, y_val, snr_levels, batch_size, device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--estimator-ckpt", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--max-files-per-bearing", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--lr-lora", type=float, default=1e-3)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-gate", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--snr-levels",
        type=str,
        default=None,
        help="Override sweep SNRs (CSV; e.g. 'inf,0,-20'). Default: full 8-level sweep.",
    )
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 2
        args.max_files_per_bearing = 4

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    Xtr, ytr, Xval, yval, Xtest, ytest = materialise(
        args.data_root,
        window=args.window,
        stride=args.stride,
        max_files_per_bearing=args.max_files_per_bearing,
        seed=args.seed,
    )
    print(
        f"[info] train={Xtr.shape[0]} val={Xval.shape[0]} test={Xtest.shape[0]} "
        f"window={args.window} stride={args.stride} "
        f"max_files_per_bearing={args.max_files_per_bearing}",
        flush=True,
    )

    # Inverse-frequency class weights for weighted CE (PU-C1 imbalance handling).
    class_weights = inverse_freq_class_weights(ytr, num_class=NUM_CLASS)
    print(f"[info] class_weights (inverse_freq, classes={CLASS_NAMES_3}): {class_weights}", flush=True)

    cfg = RoutingTrainConfig(
        estimator_ckpt=args.estimator_ckpt,
        train_estimator=args.estimator_ckpt is None,
        num_class=NUM_CLASS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        lr_gate=args.lr_gate,
        weight_decay=args.weight_decay,
        gate=GateConfig(anneal_epochs=args.epochs),  # anneal T over the full training horizon
        class_weights=class_weights,
    )
    classifier, history = train_routing(Xtr, ytr, Xval, yval, cfg)

    ckpt = args.out_dir / "checkpoints" / "snrgate.pt"
    torch.save(
        {
            "estimator": classifier.estimator.state_dict(),
            "gate": classifier.gate.state_dict(),
            "tiers": [m.state_dict() for m in classifier.tiers],
            "history": history,
            "num_class": NUM_CLASS,
            "class_names": list(CLASS_NAMES_3),
            "class_weights": class_weights,
        },
        ckpt,
    )
    print(f"[done] saved {ckpt}", flush=True)

    # SNR sweep on TEST set (PU-C1 real-damage bearings — A2R generalization).
    if args.snr_levels:
        levels = []
        for tok in args.snr_levels.split(","):
            tok = tok.strip().lower()
            levels.append(float("inf") if tok in ("inf", "clean") else float(tok))
        print(f"[info] SNR sweep override: {levels}", flush=True)
    else:
        levels = snr_sweep_levels()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    f1, routing = snr_sweep_eval(classifier, Xtest, ytest, levels, args.batch_size, device)

    sweep_path = args.out_dir / "snr_sweep.csv"
    with sweep_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "macro_f1", "share_S", "share_B", "share_L"])
        for label, score in f1.items():
            r = routing[label]
            w.writerow([label, f"{score:.4f}", f"{r['S']:.3f}", f"{r['B']:.3f}", f"{r['L']:.3f}"])
    print(f"[done] snr sweep -> {sweep_path}", flush=True)

    summary_path = args.out_dir / "summary.json"
    summary = {
        "protocol": "PU-C1 A2R 3-class (paper/DATA_PROTOCOL_LOCKED.md)",
        "num_class": NUM_CLASS,
        "class_names": list(CLASS_NAMES_3),
        "class_weights": class_weights,
        "n_train_windows": int(Xtr.shape[0]),
        "n_val_windows": int(Xval.shape[0]),
        "n_test_windows": int(Xtest.shape[0]),
        "final_centers": classifier.gate.centers.detach().cpu().tolist(),
        "final_T": float(classifier.gate.T),
        "val_f1_history": history["val_f1"],
        "T_schedule": history["T"],
        "lora_r": args.lora_r,
        # best-on-val info — sweep above was run with these weights
        "best_epoch": history.get("best_epoch", -1),
        "best_val_f1": history.get("best_val_f1", float("nan")),
        "sweep_eval_set": "test (PU-C1 real-damage bearings)",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
