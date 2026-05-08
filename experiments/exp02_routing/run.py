"""HardGateRouter ablation — same trained tiers + estimator as SNRGate, but
the learnable gate is replaced with the rule-based ``HardGateRouter``
(Notion §9: τ₁=0.33, τ₂=0.66).

This is the canonical "hard vs learnable gate" ablation paired with
``experiments/exp03_snrgate_main`` — both use the same MOMENT-S/B/L
weights so the only differing factor is the routing rule.

Loads:
  experiments/exp03_snrgate_main/checkpoints/snrgate.pt
    (must be produced first by exp03_snrgate/run.py)

Outputs:
  experiments/exp02_routing/snr_sweep.csv     per-SNR macro-F1 + S/B/L share
  experiments/exp02_routing/summary.json      router config + summary
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

from phm_routing.data.noise_injection import snr_sweep_levels  # noqa: E402
from phm_routing.data.paderborn import CLASS_NAMES_3  # noqa: E402
from phm_routing.data.protocol import NUM_CLASS, STRIDE, WINDOW, materialise  # noqa: E402
from phm_routing.eval.routing_sweep import routing_snr_sweep  # noqa: E402
from phm_routing.models.lora import inject_lora  # noqa: E402
from phm_routing.models.moment_wrapper import MomentConfig, build_moment_classifier  # noqa: E402
from phm_routing.models.noise_estimator import NoiseEstimator1D  # noqa: E402
from phm_routing.models.router import HardGateRouter, RouterConfig  # noqa: E402
from phm_routing.utils.seed import set_seed  # noqa: E402


_TIER_SIZES = ("small", "base", "large")


def build_tiers(num_class: int, lora_r: int, lora_alpha: float = 16.0):
    """Build the same 3 MOMENT tiers + LoRA injection as SNRGateClassifier."""
    tiers = []
    for size in _TIER_SIZES:
        m = build_moment_classifier(
            MomentConfig(size=size, n_channels=1, num_class=num_class, freeze_encoder=True)
        )
        inject_lora(m, r=lora_r, alpha=lora_alpha)
        tiers.append(m)
    return tiers


def load_snrgate_ckpt(ckpt_path: Path, num_class: int, lora_r: int, device: str):
    """Restore estimator + 3 tiers from a SNRGate checkpoint.

    Note: the checkpoint also stores 'gate' state but we discard it — the
    point of this ablation is to swap the learnable gate for HardGateRouter.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    estimator = NoiseEstimator1D()
    estimator.load_state_dict(state["estimator"])
    estimator.eval().to(device)

    tiers = build_tiers(num_class=num_class, lora_r=lora_r)
    if len(state["tiers"]) != len(tiers):
        raise ValueError(f"checkpoint has {len(state['tiers'])} tiers, expected {len(tiers)}")
    for tier_model, tier_state in zip(tiers, state["tiers"]):
        tier_model.load_state_dict(tier_state)
        tier_model.eval().to(device)

    return estimator, tiers


@torch.no_grad()
def hardgate_inference(
    estimator: torch.nn.Module,
    tiers,
    router: HardGateRouter,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run HardGateRouter routing → exactly one tier per sample.

    Mirrors SNRGateClassifier.inference_forward but with rule-based router.
    """
    noise = estimator(x)
    tier_idx = router(noise)
    out_dim = tiers[0](x_enc=x[:1]).logits.shape[-1]
    out = torch.empty(x.shape[0], out_dim, device=x.device)
    for k, tier_model in enumerate(tiers):
        mask = tier_idx == k
        if mask.any():
            out[mask] = tier_model(x_enc=x[mask]).logits
    return out, tier_idx


def snr_sweep_eval(estimator, tiers, router, X_val, y_val, snr_levels, batch_size, device):
    return routing_snr_sweep(
        lambda xb: hardgate_inference(estimator, tiers, router, xb),
        X_val, y_val, snr_levels, batch_size, device,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "experiments" / "exp03_snrgate_main" / "checkpoints" / "snrgate.pt",
        help="SNRGate checkpoint to source the trained tiers + estimator from.",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    p.add_argument("--num-class", type=int, default=NUM_CLASS)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-bearing", type=int, default=4,
                   help="Must match the value used to train the SNRGate ckpt for fair comparison.")
    p.add_argument("--tau1", type=float, default=0.33)
    p.add_argument("--tau2", type=float, default=0.66)
    args = p.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"SNRGate ckpt not found: {args.ckpt}\n"
                                "Run exp03_snrgate/run.py first.")

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Identical splits to exp03 (same materialise + seed). Sweep on TEST per
    # paper/DATA_PROTOCOL_LOCKED.md (PU-C1 A2R real-damage bearings).
    _Xtr, _ytr, _Xval, _yval, Xtest, ytest = materialise(
        args.data_root,
        window=WINDOW,
        stride=STRIDE,
        max_files_per_bearing=args.max_files_per_bearing,
        seed=args.seed,
    )
    print(f"[info] test={Xtest.shape[0]} window={WINDOW} stride={STRIDE} "
          f"max_files_per_bearing={args.max_files_per_bearing}", flush=True)

    print(f"[info] loading SNRGate tiers + estimator from {args.ckpt}", flush=True)
    estimator, tiers = load_snrgate_ckpt(
        args.ckpt, num_class=args.num_class, lora_r=args.lora_r, device=device
    )
    router = HardGateRouter(RouterConfig(tau1=args.tau1, tau2=args.tau2))
    print(f"[info] router: HardGateRouter(tau1={args.tau1}, tau2={args.tau2})", flush=True)

    levels = snr_sweep_levels()
    f1, routing = snr_sweep_eval(
        estimator, tiers, router, Xtest, ytest, levels, args.batch_size, device
    )

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
        "method": "HardGateRouter (rule-based ablation of SNRGate)",
        "source_ckpt": str(args.ckpt),
        "router": {"type": "HardGateRouter", "tau1": args.tau1, "tau2": args.tau2},
        "lora_r": args.lora_r,
        "max_files_per_bearing": args.max_files_per_bearing,
        "f1_per_snr": f1,
        "routing_per_snr": routing,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
