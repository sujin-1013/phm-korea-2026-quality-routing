"""Final PHM Korea 2026 quality-routing experiment.

This is the cleaned entrypoint for the final method:

    Quality-gate routing over {base=133K, x300=331K, large=524K}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from phm_routing.eval.mixed_snr import (
    add_mixed_awgn,
    estimate_noise_score,
    predict_all_models,
    select_capacity_routes,
)
from phm_routing.models.cnn1d import FINAL_CAPACITY_LADDER, ModelSize
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.training.train_cnn import (
    CLEAN_ONLY_AUGMENT,
    cap_arrays,
    load_cnn_checkpoint,
    materialise_scenario,
    save_cnn_checkpoint,
    train_cnn_classifier,
)
from phm_routing.training.train_estimator import EstimatorTrainConfig, train_estimator
from phm_routing.utils.metrics import macro_f1
from phm_routing.utils.seed import set_seed


DEFAULT_OUT_DIR = ROOT / "experiments" / "quality_routing" / "results"
DEFAULT_CKPT_DIR = ROOT / "experiments" / "quality_routing" / "checkpoints"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xv, yv, Xte, yte = materialise_scenario(
        args.data_root,
        scenario=args.scenario,
        test_condition=args.test_condition,
        seed=args.seed,
    )
    Xtr, ytr = cap_arrays(Xtr, ytr, args.cap_train, args.seed + 7)
    Xv, yv = cap_arrays(Xv, yv, args.cap_val, args.seed + 8)
    Xte, yte = cap_arrays(Xte, yte, args.cap_test, args.seed + 9)
    print(
        f"[quality-routing seed{args.seed}] scenario={args.scenario} "
        f"train={Xtr.shape[0]} val={Xv.shape[0]} test={Xte.shape[0]} device={device}",
        flush=True,
    )

    sizes = parse_sizes(args.sizes)
    ladder = parse_sizes(args.ladder)
    missing_ladder = set(ladder) - set(sizes)
    if missing_ladder:
        raise ValueError(f"ladder tiers must be included in --sizes; missing: {sorted(missing_ladder)}")
    if args.reference_tier not in sizes:
        raise ValueError(f"--reference-tier must be included in --sizes, got {args.reference_tier!r}")
    models: dict[str, torch.nn.Module] = {}
    parameter_counts: dict[str, int] = {}
    for size in sizes:
        ckpt_path = args.ckpt_dir / f"cnn_{size}_s{args.seed}.pt"
        if ckpt_path.exists() and not args.retrain:
            result = load_cnn_checkpoint(ckpt_path, size, device=device)
            print(f"  loaded {size} ({result.n_parameters / 1e3:.0f}K) best_val={result.best_val_f1:.3f}", flush=True)
        else:
            result = train_cnn_classifier(
                size,
                Xtr,
                ytr,
                Xv,
                yv,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                augment_snr=CLEAN_ONLY_AUGMENT,
            )
            save_cnn_checkpoint(ckpt_path, result, size)
            print(f"  trained+saved {size} ({result.n_parameters / 1e3:.0f}K) best_val={result.best_val_f1:.3f}", flush=True)
        models[size] = result.model
        parameter_counts[size] = result.n_parameters

    estimator = get_noise_estimator(
        args.estimator_ckpt,
        Xtr,
        Xv,
        device=device,
        epochs=args.estimator_epochs,
        batch_size=args.estimator_batch_size,
        retrain=args.retrain_estimator,
    )
    val_noisy, _ = add_mixed_awgn(Xv, seed=args.val_noise_seed)
    test_noisy, _ = add_mixed_awgn(Xte, seed=args.test_noise_seed)
    val_predictions = predict_all_models(models, val_noisy, device=device, batch_size=args.eval_batch_size)
    test_predictions = predict_all_models(models, test_noisy, device=device, batch_size=args.eval_batch_size)
    val_score = estimate_noise_score(estimator, val_noisy, device=device, batch_size=args.eval_batch_size)
    test_score = estimate_noise_score(estimator, test_noisy, device=device, batch_size=args.eval_batch_size)

    rows: list[tuple[str, float, float, str]] = []
    for size in sizes:
        rows.append(
            (
                f"single_{size}",
                macro_f1(yte, test_predictions[size]),
                parameter_counts[size] / 1e3,
                "",
            )
        )

    route_results = select_capacity_routes(
        y_val=yv,
        val_predictions=val_predictions,
        val_score=val_score,
        y_test=yte,
        test_predictions=test_predictions,
        test_score=test_score,
        tiers=ladder,
        parameter_counts=parameter_counts,
        estimator_overhead_k=args.estimator_overhead_k,
        reference_tier=args.reference_tier,
        val_margin=args.val_margin,
        target_cost_k=args.target_cost_k,
    )
    for name in ("iso", "accmatch", "fc"):
        result = route_results[name]
        shares = "/".join(f"{share:.3f}" for share in result.shares)
        rows.append((f"{args.route_prefix}_{name}", result.test_f1, result.test_cost_k, shares))
        print(
            f"  {args.route_prefix}_{name:9s} F1={result.test_f1:.4f} @{result.test_cost_k:.0f}K "
            f"share={result.shares} tau={result.thresholds}",
            flush=True,
        )

    out_path = args.out_dir / f"quality_routing_s{args.seed}.csv"
    write_rows(out_path, rows)
    print(f"  saved {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--estimator-ckpt", type=Path, default=DEFAULT_CKPT_DIR / "noise_est_7.6k.pt")
    parser.add_argument("--scenario", choices=["indist", "bdis", "loco", "a2r"], default="indist")
    parser.add_argument("--test-condition", default="N09_M07_F10")
    parser.add_argument("--sizes", default="small,mid,base,x300,large")
    parser.add_argument("--ladder", default=",".join(FINAL_CAPACITY_LADDER))
    parser.add_argument("--route-prefix", default="bxl")
    parser.add_argument("--reference-tier", default="large")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-noise-seed", type=int, default=111)
    parser.add_argument("--test-noise-seed", type=int, default=222)
    parser.add_argument("--cap-train", type=int, default=20_000)
    parser.add_argument("--cap-val", type=int, default=4_000)
    parser.add_argument("--cap-test", type=int, default=12_000)
    parser.add_argument("--estimator-overhead-k", type=float, default=7.6)
    parser.add_argument("--estimator-epochs", type=int, default=30)
    parser.add_argument("--estimator-batch-size", type=int, default=64)
    parser.add_argument("--val-margin", type=float, default=0.01)
    parser.add_argument("--target-cost-k", type=float, default=250.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retrain-estimator", action="store_true")
    return parser.parse_args()


def parse_sizes(value: str) -> tuple[ModelSize, ...]:
    sizes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("size list cannot be empty")
    return sizes  # type: ignore[return-value]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_noise_estimator(path: Path, *, device: str) -> NoiseEstimator1D:
    if not path.exists():
        raise FileNotFoundError(f"noise estimator checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        weights = state.get("model") or state.get("state_dict") or state
    else:
        weights = state
    model = NoiseEstimator1D(channels=(8, 16, 16, 32)).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model


def get_noise_estimator(
    path: Path,
    X_train,
    X_val,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    retrain: bool,
) -> NoiseEstimator1D:
    if path.exists() and not retrain:
        print(f"  loaded estimator {path}", flush=True)
        return load_noise_estimator(path, device=device)

    cfg = EstimatorTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        channels=(8, 16, 16, 32),
        kernel_size=7,
    )
    estimator, history = train_estimator(X_train, X_val, cfg)
    estimator.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": estimator.state_dict(),
            "npar": sum(p.numel() for p in estimator.parameters()),
            "history": history,
            "channels": cfg.channels,
            "kernel_size": cfg.kernel_size,
        },
        path,
    )
    print(f"  trained+saved estimator {path}", flush=True)
    return estimator


def write_rows(path: Path, rows: list[tuple[str, float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "f1", "cost_K", "shares"])
        for method, f1, cost_k, shares in rows:
            writer.writerow([method, f"{f1:.4f}", f"{cost_k:.1f}", shares])


if __name__ == "__main__":
    main()
