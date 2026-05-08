"""Phase 1 sanity check — MOMENT-S vs MOMENT-L linear probe across SNR sweep.

Decision gate per the plan §1.2:
    size effect ≥ 5%p (any SNR)            → ✅ proceed
    size effect < 5%p but L > S in low-SNR → △  pivot framing to low-SNR
    L ≈ S everywhere                       → ❌ fallback (negative-result paper)

Outputs ``experiments/exp00_sanity/results.csv`` with one row per
(seed, fold, SNR, model_size).
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.noise_injection import add_awgn, snr_sweep_levels
from phm_routing.data.paderborn import (
    CLASS_NAMES,
    SyntheticConfig,
    discover_files,
    load_signal,
    stratified_group_kfold_indices,
    synthesize,
)
from phm_routing.data.windowing import sliding_windows
from phm_routing.models.moment_wrapper import MomentConfig, build_moment_classifier
from phm_routing.utils.metrics import accuracy, macro_f1
from phm_routing.utils.seed import set_seed


DEFAULT_SEEDS = (42, 1234, 9999)
DEFAULT_SIZES = ("small", "large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def materialise_dataset(
    data_root: Path,
    *,
    max_seconds_per_file: float | None = 4.0,
    max_files_per_bearing: int | None = None,
    seed: int = 0,
):
    """Load full signals and group ids from disk, or synthesise if data missing."""
    entries = discover_files(data_root)
    if entries:
        if max_files_per_bearing is not None:
            from collections import defaultdict
            rng = np.random.default_rng(seed)
            per_bearing: dict[str, list] = defaultdict(list)
            for e in entries:
                per_bearing[e.bearing_id].append(e)
            kept: list = []
            for bid, lst in per_bearing.items():
                idx = rng.permutation(len(lst))[:max_files_per_bearing]
                kept.extend(lst[i] for i in idx)
            entries = kept

        signals, labels, groups = [], [], []
        n_skipped = 0
        for e in entries:
            try:
                sig = load_signal(e.path)
            except Exception as exc:
                n_skipped += 1
                print(f"[warn] skip {e.path.name}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                continue
            if max_seconds_per_file is not None:
                sig = sig[: int(64_000 * max_seconds_per_file)]
            signals.append(sig.astype(np.float32))
            labels.append(e.label_idx)
            groups.append(e.bearing_id)
        if n_skipped:
            print(f"[info] skipped {n_skipped} corrupt .mat files", flush=True)
        return signals, np.asarray(labels), np.asarray(groups), "paderborn"

    print(f"[warn] no .mat files under {data_root} — falling back to synthetic", flush=True)
    sigs, labels, groups = synthesize(SyntheticConfig(n_per_class=8))
    return list(sigs), labels, groups, "synthetic"


def make_windows(signals, labels, groups, indices, *, window: int, stride: int):
    """Cut every full signal in ``indices`` into windows."""
    Xs, ys, gs = [], [], []
    for i in indices:
        w = sliding_windows(signals[i], window, stride)
        if w.size == 0:
            continue
        Xs.append(w)
        ys.append(np.full(w.shape[0], labels[i], dtype=np.int64))
        gs.append(np.full(w.shape[0], groups[i]))
    if not Xs:
        return np.empty((0, window), np.float32), np.empty(0, np.int64), np.empty(0)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(gs)


def train_linear_probe(model, X, y, *, epochs=15, lr=1e-3, batch_size=64) -> nn.Module:
    """Train only the trainable params (the head) on (X, y)."""
    model = model.to(DEVICE)
    model.train()
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    n = X.shape[0]
    Xt = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, T)
    yt = torch.from_numpy(y).long()

    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)
            logits = model(x_enc=xb).logits
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:02d}: loss={ep_loss / n:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_at_snr(model, X, y, snr_db: float, *, batch_size: int = 64) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    Xt = torch.from_numpy(X).float().unsqueeze(1)
    yt = torch.from_numpy(y).long()
    gen = torch.Generator(device="cpu").manual_seed(0)  # deterministic noise per call
    for i in range(0, X.shape[0], batch_size):
        xb = Xt[i : i + batch_size].clone()
        xb = add_awgn(xb, snr_db=snr_db, generator=gen)
        xb = xb.to(DEVICE)
        logits = model(x_enc=xb).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    return {
        "f1": macro_f1(yt.numpy(), y_pred),
        "acc": accuracy(yt.numpy(), y_pred),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "paderborn_pu")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=2048, help="window stride (=window for non-overlap)")
    p.add_argument("--max-files-per-bearing", type=int, default=None,
                   help="subsample to N files per bearing (default: use all)")
    p.add_argument("--smoke", action="store_true", help="tiny budget for plumbing test")
    args = p.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.folds = 2
        args.epochs = 2
        args.batch_size = 16
        args.max_files_per_bearing = 4

    print(f"[info] device={DEVICE} folds={args.folds} seeds={args.seeds} sizes={args.sizes} "
          f"window={args.window} stride={args.stride} max_files_per_bearing={args.max_files_per_bearing}")
    signals, labels, groups, source = materialise_dataset(
        args.data_root,
        max_files_per_bearing=args.max_files_per_bearing,
        seed=0,
    )
    print(f"[info] dataset source={source} n_files={len(signals)} classes={CLASS_NAMES}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_out = args.out.open("w", newline="")
    writer = csv.writer(f_out)
    writer.writerow(["seed", "fold", "size", "snr_db", "f1", "acc", "n_test_windows"])

    snr_levels = snr_sweep_levels()

    # Build a class array indexed by file for SGKF.
    file_entries = [
        type("E", (), {"label_idx": int(labels[i]), "bearing_id": str(groups[i])})()  # type: ignore[arg-type]
        for i in range(len(signals))
    ]
    fold_splits = stratified_group_kfold_indices(file_entries, n_splits=args.folds, seed=0)

    for seed in args.seeds:
        set_seed(seed)
        for fold, (tr_idx, te_idx) in enumerate(fold_splits):
            print(f"\n=== seed={seed} fold={fold} ===")
            X_tr, y_tr, _ = make_windows(signals, labels, groups, tr_idx, window=args.window, stride=args.stride)
            X_te, y_te, _ = make_windows(signals, labels, groups, te_idx, window=args.window, stride=args.stride)
            print(f"  windows: train={X_tr.shape[0]} test={X_te.shape[0]}")
            for size in args.sizes:
                print(f"  -> training {size}")
                model = build_moment_classifier(
                    MomentConfig(size=size, n_channels=1, num_class=len(CLASS_NAMES))
                )
                model = train_linear_probe(
                    model, X_tr, y_tr, epochs=args.epochs, batch_size=args.batch_size
                )
                for snr in snr_levels:
                    res = eval_at_snr(model, X_te, y_te, snr_db=snr, batch_size=args.batch_size)
                    snr_label = "inf" if not np.isfinite(snr) else f"{snr:g}"
                    writer.writerow([seed, fold, size, snr_label, f"{res['f1']:.4f}", f"{res['acc']:.4f}", X_te.shape[0]])
                    f_out.flush()
                    print(f"    snr={snr_label:>5}  f1={res['f1']:.3f}  acc={res['acc']:.3f}")
                del model
                torch.cuda.empty_cache()
    f_out.close()
    print(f"\n[done] results -> {args.out}")


if __name__ == "__main__":
    main()
