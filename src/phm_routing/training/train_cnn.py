"""Training helpers for the final CNN capacity-ladder experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from phm_routing.data.noise_injection import add_awgn
from phm_routing.data.paderborn import (
    CLASS_NAMES_3,
    CLASS_TO_IDX_3,
    _load_and_window_entries,
    discover_files,
    inverse_freq_class_weights,
    materialise_pu_c1,
)
from phm_routing.models.cnn1d import ModelSize, build_cnn, count_parameters
from phm_routing.utils.metrics import macro_f1


DEFAULT_WINDOW = 2048
DEFAULT_STRIDE = 2048
DEFAULT_AUGMENT_SNR: tuple[float, ...] = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
CLEAN_ONLY_AUGMENT: tuple[float, ...] = (float("inf"),)


@dataclass(frozen=True)
class CNNTrainResult:
    model: nn.Module
    best_val_f1: float
    n_parameters: int


def stratified_entry_split(entries: Sequence, frac: float, seed: int):
    """Split file entries per class label with deterministic shuffling."""
    if not (0.0 < frac < 1.0):
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    rng = np.random.default_rng(seed)
    first: list = []
    second: list = []
    by_label: dict[str, list] = {}
    for entry in entries:
        by_label.setdefault(entry.label, []).append(entry)
    for entries_for_label in by_label.values():
        indices = rng.permutation(len(entries_for_label))
        n_first = int(round(frac * len(entries_for_label)))
        for offset, entry_idx in enumerate(indices):
            (first if offset < n_first else second).append(entries_for_label[entry_idx])
    return first, second


def materialise_scenario(
    data_root: str | Path,
    *,
    scenario: str = "indist",
    test_condition: str | None = "N09_M07_F10",
    seed: int = 42,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
):
    """Return ``Xtr, ytr, Xval, yval, Xtest, ytest`` for final CNN experiments."""
    data_root = Path(data_root)
    if scenario == "a2r":
        return materialise_pu_c1(data_root, seed=seed, window=window, stride=stride)

    entries = discover_files(data_root, classes=CLASS_NAMES_3)
    if not entries:
        raise FileNotFoundError(f"No PU-C1-compatible Paderborn .mat files under {data_root}")

    if scenario == "indist":
        train_val, test = stratified_entry_split(entries, 0.85, seed)
        train, val = stratified_entry_split(train_val, 0.82, seed + 1)
    elif scenario == "bdis":
        train, val, test = _bearing_disjoint_split(entries, seed)
    elif scenario == "loco":
        if not test_condition:
            raise ValueError("test_condition is required for scenario='loco'")
        test = [entry for entry in entries if entry.operating_condition == test_condition]
        train_val = [entry for entry in entries if entry.operating_condition != test_condition]
        train, val = stratified_entry_split(train_val, 0.85, seed)
    else:
        raise ValueError("scenario must be one of: indist, bdis, loco, a2r")

    load = lambda bucket: _load_and_window_entries(
        bucket,
        window=window,
        stride=stride,
        label_to_idx=CLASS_TO_IDX_3,
    )
    return (*load(train), *load(val), *load(test))


def _bearing_disjoint_split(entries: Sequence, seed: int):
    rng = np.random.default_rng(seed)
    by_bearing: dict[str, list] = {}
    for entry in entries:
        by_bearing.setdefault(entry.bearing_id, []).append(entry)

    by_class: dict[str, list[str]] = {}
    for bearing_id, bearing_entries in by_bearing.items():
        by_class.setdefault(bearing_entries[0].label, []).append(bearing_id)

    test_bearings: set[str] = set()
    for bearing_ids in by_class.values():
        bearing_ids = sorted(bearing_ids)
        indices = rng.permutation(len(bearing_ids))
        n_test = max(1, int(round(0.2 * len(bearing_ids))))
        test_bearings.update(bearing_ids[i] for i in indices[:n_test])

    test = [entry for entry in entries if entry.bearing_id in test_bearings]
    train_val = [entry for entry in entries if entry.bearing_id not in test_bearings]
    train, val = stratified_entry_split(train_val, 0.85, seed + 1)
    print(f"[bdis] test bearings ({len(test_bearings)}): {sorted(test_bearings)}", flush=True)
    return train, val, test


def cap_arrays(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if n <= 0 or X.shape[0] <= n:
        return X, y
    idx = np.random.default_rng(seed).permutation(X.shape[0])[:n]
    return X[idx], y[idx]


def train_cnn_classifier(
    size: ModelSize,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    class_weights: Sequence[float] | None = None,
    augment_snr: Sequence[float] = DEFAULT_AUGMENT_SNR,
) -> CNNTrainResult:
    model = build_cnn(size).to(device)
    n_parameters = count_parameters(model)
    weights = class_weights if class_weights is not None else inverse_freq_class_weights(y_train, num_class=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    aug_gen = torch.Generator().manual_seed(0)

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    augment_snr = tuple(float(v) for v in augment_snr)
    if not augment_snr:
        raise ValueError("augment_snr must include at least one SNR level")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for start in range(0, Xtr.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = Xtr[idx].clone()
            snr = float(augment_snr[torch.randint(len(augment_snr), (1,)).item()])
            xb = add_awgn(xb, snr_db=snr, generator=aug_gen).to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        preds = predict_labels(model, Xv, device=device)
        val_f1 = macro_f1(y_val, preds)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{size}] ep{ep:02d} val_f1={val_f1:.3f}{' *best' if val_f1 == best_val_f1 else ''}", flush=True)

    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    model.load_state_dict(best_state)
    return CNNTrainResult(model=model, best_val_f1=best_val_f1, n_parameters=n_parameters)


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray | torch.Tensor, *, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    preds: list[np.ndarray] = []
    for start in range(0, Xt.shape[0], batch_size):
        xb = Xt[start : start + batch_size].to(device)
        preds.append(model(xb).argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def save_cnn_checkpoint(path: str | Path, result: CNNTrainResult, size: ModelSize) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": result.model.state_dict(),
            "npar": result.n_parameters,
            "best_val": result.best_val_f1,
            "size": size,
        },
        path,
    )


def load_cnn_checkpoint(path: str | Path, size: ModelSize, *, device: str) -> CNNTrainResult:
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_cnn(size).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return CNNTrainResult(
        model=model,
        best_val_f1=float(state.get("best_val", float("nan"))),
        n_parameters=int(state.get("npar", count_parameters(model))),
    )
