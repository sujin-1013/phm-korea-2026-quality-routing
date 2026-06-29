"""Train the 1D-CNN noise estimator on mixed-SNR vibration windows.

Target = ``snr_to_target(snr_db) = sigmoid(-snr_db / 10)`` — monotonically
decreasing in SNR. Loss = MSE on this target.

Phase 2 §2.1 also calls for a fault/noise disentangle metric: the estimator
should NOT output a high noise level on a clean faulty signal (SNR=∞ but
non-zero fault signature). We measure that as ``false_positive_rate`` on
``snr=inf`` windows and report it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from phm_routing.data.noise_injection import add_awgn, random_snr_sample
from phm_routing.models.noise_estimator import DEPLOYED_ESTIMATOR_CHANNELS, NoiseEstimator1D, snr_to_target


@dataclass
class EstimatorTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    snr_levels: tuple[float, ...] = (float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0, -10.0)
    device: str = "cuda"
    channels: tuple[int, ...] = DEPLOYED_ESTIMATOR_CHANNELS
    kernel_size: int = 7


class MixedSNRDataset(Dataset):
    """Wraps ``(N, T)`` windows and yields ``(window_with_noise, target)``.

    Each ``__getitem__`` picks an SNR uniformly from ``cfg.snr_levels`` and
    injects AWGN at that level. This produces an on-the-fly augmentation.
    """

    def __init__(
        self,
        signals: np.ndarray,
        cfg: EstimatorTrainConfig,
        rng_seed: int = 0,
    ):
        self.signals = signals.astype(np.float32)
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self._gen = torch.Generator(device="cpu").manual_seed(rng_seed)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        snr = random_snr_sample(self.rng, levels=self.cfg.snr_levels)
        x = torch.from_numpy(self.signals[idx]).float().unsqueeze(0)  # (1, T)
        x_noisy = add_awgn(x, snr_db=snr, generator=self._gen)
        target = snr_to_target(snr).float()
        return x_noisy, target


def train_estimator(
    signals_train: np.ndarray,
    signals_val: np.ndarray,
    cfg: EstimatorTrainConfig | None = None,
) -> tuple[NoiseEstimator1D, dict]:
    cfg = cfg or EstimatorTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    train_ds = MixedSNRDataset(signals_train, cfg, rng_seed=42)
    val_ds = MixedSNRDataset(signals_val, cfg, rng_seed=99)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = NoiseEstimator1D(channels=cfg.channels, kernel_size=cfg.kernel_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_mse": [], "val_mse": [], "fp_rate_clean": []}
    for ep in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.shape[0]
            n_tr += x.shape[0]
        history["train_mse"].append(tr_loss / max(n_tr, 1))

        # Validation: standard MSE.
        model.eval()
        vl_loss = 0.0
        n_vl = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); y = y.to(device)
                pred = model(x)
                vl_loss += loss_fn(pred, y).item() * x.shape[0]
                n_vl += x.shape[0]
        history["val_mse"].append(vl_loss / max(n_vl, 1))

        # Disentangle check: on clean (SNR=∞) windows, estimator should output low noise_level.
        # Define "false positive" as estimator > 0.33 (would route to a larger tier).
        with torch.no_grad():
            xt = torch.from_numpy(signals_val).float().unsqueeze(1).to(device)
            preds_clean = model(xt).cpu().numpy()
        fp_rate = float((preds_clean > 0.33).mean())
        history["fp_rate_clean"].append(fp_rate)

        if ep % 5 == 0 or ep == cfg.epochs - 1:
            print(
                f"  ep {ep:02d}: train_mse={history['train_mse'][-1]:.4f}  "
                f"val_mse={history['val_mse'][-1]:.4f}  "
                f"fp_rate_clean={fp_rate:.3f}"
            )

    return model, history


@torch.no_grad()
def calibration_scan(
    model: NoiseEstimator1D,
    signals: np.ndarray,
    snr_levels: Sequence[float],
    *,
    device: str = "cuda",
) -> dict[float, np.ndarray]:
    """Return ``{snr_db: predictions_array}`` — one row per signal per SNR level."""
    model.eval()
    out: dict[float, np.ndarray] = {}
    xt0 = torch.from_numpy(signals).float().unsqueeze(1)  # (N, 1, T)
    for snr in snr_levels:
        gen = torch.Generator(device="cpu").manual_seed(0)
        x = add_awgn(xt0, snr_db=snr, generator=gen).to(device)
        out[float(snr) if np.isfinite(snr) else float("inf")] = model(x).cpu().numpy()
    return out
