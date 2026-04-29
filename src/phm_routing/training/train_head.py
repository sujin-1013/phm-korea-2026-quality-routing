"""Train a per-tier classification head on top of frozen MOMENT.

Plan §2.2: Phase 1 keeps the encoder frozen and trains only the head, with
mixed-SNR data augmentation so the head sees clean and noisy windows.

This file exposes a single function ``train_head`` that takes the size
("small"/"base"/"large"), the ``(X, y)`` train/val arrays, and returns the
trained model + history. Used by exp00_sanity (no-augmentation linear probe)
and the full pipeline (mixed-SNR augmentation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from phm_routing.data.noise_injection import add_awgn, random_snr_sample
from phm_routing.models.moment_wrapper import MomentConfig, build_moment_classifier
from phm_routing.utils.metrics import macro_f1


@dataclass
class HeadTrainConfig:
    size: str = "small"
    num_class: int = 4
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    augment_snr: tuple[float, ...] | None = None  # if set, sample SNR per batch
    device: str = "cuda"


def train_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: HeadTrainConfig | None = None,
):
    cfg = cfg or HeadTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = build_moment_classifier(
        MomentConfig(size=cfg.size, n_channels=1, num_class=cfg.num_class)
    ).to(device)

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).float().unsqueeze(1)  # (N, 1, T)
    yt = torch.from_numpy(y_tr).long()
    Xv = torch.from_numpy(X_val).float().unsqueeze(1)
    yv = torch.from_numpy(y_val).long()

    rng = np.random.default_rng(0)
    aug_gen = torch.Generator(device="cpu").manual_seed(0)
    history = {"train_loss": [], "val_f1": []}

    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        ep_loss, n = 0.0, 0
        for i in range(0, Xt.shape[0], cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = Xt[idx].clone()
            if cfg.augment_snr is not None:
                snr = random_snr_sample(rng, levels=cfg.augment_snr)
                xb = add_awgn(xb, snr_db=snr, generator=aug_gen)
            xb = xb.to(device)
            yb = yt[idx].to(device)
            logits = model(x_enc=xb).logits
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]
        history["train_loss"].append(ep_loss / max(n, 1))

        # validation (clean)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, Xv.shape[0], cfg.batch_size):
                xb = Xv[i : i + cfg.batch_size].to(device)
                preds.append(model(x_enc=xb).logits.argmax(-1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
        history["val_f1"].append(macro_f1(yv.numpy(), y_pred))
        if ep % 5 == 0 or ep == cfg.epochs - 1:
            print(
                f"  ep {ep:02d}: loss={history['train_loss'][-1]:.4f}  "
                f"val_f1={history['val_f1'][-1]:.3f}"
            )
    return model, history
