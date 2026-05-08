"""Canonical single-tier baseline training (MOMENT-S/B/L + LoRA + head).

Used by `experiments/exp_baselines/run.py` for the paper's no-routing baselines.
Mixed-SNR augmentation, weighted CE, best-on-val checkpoint restore.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from phm_routing.data.noise_injection import add_awgn, random_snr_sample
from phm_routing.models.lora import inject_lora, lora_param_count
from phm_routing.models.moment_wrapper import MomentConfig, build_moment_classifier
from phm_routing.utils.metrics import macro_f1


def train_single_tier(
    size: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lora_r: int,
    augment_snr: tuple[float, ...],
    device: str,
    num_class: int,
    use_amp: bool = True,
    lr_lora: float = 2e-4,
    lr_head: float = 2e-4,
    weight_decay: float = 1e-3,
    class_weights: list[float] | None = None,
):
    """Train one MOMENT tier with LoRA Q/V + linear head, restore best-val state.

    Defaults match the paper's conservative HP (lr=2e-4, wd=1e-3) used for both
    SNRGate and the single-tier baselines.
    """
    model = build_moment_classifier(
        MomentConfig(size=size, n_channels=1, num_class=num_class, freeze_encoder=True)
    ).to(device)
    lora_params = inject_lora(model, r=lora_r, alpha=16.0)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    print(
        f"[{size}] lora={lora_param_count(lora_params):,}  "
        f"head={sum(p.numel() for p in head_params):,}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        if cw.numel() != num_class:
            raise ValueError(f"class_weights length ({cw.numel()}) != num_class ({num_class})")
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        print(f"  [{size}] using weighted CE: weights={class_weights}", flush=True)
    else:
        loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr).float().unsqueeze(1)
    yt = torch.from_numpy(ytr).long()
    Xv = torch.from_numpy(Xval).float().unsqueeze(1)
    yv = torch.from_numpy(yval).long()

    rng = np.random.default_rng(0)
    aug_gen = torch.Generator(device="cpu").manual_seed(0)
    amp_dtype = torch.bfloat16 if use_amp and device == "cuda" else torch.float32
    history: dict = {"train_loss": [], "val_f1": []}
    best_val_f1 = -1.0
    best_epoch = -1
    best_state: dict | None = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        ep_loss, n = 0.0, 0
        for i in range(0, Xt.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx].clone()
            snr = random_snr_sample(rng, levels=augment_snr)
            xb = add_awgn(xb, snr_db=snr, generator=aug_gen)
            xb = xb.to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype,
                enabled=use_amp and device == "cuda",
            ):
                logits = model(x_enc=xb).logits
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]
        history["train_loss"].append(ep_loss / max(n, 1))

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, Xv.shape[0], batch_size):
                xb = Xv[i : i + batch_size].to(device)
                preds.append(model(x_enc=xb).logits.argmax(-1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
        val_f1 = macro_f1(yv.numpy(), y_pred)
        history["val_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = ep
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
        marker = "  ★best" if is_best else ""
        print(
            f"  [{size}] ep {ep:02d}: loss={history['train_loss'][-1]:.4f} "
            f"val_f1={val_f1:.3f}{marker}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"  [{size}] [done] restored best weights from ep {best_epoch:02d} "
            f"(val_f1={best_val_f1:.3f})",
            flush=True,
        )
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1

    return model, history


def snr_sweep_eval(model, X, y, snr_levels, batch_size, device):
    """Macro-F1 sweep across SNR levels for a single-tier classifier.

    `inf` (or non-finite SNR) means clean (no AWGN). Used by baselines and
    HardGate ablation.
    """
    aug_gen = torch.Generator(device="cpu").manual_seed(123)
    Xv = torch.from_numpy(X).float().unsqueeze(1)
    yv = torch.from_numpy(y).long()
    f1: dict = {}
    model.eval()
    for snr in snr_levels:
        preds = []
        for i in range(0, Xv.shape[0], batch_size):
            xb = Xv[i : i + batch_size].clone()
            if np.isfinite(snr):
                xb = add_awgn(xb, snr_db=float(snr), generator=aug_gen)
            xb = xb.to(device)
            with torch.no_grad():
                preds.append(model(x_enc=xb).logits.argmax(-1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
        label = "inf" if not np.isfinite(snr) else f"{snr:g}"
        f1[label] = macro_f1(yv.numpy(), y_pred)
    return f1
