"""Joint training of SNRGate (estimator + gate + per-tier LoRA + heads).

Soft-mixture training: every batch forwards through all three MOMENT tiers
(S/B/L) and combines logits via the gate's soft weights. The gate's
temperature is annealed each epoch so by the end of training it nearly
collapses to a single tier — that single-tier mode is what inference uses
(``SNRGateClassifier.inference_forward``) and is what preserves the
cost-saving claim.

Memory note (RTX 3070 8GB): three frozen MOMENT tiers + LoRA + activations
fits with ``batch_size≤4`` and ``use_amp=True``. ``grad_checkpoint=True``
trades compute for memory if needed for the L tier.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from phm_routing.data.noise_injection import add_awgn, random_snr_sample
from phm_routing.models.lora import inject_lora, lora_param_count
from phm_routing.models.moment_wrapper import (
    MomentConfig,
    MomentSize,
    build_moment_classifier,
)
from phm_routing.models.noise_estimator import NoiseEstimator1D
from phm_routing.models.router import GateConfig, LearnableSNRGate
from phm_routing.utils.metrics import macro_f1


_TIER_SIZES: tuple[MomentSize, MomentSize, MomentSize] = ("small", "base", "large")


@dataclass
class RoutingTrainConfig:
    estimator_ckpt: Path | None = None
    train_estimator: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    gate: GateConfig = field(default_factory=GateConfig)
    # PU-C1 A2R 3-class is the paper standard (paper/DATA_PROTOCOL_LOCKED.md).
    num_class: int = 3
    epochs: int = 30
    batch_size: int = 4
    lr_lora: float = 1e-3
    lr_head: float = 1e-3
    lr_gate: float = 1e-2
    lr_estimator: float = 1e-4
    weight_decay: float = 1e-4
    augment_snr: tuple[float, ...] = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
    use_amp: bool = True
    device: str = "cuda"
    # Per-class loss weights — None → unweighted. Caller passes inverse-freq weights
    # for PU-C1 imbalance handling (1 healthy bearing in train vs. 16 damage bearings).
    class_weights: list[float] | None = None


class SNRGateClassifier(nn.Module):
    """Bundle of estimator + gate + 3 LoRA-adapted MOMENT tiers."""

    def __init__(
        self,
        estimator: NoiseEstimator1D,
        gate: LearnableSNRGate,
        tiers: Sequence[nn.Module],
    ):
        super().__init__()
        if len(tiers) != LearnableSNRGate.NUM_TIERS:
            raise ValueError(f"need {LearnableSNRGate.NUM_TIERS} tiers, got {len(tiers)}")
        self.estimator = estimator
        self.gate = gate
        self.tiers = nn.ModuleList(tiers)

    def _tier_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Run all tiers and stack: returns ``(B, K, num_class)``."""
        per_tier = [m(x_enc=x).logits for m in self.tiers]
        return torch.stack(per_tier, dim=1)

    def train_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft-mixture forward — used in training.

        Returns ``(mixed_logits[B,C], gate_weights[B,K], noise_level[B])``.
        """
        noise = self.estimator(x)  # (B,)
        p = self.gate(noise)  # (B, K)
        per_tier = self._tier_logits(x)  # (B, K, C)
        mixed = (p.unsqueeze(-1) * per_tier).sum(dim=1)  # (B, C)
        return mixed, p, noise

    @torch.no_grad()
    def inference_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard-gate forward — exactly one tier per example.

        Returns ``(logits[B,C], tier_idx[B])``. Each example only runs through
        its selected tier (cost-saving claim) at the cost of a per-example loop;
        for batched eval you can pad-and-mask per-tier instead.
        """
        noise = self.estimator(x)
        tier_idx = self.gate.hard_tier(noise)
        out = torch.empty(x.shape[0], self.tiers[0](x_enc=x[:1]).logits.shape[-1], device=x.device)
        for k, tier_model in enumerate(self.tiers):
            mask = tier_idx == k
            if mask.any():
                out[mask] = tier_model(x_enc=x[mask]).logits
        return out, tier_idx


def build_classifier(cfg: RoutingTrainConfig) -> tuple[SNRGateClassifier, dict[str, list]]:
    """Construct estimator + 3 MOMENT tiers (LoRA-injected) + gate.

    Returns the bundled classifier and a dict of trainable param groups
    (``estimator``, ``lora``, ``head``, ``gate``) ready for the optimizer.
    """
    estimator = NoiseEstimator1D()
    if cfg.estimator_ckpt is not None:
        state = torch.load(cfg.estimator_ckpt, map_location="cpu")
        estimator.load_state_dict(state.get("model", state))
    if not cfg.train_estimator:
        for p in estimator.parameters():
            p.requires_grad_(False)

    tiers: list[nn.Module] = []
    lora_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for size in _TIER_SIZES:
        m = build_moment_classifier(
            MomentConfig(size=size, n_channels=1, num_class=cfg.num_class, freeze_encoder=True)
        )
        # build_moment_classifier already left the head trainable; inject LoRA on top
        lora_params.extend(inject_lora(m, r=cfg.lora_r, alpha=cfg.lora_alpha))
        head_params.extend(p for p in m.head.parameters() if p.requires_grad)
        tiers.append(m)

    gate = LearnableSNRGate(cfg.gate)
    classifier = SNRGateClassifier(estimator, gate, tiers)

    groups = {
        "estimator": [p for p in estimator.parameters() if p.requires_grad],
        "lora": lora_params,
        "head": head_params,
        "gate": [gate.centers],
    }
    return classifier, groups


def _make_optimizer(groups: dict[str, list], cfg: RoutingTrainConfig) -> torch.optim.Optimizer:
    param_groups = []
    if groups["estimator"]:
        param_groups.append({"params": groups["estimator"], "lr": cfg.lr_estimator})
    if groups["lora"]:
        param_groups.append({"params": groups["lora"], "lr": cfg.lr_lora})
    if groups["head"]:
        param_groups.append({"params": groups["head"], "lr": cfg.lr_head})
    if groups["gate"]:
        param_groups.append({"params": groups["gate"], "lr": cfg.lr_gate})
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train_routing(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RoutingTrainConfig | None = None,
) -> tuple[SNRGateClassifier, dict[str, list]]:
    """Joint training entry — returns trained classifier + history."""
    cfg = cfg or RoutingTrainConfig()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    classifier, groups = build_classifier(cfg)
    classifier.to(device)
    opt = _make_optimizer(groups, cfg)
    if cfg.class_weights is not None:
        cw = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
        if cw.numel() != cfg.num_class:
            raise ValueError(
                f"class_weights length ({cw.numel()}) != num_class ({cfg.num_class})"
            )
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        print(f"[train_routing] using weighted CE: weights={cfg.class_weights}", flush=True)
    else:
        loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).float().unsqueeze(1)
    yt = torch.from_numpy(y_tr).long()
    Xv = torch.from_numpy(X_val).float().unsqueeze(1)
    yv = torch.from_numpy(y_val).long()

    rng = np.random.default_rng(0)
    aug_gen = torch.Generator(device="cpu").manual_seed(0)
    history: dict[str, list] = {"train_loss": [], "val_f1": [], "T": [], "centers": []}
    best_val_f1: float = -1.0
    best_epoch: int = -1
    best_state: dict | None = None
    amp_dtype = torch.bfloat16 if cfg.use_amp and device == "cuda" else torch.float32

    print(
        f"trainable: lora={lora_param_count(groups['lora']):,}  "
        f"head={sum(p.numel() for p in groups['head']):,}  "
        f"estimator={sum(p.numel() for p in groups['estimator']):,}  "
        f"gate={sum(p.numel() for p in groups['gate']):,}"
    )

    import time
    for ep in range(cfg.epochs):
        T = classifier.gate.anneal_step(ep)
        history["T"].append(T)

        classifier.train()
        perm = torch.randperm(Xt.shape[0])
        ep_loss, n = 0.0, 0
        n_steps = (Xt.shape[0] + cfg.batch_size - 1) // cfg.batch_size
        t0 = time.time()
        for step, i in enumerate(range(0, Xt.shape[0], cfg.batch_size)):
            idx = perm[i : i + cfg.batch_size]
            xb = Xt[idx].clone()
            snr = random_snr_sample(rng, levels=cfg.augment_snr)
            xb = add_awgn(xb, snr_db=snr, generator=aug_gen)
            xb = xb.to(device)
            yb = yt[idx].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp and device == "cuda"):
                mixed, _p, _n = classifier.train_forward(xb)
                cls_loss = loss_fn(mixed, yb)
                # Soft monotonicity penalty: μ_S ≤ μ_B ≤ μ_L (with margin 0.05).
                # Without this the previous run had centers collapse to (S=0.343, B=0.316, L=0.804) —
                # binary routing instead of 3-tier. Penalty pushes centers back into order.
                gc = classifier.gate.centers
                order_penalty = (
                    torch.relu(gc[0] - gc[1] + 0.05).pow(2)
                    + torch.relu(gc[1] - gc[2] + 0.05).pow(2)
                )
                loss = cls_loss + 1.0 * order_penalty
            loss.backward()
            opt.step()
            classifier.gate.clamp_centers()  # keep μ_k ∈ [0,1] for valid soft mixture

            ep_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (n_steps - step) / max(rate, 1e-9)
                print(f"    ep{ep:02d} step {step}/{n_steps} loss={loss.item():.3f} "
                      f"rate={rate:.2f} steps/s eta={eta:.0f}s", flush=True)
        history["train_loss"].append(ep_loss / max(n, 1))
        history["centers"].append(classifier.gate.centers.detach().cpu().tolist())

        # validation: hard-gate (inference path) on clean
        classifier.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, Xv.shape[0], cfg.batch_size):
                xb = Xv[i : i + cfg.batch_size].to(device)
                logits, _ = classifier.inference_forward(xb)
                preds.append(logits.argmax(-1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
        val_f1 = macro_f1(yv.numpy(), y_pred)
        history["val_f1"].append(val_f1)

        # Track best-on-val (clean hard-gate eval). Snapshot the full classifier
        # state so we can restore it at the end — this avoids relying on the
        # final-epoch weights, which we observed to be worse than mid-training
        # weights once T anneals very low.
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = ep
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()})

        c = history["centers"][-1]
        marker = "  ★best" if is_best else ""
        print(
            f"  ep {ep:02d}: T={T:.3f}  loss={history['train_loss'][-1]:.4f}  "
            f"val_f1={history['val_f1'][-1]:.3f}  μ=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}){marker}",
            flush=True,
        )

    # Restore best-on-val weights so callers (sweep eval, checkpoint save) see
    # the strongest model rather than the final epoch.
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1
    if best_state is not None:
        classifier.load_state_dict(best_state)
        print(f"  [done] restored best weights from ep {best_epoch:02d} (val_f1={best_val_f1:.3f})", flush=True)

    return classifier, history
