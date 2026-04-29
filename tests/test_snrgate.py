"""Unit tests for SNRGate building blocks (LearnableSNRGate + LoRA)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from phm_routing.models.lora import LoRALinear, inject_lora, lora_param_count
from phm_routing.models.router import GateConfig, HardGateRouter, LearnableSNRGate, RouterConfig


# ---------- LearnableSNRGate ----------------------------------------------

def test_gate_init_centers_match_config():
    g = LearnableSNRGate(GateConfig(init_centers=(0.1, 0.4, 0.9)))
    assert torch.allclose(g.centers.detach(), torch.tensor([0.1, 0.4, 0.9]))


def test_gate_soft_weights_sum_to_one():
    g = LearnableSNRGate()
    n = torch.linspace(0, 1, 11)
    p = g(n)
    assert p.shape == (11, 3)
    assert torch.allclose(p.sum(dim=-1), torch.ones(11), atol=1e-5)


def test_gate_high_T_is_more_uniform_than_low_T():
    """Sanity: with T high, distribution is closer to uniform; T low → peaked."""
    g = LearnableSNRGate()
    n = torch.tensor([0.5])  # equidistant from S/B/L when centers are (0.17,0.5,0.83)
    g.set_temperature(5.0)
    p_hi = g(n)[0]
    g.set_temperature(0.01)
    p_lo = g(n)[0]
    # The high-T distribution should have higher entropy than low-T.
    h_hi = -(p_hi * (p_hi.clamp_min(1e-9)).log()).sum()
    h_lo = -(p_lo * (p_lo.clamp_min(1e-9)).log()).sum()
    assert h_hi > h_lo


def test_gate_low_T_collapses_to_argmin():
    g = LearnableSNRGate(GateConfig(init_centers=(0.17, 0.50, 0.83)))
    g.set_temperature(0.001)
    n = torch.tensor([0.05, 0.50, 0.95])
    p = g(n)
    chosen = p.argmax(dim=-1)
    expected = g.hard_tier(n)
    assert torch.equal(chosen, expected)
    assert torch.equal(chosen, torch.tensor([0, 1, 2]))


def test_gate_hard_tier_picks_nearest_center():
    g = LearnableSNRGate(GateConfig(init_centers=(0.17, 0.50, 0.83)))
    # midpoints: (0.17+0.5)/2=0.335, (0.5+0.83)/2=0.665
    n = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0])
    tiers = g.hard_tier(n)
    assert torch.equal(tiers, torch.tensor([0, 0, 1, 1, 2]))


def test_gate_anneal_monotonic_decrease():
    g = LearnableSNRGate(GateConfig(T_init=1.0, T_min=0.05, anneal_epochs=10))
    Ts = [g.anneal_step(ep) for ep in range(11)]
    # exponentially decreasing
    for prev, cur in zip(Ts, Ts[1:]):
        assert cur <= prev + 1e-6
    assert math.isclose(Ts[0], 1.0, rel_tol=1e-3)
    assert math.isclose(Ts[-1], 0.05, rel_tol=1e-3)


def test_gate_centers_receive_gradient():
    g = LearnableSNRGate()
    n = torch.tensor([0.2, 0.5, 0.8], requires_grad=False)
    p = g(n)
    p.sum().backward()
    assert g.centers.grad is not None
    assert torch.isfinite(g.centers.grad).all()


def test_gate_rejects_bad_input_shape():
    g = LearnableSNRGate()
    with pytest.raises(ValueError):
        g(torch.zeros(4, 1))


def test_gate_config_validation():
    with pytest.raises(ValueError):
        LearnableSNRGate(GateConfig(init_centers=(0.5, 0.5)))  # too few
    with pytest.raises(ValueError):
        LearnableSNRGate(GateConfig(init_centers=(-0.1, 0.5, 0.9)))  # out of [0,1]


# ---------- HardGateRouter (regression / unchanged behavior) ----------------

def test_hard_router_unchanged_thresholds():
    r = HardGateRouter(RouterConfig(tau1=0.33, tau2=0.66))
    n = torch.tensor([0.10, 0.40, 0.80])
    out = r(n)
    assert torch.equal(out, torch.tensor([0, 1, 2]))


# ---------- LoRA -------------------------------------------------------------

def test_lora_init_is_identity():
    """Right after wrapping, LoRA contributes 0 — output equals base Linear."""
    base = nn.Linear(8, 4)
    base.weight.data.normal_()
    base.bias.data.normal_()
    base_copy = nn.Linear(8, 4)
    base_copy.load_state_dict(base.state_dict())

    lora = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base_copy(x), atol=1e-6)


def test_lora_freezes_base_weights():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=4)
    assert not base.weight.requires_grad
    assert not base.bias.requires_grad
    assert lora.A.requires_grad
    assert lora.B.requires_grad


def test_lora_grad_flows_only_to_AB():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=4)
    x = torch.randn(2, 8)
    lora(x).sum().backward()
    assert base.weight.grad is None
    assert lora.A.grad is not None and torch.isfinite(lora.A.grad).all()
    assert lora.B.grad is not None and torch.isfinite(lora.B.grad).all()


def test_inject_lora_wraps_named_targets():
    """A toy module mimicking T5 attention: q/k/v/o linears."""
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(16, 16)
            self.k = nn.Linear(16, 16)
            self.v = nn.Linear(16, 16)
            self.o = nn.Linear(16, 16)

    model = nn.Sequential(FakeAttn(), FakeAttn())
    params = inject_lora(model, r=4, alpha=8.0, target_names=("q", "v"))
    # 2 attention modules × {q, v} × {A, B} = 8 LoRA tensors.
    assert len(params) == 8
    for attn in model:
        assert isinstance(attn.q, LoRALinear)
        assert isinstance(attn.v, LoRALinear)
        assert isinstance(attn.k, nn.Linear) and not isinstance(attn.k, LoRALinear)
        assert isinstance(attn.o, nn.Linear) and not isinstance(attn.o, LoRALinear)
    # Sanity on count: each LoRA pair has r·in + out·r = 4·16 + 16·4 = 128 elements.
    assert lora_param_count(params) == 4 * 128


def test_inject_lora_raises_on_no_targets():
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.unrelated = nn.Linear(8, 8)

    with pytest.raises(RuntimeError):
        inject_lora(FakeAttn(), target_names=("q",))
