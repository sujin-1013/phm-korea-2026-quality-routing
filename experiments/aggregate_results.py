"""Aggregate SNRGate + baseline results into paper-ready table + figures.

Inputs:
  experiments/exp03_snrgate_main/snr_sweep.csv    — SNRGate (proposed)
  experiments/exp_baselines/baselines_sweep.csv   — single-tier baselines

Outputs:
  paper/figures/F2_snr_f1_lines.pdf               — SNR-stratified Macro-F1
  paper/figures/F3_routing_dist.pdf               — SNR vs S/B/L share
  paper/figures/F4_pareto.pdf                     — FLOPs-F1 Pareto (params as proxy)
  paper/figures/results_table.md                  — Markdown table for paper
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# rough param counts (ICML2024 momentfm builds)
TIER_PARAMS = {"small": 35.3e6, "base": 109.6e6, "large": 341.2e6}
SNR_ORDER = ["inf", "20", "10", "5", "0", "-5", "-10", "-20"]


def load_snrgate():
    p = ROOT / "experiments" / "exp03_snrgate_main" / "snr_sweep.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p, dtype={"snr_db": str})
    f1 = df.set_index("snr_db")["macro_f1"].to_dict()
    routing = df.set_index("snr_db")[["share_S", "share_B", "share_L"]].to_dict("index")
    return f1, routing


def load_baselines():
    p = ROOT / "experiments" / "exp_baselines" / "baselines_sweep.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"snr_db": str, "method": str})
    return df


def load_hardgate():
    """HardGateRouter ablation row: same trained tiers as SNRGate but rule-based gate."""
    p = ROOT / "experiments" / "exp02_routing" / "snr_sweep.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p, dtype={"snr_db": str})
    f1 = df.set_index("snr_db")["macro_f1"].to_dict()
    routing = df.set_index("snr_db")[["share_S", "share_B", "share_L"]].to_dict("index")
    return f1, routing


def make_table(f1_snrgate, df_baselines, f1_hardgate=None):
    rows = []
    if df_baselines is not None:
        for size in ["small", "base", "large"]:
            sub = df_baselines[df_baselines.method == f"MOMENT-{size}-only"]
            row = {"method": f"MOMENT-{size}"}
            for snr in SNR_ORDER:
                m = sub[sub.snr_db == snr]
                row[snr] = float(m.macro_f1.iloc[0]) if len(m) else float("nan")
            rows.append(row)
    if f1_hardgate is not None:
        rows.append({"method": "HardGate (rule, ablation)", **{snr: f1_hardgate.get(snr, float("nan")) for snr in SNR_ORDER}})
    if f1_snrgate is not None:
        rows.append({"method": "SNRGate (ours)", **{snr: f1_snrgate.get(snr, float("nan")) for snr in SNR_ORDER}})
    return pd.DataFrame(rows)


def expected_flops(method: str, routing: dict | None) -> float:
    """rough cost proxy in millions of params (FLOPs ∝ params for transformer fwd)."""
    if method.startswith("MOMENT-small"):
        return TIER_PARAMS["small"]
    if method.startswith("MOMENT-base"):
        return TIER_PARAMS["base"]
    if method.startswith("MOMENT-large"):
        return TIER_PARAMS["large"]
    if method.startswith("SNRGate"):
        # average over SNR conditions, weighted equally
        if routing is None:
            return float("nan")
        total = 0.0
        n = 0
        for snr_label, share in routing.items():
            cost = (
                share["share_S"] * TIER_PARAMS["small"]
                + share["share_B"] * TIER_PARAMS["base"]
                + share["share_L"] * TIER_PARAMS["large"]
            )
            total += cost
            n += 1
        return total / max(n, 1)
    return float("nan")


def plot_snr_f1_lines(table: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in table.iterrows():
        ys = [row[snr] for snr in SNR_ORDER]
        ax.plot(SNR_ORDER, ys, marker="o", label=row["method"])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Bearing fault diagnosis: SNR-stratified Macro-F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[done] {out_path}")


def plot_routing_dist(routing: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    snrs = SNR_ORDER
    s_share = [routing.get(snr, {}).get("share_S", 0) for snr in snrs]
    b_share = [routing.get(snr, {}).get("share_B", 0) for snr in snrs]
    l_share = [routing.get(snr, {}).get("share_L", 0) for snr in snrs]
    ax.bar(snrs, s_share, label="MOMENT-S", color="#67a9cf")
    ax.bar(snrs, b_share, bottom=s_share, label="MOMENT-B", color="#f7f7f7", edgecolor="gray")
    ax.bar(
        snrs,
        l_share,
        bottom=[s + b for s, b in zip(s_share, b_share)],
        label="MOMENT-L",
        color="#ef8a62",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Routing share")
    ax.set_title("SNRGate routing distribution per SNR condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[done] {out_path}")


def plot_pareto(table: pd.DataFrame, routing: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    avg_f1 = {row["method"]: np.nanmean([row[snr] for snr in SNR_ORDER]) for _, row in table.iterrows()}
    for method in table["method"]:
        flops = expected_flops(method, routing)
        if np.isnan(flops) or flops <= 0:
            continue
        ax.scatter(flops / 1e6, avg_f1[method], s=80)
        ax.annotate(method, (flops / 1e6, avg_f1[method]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Effective param count (M, ∝ FLOPs)")
    ax.set_ylabel("Mean Macro-F1 across 7 SNR conditions")
    ax.set_title("Cost-accuracy Pareto")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[done] {out_path}")


def main():
    figures_dir = ROOT / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    f1_snrgate, routing_snrgate = load_snrgate()
    df_baselines = load_baselines()
    f1_hardgate, routing_hardgate = load_hardgate()

    if f1_snrgate is None and df_baselines is None and f1_hardgate is None:
        print("[error] no results found")
        sys.exit(1)

    table = make_table(f1_snrgate, df_baselines, f1_hardgate=f1_hardgate)
    table_md = table.to_markdown(index=False, floatfmt=".3f")
    (figures_dir / "results_table.md").write_text(table_md + "\n")
    print(f"[done] {figures_dir / 'results_table.md'}")
    print("\n" + table_md + "\n")

    if df_baselines is not None or f1_snrgate is not None:
        plot_snr_f1_lines(table, figures_dir / "F2_snr_f1_lines.pdf")
    if routing_snrgate is not None:
        plot_routing_dist(routing_snrgate, figures_dir / "F3_routing_dist.pdf")
    if f1_snrgate is not None and df_baselines is not None:
        plot_pareto(table, routing_snrgate, figures_dir / "F4_pareto.pdf")


if __name__ == "__main__":
    main()
