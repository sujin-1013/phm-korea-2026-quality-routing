"""Cost-accuracy Pareto utilities.

Given a list of ``(name, flops, f1)`` points we compute the Pareto frontier
(higher F1, lower FLOPs is better) and produce a matplotlib plot suitable
for the paper's main figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ParetoPoint:
    name: str
    flops: float
    f1: float


def pareto_front(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return points that are not dominated (lower flops AND ≥ f1)."""
    pts = sorted(points, key=lambda p: p.flops)
    front: list[ParetoPoint] = []
    best_f1 = -np.inf
    for p in pts:
        if p.f1 > best_f1:
            front.append(p)
            best_f1 = p.f1
    return front


def plot_pareto(
    points: Iterable[ParetoPoint],
    out: Path,
    *,
    title: str = "Cost vs accuracy",
    xlabel: str = "FLOPs (per window)",
    ylabel: str = "macro-F1",
):
    import matplotlib.pyplot as plt

    pts = list(points)
    front = set(p.name for p in pareto_front(pts))
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    for p in pts:
        on_front = p.name in front
        ax.scatter(
            p.flops, p.f1,
            s=80 if on_front else 40,
            c="#d79b00" if on_front else "#6c8ebf",
            edgecolors="black", linewidths=0.7, alpha=0.9,
            zorder=3 if on_front else 2,
        )
        ax.annotate(p.name, (p.flops, p.f1), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    fr = sorted([p for p in pts if p.name in front], key=lambda q: q.flops)
    if len(fr) >= 2:
        ax.plot([p.flops for p in fr], [p.f1 for p in fr],
                c="#d79b00", lw=1.2, alpha=0.6, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
