#!/usr/bin/env python3
"""Autonomous 8-hour experiment orchestrator with adaptive HP tuning.

Pipeline:
    Phase 1  Baselines (S/B/L), 5 epoch, 3-SNR sweep, default HP
    Phase 1b If ≥2 sizes show ep0-collapse → re-run baselines with conservative HP
    Phase 2  SNRGate full (8 epoch, 8-SNR sweep) with the chosen HP
    Phase 3  HardGate ablation (loads SNRGate ckpt, swaps gate, sweep on TEST)
    Phase 4  Aggregate -> paper/figures/

State + per-stage logs land in logs/orchestrator_*.{log,json}.
Run via: nohup python -u experiments/orchestrator.py > logs/orch_nohup.out 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
EXPS = ROOT / "experiments"


@dataclass
class HPSet:
    name: str
    lr_lora: float
    lr_head: float
    weight_decay: float


# Defaults from existing exp_baselines / RoutingTrainConfig.
DEFAULT_HP = HPSet("default", lr_lora=1e-3, lr_head=1e-3, weight_decay=1e-4)
# Conservative: lower lr + stronger wd to fight ep0-collapse pattern observed in Stage B.
CONSERVATIVE_HP = HPSet("conservative", lr_lora=2e-4, lr_head=2e-4, weight_decay=1e-3)


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


STARTED_AT = time.time()
RUN_ID = ts()
STATE_PATH = LOGS / f"orchestrator_state_{RUN_ID}.json"
MAIN_LOG = LOGS / f"orchestrator_{RUN_ID}.log"

state: dict = {
    "run_id": RUN_ID,
    "started": STARTED_AT,
    "stages": [],
}


def save_state() -> None:
    state["elapsed_min"] = round((time.time() - STARTED_AT) / 60, 1)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MAIN_LOG.open("a") as f:
        f.write(line + "\n")


def run_subprocess(args: list[str], log_prefix: str) -> tuple[int, Path]:
    """Run subprocess with stdout/stderr streamed to a per-stage log."""
    plog = LOGS / f"{log_prefix}_{ts()}.log"
    log(f"running: {' '.join(map(str, args))}")
    log(f"  log -> {plog}")
    with plog.open("w") as f:
        f.write(f"# args: {args}\n# started: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n")
        f.flush()
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"  rc={rc}  log_size={plog.stat().st_size}")
    return rc, plog


def has_ep0_collapse(history: list[float]) -> bool:
    """True if best val_f1 was at epoch 0 (model peaked immediately, then degraded)."""
    if not history or len(history) < 2:
        return False
    best_idx = max(range(len(history)), key=lambda i: history[i])
    return best_idx == 0


def stage_baselines(hp: HPSet, sizes: str, snr_levels: str, epochs: int, suffix: str = "") -> tuple[int, dict | None, Path]:
    out_dir = EXPS / "exp_baselines"  # canonical path; latest run wins
    args = [
        sys.executable, "-u", str(EXPS / "exp_baselines/run.py"),
        "--sizes", sizes,
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--max-files-per-bearing", "8",
        "--lora-r", "8",
        "--seed", "42",
        "--snr-levels", snr_levels,
        "--lr-lora", str(hp.lr_lora),
        "--lr-head", str(hp.lr_head),
        "--weight-decay", str(hp.weight_decay),
        "--out-dir", str(out_dir),
    ]
    rc, plog = run_subprocess(args, f"baselines_{hp.name}{suffix}")
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return rc, summary, out_dir


def stage_snrgate(hp: HPSet, snr_levels: str | None, epochs: int) -> tuple[int, Path]:
    out_dir = EXPS / "exp03_snrgate_main"
    args = [
        sys.executable, "-u", str(EXPS / "exp03_snrgate/run.py"),
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--max-files-per-bearing", "8",
        "--lora-r", "8",
        "--seed", "42",
        "--lr-lora", str(hp.lr_lora),
        "--lr-head", str(hp.lr_head),
        "--weight-decay", str(hp.weight_decay),
        "--out-dir", str(out_dir),
    ]
    if snr_levels:
        args.extend(["--snr-levels", snr_levels])
    rc, plog = run_subprocess(args, f"snrgate_{hp.name}")
    return rc, out_dir


def stage_hardgate(snrgate_ckpt: Path) -> int:
    args = [
        sys.executable, "-u", str(EXPS / "exp02_routing/run.py"),
        "--ckpt", str(snrgate_ckpt),
        "--max-files-per-bearing", "8",
        "--batch-size", "8",
        "--seed", "42",
    ]
    rc, _ = run_subprocess(args, "hardgate")
    return rc


def stage_aggregate() -> int:
    rc, _ = run_subprocess([sys.executable, "-u", str(EXPS / "aggregate_results.py")], "aggregate")
    return rc


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    log(f"=== orchestrator start {RUN_ID} ===")
    log(f"state file: {STATE_PATH}")
    save_state()

    # ----------------------------------------------------------------
    # Phase 1: Baselines (conservative HP, full 8-SNR sweep)
    #
    # Skip-to-conservative: 2026-05-07 run already proved DEFAULT HP causes
    # ep0-collapse in 2/3 sizes (logs/baselines_default_20260507_000005.log).
    # Going straight to CONSERVATIVE saves the wasted ~2h on default and gives
    # the paper main figure (F2) full 8-SNR coverage instead of just 3 points.
    # ----------------------------------------------------------------
    log("Phase 1/4: baselines (conservative HP, full 8-SNR sweep)")
    rc, summary, out_dir = stage_baselines(
        CONSERVATIVE_HP,
        sizes="small,base,large",
        snr_levels="inf,20,10,5,0,-5,-10,-20",
        epochs=5,
    )
    state["stages"].append({
        "phase": 1, "name": "baselines_conservative", "hp": asdict(CONSERVATIVE_HP),
        "rc": rc, "out_dir": str(out_dir),
    })
    save_state()

    if rc != 0 or summary is None:
        log(f"!! baselines FAILED rc={rc}; aborting.")
        return 1

    for size, hist in summary.get("histories", {}).items():
        log(f"  {size} max_val_f1={max(hist):.3f} at ep{hist.index(max(hist))}  (history={[round(v, 3) for v in hist]})")

    chosen_hp = CONSERVATIVE_HP

    # ----------------------------------------------------------------
    # Phase 2: SNRGate full (8-SNR sweep) with chosen HP
    # ----------------------------------------------------------------
    log(f"Phase 2/4: SNRGate (HP={chosen_hp.name}, 8 epoch, full 8-SNR sweep)")
    rc, snrgate_dir = stage_snrgate(chosen_hp, snr_levels=None, epochs=8)
    state["stages"].append({
        "phase": 2, "name": "snrgate", "hp": asdict(chosen_hp),
        "rc": rc, "out_dir": str(snrgate_dir),
    })
    save_state()
    if rc != 0:
        log(f"!! SNRGate FAILED rc={rc}")

    # ----------------------------------------------------------------
    # Phase 3: HardGate ablation
    # ----------------------------------------------------------------
    snrgate_ckpt = snrgate_dir / "checkpoints" / "snrgate.pt"
    if snrgate_ckpt.exists():
        log("Phase 3/4: HardGate ablation")
        rc = stage_hardgate(snrgate_ckpt)
        state["stages"].append({"phase": 3, "name": "hardgate", "rc": rc})
        save_state()
    else:
        log(f"!! skipping HardGate: ckpt not found at {snrgate_ckpt}")

    # ----------------------------------------------------------------
    # Phase 4: Aggregate
    # ----------------------------------------------------------------
    log("Phase 4/4: aggregate")
    rc = stage_aggregate()
    state["stages"].append({"phase": 4, "name": "aggregate", "rc": rc})
    save_state()

    state["finished"] = time.time()
    state["duration_h"] = round((state["finished"] - STARTED_AT) / 3600, 2)
    save_state()
    log(f"=== orchestrator done in {state['duration_h']}h ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
