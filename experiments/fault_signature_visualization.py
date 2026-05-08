"""Fault signature visualization for KA15/KA22 (failures) vs KA16/KA30 (success) + K001.

Compares raw vibration signal characteristics across the 5 test bearings to
diagnose WHY KA15 and KA22 are misclassified despite same nominal "real OR" label.

Plots produced:
  Fig. T1: time-domain segment + envelope (Hilbert) per bearing
  Fig. T2: FFT magnitude spectrum + envelope spectrum per bearing
  Fig. T3: kurtosis over sliding windows per bearing
  Fig. T4: combined paper figure (3 rows x 5 cols)

All saved to paper/figures/ for paper inclusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_routing.data.paderborn import discover_files, load_signal, pu_c1_a2r_split, CLASS_NAMES_3

DATA_ROOT = ROOT / "data" / "raw" / "paderborn_pu"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 64_000

# Bearings to compare — 2 catastrophic failures + 2 successes + 1 healthy
TARGET_BEARINGS = [
    ("KA15", "OR (real)", "fail", "tab:red"),
    ("KA22", "OR (real)", "fail", "tab:red"),
    ("KA16", "OR (real)", "success", "tab:green"),
    ("KA30", "OR (real)", "success", "tab:green"),
    ("K001", "Healthy", "neutral", "tab:blue"),
]


def load_one_file_per_bearing(target_bearings) -> dict[str, dict]:
    """Pick one .mat file per target bearing — preferably N15_M07_F10 condition."""
    entries = discover_files(DATA_ROOT, classes=CLASS_NAMES_3)
    by_bearing = {}
    for entry in entries:
        if entry.bearing_id in {b for b, *_ in target_bearings}:
            by_bearing.setdefault(entry.bearing_id, []).append(entry)

    out = {}
    for bearing, label, status, color in target_bearings:
        candidates = by_bearing.get(bearing, [])
        # prefer N15_M07_F10 operating condition for consistency
        cand = [c for c in candidates if "N15_M07_F10" in (c.operating_condition or "")]
        if not cand:
            cand = candidates
        if not cand:
            print(f"[warn] no file found for {bearing}")
            continue
        # take first available (sorted)
        sel = sorted(cand, key=lambda c: c.path.name)[0]
        sig = load_signal(sel.path)[: SAMPLE_RATE * 4].astype(np.float32)
        out[bearing] = {
            "label": label,
            "status": status,
            "color": color,
            "signal": sig,
            "file": sel.path.name,
        }
        print(f"[info] {bearing} ({label}, {status}): {sel.path.name} ({len(sig)} samples)")
    return out


def fig_time_envelope(data: dict, out_path: Path):
    """Time-domain: 0.04s segment + envelope per bearing."""
    bearings = list(data.keys())
    fig, axes = plt.subplots(len(bearings), 1, figsize=(10, 1.6 * len(bearings)),
                              sharex=True, sharey=False)
    if len(bearings) == 1:
        axes = [axes]
    seg_n = int(0.04 * SAMPLE_RATE)  # 40 ms = 2560 samples
    t = np.arange(seg_n) / SAMPLE_RATE * 1000  # ms
    for ax, bearing in zip(axes, bearings):
        d = data[bearing]
        sig = d["signal"][:seg_n]
        env = np.abs(hilbert(sig))
        ax.plot(t, sig, color="lightgray", linewidth=0.5, label="raw")
        ax.plot(t, env, color=d["color"], linewidth=1.2, label="envelope")
        ax.set_ylabel(f"{bearing}\n({d['label']})", fontsize=9)
        ax.text(0.99, 0.95, f"status: {d['status']}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor=d["color"], alpha=0.9))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("Time-domain: raw + Hilbert envelope (40 ms segment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[done] {out_path}")


def fig_spectrum(data: dict, out_path: Path):
    """FFT magnitude spectrum + envelope spectrum per bearing (0–8 kHz)."""
    bearings = list(data.keys())
    n_window = 4096
    fig, axes = plt.subplots(len(bearings), 2, figsize=(14, 1.6 * len(bearings)),
                              sharex="col")
    if len(bearings) == 1:
        axes = axes.reshape(1, -1)
    for row, bearing in enumerate(bearings):
        d = data[bearing]
        sig = d["signal"][:n_window]
        env = np.abs(hilbert(sig))
        env = env - env.mean()
        # FFT (raw)
        f = np.fft.rfftfreq(n_window, 1 / SAMPLE_RATE)
        s_raw = np.abs(np.fft.rfft(sig))
        s_env = np.abs(np.fft.rfft(env))
        # raw spectrum 0–8 kHz
        mask = f <= 8000
        axes[row, 0].semilogy(f[mask], s_raw[mask], color=d["color"], linewidth=0.7)
        axes[row, 0].set_ylabel(f"{bearing}\n({d['label']})", fontsize=8)
        axes[row, 0].grid(True, alpha=0.3, which="both")
        # envelope spectrum 0–1 kHz (for fault frequencies)
        emask = f <= 1000
        axes[row, 1].plot(f[emask], s_env[emask], color=d["color"], linewidth=0.8)
        axes[row, 1].grid(True, alpha=0.3)
    axes[0, 0].set_title("Raw FFT magnitude (0–8 kHz, log-y)")
    axes[0, 1].set_title("Envelope spectrum (0–1 kHz, fault frequencies)")
    axes[-1, 0].set_xlabel("frequency (Hz)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[done] {out_path}")


def fig_kurtosis(data: dict, out_path: Path):
    """Kurtosis over 50 sliding windows per bearing — captures impulsivity."""
    bearings = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    win = 4096
    stride = 2048
    for bearing in bearings:
        d = data[bearing]
        sig = d["signal"]
        kvals = []
        for i in range(0, len(sig) - win + 1, stride):
            kvals.append(kurtosis(sig[i:i + win], fisher=False))  # Fisher=False → Pearson kurt
        kvals = np.array(kvals)
        marker = "o" if d["status"] == "success" else ("x" if d["status"] == "fail" else "s")
        ax.plot(kvals, marker=marker, color=d["color"], linewidth=1.0, alpha=0.8,
                label=f"{bearing} ({d['label']}, {d['status']})")
    ax.set_xlabel("window index (4096-sample window, stride=2048)")
    ax.set_ylabel("Kurtosis (Pearson)")
    ax.set_title("Kurtosis over sliding windows — impulsivity signature")
    ax.axhline(3, linestyle="--", color="gray", alpha=0.5, label="Gaussian baseline (κ=3)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[done] {out_path}")


def fig_combined(data: dict, out_path: Path):
    """Combined paper figure: per-bearing rows × {time, FFT, env spectrum} cols."""
    bearings = list(data.keys())
    fig, axes = plt.subplots(len(bearings), 3, figsize=(14, 2 * len(bearings)),
                              sharex="col")
    seg_n = int(0.02 * SAMPLE_RATE)  # 20 ms
    t = np.arange(seg_n) / SAMPLE_RATE * 1000
    n_fft = 4096
    f = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    for row, bearing in enumerate(bearings):
        d = data[bearing]
        sig = d["signal"][:seg_n]
        env_full = np.abs(hilbert(d["signal"][:n_fft]))
        env_full = env_full - env_full.mean()
        # col 0: time + envelope
        ax0 = axes[row, 0]
        ax0.plot(t, sig, color="lightgray", linewidth=0.5)
        ax0.plot(t, np.abs(hilbert(sig)), color=d["color"], linewidth=1.0)
        ax0.set_ylabel(f"{bearing}\n{d['label']}\n({d['status']})", fontsize=8)
        ax0.grid(True, alpha=0.3)
        # col 1: raw FFT 0–8 kHz log
        ax1 = axes[row, 1]
        s_raw = np.abs(np.fft.rfft(d["signal"][:n_fft]))
        m = f <= 8000
        ax1.semilogy(f[m], s_raw[m], color=d["color"], linewidth=0.7)
        ax1.grid(True, alpha=0.3, which="both")
        # col 2: envelope spectrum 0–1 kHz
        ax2 = axes[row, 2]
        s_env = np.abs(np.fft.rfft(env_full))
        em = f <= 1000
        ax2.plot(f[em], s_env[em], color=d["color"], linewidth=0.7)
        ax2.grid(True, alpha=0.3)
    axes[0, 0].set_title("Time-domain (20 ms) + Hilbert envelope")
    axes[0, 1].set_title("FFT magnitude (0–8 kHz, log-y)")
    axes[0, 2].set_title("Envelope spectrum (0–1 kHz)")
    axes[-1, 0].set_xlabel("time (ms)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    axes[-1, 2].set_xlabel("frequency (Hz)")
    fig.suptitle("Fault signature comparison — KA15/KA22 (catastrophic failure) vs "
                 "KA16/KA30 (success) vs K001 (healthy)", fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[done] {out_path}")


def main() -> None:
    print("[info] loading test bearing files...")
    data = load_one_file_per_bearing(TARGET_BEARINGS)
    if not data:
        print("[error] no files loaded")
        return

    fig_time_envelope(data, OUT_DIR / "T1_time_envelope_KA15_KA22.pdf")
    fig_spectrum(data, OUT_DIR / "T2_fft_envelope_spectrum.pdf")
    fig_kurtosis(data, OUT_DIR / "T3_kurtosis.pdf")
    fig_combined(data, OUT_DIR / "T4_combined_fault_signature.pdf")

    # Quick numerical summary for paper text
    print("\n=== Kurtosis summary (full 4-second signal) ===")
    print(f"{'bearing':<8} {'label':<14} {'status':<10} {'kurtosis':<10} {'std':<10}")
    for bearing, d in data.items():
        k = kurtosis(d["signal"], fisher=False)
        s = float(np.std(d["signal"]))
        print(f"{bearing:<8} {d['label']:<14} {d['status']:<10} {k:<10.3f} {s:<10.4f}")


if __name__ == "__main__":
    main()
