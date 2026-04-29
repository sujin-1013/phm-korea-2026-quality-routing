"""Paderborn University bearing dataset (KAt Lessmeier et al., 2016).

Standard 4-class mapping used here:
- ``N``  (normal)            : K001–K006        — 6 healthy bearings
- ``IR`` (inner race damage) : KI01..KI21       — artificial + real fatigue
- ``OR`` (outer race damage) : KA01..KA30       — artificial + real fatigue
- ``RE`` (rolling element)   : KB23..KB27       — real combined / ball damage

Each bearing yields ~80 measurements (4 operating conditions × 20 trials),
each ~4 s @ 64 kHz. Files are MATLAB v7.3 (HDF5) — read via ``h5py``.

Cite:
    Lessmeier, C., Kimotho, J. K., Zimmer, D., Sextro, W. (2016).
    Condition Monitoring of Bearing Damage in Electromechanical Drive Systems...
    PHME European Conference 2016.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CLASS_NAMES = ("N", "IR", "OR", "RE")
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# Bearing-id → class. Source: Lessmeier 2016 Table 4 + KAt website annotations.
BEARING_CLASS: dict[str, str] = {
    # Healthy
    **{f"K00{i}": "N" for i in range(1, 7)},
    # Inner race (artificial KI01..KI09 + real KI04, KI14, KI16-18, KI21)
    **{f"KI0{i}": "IR" for i in range(1, 10)},
    "KI14": "IR", "KI16": "IR", "KI17": "IR", "KI18": "IR", "KI21": "IR",
    # Outer race (artificial KA01..KA09 + real KA15-22, KA30)
    **{f"KA0{i}": "OR" for i in range(1, 10)},
    **{f"KA{i}": "OR" for i in (15, 16, 17, 18, 22, 30)},
    # Rolling-element / combined (real KB23-KB27)
    **{f"KB{i}": "RE" for i in (23, 24, 27)},
}


@dataclass
class PaderbornFileEntry:
    bearing_id: str
    label: str          # "N" | "IR" | "OR" | "RE"
    label_idx: int
    path: Path
    operating_condition: str | None = None  # e.g., "N15_M07_F10"
    trial: int | None = None


# Paderborn filenames look like: N15_M07_F10_K001_1.mat — operating-cond_bearing_trial
_FNAME_RE = re.compile(
    r"^(?P<oc>[NMF\d_]+?)_(?P<bid>K[A-Z0-9]+)_(?P<trial>\d+)\.mat$"
)


def discover_files(root: str | Path, *, classes: Sequence[str] = CLASS_NAMES) -> list[PaderbornFileEntry]:
    """Walk ``root`` for Paderborn ``.mat`` files matching one of ``classes``."""
    root = Path(root)
    entries: list[PaderbornFileEntry] = []
    if not root.exists():
        return entries
    for path in root.rglob("*.mat"):
        m = _FNAME_RE.match(path.name)
        if m is None:
            continue
        bid = m.group("bid")
        label = BEARING_CLASS.get(bid)
        if label is None or label not in classes:
            continue
        entries.append(
            PaderbornFileEntry(
                bearing_id=bid,
                label=label,
                label_idx=CLASS_TO_IDX[label],
                path=path,
                operating_condition=m.group("oc"),
                trial=int(m.group("trial")),
            )
        )
    return entries


SAMPLE_RATE_HZ = 64_000  # Paderborn-PU vibration sampling rate


def load_signal(path: Path | str, *, channel: str = "vibration_1") -> np.ndarray:
    """Load a named channel from a Paderborn-PU .mat file.

    The KAt files are MATLAB v5/v7 (NOT v7.3) — read via ``scipy.io.loadmat``.
    Each file's top-level key matches the file basename (e.g., ``N09_M07_F10_K001_1``);
    inside there's a struct with ``Y`` holding 7 named channels. We pick the
    one whose ``Name`` field equals ``channel`` (default: ``vibration_1``,
    the accelerometer at the bearing housing — the standard PHM signal).
    """
    from scipy.io import loadmat

    raw = loadmat(str(path), squeeze_me=False)
    keys = [k for k in raw.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"no MATLAB struct in {path}")
    top = raw[keys[0]][0, 0]
    Y = top["Y"][0, 0] if top["Y"].dtype.names else top["Y"]
    # Y is a (1, 7) struct array — each entry has Name + Data fields.
    Y_arr = top["Y"]  # shape (1, 7)
    n_channels = Y_arr.shape[1]
    for i in range(n_channels):
        ch = Y_arr[0, i]
        name_arr = ch["Name"]
        # Name is a numpy 1-element array of unicode string.
        name = name_arr.item() if name_arr.size == 1 else str(name_arr.flatten()[0])
        if name == channel:
            data = ch["Data"]
            return np.asarray(data, dtype=np.float32).squeeze()
    available = []
    for i in range(n_channels):
        ch = Y_arr[0, i]
        n = ch["Name"].item() if ch["Name"].size == 1 else str(ch["Name"].flatten()[0])
        available.append(n)
    raise KeyError(
        f"channel {channel!r} not in {path.name}; available: {available}"
    )


def stratified_group_kfold_indices(
    entries: Sequence[PaderbornFileEntry],
    n_splits: int = 4,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split by bearing_id, stratifying as much as possible by class.

    Uses sklearn.model_selection.StratifiedGroupKFold so each fold's test set
    contains files from a *disjoint* set of bearings — preventing data leakage
    where windows from the same bearing appear in train + test.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([e.label_idx for e in entries])
    groups = np.array([e.bearing_id for e in entries])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sgkf.split(np.zeros_like(y), y, groups)]


# ---------------------------------------------------------------------------
# Synthetic fallback — exercise the pipeline before data lands.
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    n_per_class: int = 8
    sample_rate: int = 64_000
    duration_s: float = 4.0
    seed: int = 42
    # class-conditional fault frequencies (rough proxy for BPFO/BPFI/BSF)
    fault_hz: dict[str, float] = field(
        default_factory=lambda: {"N": 0.0, "IR": 230.0, "OR": 154.0, "RE": 195.0}
    )


def synthesize(cfg: SyntheticConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(signals, labels, group_ids)`` for a synthetic Paderborn-like split.

    Each "bearing" produces 1 long signal; downstream windowing decides the
    sample count. The class-conditional component is a series of impulses at
    ``fault_hz`` plus broadband noise — a CWRU/Paderborn-style proxy.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    sigs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for cls in CLASS_NAMES:
        for k in range(cfg.n_per_class):
            n = int(cfg.sample_rate * cfg.duration_s)
            t = np.arange(n) / cfg.sample_rate
            base = rng.standard_normal(n).astype(np.float32) * 0.05
            f = cfg.fault_hz[cls]
            if f > 0:
                impulse_period = 1.0 / f
                ticks = np.zeros(n, dtype=np.float32)
                step = max(1, int(impulse_period * cfg.sample_rate))
                ticks[::step] = 1.0
                # Ring each impulse with a damped sinusoid at ~3 kHz.
                ring_freq = 3000.0
                ring_decay = 800.0
                kernel_len = 256
                kt = np.arange(kernel_len) / cfg.sample_rate
                kernel = np.exp(-ring_decay * kt) * np.sin(2 * np.pi * ring_freq * kt)
                kernel = kernel.astype(np.float32)
                ticks = np.convolve(ticks, kernel, mode="same").astype(np.float32)
                base += 0.5 * ticks
            sigs.append(base)
            labels.append(CLASS_TO_IDX[cls])
            groups.append(f"{cls}_synthetic_{k:02d}")
    return np.stack(sigs), np.asarray(labels), np.asarray(groups)
