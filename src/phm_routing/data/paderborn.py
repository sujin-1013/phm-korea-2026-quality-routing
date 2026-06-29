"""Paderborn University bearing dataset (KAt Lessmeier et al., 2016).

Two class mappings supported:

* **4-class** (N/IR/OR/RE) — historical mapping retained for parser completeness.
  - ``N``  (normal)            : K001–K006        — 6 healthy bearings
  - ``IR`` (inner race damage) : KI01..KI21       — artificial + real fatigue
  - ``OR`` (outer race damage) : KA01..KA30       — artificial + real fatigue
  - ``RE`` (rolling element)   : KB23..KB27       — real combined / ball damage

* **3-class** (N/IR/OR) — locked for this quality-routing repository.
  RE class dropped (only 3 bearings — insufficient for A2R protocol). Final
  experiments use this mapping via ``pu_c1_a2r_split`` / ``materialise_pu_c1``.

Each bearing yields ~80 measurements (4 operating conditions × 20 trials),
each ~4 s @ 64 kHz. Files are MATLAB v5/v7 — read via ``scipy.io.loadmat``.

Cite:
    Lessmeier, C., Kimotho, J. K., Zimmer, D., Sextro, W. (2016).
    Condition Monitoring of Bearing Damage in Electromechanical Drive Systems...
    PHME European Conference 2016.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CLASS_NAMES = ("N", "IR", "OR", "RE")
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# 3-class mapping — paper standard (PU-C1 A2R, locked 2026-05-06).
# ---------------------------------------------------------------------------
CLASS_NAMES_3 = ("N", "IR", "OR")
CLASS_TO_IDX_3 = {n: i for i, n in enumerate(CLASS_NAMES_3)}

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


# ---------------------------------------------------------------------------
# PU-C1 Artificial-to-Real (A2R) protocol locked for quality-routing runs.
# ---------------------------------------------------------------------------

# Train: 13 bearings (artificial damage) actually available locally.
# Spec L85 originally listed KA01-KA09 + KI01-KI09 = 18 bearings; in practice the
# KAt server returns 404 for KA02, KI02, KI06, KI09 (deprecated/removed), so only
# 12 of 18 artificial bearings are obtainable. We hold out KA08/KI08 for val,
# leaving 10 OR/IR artificial bearings + K002 for train. The OR side is slightly
# heavier (7 vs 5 IR).
PU_C1_TRAIN_BEARINGS: tuple[str, ...] = (
    "K002",                                                                  # healthy
    "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA09",                  # OR artificial (7; KA08 → val)
    "KI01", "KI03", "KI04", "KI05", "KI07",                                  # IR artificial (5; KI08 → val)
)

# Within-train val: 2 disjoint OR/IR bearings (last available artificial).
# K002 file-level 80/20 split is handled inside ``pu_c1_a2r_split`` because
# it's the single healthy bearing in train (no second healthy to leave out).
PU_C1_VAL_BEARINGS: tuple[str, ...] = ("KA08", "KI08")

# Test: 10 bearings (real damage / fatigue from accelerated wear). Bearing-disjoint from train.
PU_C1_TEST_BEARINGS: tuple[str, ...] = (
    "K001",                                                                  # healthy
    "KA15", "KA16", "KA22", "KA30",                                          # OR real (4)
    "KI14", "KI16", "KI17", "KI18", "KI21",                                  # IR real (5)
)

# Bearings that are excluded from PU-C1 entirely (NOT used in train/val/test).
PU_C1_EXCLUDED_BEARINGS: tuple[str, ...] = (
    "K003", "K004", "K005", "K006",                                          # healthy unused
    "KB23", "KB24", "KB27",                                                  # combined damage (RE)
)

# 3-class subset of BEARING_CLASS, for PU-C1 use only.
BEARING_CLASS_3: dict[str, str] = {
    bid: cls for bid, cls in BEARING_CLASS.items() if cls in CLASS_NAMES_3
}


def _validate_pu_c1_constants() -> None:
    """Ensure no overlap and full coverage of expected bearings."""
    train = set(PU_C1_TRAIN_BEARINGS)
    val = set(PU_C1_VAL_BEARINGS)
    test = set(PU_C1_TEST_BEARINGS)
    excluded = set(PU_C1_EXCLUDED_BEARINGS)
    if train & val:
        # K002 file-split is documented in pu_c1_a2r_split; the bearing
        # appears only in TRAIN constants but contributes a small file-level
        # subset to val. So train/val constants must be bearing-disjoint here.
        raise AssertionError(f"PU-C1 train/val bearing constants overlap: {train & val}")
    if train & test:
        raise AssertionError(f"PU-C1 train/test overlap: {train & test}")
    if val & test:
        raise AssertionError(f"PU-C1 val/test overlap: {val & test}")
    if (train | val | test) & excluded:
        raise AssertionError("PU-C1 train/val/test contains excluded bearing")
    # All listed bearings must be in the 3-class mapping.
    for bid in train | val | test:
        if bid not in BEARING_CLASS_3:
            raise AssertionError(f"{bid} not in BEARING_CLASS_3 (3-class mapping)")


_validate_pu_c1_constants()


@dataclass
class PUC1Split:
    """Result of ``pu_c1_a2r_split`` — three lists of file entries."""

    train: list[PaderbornFileEntry]
    val: list[PaderbornFileEntry]
    test: list[PaderbornFileEntry]

    def summary(self) -> dict[str, dict[str, int]]:
        """Per-split per-class file count."""
        out: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        for split_name, entries in (("train", self.train), ("val", self.val), ("test", self.test)):
            counts: dict[str, int] = {c: 0 for c in CLASS_NAMES_3}
            for e in entries:
                counts[e.label] = counts.get(e.label, 0) + 1
            out[split_name] = counts
        return out


def pu_c1_a2r_split(
    entries: Sequence[PaderbornFileEntry],
    *,
    max_files_per_bearing: int = 8,
    seed: int = 42,
    val_file_ratio_for_K002: float = 0.2,
) -> PUC1Split:
    """Apply PU-C1 A2R protocol to a list of file entries.

    Returns ``PUC1Split(train, val, test)`` with bearing-disjoint splits
    (except K002, which uses a deterministic file-level 80/20 split because
    it is the single healthy bearing assigned to train).

    Files outside the 3-class mapping (KB / K003-K006) are silently dropped.
    Each kept bearing contributes at most ``max_files_per_bearing`` files,
    sampled deterministically by ``seed``.
    """
    if not (0 < val_file_ratio_for_K002 < 1):
        raise ValueError("val_file_ratio_for_K002 must lie in (0, 1)")

    # Group by bearing, dropping anything outside the 3-class mapping.
    by_bearing: dict[str, list[PaderbornFileEntry]] = defaultdict(list)
    for e in entries:
        if e.bearing_id in BEARING_CLASS_3:
            by_bearing[e.bearing_id].append(e)

    rng = np.random.default_rng(seed)

    train: list[PaderbornFileEntry] = []
    val: list[PaderbornFileEntry] = []
    test: list[PaderbornFileEntry] = []

    for bearing_id in sorted(by_bearing.keys()):  # sort for determinism
        files = by_bearing[bearing_id]
        # Deterministic per-bearing file selection (cap at max_files_per_bearing).
        idx = rng.permutation(len(files))[:max_files_per_bearing]
        kept = [files[i] for i in idx]

        if bearing_id in PU_C1_TEST_BEARINGS:
            test.extend(kept)
        elif bearing_id in PU_C1_VAL_BEARINGS:
            val.extend(kept)
        elif bearing_id == "K002":
            # File-level 80/20 split — single healthy bearing in train.
            split_rng = np.random.default_rng(seed + 1)
            file_perm = split_rng.permutation(len(kept))
            n_val = max(1, int(round(len(kept) * val_file_ratio_for_K002)))
            val_set = set(file_perm[:n_val].tolist())
            for i, f in enumerate(kept):
                (val if i in val_set else train).append(f)
        elif bearing_id in PU_C1_TRAIN_BEARINGS:
            train.extend(kept)
        # Any other bearing (e.g. K003-K006) is silently dropped per spec.

    return PUC1Split(train=train, val=val, test=test)


def _load_and_window_entries(
    entries: Sequence[PaderbornFileEntry],
    *,
    window: int,
    stride: int,
    label_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load + window a list of entries, label-mapping via ``label_to_idx``.

    Mirrors the loop previously inlined inside ``exp03_snrgate.run.materialise``,
    extracted here so all experiments share a single windowing path.
    """
    from .windowing import sliding_windows

    win_list: list[np.ndarray] = []
    lab_list: list[np.ndarray] = []
    n_skipped = 0
    for e in entries:
        try:
            sig = load_signal(e.path)[: 64_000 * 4].astype(np.float32)
        except Exception as exc:
            n_skipped += 1
            print(
                f"[warn] skip {e.path.name}: {type(exc).__name__}: {str(exc)[:80]}",
                flush=True,
            )
            continue
        w = sliding_windows(sig, window, stride)
        win_list.append(w)
        lab_list.append(np.full(len(w), label_to_idx[e.label], dtype=np.int64))
    if n_skipped:
        print(f"[info] skipped {n_skipped} corrupt files", flush=True)
    if not win_list:
        return (
            np.empty((0, window), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.concatenate(win_list), np.concatenate(lab_list)


def materialise_pu_c1(
    data_root: str | Path,
    *,
    window: int = 2048,
    stride: int = 2048,
    max_files_per_bearing: int = 8,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray,  # train (X, y)
    np.ndarray, np.ndarray,  # val   (X, y)
    np.ndarray, np.ndarray,  # test  (X, y)
]:
    """Single-call materialisation of the PU-C1 A2R 3-class split.

    Returns six ``np.ndarray`` arrays: train / val / test windows + labels.
    Labels are 0=N, 1=IR, 2=OR per ``CLASS_TO_IDX_3``.

    This split is pinned by ``tests/test_paderborn_pu_c1.py``.
    """
    entries = discover_files(data_root, classes=CLASS_NAMES_3)
    if not entries:
        raise FileNotFoundError(
            f"No PU-C1-compatible Paderborn .mat files under {data_root}. "
            f"Synthetic fallback is intentionally NOT supported for the "
            f"locked PU-C1 A2R protocol — point --data-root to real data."
        )

    split = pu_c1_a2r_split(
        entries,
        max_files_per_bearing=max_files_per_bearing,
        seed=seed,
    )
    summary = split.summary()
    print(
        "[pu_c1] split (file counts):  "
        f"train={summary['train']}  val={summary['val']}  test={summary['test']}",
        flush=True,
    )

    X_tr, y_tr = _load_and_window_entries(
        split.train, window=window, stride=stride, label_to_idx=CLASS_TO_IDX_3
    )
    X_val, y_val = _load_and_window_entries(
        split.val, window=window, stride=stride, label_to_idx=CLASS_TO_IDX_3
    )
    X_test, y_test = _load_and_window_entries(
        split.test, window=window, stride=stride, label_to_idx=CLASS_TO_IDX_3
    )
    print(
        "[pu_c1] windows:  "
        f"train={X_tr.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}  "
        f"window={window} stride={stride}",
        flush=True,
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def inverse_freq_class_weights(
    y: np.ndarray, num_class: int = 3
) -> "list[float]":
    """Per-class weight = total / (num_class * count_c). Standard inverse-freq.

    Returned as a plain Python list (no torch dependency at this layer); the
    training loop wraps it into a ``torch.Tensor`` of the appropriate dtype/device.
    """
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero on empty classes
    weights = counts.sum() / (num_class * counts)
    return weights.astype(np.float32).tolist()
