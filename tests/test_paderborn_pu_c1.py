"""Unit tests for the locked PU-C1 A2R 3-class protocol.

Pins the data-split contract used by the quality-routing experiments. If any of
these tests start failing without an explicit protocol update, that's a leakage
or regression signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phm_routing.data.paderborn import (
    BEARING_CLASS_3,
    CLASS_NAMES_3,
    CLASS_TO_IDX_3,
    PaderbornFileEntry,
    PU_C1_EXCLUDED_BEARINGS,
    PU_C1_TEST_BEARINGS,
    PU_C1_TRAIN_BEARINGS,
    PU_C1_VAL_BEARINGS,
    inverse_freq_class_weights,
    pu_c1_a2r_split,
)


# ---------------------------------------------------------------------------
# Bearing-list invariants (catches accidental edits to the locked constants)
# ---------------------------------------------------------------------------

def test_train_has_expected_bearings():
    """Locked-in (2026-05-06 sanity + post-download corrected) train list:
    K002 + KA01/03/04/05/06/07/09 (7 OR) + KI01/03/04/05/07 (5 IR) = 13 bearings.
    KA08/KI08 are held out for val; KA02, KI02, KI06, KI09 are 404 on KAt server.
    """
    assert len(PU_C1_TRAIN_BEARINGS) == 13
    assert "K002" in PU_C1_TRAIN_BEARINGS
    # KA06, KA09 added after secondary download (server returned 200).
    assert "KA06" in PU_C1_TRAIN_BEARINGS
    assert "KA09" in PU_C1_TRAIN_BEARINGS
    # Held out for val.
    assert "KA08" not in PU_C1_TRAIN_BEARINGS
    assert "KI08" not in PU_C1_TRAIN_BEARINGS
    # Server-side 404 — must NOT be referenced.
    for missing in ("KA02", "KI02", "KI06", "KI09"):
        assert missing not in PU_C1_TRAIN_BEARINGS, f"{missing} returned 404 on KAt"
        assert missing not in PU_C1_VAL_BEARINGS, f"{missing} returned 404 on KAt"


def test_test_has_10_real_damage_bearings():
    """Spec: K001 + KA15/16/22/30 + KI14/16/17/18/21 = 10 bearings."""
    assert len(PU_C1_TEST_BEARINGS) == 10
    assert "K001" in PU_C1_TEST_BEARINGS
    for b in ("KA15", "KA16", "KA22", "KA30"):
        assert b in PU_C1_TEST_BEARINGS
    for b in ("KI14", "KI16", "KI17", "KI18", "KI21"):
        assert b in PU_C1_TEST_BEARINGS


def test_val_has_2_disjoint_bearings():
    """Last available artificial-damage bearings (KA08, KI08) — see DATA_PROTOCOL_LOCKED.md."""
    assert set(PU_C1_VAL_BEARINGS) == {"KA08", "KI08"}


def test_train_val_test_pairwise_disjoint():
    train = set(PU_C1_TRAIN_BEARINGS)
    val = set(PU_C1_VAL_BEARINGS)
    test = set(PU_C1_TEST_BEARINGS)
    assert not (train & val), "train/val bearings overlap"
    assert not (train & test), "train/test bearings overlap"
    assert not (val & test), "val/test bearings overlap"


def test_excluded_does_not_appear_in_any_split():
    excluded = set(PU_C1_EXCLUDED_BEARINGS)
    used = set(PU_C1_TRAIN_BEARINGS) | set(PU_C1_VAL_BEARINGS) | set(PU_C1_TEST_BEARINGS)
    assert not (excluded & used), f"excluded bearing leaked into splits: {excluded & used}"
    # Spec also pins which bearings are excluded.
    assert {"K003", "K004", "K005", "K006"}.issubset(excluded)
    assert {"KB23", "KB24", "KB27"}.issubset(excluded)


def test_3class_mapping_pins_three_labels():
    assert CLASS_NAMES_3 == ("N", "IR", "OR")
    assert CLASS_TO_IDX_3 == {"N": 0, "IR": 1, "OR": 2}


def test_all_pu_c1_bearings_have_3class_label():
    """Every train/val/test bearing must map to N, IR, or OR."""
    for bid in PU_C1_TRAIN_BEARINGS + PU_C1_VAL_BEARINGS + PU_C1_TEST_BEARINGS:
        assert bid in BEARING_CLASS_3, f"{bid} missing from 3-class mapping"
        assert BEARING_CLASS_3[bid] in CLASS_NAMES_3


# ---------------------------------------------------------------------------
# pu_c1_a2r_split() behaviour
# ---------------------------------------------------------------------------

def _fake_entry(bearing_id: str, trial: int) -> PaderbornFileEntry:
    """Construct a minimal entry without touching disk."""
    label = BEARING_CLASS_3.get(bearing_id, "N")
    return PaderbornFileEntry(
        bearing_id=bearing_id,
        label=label,
        label_idx=CLASS_TO_IDX_3.get(label, 0),
        path=Path(f"/tmp/{bearing_id}_{trial:02d}.mat"),
        operating_condition="N09_M07_F10",
        trial=trial,
    )


def _all_pu_c1_entries(files_per_bearing: int = 8) -> list[PaderbornFileEntry]:
    """Build a synthetic pool: 8 fake files per train/val/test bearing + KB / K003-K006."""
    out: list[PaderbornFileEntry] = []
    for bid in PU_C1_TRAIN_BEARINGS + PU_C1_VAL_BEARINGS + PU_C1_TEST_BEARINGS:
        for t in range(files_per_bearing):
            out.append(_fake_entry(bid, t + 1))
    # Excluded bearings — must be silently dropped by the split.
    for bid in ("K003", "K004", "KB23", "KB27"):
        for t in range(4):
            label = "N" if bid.startswith("K0") else "RE"
            out.append(
                PaderbornFileEntry(
                    bearing_id=bid,
                    label=label,
                    label_idx=0,
                    path=Path(f"/tmp/{bid}_{t}.mat"),
                )
            )
    return out


def test_split_produces_bearing_disjoint_sets_except_K002():
    """Train and val bearings (excluding K002) must not overlap; train/test fully disjoint."""
    entries = _all_pu_c1_entries()
    split = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)

    train_bids = {e.bearing_id for e in split.train}
    val_bids = {e.bearing_id for e in split.val}
    test_bids = {e.bearing_id for e in split.test}

    # K002 may appear in both train and val (file-level 80/20 split is documented).
    assert (train_bids & val_bids).issubset({"K002"}), \
        f"unexpected train/val bearing overlap: {(train_bids & val_bids) - {'K002'}}"
    assert not (train_bids & test_bids), "train/test bearing leakage"
    assert not (val_bids & test_bids), "val/test bearing leakage"


def test_split_drops_excluded_bearings():
    """KB / K003-K006 must NOT appear in any split."""
    entries = _all_pu_c1_entries()
    split = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)
    excluded = set(PU_C1_EXCLUDED_BEARINGS)
    for bucket_name, bucket in (("train", split.train), ("val", split.val), ("test", split.test)):
        bids = {e.bearing_id for e in bucket}
        assert not (bids & excluded), f"{bucket_name} contains excluded bearings: {bids & excluded}"


def test_split_caps_files_per_bearing():
    """If we hand it 20 files per bearing but max=8, no bearing exceeds 8 in any split.

    K002 is split file-level (~6 train + ~2 val), so its train+val sum equals the cap.
    """
    entries: list[PaderbornFileEntry] = []
    for bid in PU_C1_TRAIN_BEARINGS + PU_C1_VAL_BEARINGS + PU_C1_TEST_BEARINGS:
        for t in range(20):
            entries.append(_fake_entry(bid, t))

    split = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)

    for bid in PU_C1_TEST_BEARINGS:
        assert sum(1 for e in split.test if e.bearing_id == bid) == 8
    for bid in PU_C1_VAL_BEARINGS:
        assert sum(1 for e in split.val if e.bearing_id == bid) == 8
    for bid in PU_C1_TRAIN_BEARINGS:
        if bid == "K002":
            n_tr = sum(1 for e in split.train if e.bearing_id == bid)
            n_va = sum(1 for e in split.val if e.bearing_id == bid)
            assert n_tr + n_va == 8, f"K002 file split should sum to cap, got {n_tr}+{n_va}"
            assert n_va >= 1
        else:
            assert sum(1 for e in split.train if e.bearing_id == bid) == 8


def test_split_is_deterministic_given_seed():
    entries = _all_pu_c1_entries()
    a = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)
    b = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)
    assert [str(e.path) for e in a.train] == [str(e.path) for e in b.train]
    assert [str(e.path) for e in a.val] == [str(e.path) for e in b.val]
    assert [str(e.path) for e in a.test] == [str(e.path) for e in b.test]


def test_split_summary_counts_balance():
    """The split summary should account for every PU-C1-eligible entry.

    K003-K006 map to N in ``BEARING_CLASS_3`` but are *excluded* by PU-C1
    (see ``PU_C1_EXCLUDED_BEARINGS``), so they must NOT appear in any split.
    """
    entries = _all_pu_c1_entries()
    split = pu_c1_a2r_split(entries, max_files_per_bearing=8, seed=42)
    summary = split.summary()
    total_in_splits = sum(sum(d.values()) for d in summary.values())
    expected_pu_c1 = sum(
        1
        for e in entries
        if e.bearing_id in BEARING_CLASS_3
        and e.bearing_id not in PU_C1_EXCLUDED_BEARINGS
    )
    assert total_in_splits == expected_pu_c1


# ---------------------------------------------------------------------------
# inverse_freq_class_weights()
# ---------------------------------------------------------------------------

def test_inverse_freq_weights_sum_to_num_class():
    """For weights w_c = N / (K · n_c), Σ w_c · n_c / N = 1, and Σ w_c approx K when balanced."""
    y = np.array([0, 0, 0, 1, 1, 2])  # counts: N=3, IR=2, OR=1
    w = inverse_freq_class_weights(y, num_class=3)
    assert len(w) == 3
    # w_c = 6 / (3 · n_c) → (2/3, 1.0, 2.0). Sum-weighted-by-count should equal num_class.
    weighted_total = sum(w[c] * (y == c).sum() for c in range(3))
    assert abs(weighted_total - 6.0) < 1e-5  # = total samples


def test_inverse_freq_handles_empty_class():
    """An absent class shouldn't crash (count → 1 fallback)."""
    y = np.array([0, 0, 1, 1])  # class 2 absent
    w = inverse_freq_class_weights(y, num_class=3)
    assert len(w) == 3
    assert all(np.isfinite(w))
