"""Evaluation metrics for fault classification."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return np.asarray(f1, dtype=np.float64)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def confusion(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
