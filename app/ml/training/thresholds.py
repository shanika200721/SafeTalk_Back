"""Validation-only threshold selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn import metrics as sk_metrics

from app.ml.training.schemas import ThresholdStrategy


@dataclass(frozen=True)
class ThresholdSelection:
    strategy: ThresholdStrategy
    threshold: float | None
    objective: str
    validation_metric: float | None
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "threshold": self.threshold,
            "objective": self.objective,
            "validation_metric": self.validation_metric,
            "warnings": self.warnings,
            "note": "Selected with validation data only; not clinically validated.",
        }


def _candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    values = np.unique(np.clip(probabilities.astype(float), 0.0, 1.0))
    grid = np.unique(np.concatenate([[0.0, 0.5, 1.0], values]))
    return np.sort(grid)


def labels_from_threshold(probabilities: Sequence[float], threshold: float, *, positive_label=1, negative_label=0):
    return np.where(np.asarray(probabilities, dtype=float) >= threshold, positive_label, negative_label)


def select_binary_threshold(
    y_validation_true: Sequence,
    y_validation_probability: Sequence[float],
    *,
    strategy: ThresholdStrategy | str = ThresholdStrategy.DEFAULT,
    positive_label=1,
    fixed_threshold: float | None = None,
    min_recall: float | None = None,
    min_precision: float | None = None,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> ThresholdSelection:
    strategy = ThresholdStrategy(strategy)
    y_true = np.asarray(y_validation_true)
    proba = np.asarray(y_validation_probability, dtype=float)
    if proba.ndim != 1 or len(proba) != len(y_true):
        raise ValueError("threshold selection requires validation probabilities matching labels")
    if np.any(~np.isfinite(proba)) or np.any((proba < 0) | (proba > 1)):
        raise ValueError("validation probabilities must be finite values between 0 and 1")
    binary_true = (y_true == positive_label).astype(int)
    warnings: list[str] = []
    if len(set(binary_true)) < 2:
        warnings.append("validation labels contain one class; threshold metrics may be unstable")

    if strategy == ThresholdStrategy.DEFAULT:
        return ThresholdSelection(strategy, 0.5, "default probability threshold", None, warnings)
    if strategy == ThresholdStrategy.FIXED:
        if fixed_threshold is None:
            raise ValueError("fixed threshold strategy requires fixed_threshold")
        if not 0 <= fixed_threshold <= 1:
            raise ValueError("fixed_threshold must be between 0 and 1")
        return ThresholdSelection(strategy, float(fixed_threshold), "fixed threshold supplied by config", None, warnings)

    best_threshold = 0.5
    best_value: float | None = None
    thresholds = _candidate_thresholds(proba)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        precision = sk_metrics.precision_score(binary_true, pred, zero_division=0)
        recall = sk_metrics.recall_score(binary_true, pred, zero_division=0)
        f1 = sk_metrics.f1_score(binary_true, pred, zero_division=0)
        cm = sk_metrics.confusion_matrix(binary_true, pred, labels=[0, 1])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])

        if strategy == ThresholdStrategy.MAX_F1:
            value = float(f1)
            eligible = True
            objective = "maximize validation F1"
        elif strategy == ThresholdStrategy.RECALL_PRIORITY:
            target = 0.8 if min_recall is None else min_recall
            eligible = recall >= target
            value = float(precision)
            objective = f"maximize precision subject to validation recall >= {target}"
        elif strategy == ThresholdStrategy.PRECISION_PRIORITY:
            target = 0.8 if min_precision is None else min_precision
            eligible = precision >= target
            value = float(recall)
            objective = f"maximize recall subject to validation precision >= {target}"
        elif strategy == ThresholdStrategy.COST_SENSITIVE:
            eligible = True
            value = -float(false_negative_cost * fn + false_positive_cost * fp)
            objective = "minimize validation false-negative/false-positive cost"
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy}")

        if eligible and (best_value is None or value > best_value or (value == best_value and threshold > best_threshold)):
            best_value = value
            best_threshold = float(threshold)

    if best_value is None:
        warnings.append("no threshold satisfied the validation constraint; defaulting to 0.5")
        best_value = None
        best_threshold = 0.5
    return ThresholdSelection(strategy, best_threshold, objective, best_value, warnings)


def select_multiclass_thresholds(*, strategy: str = "argmax") -> dict:
    if strategy != "argmax":
        return {
            "strategy": strategy,
            "thresholds": None,
            "note": "One-vs-rest multiclass thresholds are a future interface only in Phase 3B.",
        }
    return {"strategy": "argmax", "thresholds": None}
