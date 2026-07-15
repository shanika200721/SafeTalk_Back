"""Common deterministic metrics for binary and multiclass classifiers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import label_binarize

from app.ml.training.schemas import MetricSet


def _labels(y_true: Sequence, y_pred: Sequence | None = None, labels: Sequence | None = None) -> list:
    if labels is not None:
        return list(labels)
    values = list(y_true)
    if y_pred is not None:
        values += list(y_pred)
    return sorted(set(values), key=lambda item: str(item))


def _warnings_for_absent_classes(y_true: Sequence, labels: Sequence) -> list[str]:
    present = set(y_true)
    return [f"class absent from y_true: {label}" for label in labels if label not in present]


def _as_array(values) -> np.ndarray:
    return np.asarray(values)


def _validate_probability_vector(probabilities: Sequence[float], expected: int) -> np.ndarray:
    proba = _as_array(probabilities).astype(float)
    if proba.ndim != 1 or len(proba) != expected:
        raise ValueError("binary probabilities must be a one-dimensional vector matching y_true")
    if np.any(~np.isfinite(proba)) or np.any((proba < 0) | (proba > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1")
    return proba


def evaluate_binary_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    y_probability: Sequence[float] | None = None,
    positive_label=1,
    threshold: float | None = None,
    labels: Sequence | None = None,
) -> MetricSet:
    label_values = _labels(y_true, y_pred, labels)
    warnings = _warnings_for_absent_classes(y_true, label_values)
    if positive_label not in label_values:
        label_values.append(positive_label)
        label_values = _labels(label_values)
        warnings.append(f"positive label absent from labels: {positive_label}")

    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=label_values)
    positive_index = label_values.index(positive_label)
    tp = int(cm[positive_index, positive_index])
    fn = int(cm[positive_index, :].sum() - tp)
    fp = int(cm[:, positive_index].sum() - tp)
    tn = int(cm.sum() - tp - fn - fp)
    specificity = tn / (tn + fp) if (tn + fp) else None

    metric = MetricSet(
        accuracy=float(sk_metrics.accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(sk_metrics.balanced_accuracy_score(y_true, y_pred)),
        precision_macro=float(sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=float(sk_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_macro=float(sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_weighted=float(sk_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_macro=float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        specificity=specificity,
        confusion_matrix=cm.astype(int).tolist(),
        support={str(label): int(np.sum(_as_array(y_true) == label)) for label in label_values},
        threshold=threshold,
        warnings=warnings,
        false_negative_count=fn,
        false_positive_count=fp,
    )

    if y_probability is not None:
        proba = _validate_probability_vector(y_probability, len(y_true))
        binary_true = (_as_array(y_true) == positive_label).astype(int)
        if len(set(binary_true)) < 2:
            metric.warnings.append("ROC-AUC and PR-AUC undefined because y_true has one class")
        else:
            metric.roc_auc = float(sk_metrics.roc_auc_score(binary_true, proba))
            metric.pr_auc = float(sk_metrics.average_precision_score(binary_true, proba))
        metric.log_loss = float(sk_metrics.log_loss(binary_true, np.column_stack([1 - proba, proba]), labels=[0, 1]))
        metric.brier_score = float(sk_metrics.brier_score_loss(binary_true, proba))
    return metric


def evaluate_multiclass_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    y_probability=None,
    labels: Sequence | None = None,
) -> MetricSet:
    label_values = _labels(y_true, y_pred, labels)
    warnings = _warnings_for_absent_classes(y_true, label_values)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=label_values)
    report = sk_metrics.classification_report(
        y_true,
        y_pred,
        labels=label_values,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        str(label): {
            "precision": float(report[str(label)]["precision"]),
            "recall": float(report[str(label)]["recall"]),
            "f1": float(report[str(label)]["f1-score"]),
            "support": int(report[str(label)]["support"]),
        }
        for label in label_values
        if str(label) in report
    }
    metric = MetricSet(
        accuracy=float(sk_metrics.accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(sk_metrics.balanced_accuracy_score(y_true, y_pred)),
        precision_macro=float(sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=float(sk_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_macro=float(sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_weighted=float(sk_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_macro=float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
        per_class_metrics=per_class,
        support={str(label): int(np.sum(_as_array(y_true) == label)) for label in label_values},
        warnings=warnings,
    )
    if y_probability is not None:
        proba = _as_array(y_probability).astype(float)
        if proba.ndim != 2 or proba.shape[0] != len(y_true) or proba.shape[1] != len(label_values):
            raise ValueError("multiclass probability array must be n_samples x n_classes")
        if np.any(~np.isfinite(proba)) or np.any((proba < 0) | (proba > 1)):
            raise ValueError("probabilities must be finite values between 0 and 1")
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("multiclass probability rows must sum to 1")
        if len(set(y_true)) < 2:
            metric.warnings.append("ROC-AUC and PR-AUC undefined because y_true has one class")
        else:
            y_bin = label_binarize(y_true, classes=label_values)
            try:
                metric.roc_auc = float(sk_metrics.roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
            except ValueError as exc:
                metric.warnings.append(f"one-vs-rest ROC-AUC unavailable: {exc}")
            try:
                metric.pr_auc = float(sk_metrics.average_precision_score(y_bin, proba, average="macro"))
            except ValueError as exc:
                metric.warnings.append(f"macro PR-AUC unavailable: {exc}")
        metric.log_loss = float(sk_metrics.log_loss(y_true, proba, labels=label_values))
    return metric
