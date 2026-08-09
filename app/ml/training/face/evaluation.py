"""Evaluation helpers for Phase 3I Face baselines."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from sklearn import metrics as sk_metrics

from app.ml.training.face.constants import FACE_LABELS


def _prediction_scores(estimator, X: np.ndarray) -> tuple[list[str], list[float]]:
    pred = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        confidence = np.max(proba, axis=1).astype(float).tolist()
    elif hasattr(estimator, "decision_function"):
        margins = np.asarray(estimator.decision_function(X))
        confidence = np.max(margins, axis=1).astype(float).tolist() if margins.ndim == 2 else np.abs(margins).astype(float).tolist()
    else:
        confidence = [0.0 for _ in pred]
    return [str(item) for item in pred], confidence


def evaluate_face_predictions(y_true: Sequence[str], y_pred: Sequence[str], *, split_name: str) -> dict[str, Any]:
    labels = list(FACE_LABELS)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    report = sk_metrics.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {}
    false_negatives = {}
    for index, label in enumerate(labels):
        row = cm[index, :]
        fn = int(row.sum() - cm[index, index])
        false_negatives[label] = fn
        per_class[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
            "false_negatives": fn,
        }
    recalls = {label: per_class[label]["recall"] for label in labels}
    pred_counts = Counter(y_pred)
    overpredicts = [label for label in ("happy", "neutral") if pred_counts[label] > max(1, len(y_pred) * 0.35)]
    worst_class = min(labels, key=lambda label: (recalls[label], label))
    return {
        "split": split_name,
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": labels,
        "per_class": per_class,
        "false_negatives_by_class": false_negatives,
        "minimum_class_recall": float(recalls[worst_class]),
        "worst_performing_class": worst_class,
        "disgust_recall": float(recalls["disgust"]),
        "prediction_distribution": dict(sorted(pred_counts.items())),
        "overpredicts_happy_or_neutral": overpredicts,
    }


def evaluate_face_split(estimator, X: np.ndarray, y_true: Sequence[str], *, split_name: str) -> dict[str, Any]:
    y_pred, _ = _prediction_scores(estimator, X)
    return evaluate_face_predictions(y_true, y_pred, split_name=split_name)


def train_validation_gap(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "macro_f1_train_minus_validation": float(train_metrics.get("macro_f1", 0.0) - validation_metrics.get("macro_f1", 0.0)),
        "macro_recall_train_minus_validation": float(train_metrics.get("macro_recall", 0.0) - validation_metrics.get("macro_recall", 0.0)),
        "balanced_accuracy_train_minus_validation": float(
            train_metrics.get("balanced_accuracy", 0.0) - validation_metrics.get("balanced_accuracy", 0.0)
        ),
    }


def selection_score(validation_metrics: dict[str, Any], gap: dict[str, float], estimator_type: str) -> tuple[Any, ...]:
    simplicity = {"logistic_regression": 3, "linear_svm": 2, "random_forest": 1}.get(estimator_type, 0)
    inference_cost = {"linear_svm": 3, "logistic_regression": 2, "random_forest": 1}.get(estimator_type, 0)
    return (
        float(validation_metrics.get("macro_f1", 0.0)),
        float(validation_metrics.get("macro_recall", 0.0)),
        float(validation_metrics.get("balanced_accuracy", 0.0)),
        float(validation_metrics.get("minimum_class_recall", 0.0)),
        float(validation_metrics.get("disgust_recall", 0.0)),
        -abs(float(gap.get("macro_f1_train_minus_validation", 0.0))),
        simplicity,
        inference_cost,
    )


def error_analysis_rows(rows, y_true: Sequence[str], y_pred: Sequence[str], confidence: Sequence[float]) -> list[dict[str, Any]]:
    output = []
    for (_, row), truth, pred, score in zip(rows.iterrows(), y_true, y_pred, confidence):
        if truth == pred:
            continue
        output.append(
            {
                "record_id": row["record_id"],
                "true_label": truth,
                "predicted_label": pred,
                "confidence_or_margin": float(score),
                "original_source_split": row.get("source_split", ""),
                "image_hash_prefix": str(row.get("image_hash", ""))[:12],
                "width": 48,
                "height": 48,
            }
        )
    return output

