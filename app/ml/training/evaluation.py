"""Common evaluation orchestration for Phase 3B candidate models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from app.ml.training.metrics import evaluate_binary_metrics, evaluate_multiclass_metrics
from app.ml.training.schemas import MetricSet, ThresholdStrategy, TrainingTask
from app.ml.training.thresholds import labels_from_threshold, select_binary_threshold, select_multiclass_thresholds


def _predict_proba(estimator: Any, x) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
    return None


def _positive_probability(estimator: Any, x, positive_label=1) -> np.ndarray | None:
    proba = _predict_proba(estimator, x)
    if proba is None:
        return None
    classes = list(getattr(estimator, "classes_", [0, positive_label]))
    index = classes.index(positive_label) if positive_label in classes else min(1, proba.shape[1] - 1)
    return np.asarray(proba)[:, index]


def _binary_class_labels(estimator: Any, y_true, positive_label):
    classes = list(getattr(estimator, "classes_", sorted(set(y_true), key=str)))
    if positive_label is None:
        positive_label = classes[-1]
    negative_label = next((label for label in classes if label != positive_label), 0)
    return positive_label, negative_label, classes


def evaluate_binary_classifier(
    estimator: Any,
    x,
    y_true,
    *,
    threshold: float = 0.5,
    positive_label=None,
    labels: Sequence | None = None,
) -> MetricSet:
    positive_label, negative_label, class_labels = _binary_class_labels(estimator, y_true, positive_label)
    proba = _positive_probability(estimator, x, positive_label=positive_label)
    if proba is not None:
        y_pred = labels_from_threshold(proba, threshold, positive_label=positive_label, negative_label=negative_label)
    else:
        y_pred = estimator.predict(x)
    return evaluate_binary_metrics(y_true, y_pred, y_probability=proba, positive_label=positive_label, threshold=threshold, labels=labels or class_labels)


def evaluate_multiclass_classifier(estimator: Any, x, y_true, *, labels: Sequence | None = None) -> MetricSet:
    y_pred = estimator.predict(x)
    proba = _predict_proba(estimator, x)
    classes = list(getattr(estimator, "classes_", labels or sorted(set(y_true), key=str)))
    return evaluate_multiclass_metrics(y_true, y_pred, y_probability=proba, labels=classes)


def evaluate_train_validation_test(
    estimator: Any,
    *,
    task: TrainingTask | str,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    threshold_strategy: ThresholdStrategy | str = ThresholdStrategy.DEFAULT,
    positive_label=None,
    min_recall: float | None = None,
) -> dict[str, Any]:
    task = TrainingTask(task)
    if task == TrainingTask.BINARY_CLASSIFICATION:
        positive_label, _, class_labels = _binary_class_labels(estimator, y_validation, positive_label)
        validation_proba = _positive_probability(estimator, x_validation, positive_label=positive_label)
        if validation_proba is None:
            selection = select_binary_threshold(y_validation, np.asarray(estimator.predict(x_validation)) == positive_label, strategy=ThresholdStrategy.DEFAULT)
        else:
            selection = select_binary_threshold(
                y_validation,
                validation_proba,
                strategy=threshold_strategy,
                positive_label=positive_label,
                min_recall=min_recall,
            )
        threshold = selection.threshold if selection.threshold is not None else 0.5
        return {
            "train_metrics": evaluate_binary_classifier(estimator, x_train, y_train, threshold=threshold, positive_label=positive_label, labels=class_labels),
            "validation_metrics": evaluate_binary_classifier(estimator, x_validation, y_validation, threshold=threshold, positive_label=positive_label, labels=class_labels),
            "test_metrics": evaluate_binary_classifier(estimator, x_test, y_test, threshold=threshold, positive_label=positive_label, labels=class_labels),
            "selected_thresholds": selection.to_dict(),
        }
    return {
        "train_metrics": evaluate_multiclass_classifier(estimator, x_train, y_train),
        "validation_metrics": evaluate_multiclass_classifier(estimator, x_validation, y_validation),
        "test_metrics": evaluate_multiclass_classifier(estimator, x_test, y_test),
        "selected_thresholds": select_multiclass_thresholds(),
    }


def calibration_summary(metric: MetricSet) -> dict[str, float | None]:
    return {"brier_score": metric.brier_score, "log_loss": metric.log_loss}


def error_analysis_summary(metric: MetricSet) -> dict[str, Any]:
    return {
        "confusion_matrix": metric.confusion_matrix,
        "false_negative_count": metric.false_negative_count,
        "false_positive_count": metric.false_positive_count,
        "warnings": metric.warnings,
    }


def class_distribution_summary(y) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _metric_value(metric: MetricSet, name: str) -> float | None:
    return getattr(metric, name, None)


def compare_candidate_results(results: Sequence[Mapping[str, Any]], *, primary_metric: str) -> list[dict[str, Any]]:
    ranked = []
    for index, result in enumerate(results):
        validation_metrics = result["validation_metrics"]
        ranked.append(
            {
                "index": index,
                "run_id": result.get("run_id"),
                "primary_metric": primary_metric,
                "validation_value": _metric_value(validation_metrics, primary_metric),
                "test_metrics_ignored_for_selection": True,
            }
        )
    return sorted(ranked, key=lambda item: (item["validation_value"] is not None, item["validation_value"] or float("-inf")), reverse=True)


def select_candidate_model(
    results: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    minimum_recall: float | None = None,
) -> dict[str, Any]:
    eligible = []
    for result in results:
        metric: MetricSet = result["validation_metrics"]
        if minimum_recall is not None and (metric.recall_macro is None or metric.recall_macro < minimum_recall):
            continue
        value = _metric_value(metric, primary_metric)
        if value is not None:
            eligible.append((value, _simplicity_score(result), result))
    if not eligible:
        return {"selected": None, "reason": "no candidate satisfies validation constraints"}
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return {
        "selected": eligible[0][2],
        "reason": "selected by validation metrics only with simplicity tie-break",
        "test_metrics_used_for_selection": False,
    }


def _simplicity_score(result: Mapping[str, Any]) -> int:
    estimator_type = str(result.get("estimator_type", "")).lower()
    if "logistic" in estimator_type or "linear" in estimator_type:
        return 3
    if "tree" in estimator_type or "forest" in estimator_type:
        return 2
    return 1
