"""Metrics for Speech corpus domain-shift evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def evaluate_domain_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    observed = set(str(value) for value in y_true)
    eval_labels = [str(label) for label in labels if str(label) in observed]
    if not eval_labels:
        raise ValueError(f"no supported labels are present in {split_name}")
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    cm = sk_metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=eval_labels)
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=eval_labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": safe_float(precision[index]),
            "recall": safe_float(recall[index]),
            "f1": safe_float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(eval_labels)
    }
    missing = [str(label) for label in labels if str(label) not in observed]
    recalls = [value["recall"] for value in per_class.values() if value["recall"] is not None]
    return {
        "split": split_name,
        "labels": eval_labels,
        "missing_classes": missing,
        "accuracy": safe_float(sk_metrics.accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": safe_float(sk_metrics.balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": safe_float(sk_metrics.precision_score(y_true_arr, y_pred_arr, labels=eval_labels, average="macro", zero_division=0)),
        "macro_recall": safe_float(sk_metrics.recall_score(y_true_arr, y_pred_arr, labels=eval_labels, average="macro", zero_division=0)),
        "macro_f1": safe_float(sk_metrics.f1_score(y_true_arr, y_pred_arr, labels=eval_labels, average="macro", zero_division=0)),
        "weighted_f1": safe_float(sk_metrics.f1_score(y_true_arr, y_pred_arr, labels=eval_labels, average="weighted", zero_division=0)),
        "worst_class_recall": safe_float(min(recalls)) if recalls else None,
        "support": {label: int((y_true_arr == label).sum()) for label in eval_labels} | {"total": int(len(y_true_arr))},
        "per_class": per_class,
        "confusion_matrix": {"labels": eval_labels, "matrix": cm.astype(int).tolist()},
        "warnings": [f"Absent class excluded from {split_name} metric denominator: {label}" for label in missing],
    }


def per_class_rows(fold_name: str, split_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"fold_name": fold_name, "split": split_name, "label": label, **values}
        for label, values in (metrics.get("per_class") or {}).items()
    ]


def corpus_generalization_gap(pooled_macro_f1: float | None, loco_macro_f1_values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in loco_macro_f1_values if value is not None and np.isfinite(float(value))]
    if not values:
        return {
            "pooled_test_macro_f1": pooled_macro_f1,
            "loco_mean_macro_f1": None,
            "loco_std_macro_f1": None,
            "minimum_corpus_macro_f1": None,
            "maximum_corpus_macro_f1": None,
            "corpus_generalization_gap": None,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return {
        "pooled_test_macro_f1": pooled_macro_f1,
        "loco_mean_macro_f1": safe_float(mean),
        "loco_std_macro_f1": safe_float(std),
        "minimum_corpus_macro_f1": safe_float(min(values)),
        "maximum_corpus_macro_f1": safe_float(max(values)),
        "corpus_generalization_gap": safe_float((pooled_macro_f1 - mean) if pooled_macro_f1 is not None else max(values) - min(values)),
    }


def feature_distribution_rows(records: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    focus = [
        feature
        for feature in features
        if feature == "duration_seconds"
        or feature.startswith("pitch")
        or feature.startswith("rms_energy")
        or feature.startswith("spectral_")
        or feature.startswith("mfcc_01")
    ]
    rows: list[dict[str, Any]] = []
    for corpus, group in records.groupby("corpus_name", sort=True):
        numeric = group[focus].apply(pd.to_numeric, errors="coerce")
        for feature in focus:
            rows.append(
                {
                    "corpus": str(corpus),
                    "feature": feature,
                    "count": int(numeric[feature].count()),
                    "mean": safe_float(numeric[feature].mean()),
                    "std": safe_float(numeric[feature].std(ddof=0)),
                    "min": safe_float(numeric[feature].min()),
                    "max": safe_float(numeric[feature].max()),
                }
            )
    return rows

