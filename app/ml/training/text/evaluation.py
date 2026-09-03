"""Evaluation and aggregate interpretation for Text baselines."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from app.ml.training.text.constants import DEPRESSION_LABEL, NORMAL_LABEL, SUICIDAL_LABEL, TEXT_LABELS


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _prediction_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X), "probability"
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X), "margin"
    return None, "none"


def confusion_matrix_rows(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = TEXT_LABELS) -> list[dict[str, Any]]:
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=list(labels))
    rows = []
    for i, true_label in enumerate(labels):
        row = {"true_label": true_label}
        for j, predicted_label in enumerate(labels):
            row[f"predicted_{predicted_label}"] = int(cm[i, j])
        rows.append(row)
    return rows


def per_class_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"label": label, **values} for label, values in metrics.get("per_class_metrics", {}).items()]


def evaluate_text_split(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = TEXT_LABELS,
    scores=None,
    score_kind: str = "none",
    split_name: str,
) -> dict[str, Any]:
    labels = list(labels)
    report = sk_metrics.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    suicidal_index = labels.index(SUICIDAL_LABEL)
    normal_index = labels.index(NORMAL_LABEL)
    depression_index = labels.index(DEPRESSION_LABEL)
    suicidal_tp = int(cm[suicidal_index, suicidal_index])
    suicidal_fn = int(cm[suicidal_index, :].sum() - suicidal_tp)
    suicidal_fp = int(cm[:, suicidal_index].sum() - suicidal_tp)
    result = {
        "split": split_name,
        "accuracy": _safe_float(sk_metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(sk_metrics.balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": _safe_float(sk_metrics.precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": _safe_float(sk_metrics.recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": _safe_float(sk_metrics.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": _safe_float(sk_metrics.precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": _safe_float(sk_metrics.recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": _safe_float(sk_metrics.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": labels,
        "support": {label: int(np.sum(np.asarray(y_true) == label)) for label in labels},
        "per_class_metrics": {
            label: {
                "precision": _safe_float(report[label]["precision"]),
                "recall": _safe_float(report[label]["recall"]),
                "f1": _safe_float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "suicidal_class": {
            "precision": _safe_float(report[SUICIDAL_LABEL]["precision"]),
            "recall": _safe_float(report[SUICIDAL_LABEL]["recall"]),
            "f1": _safe_float(report[SUICIDAL_LABEL]["f1-score"]),
            "false_negatives": suicidal_fn,
            "false_positives": suicidal_fp,
            "suicidal_predicted_as_normal": int(cm[suicidal_index, normal_index]),
            "suicidal_predicted_as_depression": int(cm[suicidal_index, depression_index]),
            "depression_predicted_as_suicidal": int(cm[depression_index, suicidal_index]),
        },
        "score_kind": score_kind,
        "warnings": [],
    }
    if scores is not None and score_kind == "probability":
        proba = np.asarray(scores, dtype=float)
        try:
            result["log_loss"] = _safe_float(sk_metrics.log_loss(y_true, proba, labels=labels))
        except ValueError as exc:
            result["warnings"].append(f"log loss unavailable: {exc}")
        try:
            result["roc_auc_macro_ovr"] = _safe_float(sk_metrics.roc_auc_score(y_true, proba, labels=labels, multi_class="ovr", average="macro"))
        except ValueError as exc:
            result["warnings"].append(f"ROC-AUC unavailable: {exc}")
    else:
        result["log_loss"] = None
        result["roc_auc_macro_ovr"] = None
    return result


def train_validation_gap(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1_train_minus_validation": _safe_float((train_metrics.get("macro_f1") or 0) - (validation_metrics.get("macro_f1") or 0)),
        "suicidal_recall_train_minus_validation": _safe_float(
            ((train_metrics.get("suicidal_class") or {}).get("recall") or 0) - ((validation_metrics.get("suicidal_class") or {}).get("recall") or 0)
        ),
        "balanced_accuracy_train_minus_validation": _safe_float(
            (train_metrics.get("balanced_accuracy") or 0) - (validation_metrics.get("balanced_accuracy") or 0)
        ),
    }


def aggregate_feature_interpretation(estimator, feature_names: list[str], *, top_n: int = 20) -> list[dict[str, Any]]:
    if not hasattr(estimator, "coef_"):
        return []
    coefficients = np.asarray(estimator.coef_)
    classes = [str(value) for value in estimator.classes_]
    rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(classes):
        weights = coefficients[class_index]
        top_indexes = np.argsort(weights)[-top_n:][::-1]
        for rank, feature_index in enumerate(top_indexes, start=1):
            feature = feature_names[int(feature_index)]
            lower = feature.lower()
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "feature": feature,
                    "weight": _safe_float(weights[int(feature_index)]),
                    "contains_privacy_token": "<" in feature and ">" in feature,
                    "shortcut_warning": any(token in lower for token in [label, "suicid", "depress", "anxiety", "normal"]),
                    "interpretation_warning": "Aggregate association only; no raw text, causal claim, clinical interpretation, or crisis rule.",
                }
            )
    return rows


def privacy_safe_error_analysis(
    df: pd.DataFrame,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores=None,
    *,
    score_kind: str,
    max_rows: int = 500,
) -> list[dict[str, Any]]:
    score_array = None if scores is None else np.asarray(scores)
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        if str(y_true[position]) == str(y_pred[position]):
            continue
        score_value = None
        if score_array is not None:
            values = score_array[position]
            score_value = _safe_float(np.max(values)) if np.ndim(values) else _safe_float(values)
        rows.append(
            {
                "record_id": row.get("record_id"),
                "true_label": str(y_true[position]),
                "predicted_label": str(y_pred[position]),
                "score_kind": score_kind,
                "score": score_value,
                "text_hash": row.get("text_hash"),
                "text_length": int(row.get("character_count", 0) or 0),
                "placeholder_count": int(row.get("placeholder_count", 0) or 0),
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def predict_with_scores(estimator, X):
    y_pred = estimator.predict(X)
    scores, kind = _prediction_scores(estimator, X)
    return y_pred, scores, kind

