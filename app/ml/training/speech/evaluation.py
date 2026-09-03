"""Metrics, corpus slices, interpretation, and privacy-safe errors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from app.ml.training.speech.constants import SPEECH_CORPUS_COLUMN, SPEECH_LABELS, SPEECH_RECORD_ID_COLUMN, SPEECH_TARGET_COLUMN


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def predict_with_scores(estimator, X):
    pred = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        return pred, estimator.predict_proba(X), "probability"
    if hasattr(estimator, "decision_function"):
        return pred, estimator.decision_function(X), "decision_function"
    return pred, None, None


def evaluate_speech_split(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    scores=None,
    score_kind: str | None = None,
    split_name: str,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    labels = SPEECH_LABELS
    cm = sk_metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(y_true_arr, y_pred_arr, labels=labels, zero_division=0)
    per_class = {
        label: {
            "precision": _safe_float(precision[idx]),
            "recall": _safe_float(recall[idx]),
            "f1": _safe_float(f1[idx]),
            "support": int(support[idx]),
            "false_negatives": int(cm[idx, :].sum() - cm[idx, idx]),
        }
        for idx, label in enumerate(labels)
    }
    result: dict[str, Any] = {
        "split": split_name,
        "accuracy": _safe_float(sk_metrics.accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": _safe_float(sk_metrics.balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": _safe_float(sk_metrics.precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "macro_recall": _safe_float(sk_metrics.recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "macro_f1": _safe_float(sk_metrics.f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": _safe_float(sk_metrics.precision_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": _safe_float(sk_metrics.recall_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": _safe_float(sk_metrics.f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)),
        "support": {label: int((y_true_arr == label).sum()) for label in labels} | {"total": int(len(y_true_arr))},
        "confusion_matrix": {"labels": labels, "matrix": cm.astype(int).tolist()},
        "per_class": per_class,
        "class_focus": {label: per_class[label] for label in ("sad", "fearful", "neutral", "surprised", "calm")},
        "class_false_negatives": {label: per_class[label]["false_negatives"] for label in labels},
    }
    if scores is not None and score_kind == "probability":
        try:
            result["log_loss"] = _safe_float(sk_metrics.log_loss(y_true_arr, scores, labels=labels))
        except Exception:
            result["log_loss"] = None
        try:
            result["roc_auc_ovr"] = _safe_float(sk_metrics.roc_auc_score(y_true_arr, scores, labels=labels, multi_class="ovr", average="macro"))
        except Exception:
            result["roc_auc_ovr"] = None
    else:
        result["log_loss"] = None
        result["roc_auc_ovr"] = None
    return result


def confusion_matrix_rows(y_true: Sequence[str], y_pred: Sequence[str]) -> list[dict[str, Any]]:
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=SPEECH_LABELS)
    return [
        {"true_label": SPEECH_LABELS[row], **{f"predicted_{SPEECH_LABELS[col]}": int(cm[row, col]) for col in range(len(SPEECH_LABELS))}}
        for row in range(len(SPEECH_LABELS))
    ]


def per_class_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"label": label, **values} for label, values in (metrics.get("per_class") or {}).items()]


def train_validation_gap(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1_train_minus_validation": _safe_float((train_metrics.get("macro_f1") or 0) - (validation_metrics.get("macro_f1") or 0)),
        "macro_recall_train_minus_validation": _safe_float((train_metrics.get("macro_recall") or 0) - (validation_metrics.get("macro_recall") or 0)),
        "balanced_accuracy_train_minus_validation": _safe_float((train_metrics.get("balanced_accuracy") or 0) - (validation_metrics.get("balanced_accuracy") or 0)),
    }


def corpus_distribution(splits: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for split_name, df in splits.items():
        counts = df[SPEECH_CORPUS_COLUMN].value_counts().sort_index()
        for corpus, count in counts.items():
            rows.append({"split": split_name, "corpus": corpus, "count": int(count), "proportion": _safe_float(count / max(len(df), 1))})
    return rows


def corpus_metrics(df: pd.DataFrame, y_true: Sequence[str], y_pred: Sequence[str], *, min_support: int = 30) -> dict[str, Any]:
    y_true_series = pd.Series(list(y_true), index=df.index)
    y_pred_series = pd.Series(list(y_pred), index=df.index)
    output: dict[str, Any] = {"minimum_support": min_support, "by_corpus": {}, "warnings": []}
    macro_f1_values = []
    for corpus, group in df.groupby(SPEECH_CORPUS_COLUMN, dropna=False):
        if len(group) < min_support:
            output["by_corpus"][str(corpus)] = {"status": "insufficient_support", "support": int(len(group))}
            output["warnings"].append(f"unsupported corpus/split combination for {corpus}: support {len(group)}")
            continue
        metrics = evaluate_speech_split(y_true_series.loc[group.index], y_pred_series.loc[group.index], split_name=f"corpus:{corpus}")
        output["by_corpus"][str(corpus)] = metrics
        if metrics.get("macro_f1") is not None:
            macro_f1_values.append(float(metrics["macro_f1"]))
    output["performance_gap_macro_f1"] = _safe_float(max(macro_f1_values) - min(macro_f1_values)) if len(macro_f1_values) >= 2 else None
    output["warning"] = "Corpus metrics are for domain-shift analysis only; corpus is not a primary predictive feature."
    return output


def feature_interpretation(estimator, feature_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        classes = list(getattr(estimator, "classes_", []))
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        aggregate = np.mean(np.abs(coef), axis=0)
        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature,
                    "aggregate_abs_coefficient": _safe_float(aggregate[idx]),
                    "top_class": classes[int(np.argmax(np.abs(coef[:, idx])))] if classes else None,
                    "warning": "Associational only; no causal, depression, or suicide-risk interpretation.",
                }
            )
        return sorted(rows, key=lambda row: row.get("aggregate_abs_coefficient") or 0, reverse=True)
    if hasattr(estimator, "feature_importances_"):
        for feature, value in zip(feature_names, estimator.feature_importances_):
            rows.append(
                {
                    "feature": feature,
                    "importance": _safe_float(value),
                    "warning": "Impurity importances can be biased; no causal, depression, or suicide-risk interpretation.",
                }
            )
        return sorted(rows, key=lambda row: row.get("importance") or 0, reverse=True)
    return [{"feature": feature, "warning": "Nonlinear SVM has no safe aggregate feature interpretation in this baseline."} for feature in feature_names]


def privacy_safe_error_analysis(df: pd.DataFrame, y_true: Sequence[str], y_pred: Sequence[str], scores=None, *, score_kind: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confidences: list[float | None] = [None] * len(df)
    if scores is not None:
        score_array = np.asarray(scores)
        if score_array.ndim == 2:
            if score_kind == "probability":
                confidences = [_safe_float(np.max(row)) for row in score_array]
            else:
                confidences = [_safe_float(np.max(row) - np.partition(row, -2)[-2]) if row.size > 1 else _safe_float(row[0]) for row in score_array]
    for idx, (_, row) in enumerate(df.iterrows()):
        true_label = str(list(y_true)[idx])
        pred_label = str(list(y_pred)[idx])
        if true_label == pred_label:
            continue
        rows.append(
            {
                "record_id": row.get(SPEECH_RECORD_ID_COLUMN),
                "true_emotion": true_label,
                "predicted_emotion": pred_label,
                "confidence_or_margin": confidences[idx],
                "corpus": row.get(SPEECH_CORPUS_COLUMN),
                "duration": row.get("duration_seconds"),
                "feature_completeness": "complete",
            }
        )
    return rows

