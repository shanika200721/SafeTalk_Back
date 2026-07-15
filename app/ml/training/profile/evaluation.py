"""Evaluation, thresholding, fairness slices, and interpretation helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from app.ml.training.thresholds import labels_from_threshold, select_binary_threshold
from app.ml.training.profile.constants import PROFILE_NEGATIVE_LABEL, PROFILE_POSITIVE_LABEL


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _probability_summary(probabilities: Sequence[float]) -> dict[str, Any]:
    proba = np.asarray(probabilities, dtype=float)
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    counts, _ = np.histogram(proba, bins=bins)
    return {
        "min": _safe_float(np.min(proba)) if len(proba) else None,
        "max": _safe_float(np.max(proba)) if len(proba) else None,
        "mean": _safe_float(np.mean(proba)) if len(proba) else None,
        "median": _safe_float(np.median(proba)) if len(proba) else None,
        "bins": {"[0.00,0.25)": int(counts[0]), "[0.25,0.50)": int(counts[1]), "[0.50,0.75)": int(counts[2]), "[0.75,1.00]": int(counts[3])},
    }


def calibration_summary(y_true: Sequence[str], probabilities: Sequence[float], *, bins: int = 5) -> dict[str, Any]:
    y_binary = (np.asarray(y_true) == PROFILE_POSITIVE_LABEL).astype(int)
    proba = np.asarray(probabilities, dtype=float)
    if len(y_binary) == 0:
        return {"bins": [], "warning": "empty split"}
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (proba >= left) & (proba <= right)
        else:
            mask = (proba >= left) & (proba < right)
        count = int(mask.sum())
        rows.append(
            {
                "bin": f"{left:.2f}-{right:.2f}",
                "count": count,
                "mean_probability": _safe_float(proba[mask].mean()) if count else None,
                "observed_positive_rate": _safe_float(y_binary[mask].mean()) if count else None,
            }
        )
    return {
        "bins": rows,
        "warning": "Calibration is descriptive only; split has too few records for stable calibration.",
    }


def evaluate_profile_split(
    y_true: Sequence[str],
    probabilities: Sequence[float],
    *,
    threshold: float,
    feature_count: int,
    split_name: str,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    proba = np.asarray(probabilities, dtype=float)
    y_pred = labels_from_threshold(
        proba,
        threshold,
        positive_label=PROFILE_POSITIVE_LABEL,
        negative_label=PROFILE_NEGATIVE_LABEL,
    )
    binary_true = (y_true_arr == PROFILE_POSITIVE_LABEL).astype(int)
    binary_pred = (y_pred == PROFILE_POSITIVE_LABEL).astype(int)
    cm = sk_metrics.confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    warnings = []
    if len(y_true_arr) <= 15:
        warnings.append("Metrics are unstable because this split contains 15 or fewer records.")
    if len(set(binary_true)) < 2:
        warnings.append("ROC-AUC, PR-AUC, and log loss may be undefined because y_true has one class.")

    result = {
        "split": split_name,
        "accuracy": _safe_float(sk_metrics.accuracy_score(binary_true, binary_pred)),
        "balanced_accuracy": _safe_float(sk_metrics.balanced_accuracy_score(binary_true, binary_pred)),
        "precision": _safe_float(sk_metrics.precision_score(binary_true, binary_pred, zero_division=0)),
        "recall": _safe_float(sk_metrics.recall_score(binary_true, binary_pred, zero_division=0)),
        "f1": _safe_float(sk_metrics.f1_score(binary_true, binary_pred, zero_division=0)),
        "specificity": _safe_float(tn / (tn + fp)) if (tn + fp) else None,
        "brier_score": _safe_float(sk_metrics.brier_score_loss(binary_true, proba)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "labels": ["no", "yes"]},
        "support": {"no": int((binary_true == 0).sum()), "yes": int((binary_true == 1).sum()), "total": int(len(binary_true))},
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_probability_distribution": _probability_summary(proba),
        "calibration_summary": calibration_summary(y_true_arr, proba),
        "feature_count": int(feature_count),
        "class_balance": {"no": int((binary_true == 0).sum()), "yes": int((binary_true == 1).sum())},
        "threshold": float(threshold),
        "confidence_intervals": {
            "available": False,
            "reason": "Bootstrap confidence intervals are unstable for the 15-record validation/test splits.",
        },
        "warnings": warnings,
    }
    if len(set(binary_true)) >= 2:
        result["roc_auc"] = _safe_float(sk_metrics.roc_auc_score(binary_true, proba))
        result["pr_auc"] = _safe_float(sk_metrics.average_precision_score(binary_true, proba))
        result["log_loss"] = _safe_float(sk_metrics.log_loss(binary_true, np.column_stack([1 - proba, proba]), labels=[0, 1]))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
        result["log_loss"] = None
    return result


def select_profile_threshold(
    y_validation: Sequence[str],
    probabilities: Sequence[float],
    *,
    strategy: str,
    min_recall: float | None = None,
) -> dict[str, Any]:
    selection = select_binary_threshold(
        y_validation,
        probabilities,
        strategy=strategy,
        positive_label=PROFILE_POSITIVE_LABEL,
        min_recall=min_recall,
    )
    return selection.to_dict()


def overfitting_gap(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "f1_train_minus_validation": _safe_float((train_metrics.get("f1") or 0) - (validation_metrics.get("f1") or 0)),
        "recall_train_minus_validation": _safe_float((train_metrics.get("recall") or 0) - (validation_metrics.get("recall") or 0)),
        "balanced_accuracy_train_minus_validation": _safe_float(
            (train_metrics.get("balanced_accuracy") or 0) - (validation_metrics.get("balanced_accuracy") or 0)
        ),
    }


def fairness_exploration(
    df: pd.DataFrame,
    y_true: Sequence[str],
    probabilities: Sequence[float],
    *,
    threshold: float,
    sensitive_columns: Sequence[str],
    min_support: int = 5,
) -> dict[str, Any]:
    if not sensitive_columns:
        return {
            "enabled": False,
            "reason": "Sensitive-context exploratory mode was not enabled.",
            "limitations": ["Primary baseline does not use sensitive attributes."],
        }
    y_true_series = pd.Series(list(y_true), index=df.index)
    probability_series = pd.Series(list(probabilities), index=df.index)
    output: dict[str, Any] = {"enabled": True, "minimum_subgroup_support": min_support, "slices": {}, "limitations": []}
    for column in sensitive_columns:
        if column not in df.columns:
            output["slices"][column] = {"status": "unavailable", "reason": "column absent from canonical data"}
            continue
        groups = {}
        for value, group_df in df.groupby(column, dropna=False):
            if len(group_df) < min_support:
                groups[str(value)] = {"status": "insufficient sample", "support": int(len(group_df))}
                continue
            groups[str(value)] = evaluate_profile_split(
                y_true_series.loc[group_df.index],
                probability_series.loc[group_df.index],
                threshold=threshold,
                feature_count=0,
                split_name=f"slice:{column}",
            )
        output["slices"][column] = groups
    output["limitations"].append("Subgroup metrics are exploratory and must not drive candidate selection alone.")
    return output


def feature_interpretation(estimator, feature_names: list[str]) -> list[dict[str, Any]]:
    if hasattr(estimator, "coef_"):
        coefficients = np.ravel(estimator.coef_)
        rows = []
        for name, coefficient in zip(feature_names, coefficients):
            rows.append(
                {
                    "feature": name,
                    "coefficient": _safe_float(coefficient),
                    "odds_ratio_style": _safe_float(np.exp(coefficient)),
                    "warning": "Coefficient is unstable due to tiny sample size; no causal or clinical interpretation.",
                }
            )
        return rows
    if hasattr(estimator, "feature_importances_"):
        return [
            {
                "feature": name,
                "importance": _safe_float(value),
                "warning": "Impurity importance can be biased; no causal or clinical interpretation.",
            }
            for name, value in zip(feature_names, estimator.feature_importances_)
        ]
    return [{"feature": name, "warning": "Estimator has no built-in global interpretation."} for name in feature_names]
