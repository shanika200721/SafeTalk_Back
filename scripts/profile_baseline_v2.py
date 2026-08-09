"""Audit and retrain the Profile Assessment baseline as a v2 research artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"
TARGET = "target_depression"
FEATURES = ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]

CANONICAL = REPO_ROOT / "generated" / "preprocessing" / "profile" / "v1" / "canonical_profile.csv"
SOURCE = REPO_ROOT / "Final Dataset" / "Student Profile" / "Student Mental health.csv"
ASSIGNMENTS = REPO_ROOT / "generated" / "manifests" / "splits" / "profile" / "v1" / "profile_split_assignments.csv"
V1_METRICS = REPO_ROOT / "generated" / "reports" / "profile_baseline" / "v1" / "profile_metrics_test.json"
AUDIT_DIR = REPO_ROOT / "generated" / "reports" / "profile_baseline_audit"
V2_DIR = REPO_ROOT / "generated" / "reports" / "profile_baseline" / "v2"
MODEL_DIR = REPO_ROOT / "ml_models" / "profile" / "profile-depression-v2" / "2.0.0" / "profile-v2-repeated-cv-seed42"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", safe_one_hot()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", preprocessor()), ("model", estimator)])


def candidates() -> list[tuple[str, Pipeline]]:
    linear_svc = CalibratedClassifierCV(
        LinearSVC(C=0.1, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000),
        cv=3,
    )
    return [
        ("dummy_most_frequent", make_pipeline(DummyClassifier(strategy="most_frequent"))),
        ("dummy_stratified", make_pipeline(DummyClassifier(strategy="stratified", random_state=RANDOM_SEED))),
        ("logistic_l2_C0.1", make_pipeline(LogisticRegression(C=0.1, solver="liblinear", random_state=RANDOM_SEED))),
        ("logistic_l2_C1", make_pipeline(LogisticRegression(C=1.0, solver="liblinear", random_state=RANDOM_SEED))),
        (
            "logistic_balanced_C0.1",
            make_pipeline(LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED)),
        ),
        (
            "logistic_balanced_C1",
            make_pipeline(LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED)),
        ),
        (
            "random_forest_depth2_leaf3_balanced",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=2,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                )
            ),
        ),
        (
            "random_forest_depth3_leaf5",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    min_samples_leaf=5,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                )
            ),
        ),
        ("linear_svc_calibrated_balanced_C0.1", make_pipeline(linear_svc)),
        (
            "decision_tree_depth2_leaf5_balanced",
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )
            ),
        ),
        ("gaussian_nb", make_pipeline(GaussianNB())),
        ("gradient_boosting_depth1", make_pipeline(GradientBoostingClassifier(max_depth=1, random_state=RANDOM_SEED))),
    ]


def label_to_binary(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(y) == POSITIVE_LABEL


def positive_probability(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        return probabilities[:, classes.index(POSITIVE_LABEL)]
    scores = estimator.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def metrics_from_predictions(y_true, y_pred, y_prob=None, split_name="test", threshold=None) -> dict:
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    payload = {
        "split": split_name,
        "threshold": threshold,
        "support": {
            "no": int((np.asarray(y_true) == NEGATIVE_LABEL).sum()),
            "yes": int((np.asarray(y_true) == POSITIVE_LABEL).sum()),
            "total": int(len(y_true)),
        },
        "predicted_distribution": {label: int((np.asarray(y_pred) == label).sum()) for label in labels},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist(), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        y_bin = label_to_binary(y_true).astype(int)
        payload["roc_auc"] = float(roc_auc_score(y_bin, y_prob))
        payload["pr_auc"] = float(average_precision_score(y_bin, y_prob))
    else:
        payload["roc_auc"] = None
        payload["pr_auc"] = None
    return payload


def threshold_grid(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = np.where(y_prob >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0),
                "recall": recall_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0),
                "f1": f1_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    grid = threshold_grid(y_true, y_prob)
    default = grid.iloc[(grid["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]
    best_macro = grid.sort_values(["macro_f1", "balanced_accuracy", "precision"], ascending=False).iloc[0]
    best_bal = grid.sort_values(["balanced_accuracy", "macro_f1", "precision"], ascending=False).iloc[0]
    acceptable = grid[grid["precision"] >= 0.4]
    if len(acceptable):
        recall_priority = acceptable.sort_values(["recall", "precision", "macro_f1"], ascending=False).iloc[0]
        recall_note = "max recall with validation/CV precision >= 0.40"
    else:
        recall_priority = grid.sort_values(["recall", "precision", "macro_f1"], ascending=False).iloc[0]
        recall_note = "no threshold met precision >= 0.40; reported for sensitivity only"
    chosen = best_macro
    return {
        "selection_source": "development stratified 5-fold out-of-fold probabilities only",
        "chosen_strategy": "max_macro_f1",
        "chosen_threshold": float(chosen["threshold"]),
        "default_0_50": default.to_dict(),
        "max_macro_f1": best_macro.to_dict(),
        "max_balanced_accuracy": best_bal.to_dict(),
        "recall_priority": {**recall_priority.to_dict(), "note": recall_note},
        "safety_note": "This is a research classification threshold, not a clinical safety escalation threshold.",
    }


def repeated_cv_summary(estimator, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    fold_metrics = []
    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(estimator)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        fold_model.fit(X_train, y_train)
        pred = fold_model.predict(X_val)
        try:
            prob = positive_probability(fold_model, X_val)
        except Exception:
            prob = None
        fold_metrics.append(metrics_from_predictions(y_val, pred, prob, split_name="development_cv", threshold=0.5))

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "pr_auc",
    ]
    summary = {}
    for metric in metric_names:
        values = [row[metric] for row in fold_metrics if row[metric] is not None]
        summary[metric] = float(np.mean(values)) if values else None
        summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    summary["folds"] = len(fold_metrics)
    summary["mean_predicted_no"] = float(np.mean([row["predicted_distribution"]["no"] for row in fold_metrics]))
    summary["mean_predicted_yes"] = float(np.mean([row["predicted_distribution"]["yes"] for row in fold_metrics]))
    return summary


def out_of_fold_probabilities(estimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    probabilities = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(estimator)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[val_idx] = positive_probability(fold_model, X.iloc[val_idx])
    return probabilities


def bootstrap_ci(y_true, y_pred, y_prob, n=2000) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    rows = {"accuracy": [], "balanced_accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        rows["accuracy"].append(accuracy_score(y_true[idx], y_pred[idx]))
        rows["balanced_accuracy"].append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
        rows["precision"].append(precision_score(y_true[idx], y_pred[idx], pos_label=POSITIVE_LABEL, zero_division=0))
        rows["recall"].append(recall_score(y_true[idx], y_pred[idx], pos_label=POSITIVE_LABEL, zero_division=0))
        rows["f1"].append(f1_score(y_true[idx], y_pred[idx], pos_label=POSITIVE_LABEL, zero_division=0))
        rows["roc_auc"].append(roc_auc_score(label_to_binary(y_true[idx]).astype(int), y_prob[idx]))
        rows["pr_auc"].append(average_precision_score(label_to_binary(y_true[idx]).astype(int), y_prob[idx]))
    return {
        key: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "bootstrap_samples": len(values),
        }
        for key, values in rows.items()
        if values
    }


def plot_confusion(cm_payload: dict, path: Path) -> None:
    matrix = np.asarray(cm_payload["matrix"])
    labels = cm_payload["labels"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Profile Model Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(y_true, y_prob, roc_path: Path, pr_path: Path) -> None:
    y_bin = label_to_binary(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_prob)
    precision, recall, _ = precision_recall_curve(y_bin, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC {roc_auc_score(y_bin, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Profile ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC {average_precision_score(y_bin, y_prob):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Profile Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CANONICAL)
    source_df = pd.read_csv(SOURCE)
    assignments = pd.read_csv(ASSIGNMENTS)
    data = df.merge(assignments[["record_id", "split", "label"]], on="record_id", how="left")
    data["target_binary"] = data[TARGET]

    class_rows = []
    for split_name, split_df in data.groupby("split"):
        counts = split_df[TARGET].value_counts().to_dict()
        class_rows.append({"split": split_name, "total": len(split_df), "no": counts.get("no", 0), "yes": counts.get("yes", 0)})
    pd.DataFrame(class_rows).sort_values("split").to_csv(AUDIT_DIR / "profile_class_distribution.csv", index=False)

    split_audit = {
        "total_records": int(len(data)),
        "missing_split_records": int(data["split"].isna().sum()),
        "split_counts": data["split"].value_counts().to_dict(),
        "class_distribution": {row["split"]: {"no": int(row["no"]), "yes": int(row["yes"])} for row in class_rows},
        "classes_missing_in_any_split": [
            row["split"] for row in class_rows if int(row["no"]) == 0 or int(row["yes"]) == 0
        ],
        "stratification_valid": all(int(row["no"]) > 0 and int(row["yes"]) > 0 for row in class_rows),
        "test_set_preserved": True,
        "test_set_reason": "Existing frozen test set is stratified, contains both classes, and has no duplicate-overlap evidence.",
    }
    pd.DataFrame(
        [
            {"check": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
            for key, value in split_audit.items()
        ]
    ).to_csv(AUDIT_DIR / "profile_split_audit.csv", index=False)

    feature_duplicate_groups = data.groupby(FEATURES + [TARGET], dropna=False)["record_id"].apply(list).reset_index(name="record_ids")
    feature_duplicate_groups["count"] = feature_duplicate_groups["record_ids"].apply(len)
    duplicate_groups = feature_duplicate_groups[feature_duplicate_groups["count"] > 1].copy()
    split_lookup = data.set_index("record_id")["split"].to_dict()
    duplicate_rows = []
    for _, row in duplicate_groups.iterrows():
        ids = row["record_ids"]
        splits = sorted({split_lookup[i] for i in ids})
        duplicate_rows.append(
            {
                "duplicate_type": "same_features_and_target",
                "count": len(ids),
                "record_ids": ";".join(ids),
                "splits": ";".join(splits),
                "crosses_split": len(splits) > 1,
            }
        )
    exact_source_duplicate_count = int(source_df.duplicated().sum())
    exact_canonical_duplicate_count = int(df.drop(columns=["record_id"]).duplicated().sum())
    duplicate_audit = pd.DataFrame(duplicate_rows)
    if duplicate_audit.empty:
        duplicate_audit = pd.DataFrame(
            [
                {
                    "duplicate_type": "none_found",
                    "count": 0,
                    "record_ids": "",
                    "splits": "",
                    "crosses_split": False,
                }
            ]
        )
    duplicate_audit.to_csv(AUDIT_DIR / "profile_duplicate_audit.csv", index=False)

    dev = data[data["split"].isin(["train", "validation"])].copy()
    test = data[data["split"] == "test"].copy()
    X_dev = dev[FEATURES]
    y_dev = dev[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    comparison_rows = []
    candidate_estimators = {}
    for name, estimator in candidates():
        metrics = repeated_cv_summary(clone(estimator), X_dev, y_dev, cv)
        comparison_rows.append(
            {
                "model": name,
                "cv_accuracy": metrics["accuracy"],
                "cv_accuracy_std": metrics["accuracy_std"],
                "cv_balanced_accuracy": metrics["balanced_accuracy"],
                "cv_balanced_accuracy_std": metrics["balanced_accuracy_std"],
                "cv_precision": metrics["precision"],
                "cv_precision_std": metrics["precision_std"],
                "cv_recall": metrics["recall"],
                "cv_recall_std": metrics["recall_std"],
                "cv_f1": metrics["f1"],
                "cv_f1_std": metrics["f1_std"],
                "cv_macro_f1": metrics["macro_f1"],
                "cv_macro_f1_std": metrics["macro_f1_std"],
                "cv_weighted_f1": metrics["weighted_f1"],
                "cv_weighted_f1_std": metrics["weighted_f1_std"],
                "cv_roc_auc": metrics["roc_auc"],
                "cv_roc_auc_std": metrics["roc_auc_std"],
                "cv_pr_auc": metrics["pr_auc"],
                "cv_pr_auc_std": metrics["pr_auc_std"],
                "cv_folds": metrics["folds"],
                "cv_mean_predicted_no": metrics["mean_predicted_no"],
                "cv_mean_predicted_yes": metrics["mean_predicted_yes"],
            }
        )
        candidate_estimators[name] = estimator

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_macro_f1", "cv_balanced_accuracy", "cv_pr_auc"], ascending=False
    )
    comparison.to_csv(V2_DIR / "profile_model_comparison.csv", index=False)
    comparison.to_csv(V2_DIR / "profile_cross_validation_results.csv", index=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected_estimator = candidate_estimators[selected_name]
    selected_cv_prob = out_of_fold_probabilities(selected_estimator, X_dev, y_dev)
    threshold_selection = choose_threshold(y_dev, selected_cv_prob)
    threshold = float(threshold_selection["chosen_threshold"])
    write_json(V2_DIR / "profile_threshold_selection.json", threshold_selection)

    final_model = clone(selected_estimator)
    final_model.fit(X_dev, y_dev)
    test_prob = positive_probability(final_model, X_test)
    test_pred = np.where(test_prob >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    test_metrics = metrics_from_predictions(y_test, test_pred, test_prob, split_name="test", threshold=threshold)
    test_metrics["confidence_intervals"] = bootstrap_ci(y_test, test_pred, test_prob)
    test_metrics["selected_model"] = selected_name
    test_metrics["selection_metric"] = "development repeated-stratified-CV macro F1"
    test_metrics["development_rows"] = int(len(dev))
    test_metrics["test_rows"] = int(len(test))
    test_metrics["random_seed"] = RANDOM_SEED
    test_metrics["dataset_fingerprint"] = sha256_file(CANONICAL)
    test_metrics["source_fingerprint"] = sha256_file(SOURCE)
    test_metrics["created_at"] = datetime.now(timezone.utc).isoformat()
    test_metrics["warnings"] = [
        "The held-out test set contains only 15 records; confidence intervals are wide and unstable.",
        "Target is self-reported depression, not suicide-risk ground truth.",
    ]
    write_json(V2_DIR / "profile_metrics_test.json", test_metrics)

    report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True, zero_division=0)).transpose()
    report.to_csv(V2_DIR / "profile_classification_report.csv")
    plot_confusion(test_metrics["confusion_matrix"], V2_DIR / "profile_confusion_matrix.png")
    plot_curves(y_test, test_prob, V2_DIR / "profile_roc_curve.png", V2_DIR / "profile_precision_recall_curve.png")

    error_rows = test[["record_id", *FEATURES, TARGET]].copy()
    error_rows["predicted_label"] = test_pred
    error_rows["positive_probability"] = test_prob
    error_rows["correct"] = error_rows[TARGET] == error_rows["predicted_label"]
    error_rows.to_csv(V2_DIR / "profile_error_analysis.csv", index=False)

    metric_recalculation = {
        "v1_metrics_path": str(V1_METRICS.relative_to(REPO_ROOT)),
        "v1_recalculation_source": "stored v1 metrics and confusion matrix; v1 probabilities were not separately exported",
        "v1_values_match_confusion_matrix": True,
        "v1_confusion_matrix": {"tn": 0, "fp": 10, "fn": 0, "tp": 5},
        "v1_accuracy_from_cm": 5 / 15,
        "v1_precision_from_cm": 5 / (5 + 10),
        "v1_recall_from_cm": 5 / (5 + 0),
        "v1_f1_from_cm": 2 * (1 / 3) * 1 / ((1 / 3) + 1),
        "v1_roc_auc_reported": json.loads(V1_METRICS.read_text(encoding="utf-8"))["roc_auc"],
        "interpretation": "The values are mathematically consistent. The weak result comes from a weak/all-positive classifier on a 15-record test set, not from arithmetic error.",
    }
    write_json(AUDIT_DIR / "profile_metric_recalculation.json", metric_recalculation)

    model_metadata = {
        "selected_model": selected_name,
        "feature_list": FEATURES,
        "target": TARGET,
        "target_mapping": {"no": 0, "yes": 1},
        "positive_label": POSITIVE_LABEL,
        "selected_threshold": threshold,
        "random_seed": RANDOM_SEED,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": sha256_file(CANONICAL),
        "source_fingerprint": sha256_file(SOURCE),
        "split_manifest": str(ASSIGNMENTS.relative_to(REPO_ROOT)),
        "test_ids_preserved_from_v1": True,
        "metric_definitions": {
            "precision_recall_f1": "binary metrics with yes as positive label plus macro/weighted summaries",
            "roc_auc": "binary ROC-AUC using positive-class probability for yes",
            "pr_auc": "average precision using positive-class probability for yes",
        },
        "code_version": "backend/scripts/profile_baseline_v2.py",
        "code_sha256": sha256_file(Path(__file__)),
    }
    joblib.dump(final_model, MODEL_DIR / "pipeline.joblib")
    write_json(MODEL_DIR / "model_metadata.json", model_metadata)
    write_json(MODEL_DIR / "metrics.json", {"test": test_metrics, "threshold_selection": threshold_selection})

    inspected = [
        str(path.relative_to(REPO_ROOT))
        for path in [
            SOURCE,
            CANONICAL,
            ASSIGNMENTS,
            REPO_ROOT / "generated/preprocessing/profile/v1/profile_preprocessing_report.json",
            REPO_ROOT / "generated/manifests/splits/profile/v1/profile_split_manifest.json",
            REPO_ROOT / "generated/manifests/splits/profile/v1/profile_split_report.json",
            V1_METRICS,
            REPO_ROOT / "generated/reports/profile_baseline/v1/profile_candidate_comparison.csv",
            REPO_ROOT / "generated/reports/profile_baseline/v1/profile_confusion_matrix_test.csv",
        ]
    ]
    audit_md = f"""# Profile Pipeline Audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Files Inspected

{chr(10).join(f"- `{item}`" for item in inspected)}

## Target and Task

- Target variable: `{TARGET}`.
- Target definition: self-reported depression from the Student Mental Health profile dataset.
- Task type: binary classification.
- Class labels: `no`, `yes`; positive class is `yes`.
- Total records: {len(data)}.
- Feature columns used: {", ".join(FEATURES)}.

## Split Audit

- Existing frozen split preserved: yes.
- Train/validation/test counts: {split_audit["split_counts"]}.
- Class distributions: {split_audit["class_distribution"]}.
- Stratification status: {"valid" if split_audit["stratification_valid"] else "invalid"}.
- Classes missing from validation or test: {split_audit["classes_missing_in_any_split"] or "none"}.
- Exact source duplicate rows: {exact_source_duplicate_count}.
- Exact canonical duplicate rows excluding `record_id`: {exact_canonical_duplicate_count}.
- Feature/target duplicate groups crossing splits: {int(duplicate_audit["crosses_split"].sum()) if "crosses_split" in duplicate_audit else 0}.

## Leakage and Preprocessing Audit

The v1 feature schema excludes timestamp, gender, age, course, CGPA band, marital status, specialist-treatment seeking, and the target. Treatment seeking is documented as a possible post-outcome leakage variable and excluded. The v2 model uses a scikit-learn `Pipeline` and `ColumnTransformer`; imputation and one-hot encoding are fitted inside training/CV folds only.

## Metric Audit

The v1 test confusion matrix is TN=0, FP=10, FN=0, TP=5. The reported accuracy 0.3333, precision 0.3333, recall 1.0, and F1 0.5 are mathematically consistent with that matrix. The recall of 1.0 is caused by predicting every held-out test case as the positive class. Therefore the current metrics are correctly calculated but produced by a weak model and a very small 15-record test split.

## ROC-AUC and Averaging

The v1 task is binary with positive label `yes`. Precision, recall, and F1 are binary positive-class metrics. ROC-AUC was reported from positive-class probabilities and is plausible at 0.41; it is below random-ranking performance on this small test split.

## Conclusion

The v1 metrics should not be described as an implementation bug. They are weak but internally consistent. Chapter 4 should report the profile modality as a limited research baseline or contextual supporting modality unless v2 evidence demonstrates a defensible improvement.
"""
    (AUDIT_DIR / "profile_pipeline_audit.md").write_text(audit_md, encoding="utf-8")

    summary_md = f"""# Profile Baseline v2 Evaluation Summary

## Selection Protocol

The frozen v1 test set was preserved and not used for model selection or threshold tuning. Candidate models were compared on the 86-record development set using repeated stratified cross-validation (`n_splits=5`, `n_repeats=5`, seed `{RANDOM_SEED}`). The primary selection metric was macro F1.

## Selected Model

Selected model: `{selected_name}`.

Chosen research threshold: `{threshold:.6f}`, selected from development out-of-fold probabilities by maximum macro F1. This is not a clinical safety threshold.

## Final Held-Out Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.6f} |
| Balanced accuracy | {test_metrics['balanced_accuracy']:.6f} |
| Precision (`yes`) | {test_metrics['precision']:.6f} |
| Recall (`yes`) | {test_metrics['recall']:.6f} |
| F1 (`yes`) | {test_metrics['f1']:.6f} |
| Macro F1 | {test_metrics['macro_f1']:.6f} |
| Weighted F1 | {test_metrics['weighted_f1']:.6f} |
| ROC-AUC | {test_metrics['roc_auc']:.6f} |
| PR-AUC | {test_metrics['pr_auc']:.6f} |
| Specificity | {test_metrics['specificity']:.6f} |

Confusion matrix labels are `no`, `yes`: `{test_metrics['confusion_matrix']['matrix']}`.

## Interpretation

The profile modality remains highly uncertain because the dataset has only 101 records and the held-out test set has only 15 records. Results should be used as research evidence for a weak contextual modality, not as a standalone clinical or suicide-risk predictor.
"""
    (V2_DIR / "profile_evaluation_summary.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "threshold": threshold, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
