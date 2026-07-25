"""Offline synthetic late-fusion evaluation runner."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from .synthetic import (
    FEATURE_COLUMNS,
    LATENT_COLUMN,
    TARGET_COLUMN,
    generate_synthetic_fusion_dataset,
    load_fusion_config,
    validate_synthetic_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "ml-research" / "configs" / "fusion.synthetic_late_fusion.v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "generated" / "reports" / "fusion_evaluation"
THESIS_REPORT = REPO_ROOT / "ml-research" / "thesis_evidence" / "reports" / "fusion_evaluation_report.md"


@dataclass
class ModelResult:
    model_name: str
    estimator: Any
    validation_macro_f1: float
    hyperparameters: dict[str, Any]
    split_metrics: dict[str, dict[str, Any]]
    inference_time_ms_per_observation: float


class WeightedRuleBaseline:
    """Offline clone of the existing weighted composite formula."""

    def __init__(self, weights: dict[str, float], thresholds: list[float], labels: list[str]) -> None:
        self.weights = weights
        self.thresholds = thresholds
        self.labels = labels
        self.train_medians: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "WeightedRuleBaseline":
        self.train_medians = X.median(numeric_only=True)
        return self

    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        if self.train_medians is None:
            raise RuntimeError("WeightedRuleBaseline must be fitted before prediction.")
        filled = X.fillna(self.train_medians)
        scores = np.zeros(len(filled), dtype=float)
        for feature in FEATURE_COLUMNS:
            weight = float(self.weights.get(feature, 0.0))
            scores += filled[feature].to_numpy(dtype=float) * weight
        return scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._scores(X)
        predictions = []
        for score in scores:
            if score < self.thresholds[0]:
                predictions.append(self.labels[0])
            elif score < self.thresholds[1]:
                predictions.append(self.labels[1])
            elif score < self.thresholds[2]:
                predictions.append(self.labels[2])
            else:
                predictions.append(self.labels[3])
        return np.asarray(predictions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def _probability_metrics(y_true: pd.Series, probabilities: np.ndarray | None, labels: list[str]) -> dict[str, Any]:
    if probabilities is None:
        return {
            "roc_auc_ovr": None,
            "average_precision_ovr": None,
            "calibration": {"available": False, "reason": "No class probabilities are produced by this baseline."},
        }
    y_bin = label_binarize(y_true, classes=labels)
    metrics: dict[str, Any] = {}
    if y_bin.shape[1] == probabilities.shape[1] and all(y_bin[:, idx].sum() > 0 for idx in range(y_bin.shape[1])):
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr"))
        metrics["average_precision_ovr"] = float(average_precision_score(y_bin, probabilities, average="macro"))
    else:
        metrics["roc_auc_ovr"] = None
        metrics["average_precision_ovr"] = None

    one_hot = pd.get_dummies(pd.Categorical(y_true, categories=labels)).to_numpy(dtype=float)
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    predicted_confidence = probabilities.max(axis=1)
    correctness = (np.asarray(labels, dtype=object)[probabilities.argmax(axis=1)] == y_true.to_numpy()).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    calibration_bins = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (predicted_confidence >= lower) & (predicted_confidence < upper if upper < 1 else predicted_confidence <= upper)
        if mask.any():
            accuracy = float(correctness[mask].mean())
            confidence = float(predicted_confidence[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - confidence)
            calibration_bins.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "count": int(mask.sum()),
                    "mean_confidence": confidence,
                    "observed_accuracy": accuracy,
                }
            )
    metrics["calibration"] = {
        "available": True,
        "multiclass_brier_score": brier,
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "expected_calibration_error": float(ece),
        "bins": calibration_bins,
    }
    return metrics


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    high_idx = labels.index("high")
    severe_idx = labels.index("severe")
    false_negatives = {
        "high": int(matrix[high_idx, :].sum() - matrix[high_idx, high_idx]),
        "severe": int(matrix[severe_idx, :].sum() - matrix[severe_idx, severe_idx]),
    }
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    elevated_true = y_true.isin(["high", "severe"]).astype(int)
    elevated_pred = pd.Series(y_pred).isin(["high", "severe"]).astype(int)
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
        "false_negatives_high_severe": false_negatives,
        "binary_elevated_sensitivity": {
            "positive_definition": "high or severe synthetic risk category",
            "recall": float(recall_score(elevated_true, elevated_pred, zero_division=0)),
            "f1": float(f1_score(elevated_true, elevated_pred, zero_division=0)),
        },
    }
    payload.update(_probability_metrics(y_true, probabilities, labels))
    return payload


def model_candidates(config: dict[str, Any]) -> dict[str, list[Any]]:
    seed = int(config["random_seed"])
    return {
        "logistic_regression": [
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=C,
                            class_weight=class_weight,
                            max_iter=1000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for C in config["models"]["logistic_regression"]["C"]
            for class_weight in config["models"]["logistic_regression"]["class_weight"]
        ],
        "random_forest": [
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=int(n_estimators),
                            max_depth=max_depth,
                            min_samples_leaf=int(min_samples_leaf),
                            class_weight=class_weight,
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            for n_estimators in config["models"]["random_forest"]["n_estimators"]
            for max_depth in config["models"]["random_forest"]["max_depth"]
            for min_samples_leaf in config["models"]["random_forest"]["min_samples_leaf"]
            for class_weight in config["models"]["random_forest"]["class_weight"]
        ],
        "gaussian_naive_bayes": [
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", GaussianNB(var_smoothing=float(var_smoothing))),
                ]
            )
            for var_smoothing in config["models"]["gaussian_naive_bayes"]["var_smoothing"]
        ],
    }


def estimator_params(estimator: Any) -> dict[str, Any]:
    classifier = estimator.named_steps["classifier"]
    params = classifier.get_params()
    wanted = ["C", "class_weight", "n_estimators", "max_depth", "min_samples_leaf", "var_smoothing"]
    return {key: params.get(key) for key in wanted if key in params}


def fit_weighted_baseline(df: pd.DataFrame, config: dict[str, Any]) -> ModelResult:
    labels = list(config["risk_labels"])
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    X_train, y_train = split_xy(train)
    baseline = WeightedRuleBaseline(
        weights=config["weighted_rule_baseline"]["weights"],
        thresholds=config["weighted_rule_baseline"]["thresholds"],
        labels=labels,
    ).fit(X_train, y_train)
    split_metrics = {}
    start = time.perf_counter()
    for split_name, split_df in [("train", train), ("validation", validation), ("test", test)]:
        X_split, y_split = split_xy(split_df)
        pred = baseline.predict(X_split)
        split_metrics[split_name] = evaluate_predictions(y_split, pred, labels)
    elapsed = time.perf_counter() - start
    return ModelResult(
        model_name="weighted_rule_baseline_original_formula",
        estimator=baseline,
        validation_macro_f1=split_metrics["validation"]["macro_f1"],
        hyperparameters={
            "source": "backend/app/models/weights.csv",
            "thresholds": config["weighted_rule_baseline"]["thresholds"],
            "missing_values": "train median imputation fitted on training split only",
        },
        split_metrics=split_metrics,
        inference_time_ms_per_observation=(elapsed / max(len(df), 1)) * 1000,
    )


def fit_learned_model(model_name: str, candidates: list[Any], df: pd.DataFrame, labels: list[str]) -> ModelResult:
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    X_train, y_train = split_xy(train)
    X_validation, y_validation = split_xy(validation)

    best_estimator = None
    best_score = -1.0
    best_params: dict[str, Any] = {}
    for candidate in candidates:
        estimator = clone(candidate)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_validation)
        score = f1_score(y_validation, pred, labels=labels, average="macro", zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_estimator = estimator
            best_params = estimator_params(estimator)

    if best_estimator is None:
        raise RuntimeError(f"No estimator selected for {model_name}.")

    split_metrics = {}
    total_elapsed = 0.0
    total_rows = 0
    for split_name, split_df in [("train", train), ("validation", validation), ("test", test)]:
        X_split, y_split = split_xy(split_df)
        start = time.perf_counter()
        pred = best_estimator.predict(X_split)
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        total_rows += len(split_df)
        probabilities = best_estimator.predict_proba(X_split) if hasattr(best_estimator, "predict_proba") else None
        split_metrics[split_name] = evaluate_predictions(y_split, pred, labels, probabilities)

    return ModelResult(
        model_name=model_name,
        estimator=best_estimator,
        validation_macro_f1=float(best_score),
        hyperparameters=best_params,
        split_metrics=split_metrics,
        inference_time_ms_per_observation=(total_elapsed / max(total_rows, 1)) * 1000,
    )


def run_model_comparison(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, ModelResult]:
    labels = list(config["risk_labels"])
    results = {"weighted_rule_baseline_original_formula": fit_weighted_baseline(df, config)}
    for model_name, candidates in model_candidates(config).items():
        results[model_name] = fit_learned_model(model_name, candidates, df, labels)
    return results


def result_rows(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        test = result.split_metrics["test"]
        train = result.split_metrics["train"]
        validation = result.split_metrics["validation"]
        rows.append(
            {
                "model": name,
                "selection_metric": "validation_macro_f1",
                "validation_macro_f1": result.validation_macro_f1,
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_precision": test["macro_precision"],
                "test_macro_recall": test["macro_recall"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
                "test_high_recall": test["per_class"]["high"]["recall"],
                "test_severe_recall": test["per_class"]["severe"]["recall"],
                "test_high_false_negatives": test["false_negatives_high_severe"]["high"],
                "test_severe_false_negatives": test["false_negatives_high_severe"]["severe"],
                "test_roc_auc_ovr": test["roc_auc_ovr"],
                "test_average_precision_ovr": test["average_precision_ovr"],
                "train_validation_macro_f1_gap": train["macro_f1"] - validation["macro_f1"],
                "validation_test_macro_f1_gap": validation["macro_f1"] - test["macro_f1"],
                "inference_ms_per_observation": result.inference_time_ms_per_observation,
            }
        )
    return pd.DataFrame(rows).sort_values(["validation_macro_f1", "test_macro_f1"], ascending=False)


def per_class_rows(results: dict[str, ModelResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        for label, values in result.split_metrics[split]["per_class"].items():
            rows.append({"model": name, "split": split, "synthetic_risk_category": label, **values})
    return pd.DataFrame(rows)


def confusion_rows(results: dict[str, ModelResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        labels = result.split_metrics[split]["confusion_matrix"]["labels"]
        matrix = result.split_metrics[split]["confusion_matrix"]["matrix"]
        for true_label, row in zip(labels, matrix):
            for predicted_label, count in zip(labels, row):
                rows.append(
                    {
                        "model": name,
                        "split": split,
                        "true_synthetic_risk_category": true_label,
                        "predicted_synthetic_risk_category": predicted_label,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def high_severe_recall(metrics: dict[str, Any]) -> float:
    return float(np.mean([metrics["per_class"]["high"]["recall"], metrics["per_class"]["severe"]["recall"]]))


def run_ablation(
    df: pd.DataFrame,
    config: dict[str, Any],
    selected_model: Any,
    selected_model_name: str,
) -> pd.DataFrame:
    labels = list(config["risk_labels"])
    scenarios: dict[str, list[str]] = {"all_modalities": FEATURE_COLUMNS.copy()}
    for feature in FEATURE_COLUMNS:
        scenarios[f"without_{feature}"] = [column for column in FEATURE_COLUMNS if column != feature]
    scenarios["dass21_only"] = ["dass21_score"]
    scenarios["text_only"] = ["text_score"]
    scenarios["non_psychometric_modalities_only"] = ["speech_score", "face_score", "behavioral_score"]

    base_macro = None
    base_recall = None
    rows = []
    for scenario, features in scenarios.items():
        model = clone(selected_model)
        model.fit(df.loc[df["split"] == "train", features], df.loc[df["split"] == "train", TARGET_COLUMN])
        X_test = df.loc[df["split"] == "test", features]
        y_test = df.loc[df["split"] == "test", TARGET_COLUMN]
        pred = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, pred, labels, probabilities)
        if scenario == "all_modalities":
            base_macro = metrics["macro_f1"]
            base_recall = high_severe_recall(metrics)
        rows.append(
            {
                "scenario": scenario,
                "features": ",".join(features),
                "test_macro_f1": metrics["macro_f1"],
                "delta_macro_f1": 0.0 if base_macro is None else metrics["macro_f1"] - base_macro,
                "high_severe_mean_recall": high_severe_recall(metrics),
                "delta_high_severe_mean_recall": 0.0 if base_recall is None else high_severe_recall(metrics) - base_recall,
                "high_false_negatives": metrics["false_negatives_high_severe"]["high"],
                "severe_false_negatives": metrics["false_negatives_high_severe"]["severe"],
            }
        )
    return pd.DataFrame(rows)


def clone_selected_estimator(results: dict[str, ModelResult], selected_model_name: str) -> Any:
    return clone(results[selected_model_name].estimator)


def run_missing_modality_robustness(
    df: pd.DataFrame,
    config: dict[str, Any],
    selected_model: Any,
    selected_model_name: str,
) -> pd.DataFrame:
    labels = list(config["risk_labels"])
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    selected_model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    rows = []
    scenarios = [("random_10_percent", 0.10, None), ("random_20_percent", 0.20, None)]
    scenarios.extend([(f"{feature}_absent", 0.0, feature) for feature in FEATURE_COLUMNS])
    for scenario, rate, absent_feature in scenarios:
        X_test = test[FEATURE_COLUMNS].copy()
        stable_offset = int(hashlib.sha256(scenario.encode("utf-8")).hexdigest()[:8], 16) % 10000
        rng = np.random.default_rng(int(config["random_seed"]) + stable_offset)
        if absent_feature:
            X_test[absent_feature] = np.nan
        else:
            mask = rng.random(X_test.shape) < rate
            X_test = X_test.mask(mask)
        pred = selected_model.predict(X_test)
        probabilities = selected_model.predict_proba(X_test) if hasattr(selected_model, "predict_proba") else None
        metrics = evaluate_predictions(test[TARGET_COLUMN], pred, labels, probabilities)
        rows.append(
            {
                "model": selected_model_name,
                "scenario": scenario,
                "test_macro_f1": metrics["macro_f1"],
                "high_recall": metrics["per_class"]["high"]["recall"],
                "severe_recall": metrics["per_class"]["severe"]["recall"],
                "high_false_negatives": metrics["false_negatives_high_severe"]["high"],
                "severe_false_negatives": metrics["false_negatives_high_severe"]["severe"],
            }
        )
    return pd.DataFrame(rows)


def run_noise_sensitivity(df: pd.DataFrame, config: dict[str, Any], selected_model: Any, selected_model_name: str) -> pd.DataFrame:
    labels = list(config["risk_labels"])
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    selected_model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    rows = []
    for noise_sd in config["robustness"]["noise_test_sd"]:
        X_test = test[FEATURE_COLUMNS].copy()
        rng = np.random.default_rng(int(config["random_seed"]) + int(noise_sd * 1000))
        noise = rng.normal(0, float(noise_sd), X_test.shape)
        X_test = X_test + noise
        for feature in FEATURE_COLUMNS:
            X_test[feature] = X_test[feature].clip(0, 100)
        pred = selected_model.predict(X_test)
        probabilities = selected_model.predict_proba(X_test) if hasattr(selected_model, "predict_proba") else None
        metrics = evaluate_predictions(test[TARGET_COLUMN], pred, labels, probabilities)
        rows.append(
            {
                "model": selected_model_name,
                "test_noise_sd": noise_sd,
                "test_macro_f1": metrics["macro_f1"],
                "high_recall": metrics["per_class"]["high"]["recall"],
                "severe_recall": metrics["per_class"]["severe"]["recall"],
            }
        )
    return pd.DataFrame(rows)


def run_distribution_shift(
    df: pd.DataFrame,
    config: dict[str, Any],
    selected_model: Any,
    selected_model_name: str,
) -> pd.DataFrame:
    labels = list(config["risk_labels"])
    selected_model.fit(df.loc[df["split"] == "train", FEATURE_COLUMNS], df.loc[df["split"] == "train", TARGET_COLUMN])
    normal_test = df[df["split"] == "test"]
    shifted = generate_synthetic_fusion_dataset(config, seed=int(config["random_seed"]) + 999, shifted=True)
    shifted_test = shifted[shifted["split"] == "test"]
    rows = []
    for scenario, frame in [("original_test", normal_test), ("shifted_synthetic_test", shifted_test)]:
        pred = selected_model.predict(frame[FEATURE_COLUMNS])
        probabilities = selected_model.predict_proba(frame[FEATURE_COLUMNS]) if hasattr(selected_model, "predict_proba") else None
        metrics = evaluate_predictions(frame[TARGET_COLUMN], pred, labels, probabilities)
        rows.append(
            {
                "model": selected_model_name,
                "scenario": scenario,
                "test_macro_f1": metrics["macro_f1"],
                "high_recall": metrics["per_class"]["high"]["recall"],
                "severe_recall": metrics["per_class"]["severe"]["recall"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def run_seed_stability(config: dict[str, Any], selected_model_name: str) -> pd.DataFrame:
    rows = []
    for seed in config["robustness"]["seed_stability_seeds"]:
        seed_config = dict(config)
        seed_config["random_seed"] = int(seed)
        seed_df = generate_synthetic_fusion_dataset(seed_config)
        result = fit_learned_model(selected_model_name, model_candidates(seed_config)[selected_model_name], seed_df, list(config["risk_labels"]))
        test = result.split_metrics["test"]
        rows.append(
            {
                "seed": seed,
                "model": selected_model_name,
                "test_macro_f1": test["macro_f1"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "high_recall": test["per_class"]["high"]["recall"],
                "severe_recall": test["per_class"]["severe"]["recall"],
            }
        )
    detail = pd.DataFrame(rows)
    summary = detail.drop(columns=["seed"]).groupby("model").agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(part) for part in column if part]) for column in summary.columns.to_flat_index()]
    detail["row_type"] = "seed"
    summary["row_type"] = "summary"
    return pd.concat([detail, summary], ignore_index=True, sort=False)


def run_sample_size_sensitivity(config: dict[str, Any], selected_model_name: str) -> pd.DataFrame:
    rows = []
    labels = list(config["risk_labels"])
    for participants in config["robustness"]["sample_size_participants"]:
        size_df = generate_synthetic_fusion_dataset(config, n_participants=int(participants))
        result = fit_learned_model(selected_model_name, model_candidates(config)[selected_model_name], size_df, labels)
        test = result.split_metrics["test"]
        rows.append(
            {
                "synthetic_participants": participants,
                "model": selected_model_name,
                "train_observations": int((size_df["split"] == "train").sum()),
                "test_macro_f1": test["macro_f1"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "high_recall": test["per_class"]["high"]["recall"],
                "severe_recall": test["per_class"]["severe"]["recall"],
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=220)
    fig.savefig(figures_dir / f"{name}.svg")
    plt.close(fig)


def plot_confusion(result: ModelResult, figures_dir: Path, name: str, title: str) -> None:
    labels = result.split_metrics["test"]["confusion_matrix"]["labels"]
    matrix = np.asarray(result.split_metrics["test"]["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted synthetic risk category")
    ax.set_ylabel("Synthetic risk category")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir, name)


def generate_figures(
    df: pd.DataFrame,
    results: dict[str, ModelResult],
    selected_model_name: str,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    figures_dir: Path,
) -> None:
    labels = list(config["risk_labels"])

    fig, ax = plt.subplots(figsize=(9, 5))
    workflow = [
        "Hidden synthetic distress factor",
        "Noisy modality score generation",
        "Participant-grouped split",
        "Train-only preprocessing",
        "Late-fusion comparison",
        "Research-only reports",
    ]
    ax.axis("off")
    for idx, text in enumerate(workflow):
        ax.text(0.5, 1 - idx * 0.16, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#f2f6f8", "ec": "#4b6670"})
        if idx < len(workflow) - 1:
            ax.annotate("", xy=(0.5, 0.90 - idx * 0.16), xytext=(0.5, 0.82 - idx * 0.16), arrowprops={"arrowstyle": "->"})
    ax.set_title("Synthetic Fusion Data Generation Workflow")
    save_figure(fig, figures_dir, "synthetic_fusion_data_generation_workflow")

    class_counts = df[TARGET_COLUMN].value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(class_counts.index, class_counts.values, color="#4e79a7")
    ax.set_title("Synthetic Class Distribution")
    ax.set_xlabel("Synthetic risk category")
    ax.set_ylabel("Observation count")
    save_figure(fig, figures_dir, "synthetic_class_distribution")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[idx]
        data = [df.loc[df[TARGET_COLUMN] == label, feature].dropna() for label in labels]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(feature)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[len(FEATURE_COLUMNS):]:
        ax.axis("off")
    fig.suptitle("Modality Score Distributions By Synthetic Class")
    save_figure(fig, figures_dir, "modality_score_distributions_by_synthetic_class")

    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[FEATURE_COLUMNS].corr()
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Modality Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir, "modality_correlation_heatmap")

    comparison = tables["fusion_model_comparison"].sort_values("test_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(comparison["model"], comparison["test_macro_f1"], color="#59a14f")
    ax.set_xlabel("Test macro F1")
    ax.set_title("Fusion Model Metric Comparison")
    save_figure(fig, figures_dir, "fusion_model_metric_comparison")

    plot_confusion(results["weighted_rule_baseline_original_formula"], figures_dir, "confusion_matrix_weighted_rule", "Weighted Rule Confusion Matrix")
    plot_confusion(results["logistic_regression"], figures_dir, "confusion_matrix_logistic_regression", "Logistic Regression Confusion Matrix")
    plot_confusion(results["random_forest"], figures_dir, "confusion_matrix_random_forest", "Random Forest Confusion Matrix")

    selected = results[selected_model_name].estimator
    test = df[df["split"] == "test"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]
    probabilities = selected.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], probabilities[:, idx])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Multiclass ROC Curves Selected Model")
    ax.legend()
    save_figure(fig, figures_dir, "multiclass_roc_curves_selected_model")

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], probabilities[:, idx])
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curves Selected Model")
    ax.legend()
    save_figure(fig, figures_dir, "multiclass_precision_recall_curves_selected_model")

    fig, ax = plt.subplots(figsize=(7, 6))
    pred_conf = probabilities.max(axis=1)
    correct = (selected.predict(X_test) == y_test).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, pred_conf, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=selected_model_name)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed correctness")
    ax.set_title("Calibration Curve Selected Model")
    ax.legend()
    save_figure(fig, figures_dir, "calibration_curve_selected_model")

    fig, ax = plt.subplots(figsize=(8, 5))
    rf = results["random_forest"].estimator.named_steps["classifier"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    ax.barh(importances.index, importances.values, color="#f28e2b")
    ax.set_title("Fusion Feature Importance")
    ax.set_xlabel("Random forest impurity importance")
    save_figure(fig, figures_dir, "fusion_feature_importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    lr = results["logistic_regression"].estimator.named_steps["classifier"]
    coef = pd.DataFrame(lr.coef_, columns=FEATURE_COLUMNS, index=lr.classes_)
    image = ax.imshow(coef, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(range(len(lr.classes_)), lr.classes_)
    ax.set_title("Logistic Regression Coefficients")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir, "logistic_regression_coefficients")

    figure_names = {
        "modality_ablation_results": "modality_ablation_results",
        "missing_modality_results": "missing_modality_robustness",
        "noise_sensitivity_results": "noise_sensitivity_results",
        "distribution_shift_results": "distribution_shift_comparison",
        "sample_size_sensitivity_results": "sample_size_sensitivity",
    }
    for table_name, x_col, y_col, title in [
        ("modality_ablation_results", "scenario", "test_macro_f1", "Modality Ablation Results"),
        ("missing_modality_results", "scenario", "test_macro_f1", "Missing Modality Robustness"),
        ("noise_sensitivity_results", "test_noise_sd", "test_macro_f1", "Noise Sensitivity Results"),
        ("distribution_shift_results", "scenario", "test_macro_f1", "Distribution Shift Comparison"),
        ("sample_size_sensitivity_results", "synthetic_participants", "test_macro_f1", "Sample Size Sensitivity"),
    ]:
        if table_name in tables:
            table = tables[table_name].dropna(subset=[y_col])
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(table[x_col].astype(str), table[y_col], marker="o")
            ax.set_title(title)
            ax.set_xlabel(x_col.replace("_", " "))
            ax.set_ylabel("Test macro F1")
            ax.tick_params(axis="x", rotation=35)
            save_figure(fig, figures_dir, figure_names[table_name])

    seed_rows = tables["seed_stability_results"][tables["seed_stability_results"]["row_type"] == "seed"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        ["macro F1", "high recall", "severe recall"],
        [
            seed_rows["test_macro_f1"].mean(),
            seed_rows["high_recall"].mean(),
            seed_rows["severe_recall"].mean(),
        ],
        yerr=[
            seed_rows["test_macro_f1"].std(),
            seed_rows["high_recall"].std(),
            seed_rows["severe_recall"].std(),
        ],
        fmt="o",
        capsize=5,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Seed Stability Results")
    ax.set_ylabel("Mean +/- standard deviation")
    save_figure(fig, figures_dir, "seed_stability_results")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[LATENT_COLUMN], bins=30, color="#76b7b2")
    ax.set_title("Synthetic Risk Score Distribution")
    ax.set_xlabel("Synthetic latent research label score")
    ax.set_ylabel("Observation count")
    save_figure(fig, figures_dir, "synthetic_risk_score_distribution")


def build_tables(
    df: pd.DataFrame,
    config: dict[str, Any],
    results: dict[str, ModelResult],
    selected_model_name: str,
) -> dict[str, pd.DataFrame]:
    config_rows = [
        {"parameter": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
        for key, value in config.items()
        if key not in {"modality_generation"}
    ]
    summary = pd.DataFrame(
        [
            {"metric": "observations", "value": len(df)},
            {"metric": "synthetic_participants", "value": df["synthetic_participant_id"].nunique()},
            {"metric": "score_range", "value": "0-100"},
            {"metric": "target_policy", "value": "synthetic latent research label from hidden latent process"},
            {"metric": "selected_learned_model", "value": selected_model_name},
        ]
    )
    split_summary = df.groupby("split").agg(
        observations=("synthetic_observation_id", "count"),
        participants=("synthetic_participant_id", "nunique"),
    ).reset_index()
    class_distribution = df.groupby(["split", TARGET_COLUMN]).size().reset_index(name="count")
    modality_parameters = pd.DataFrame(
        [{"modality": key, **value} for key, value in config["modality_generation"].items()]
    )
    missingness = pd.DataFrame(
        [
            {
                "modality": feature,
                "missing_count": int(df[feature].isna().sum()),
                "missing_rate": float(df[feature].isna().mean()),
            }
            for feature in FEATURE_COLUMNS
        ]
    )
    hyperparameters = pd.DataFrame(
        [{"model": name, **result.hyperparameters} for name, result in results.items()]
    )
    claims = pd.DataFrame(
        [
            {
                "claim_area": "fusion feasibility",
                "allowed_wording": "The synthetic late-fusion experiment demonstrates technical feasibility under controlled synthetic assumptions.",
                "limitation": "It does not establish clinical validity, real-world suicide-risk prediction, or deployment readiness.",
            },
            {
                "claim_area": "target",
                "allowed_wording": "The target is a synthetic latent research label and synthetic risk category.",
                "limitation": "It is not clinical ground truth, diagnosis, patient outcome, validated suicide risk, or real-world participant truth.",
            },
            {
                "claim_area": "modalities",
                "allowed_wording": "Speech and face inputs are synthetic score variables informed by emotion-classification research artifacts.",
                "limitation": "Emotion labels are not depression labels or suicide labels.",
            },
            {
                "claim_area": "DASS21",
                "allowed_wording": "DASS21 is treated as a symptom questionnaire score within a synthetic experiment.",
                "limitation": "DASS21 is not a suicide-risk classifier.",
            },
        ]
    )
    return {
        "synthetic_dataset_configuration": pd.DataFrame(config_rows),
        "synthetic_dataset_summary": summary,
        "synthetic_split_summary": split_summary,
        "synthetic_class_distribution": class_distribution,
        "modality_generation_parameters": modality_parameters,
        "modality_missingness_summary": missingness,
        "fusion_model_hyperparameters": hyperparameters,
        "fusion_model_comparison": result_rows(results),
        "per_class_metrics": per_class_rows(results),
        "confusion_matrices_summary": confusion_rows(results),
        "fusion_claims_and_limitations": claims,
    }


def environment_report() -> dict[str, Any]:
    packages = {}
    for module_name in ["numpy", "pandas", "sklearn", "matplotlib", "joblib"]:
        module = __import__(module_name)
        packages[module_name] = getattr(module, "__version__", "unknown")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def write_model_card(path: Path, selected_model_name: str, results: dict[str, ModelResult]) -> None:
    test = results[selected_model_name].split_metrics["test"]
    text = f"""# Synthetic Late-Fusion Model Card

## Intended Use

Offline research-only synthetic late-fusion feasibility analysis.

## Prohibited Uses

- diagnosis
- suicide-risk prediction in real people
- autonomous intervention
- counselor alerting
- treatment recommendation
- production scoring
- participant care
- institutional disciplinary decisions

## Selected Research Artifact

Selected learned model: `{selected_model_name}` by validation macro F1. Test macro F1 on the synthetic risk category task: `{test['macro_f1']:.6f}`.

## Required Limitations

- The observations are participant-aligned synthetic observations.
- The labels are synthetic latent research labels and synthetic risk categories.
- There is no participant-aligned real multimodal dataset.
- The unimodal models were trained on different datasets and tasks.
- Emotion is not suicide risk.
- DASS21 is not a suicide-risk classifier.
- There is no clinical validation.
- There is no Sri Lankan pilot validation.
- The artifact is not deployment ready.
- Results are sensitive to generator assumptions.
- Circularity was reduced by generating the target from a hidden latent process rather than from the weighted formula or any single modality.
- Missing-modality and distribution-shift findings are synthetic robustness checks only.

This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system.
"""
    path.write_text(text, encoding="utf-8")


def write_reports(
    run_dir: Path,
    config: dict[str, Any],
    validation: dict[str, Any],
    selected_model_name: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    comparison = tables["fusion_model_comparison"]
    best = comparison.iloc[0]
    limitations = """# Synthetic Fusion Limitations

- Fusion inputs are synthetic score variables informed by modality research, not joined real predictions.
- No raw participant text, audio, images, identifiers, or production database records were used.
- The synthetic latent research label is not clinical ground truth, suicide-risk ground truth, diagnosis, patient outcome, validated suicide risk, or real-world participant truth.
- DASS21 is a symptom questionnaire and is not a suicide-risk classifier.
- Speech and Face emotion labels are not depression labels or suicide labels.
- No current research-model output may trigger intervention, treatment, alerts, counselor action, or participant care.
"""
    (run_dir / "limitations_report.md").write_text(limitations, encoding="utf-8")

    interpretation = f"""# Synthetic Late-Fusion Evaluation Report

## Research Question

This offline research experiment evaluates whether participant-aligned synthetic observations can support a reproducible fusion feasibility comparison.

## Why Real Supervised Fusion Was Not Scientifically Possible

The repository evidence shows no participant-aligned real multimodal dataset with a shared outcome label. The Profile, Text, Speech, and Face artifacts use different datasets, label spaces, and tasks. Joining them as if they represented the same participants would create unsupported evidence.

## Why Synthetic Participant Alignment Was Used

Synthetic participant alignment was used to test late-fusion mechanics under controlled assumptions while avoiding raw participant text, audio, images, identifiers, production database records, and unsupported real-world labels.

## Circularity Prevention

The synthetic latent research label is sampled from a hidden latent process with target noise. Observable modality scores are noisy, imperfect measurements of that hidden factor. The existing weighted rule formula is evaluated only as a baseline and does not generate the target.

## Models Compared

- Existing weighted rule-based fusion, preserved as `weighted_rule_baseline_original_formula`.
- Logistic Regression late fusion.
- Random Forest late fusion.
- Gaussian Naive Bayes late fusion, using the existing scikit-learn dependency.

## Evaluation Protocol

Participant-grouped train/validation/test splits were generated with seed `{config['random_seed']}`. Learned models were fitted on training data only, hyperparameters were selected on validation macro F1 only, and test results were evaluated once after selection. Imputers and scalers live inside scikit-learn pipelines and are fitted on training data only.

## Principal Findings

Best validation-selected comparison row: `{best['model']}` with test macro F1 `{best['test_macro_f1']:.6f}`. Accuracy is reported but macro F1 and high/severe synthetic-category recall are the primary interpretation metrics.

## What This Demonstrates

The synthetic late-fusion experiment demonstrated the technical feasibility of combining participant-aligned modality scores under controlled assumptions.

## What This Does Not Demonstrate

The results do not establish clinical validity, real-world suicide-risk prediction, participant outcome prediction, diagnosis, intervention readiness, counselor alert readiness, or deployment readiness.

## Thesis-Safe Wording

The synthetic late-fusion experiment demonstrated the technical feasibility of combining participant-aligned modality scores under controlled assumptions. The results do not establish clinical validity, real-world suicide-risk prediction, or deployment readiness.

## Prohibited Claims

- The fusion model predicts suicide risk in real people.
- The synthetic target is clinical ground truth.
- DASS21 is a suicide-risk classifier.
- Emotion labels from Speech or Face are depression or suicide labels.
- The model can trigger intervention, treatment, alerts, counselor action, or participant care.

## Integrity Status

Synthetic dataset integrity status: `{validation['status']}`.
"""
    (run_dir / "thesis_interpretation_report.md").write_text(interpretation, encoding="utf-8")
    THESIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    THESIS_REPORT.write_text(interpretation, encoding="utf-8")


def build_manifest(run_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    return {
        "artifact_manifest_version": "1.0.0",
        "research_only": True,
        "file_hashes": {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in files},
    }


def run_fusion_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> Path:
    config = load_fusion_config(config_path)
    run_name = run_id or f"synthetic_fusion_v1_seed{config['random_seed']}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = Path(output_root) / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "tables" / "csv").mkdir(parents=True)
    (run_dir / "tables" / "markdown").mkdir(parents=True)
    figures_dir = run_dir / "figures"

    df = generate_synthetic_fusion_dataset(config)
    validation = validate_synthetic_dataset(df, config)
    if validation["status"] != "passed":
        raise ValueError(f"Synthetic dataset failed integrity checks: {validation}")

    results = run_model_comparison(df, config)
    learned_results = {name: result for name, result in results.items() if name != "weighted_rule_baseline_original_formula"}
    selected_model_name = max(learned_results.values(), key=lambda result: result.validation_macro_f1).model_name
    selected_estimator = results[selected_model_name].estimator

    tables = build_tables(df, config, results, selected_model_name)
    tables["modality_ablation_results"] = run_ablation(df, config, clone(selected_estimator), selected_model_name)
    tables["missing_modality_results"] = run_missing_modality_robustness(
        df, config, clone(selected_estimator), selected_model_name
    )
    tables["noise_sensitivity_results"] = run_noise_sensitivity(df, config, clone(selected_estimator), selected_model_name)
    tables["distribution_shift_results"] = run_distribution_shift(df, config, clone(selected_estimator), selected_model_name)
    tables["seed_stability_results"] = run_seed_stability(config, selected_model_name)
    tables["sample_size_sensitivity_results"] = run_sample_size_sensitivity(config, selected_model_name)

    df.to_csv(run_dir / "synthetic_fusion_dataset.csv", index=False)
    df[["synthetic_participant_id", "synthetic_observation_id", "split"]].to_csv(run_dir / "split_manifest.csv", index=False)
    write_json(run_dir / "configuration_snapshot.json", config)
    write_json(run_dir / "dataset_integrity_report.json", validation)
    write_json(run_dir / "environment_reproducibility_report.json", environment_report())
    write_json(
        run_dir / "dataset_fingerprint.json",
        {
            "synthetic_fusion_dataset_sha256": sha256_file(run_dir / "synthetic_fusion_dataset.csv"),
            "split_manifest_sha256": sha256_file(run_dir / "split_manifest.csv"),
        },
    )

    metrics_payload = {
        "research_only": True,
        "selected_learned_model": selected_model_name,
        "primary_selection_metric": "validation_macro_f1",
        "safety_secondary_metric": "recall for high and severe synthetic risk categories",
        "models": {name: result.split_metrics for name, result in results.items()},
        "hyperparameters": {name: result.hyperparameters for name, result in results.items()},
        "inference_time_ms_per_observation": {
            name: result.inference_time_ms_per_observation for name, result in results.items()
        },
    }
    write_json(run_dir / "metrics.json", metrics_payload)

    for name, table in tables.items():
        write_table(table, run_dir / "tables" / "csv" / f"{name}.csv", run_dir / "tables" / "markdown" / f"{name}.md")

    generate_figures(df, results, selected_model_name, tables, config, figures_dir)
    joblib.dump(selected_estimator, run_dir / "selected_model.joblib")
    write_model_card(run_dir / "model_card.md", selected_model_name, results)
    write_reports(run_dir, config, validation, selected_model_name, tables)
    write_json(run_dir / "artifact_manifest.json", build_manifest(run_dir))
    return run_dir
