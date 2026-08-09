"""Run the controlled synthetic multimodal late-fusion v2 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC

from app.ml.fusion.weighted_late_fusion import CLASS_ORDER, WeightedLateFusion, validate_weights


SEED = 42
N_SYNTHETIC = 2400
MODALITIES = ["profile_score", "dass_score", "mood_score", "text_score", "speech_score", "face_score"]
AVAILABILITY_COLUMNS = [name.replace("_score", "_available") for name in MODALITIES]
FEATURE_COLUMNS = MODALITIES + AVAILABILITY_COLUMNS
TARGET = "target_risk_class"
TARGET_SCORE = "target_risk_score"

DATA_DIR = REPO_ROOT / "generated" / "datasets" / "fusion_controlled" / "v2"
REPORT_DIR = REPO_ROOT / "generated" / "reports" / "fusion_controlled" / "v2"
MANIFEST_PATH = REPO_ROOT / "generated" / "manifests" / "fusion_controlled_v2_split_manifest.json"
MODEL_DIR = REPO_ROOT / "ml_models" / "fusion" / "controlled-late-fusion-v2" / "2.0.0"
WEIGHT_CONFIG_PATH = BACKEND_ROOT / "app" / "ml" / "fusion" / "config" / "fusion_weights_v2.json"
THRESHOLD_CONFIG_PATH = BACKEND_ROOT / "app" / "ml" / "fusion" / "config" / "fusion_thresholds_v2.json"
THESIS_METRICS = REPO_ROOT / "docs" / "thesis" / "chapter4_metrics_summary.csv"
THESIS_INVENTORY = REPO_ROOT / "docs" / "thesis" / "chapter4_evidence_inventory.csv"
THESIS_SECTION = REPO_ROOT / "docs" / "thesis" / "chapter4_fusion_results_v2.md"


INITIAL_WEIGHTS = {
    "profile_score": 0.10,
    "dass_score": 0.25,
    "mood_score": 0.15,
    "text_score": 0.25,
    "speech_score": 0.13,
    "face_score": 0.12,
}
INITIAL_THRESHOLDS = {"low_moderate": 0.30, "moderate_high": 0.50, "high_severe": 0.70}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


class FusionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build score and availability-indicator features from modality scores."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FusionFeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        out = frame[MODALITIES].copy()
        for modality, available in zip(MODALITIES, AVAILABILITY_COLUMNS):
            out[available] = frame[modality].notna().astype(float)
        return out[FEATURE_COLUMNS]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    return FusionFeatureBuilder().transform(frame)


def class_from_score(score: float) -> str:
    if score < 0.30:
        return "Low"
    if score < 0.52:
        return "Moderate"
    if score < 0.72:
        return "High"
    return "Severe"


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def generate_controlled_dataset(seed: int = SEED, n: int = N_SYNTHETIC) -> pd.DataFrame:
    """Generate controlled synthetic participants from a hidden latent process."""

    rng = np.random.default_rng(seed)
    scenario_probs = {
        "aligned_low": 0.18,
        "aligned_moderate": 0.18,
        "aligned_high": 0.18,
        "aligned_severe": 0.14,
        "low_profile_high_text": 0.08,
        "high_dass_neutral_face": 0.08,
        "mood_deterioration_missing_speech": 0.08,
        "moderate_multimodal_evidence": 0.08,
    }
    scenario_names = list(scenario_probs)
    scenarios = rng.choice(scenario_names, size=n, p=np.asarray(list(scenario_probs.values())) / sum(scenario_probs.values()))

    latent_means = {
        "aligned_low": 0.20,
        "aligned_moderate": 0.43,
        "aligned_high": 0.63,
        "aligned_severe": 0.83,
        "low_profile_high_text": 0.57,
        "high_dass_neutral_face": 0.61,
        "mood_deterioration_missing_speech": 0.66,
        "moderate_multimodal_evidence": 0.49,
    }
    latent_sd = {
        "aligned_low": 0.09,
        "aligned_moderate": 0.10,
        "aligned_high": 0.10,
        "aligned_severe": 0.08,
        "low_profile_high_text": 0.12,
        "high_dass_neutral_face": 0.12,
        "mood_deterioration_missing_speech": 0.11,
        "moderate_multimodal_evidence": 0.10,
    }

    rows: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        latent = float(np.clip(rng.normal(latent_means[scenario], latent_sd[scenario]), 0.02, 0.98))
        interaction = 0.06 * max(latent - 0.55, 0.0) * rng.uniform(0.6, 1.4)
        nonlinear = 0.05 * (latent**2)
        target_score = float(np.clip(0.86 * latent + interaction + nonlinear + rng.normal(0, 0.075), 0.0, 1.0))

        shared = rng.normal(0, 0.055)
        profile = 0.42 * latent + 0.58 * rng.beta(2.1, 3.2) + shared * 0.20 + rng.normal(0, 0.105)
        dass = 0.70 * latent + 0.30 * rng.beta(2.5, 2.2) + shared * 0.25 + rng.normal(0, 0.075)
        mood = 0.54 * latent + 0.46 * rng.beta(2.0, 2.8) + shared * 0.22 + rng.normal(0, 0.105)
        text = 0.66 * latent + 0.34 * rng.beta(2.2, 2.4) + shared * 0.28 + rng.normal(0, 0.085)
        speech = 0.40 * latent + 0.60 * rng.beta(2.0, 2.9) + shared * 0.16 + rng.normal(0, 0.135)
        face = 0.34 * latent + 0.66 * rng.beta(2.1, 3.0) + shared * 0.14 + rng.normal(0, 0.145)

        if scenario == "low_profile_high_text":
            profile -= rng.uniform(0.18, 0.35)
            text += rng.uniform(0.18, 0.32)
        elif scenario == "high_dass_neutral_face":
            dass += rng.uniform(0.16, 0.28)
            face = 0.35 + rng.normal(0, 0.075)
        elif scenario == "mood_deterioration_missing_speech":
            mood += rng.uniform(0.17, 0.30)
        elif scenario == "moderate_multimodal_evidence":
            values = np.asarray([profile, dass, mood, text, speech, face])
            values = 0.48 + 0.50 * (values - values.mean()) + rng.normal(0, 0.04, size=6)
            profile, dass, mood, text, speech, face = values.tolist()

        values = dict(zip(MODALITIES, clip01(np.asarray([profile, dass, mood, text, speech, face]))))
        missing_rates = {
            "profile_score": 0.03,
            "dass_score": 0.02,
            "mood_score": 0.07,
            "text_score": 0.05,
            "speech_score": 0.09,
            "face_score": 0.10,
        }
        if scenario == "mood_deterioration_missing_speech":
            missing_rates["speech_score"] = 0.55
        for modality, rate in missing_rates.items():
            if rng.random() < rate:
                values[modality] = np.nan

        rows.append(
            {
                "synthetic_participant_id": f"fusion-v2-p{idx + 1:05d}",
                **values,
                TARGET_SCORE: round(target_score, 6),
                TARGET: class_from_score(target_score),
                "generation_seed": seed,
                "scenario_type": scenario,
            }
        )

    return pd.DataFrame(rows)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df[TARGET])
    validation, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp[TARGET])
    train = train.sort_values("synthetic_participant_id").reset_index(drop=True)
    validation = validation.sort_values("synthetic_participant_id").reset_index(drop=True)
    test = test.sort_values("synthetic_participant_id").reset_index(drop=True)
    manifest = {
        "version": "fusion_controlled_v2",
        "random_seed": SEED,
        "split_policy": "70/15/15 stratified by synthetic target_risk_class; test frozen before model selection",
        "class_order": CLASS_ORDER,
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "participant_id_hashes": {
            "train": hashlib.sha256("\n".join(train["synthetic_participant_id"]).encode()).hexdigest(),
            "validation": hashlib.sha256("\n".join(validation["synthetic_participant_id"]).encode()).hexdigest(),
            "test": hashlib.sha256("\n".join(test["synthetic_participant_id"]).encode()).hexdigest(),
        },
    }
    return train, validation, test, manifest


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    bootstrap_samples: int = 150,
) -> dict[str, Any]:
    labels = CLASS_ORDER
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    payload: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "predicted_class_distribution": {label: int((pd.Series(y_pred) == label).sum()) for label in labels},
    }
    if probabilities is not None:
        y_bin = label_binarize(y_true, classes=labels)
        payload["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr"))
        payload["roc_auc_weighted_ovr"] = float(roc_auc_score(y_bin, probabilities, average="weighted", multi_class="ovr"))
        payload["pr_auc_macro_ovr"] = float(average_precision_score(y_bin, probabilities, average="macro"))
        payload["pr_auc_weighted_ovr"] = float(average_precision_score(y_bin, probabilities, average="weighted"))
    else:
        payload["roc_auc_macro_ovr"] = None
        payload["roc_auc_weighted_ovr"] = None
        payload["pr_auc_macro_ovr"] = None
        payload["pr_auc_weighted_ovr"] = None

    rng = np.random.default_rng(SEED)
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_pred)
    n = len(y_arr)
    ci_metrics = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "severe_recall": []}
    for _ in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        sampled_true = y_arr[idx]
        sampled_pred = p_arr[idx]
        if len(set(sampled_true)) < 2:
            continue
        ci_metrics["accuracy"].append(accuracy_score(sampled_true, sampled_pred))
        ci_metrics["balanced_accuracy"].append(balanced_accuracy_score(sampled_true, sampled_pred))
        ci_metrics["macro_f1"].append(f1_score(sampled_true, sampled_pred, labels=labels, average="macro", zero_division=0))
        ci_metrics["severe_recall"].append(recall_score(sampled_true, sampled_pred, labels=["Severe"], average="macro", zero_division=0))
    payload["bootstrap_95ci"] = {
        key: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "bootstrap_samples": int(len(values)),
        }
        for key, values in ci_metrics.items()
        if values
    }
    return payload


def probabilities_for_estimator(estimator: Any, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)
    return None


@dataclass
class ModelResult:
    model_name: str
    estimator: Any
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    hyperparameters: dict[str, Any]


def candidate_models() -> dict[str, list[Any]]:
    return {
        "DummyClassifier": [
            Pipeline([("features", FusionFeatureBuilder()), ("imputer", SimpleImputer()), ("model", DummyClassifier(strategy="stratified", random_state=SEED))])
        ],
        "Multinomial Logistic Regression": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=c, max_iter=1000, random_state=SEED, class_weight=weight)),
                ]
            )
            for c in [1.0]
            for weight in ["balanced"]
        ],
        "Random Forest": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=220, max_depth=depth, min_samples_leaf=leaf, random_state=SEED, n_jobs=1, class_weight=weight)),
                ]
            )
            for depth in [8]
            for leaf in [3]
            for weight in ["balanced"]
        ],
        "Extra Trees": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", ExtraTreesClassifier(n_estimators=240, max_depth=depth, min_samples_leaf=leaf, random_state=SEED, n_jobs=1, class_weight=weight)),
                ]
            )
            for depth in [8]
            for leaf in [3]
            for weight in ["balanced"]
        ],
        "HistGradientBoosting": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=rate, max_leaf_nodes=nodes, random_state=SEED)),
                ]
            )
            for rate in [0.06]
            for nodes in [31]
        ],
        "Linear SVM": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", CalibratedClassifierCV(LinearSVC(C=c, random_state=SEED, class_weight=weight, dual="auto", max_iter=5000), cv=3)),
                ]
            )
            for c in [1.0]
            for weight in ["balanced"]
        ],
        "RBF SVM": [
            Pipeline(
                [
                    ("features", FusionFeatureBuilder()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(C=c, gamma=gamma, kernel="rbf", probability=True, random_state=SEED, class_weight=weight)),
                ]
            )
            for c in [1.0]
            for gamma in ["scale"]
            for weight in ["balanced"]
        ],
    }


def fit_model_family(name: str, candidates: list[Any], train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    X_train, y_train = train[MODALITIES], train[TARGET]
    X_val, y_val = validation[MODALITIES], validation[TARGET]
    best_estimator = None
    best_key: tuple[float, float, float] | None = None
    best_params: dict[str, Any] = {}
    for candidate in candidates:
        estimator = clone(candidate)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_val)
        key = (
            float(f1_score(y_val, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
            float(balanced_accuracy_score(y_val, pred)),
            float(recall_score(y_val, pred, labels=["Severe"], average="macro", zero_division=0)),
        )
        if best_key is None or key > best_key:
            best_estimator = estimator
            best_key = key
            best_params = estimator.get_params(deep=False)
    if best_estimator is None or best_key is None:
        raise RuntimeError(f"No estimator selected for {name}.")
    train_pred = best_estimator.predict(X_train)
    val_pred = best_estimator.predict(X_val)
    test_pred = best_estimator.predict(test[MODALITIES])
    return ModelResult(
        model_name=name,
        estimator=best_estimator,
        train_metrics=evaluate_predictions(y_train, train_pred, probabilities_for_estimator(best_estimator, X_train), bootstrap_samples=40),
        validation_metrics=evaluate_predictions(y_val, val_pred, probabilities_for_estimator(best_estimator, X_val), bootstrap_samples=60),
        test_metrics=evaluate_predictions(test[TARGET], test_pred, probabilities_for_estimator(best_estimator, test[MODALITIES]), bootstrap_samples=150),
        hyperparameters={key: str(value) for key, value in best_params.items() if key in {"model", "scaler", "imputer"}},
    )


def threshold_candidates() -> list[dict[str, float]]:
    return [
        INITIAL_THRESHOLDS,
        {"low_moderate": 0.28, "moderate_high": 0.50, "high_severe": 0.70},
        {"low_moderate": 0.30, "moderate_high": 0.52, "high_severe": 0.72},
        {"low_moderate": 0.32, "moderate_high": 0.52, "high_severe": 0.70},
        {"low_moderate": 0.30, "moderate_high": 0.48, "high_severe": 0.68},
    ]


def weight_candidates() -> list[dict[str, float]]:
    variants = [INITIAL_WEIGHTS]
    raw_variants = [
        {"profile_score": 0.10, "dass_score": 0.27, "mood_score": 0.14, "text_score": 0.25, "speech_score": 0.12, "face_score": 0.12},
        {"profile_score": 0.09, "dass_score": 0.25, "mood_score": 0.14, "text_score": 0.28, "speech_score": 0.12, "face_score": 0.12},
        {"profile_score": 0.11, "dass_score": 0.24, "mood_score": 0.17, "text_score": 0.24, "speech_score": 0.12, "face_score": 0.12},
        {"profile_score": 0.11, "dass_score": 0.26, "mood_score": 0.15, "text_score": 0.25, "speech_score": 0.10, "face_score": 0.13},
    ]
    for weights in raw_variants:
        total = sum(weights.values())
        variants.append({key: value / total for key, value in weights.items()})
    return variants


def evaluate_weighted(frame: pd.DataFrame, weights: dict[str, float], thresholds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    model = WeightedLateFusion(weights, thresholds)
    pred = model.predict(frame[MODALITIES])
    proba = model.predict_proba(frame[MODALITIES])
    return pred, proba


def select_weighted_rule(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], ModelResult, pd.DataFrame, pd.DataFrame]:
    validate_weights(INITIAL_WEIGHTS)
    rows = []
    best: tuple[float, float, float, dict[str, float], dict[str, float]] | None = None
    for weights in weight_candidates():
        for thresholds in threshold_candidates():
            pred, proba = evaluate_weighted(validation, weights, thresholds)
            metrics = evaluate_predictions(validation[TARGET], pred, proba, bootstrap_samples=20)
            row = {
                "weights": json.dumps(weights, sort_keys=True),
                "thresholds": json.dumps(thresholds, sort_keys=True),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_severe_recall": metrics["per_class"]["Severe"]["recall"],
            }
            rows.append(row)
            key = (metrics["macro_f1"], metrics["balanced_accuracy"], metrics["per_class"]["Severe"]["recall"], weights, thresholds)
            if best is None or key[:3] > best[:3]:
                best = key
    assert best is not None
    selected_weights = best[3]
    selected_thresholds = best[4]
    train_pred, train_proba = evaluate_weighted(train, selected_weights, selected_thresholds)
    val_pred, val_proba = evaluate_weighted(validation, selected_weights, selected_thresholds)
    test_pred, test_proba = evaluate_weighted(test, selected_weights, selected_thresholds)
    result = ModelResult(
        model_name="Fixed weighted late fusion",
        estimator=WeightedLateFusion(selected_weights, selected_thresholds),
        train_metrics=evaluate_predictions(train[TARGET], train_pred, train_proba, bootstrap_samples=40),
        validation_metrics=evaluate_predictions(validation[TARGET], val_pred, val_proba, bootstrap_samples=60),
        test_metrics=evaluate_predictions(test[TARGET], test_pred, test_proba, bootstrap_samples=150),
        hyperparameters={"weights": selected_weights, "thresholds": selected_thresholds},
    )
    weight_sensitivity = pd.DataFrame(rows)

    threshold_rows = []
    for thresholds in threshold_candidates():
        pred, proba = evaluate_weighted(validation, selected_weights, thresholds)
        val_metrics = evaluate_predictions(validation[TARGET], pred, proba, bootstrap_samples=20)
        test_pred, test_proba = evaluate_weighted(test, selected_weights, thresholds)
        test_metrics = evaluate_predictions(test[TARGET], test_pred, test_proba, bootstrap_samples=20)
        threshold_rows.append(
            {
                **thresholds,
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "test_macro_f1_after_finalization": test_metrics["macro_f1"],
                "test_balanced_accuracy_after_finalization": test_metrics["balanced_accuracy"],
            }
        )
    return selected_weights, selected_thresholds, result, weight_sensitivity, pd.DataFrame(threshold_rows)


def model_comparison_rows(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        val = result.validation_metrics
        test = result.test_metrics
        rows.append(
            {
                "model": result.model_name,
                "selection_metric": "validation_macro_f1",
                "validation_macro_f1": val["macro_f1"],
                "validation_balanced_accuracy": val["balanced_accuracy"],
                "validation_severe_recall": val["per_class"]["Severe"]["recall"],
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_precision": test["macro_precision"],
                "test_macro_recall": test["macro_recall"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_precision": test["weighted_precision"],
                "test_weighted_recall": test["weighted_recall"],
                "test_weighted_f1": test["weighted_f1"],
                "test_roc_auc_macro_ovr": test["roc_auc_macro_ovr"],
                "test_roc_auc_weighted_ovr": test["roc_auc_weighted_ovr"],
                "test_pr_auc_macro_ovr": test["pr_auc_macro_ovr"],
                "test_severe_recall": test["per_class"]["Severe"]["recall"],
            }
        )
    return pd.DataFrame(rows).sort_values(["validation_macro_f1", "validation_balanced_accuracy"], ascending=False)


def per_class_rows(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for label in CLASS_ORDER:
            rows.append({"model": result.model_name, "class": label, **result.test_metrics["per_class"][label]})
    return pd.DataFrame(rows)


def classification_report_rows(selected: ModelResult) -> pd.DataFrame:
    rows = []
    for label, metrics in selected.test_metrics["per_class"].items():
        rows.append({"class": label, **metrics})
    for name in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        rows.append({"class": name, "precision": "", "recall": "", "f1": selected.test_metrics[name], "support": int(sum(v["support"] for v in selected.test_metrics["per_class"].values()))})
    return pd.DataFrame(rows)


def train_learned_models(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> list[ModelResult]:
    results = []
    for name, candidates in candidate_models().items():
        results.append(fit_model_family(name, candidates, train, validation, test))
    return results


def select_learned_model(results: list[ModelResult]) -> ModelResult:
    learned = [result for result in results if result.model_name != "DummyClassifier"]
    return sorted(
        learned,
        key=lambda result: (
            result.validation_metrics["macro_f1"],
            result.validation_metrics["balanced_accuracy"],
            result.validation_metrics["per_class"]["Severe"]["recall"],
        ),
        reverse=True,
    )[0]


def evaluate_modality_combinations(validation: pd.DataFrame, test: pd.DataFrame, weighted_result: ModelResult, selected_model: ModelResult) -> pd.DataFrame:
    rows = []

    def combo_metrics(split: pd.DataFrame, modalities: list[str], weights: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
        subset_weights = {key: weights[key] for key in modalities}
        total = sum(subset_weights.values())
        subset_weights = {key: value / total for key, value in subset_weights.items()}
        model = WeightedLateFusion(subset_weights, thresholds)
        comparison_frame = split[modalities].copy()
        medians = validation[modalities].median(numeric_only=True)
        comparison_frame = comparison_frame.fillna(medians)
        pred = model.predict(comparison_frame)
        proba = model.predict_proba(comparison_frame)
        return evaluate_predictions(split[TARGET], pred, proba, bootstrap_samples=20)

    weights = dict(weighted_result.hyperparameters["weights"])
    thresholds = dict(weighted_result.hyperparameters["thresholds"])
    for modality in MODALITIES:
        val = combo_metrics(validation, [modality], weights, thresholds)
        test_m = combo_metrics(test, [modality], weights, thresholds)
        rows.append({"comparison": modality, "modalities": modality, "selection_basis": "single modality", "validation_macro_f1": val["macro_f1"], "test_macro_f1": test_m["macro_f1"], "test_balanced_accuracy": test_m["balanced_accuracy"], "test_severe_recall": test_m["per_class"]["Severe"]["recall"]})

    for size, label in [(2, "best two-modality combination"), (3, "best three-modality combination")]:
        candidates = []
        for combo in combinations(MODALITIES, size):
            val = combo_metrics(validation, list(combo), weights, thresholds)
            candidates.append((val["macro_f1"], val["balanced_accuracy"], list(combo), val))
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        _, _, best_combo, val = candidates[0]
        test_m = combo_metrics(test, best_combo, weights, thresholds)
        rows.append({"comparison": label, "modalities": ",".join(best_combo), "selection_basis": "validation macro F1", "validation_macro_f1": val["macro_f1"], "test_macro_f1": test_m["macro_f1"], "test_balanced_accuracy": test_m["balanced_accuracy"], "test_severe_recall": test_m["per_class"]["Severe"]["recall"]})

    rows.append({"comparison": "all-modality fixed weighted fusion", "modalities": ",".join(MODALITIES), "selection_basis": "validation-selected weighted rule", "validation_macro_f1": weighted_result.validation_metrics["macro_f1"], "test_macro_f1": weighted_result.test_metrics["macro_f1"], "test_balanced_accuracy": weighted_result.test_metrics["balanced_accuracy"], "test_severe_recall": weighted_result.test_metrics["per_class"]["Severe"]["recall"]})
    rows.append({"comparison": "selected learned fusion model", "modalities": ",".join(MODALITIES), "selection_basis": "validation-selected learned model", "validation_macro_f1": selected_model.validation_metrics["macro_f1"], "test_macro_f1": selected_model.test_metrics["macro_f1"], "test_balanced_accuracy": selected_model.test_metrics["balanced_accuracy"], "test_severe_recall": selected_model.test_metrics["per_class"]["Severe"]["recall"]})
    return pd.DataFrame(rows)


def apply_missing_scenario(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = frame.copy()
    mapping = {
        "All modalities available": [],
        "Profile missing": ["profile_score"],
        "DASS missing": ["dass_score"],
        "Mood missing": ["mood_score"],
        "Text missing": ["text_score"],
        "Speech missing": ["speech_score"],
        "Face missing": ["face_score"],
        "Text and speech missing": ["text_score", "speech_score"],
        "Speech and face missing": ["speech_score", "face_score"],
        "Only profile, DASS, and mood available": ["text_score", "speech_score", "face_score"],
    }
    if scenario in mapping:
        for column in mapping[scenario]:
            out[column] = np.nan
    elif scenario == "Random one-modality dropout":
        rng = np.random.default_rng(SEED)
        for idx in out.index:
            out.loc[idx, rng.choice(MODALITIES)] = np.nan
    elif scenario == "Random two-modality dropout":
        rng = np.random.default_rng(SEED + 1)
        for idx in out.index:
            for column in rng.choice(MODALITIES, size=2, replace=False):
                out.loc[idx, column] = np.nan
    else:
        raise ValueError(scenario)
    return out


def missing_modality_robustness(test: pd.DataFrame, weighted: ModelResult, selected: ModelResult) -> pd.DataFrame:
    scenarios = [
        "All modalities available",
        "Profile missing",
        "DASS missing",
        "Mood missing",
        "Text missing",
        "Speech missing",
        "Face missing",
        "Text and speech missing",
        "Speech and face missing",
        "Only profile, DASS, and mood available",
        "Random one-modality dropout",
        "Random two-modality dropout",
    ]
    rows = []
    base_macro: dict[str, float] = {}
    for scenario in scenarios:
        scenario_frame = apply_missing_scenario(test, scenario)
        for model_name, evaluator in [
            ("Fixed weighted late fusion", weighted.estimator),
            (selected.model_name, selected.estimator),
        ]:
            pred = evaluator.predict(scenario_frame[MODALITIES])
            proba = probabilities_for_estimator(evaluator, scenario_frame[MODALITIES])
            metrics = evaluate_predictions(test[TARGET], pred, proba, bootstrap_samples=20)
            key = model_name
            if scenario == "All modalities available":
                base_macro[key] = metrics["macro_f1"]
            rows.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "severe_recall": metrics["per_class"]["Severe"]["recall"],
                    "macro_f1_reduction": base_macro.get(key, metrics["macro_f1"]) - metrics["macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def run_test_sensitivity(test: pd.DataFrame, selected_weights: dict[str, float], weight_sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in weight_sensitivity.iterrows():
        weights = json.loads(row["weights"])
        thresholds = json.loads(row["thresholds"])
        pred, proba = evaluate_weighted(test, weights, thresholds)
        metrics = evaluate_predictions(test[TARGET], pred, proba, bootstrap_samples=20)
        rows.append(
            {
                **row.to_dict(),
                "test_macro_f1_after_finalization": metrics["macro_f1"],
                "test_balanced_accuracy_after_finalization": metrics["balanced_accuracy"],
                "test_severe_recall_after_finalization": metrics["per_class"]["Severe"]["recall"],
                "selected_weights": weights == selected_weights,
            }
        )
    return pd.DataFrame(rows)


def error_analysis(test: pd.DataFrame, weighted: ModelResult, selected: ModelResult) -> tuple[pd.DataFrame, str]:
    rows = []
    pattern_counts: dict[str, int] = {}
    for model_name, evaluator in [("Fixed weighted late fusion", weighted.estimator), (selected.model_name, selected.estimator)]:
        pred = evaluator.predict(test[MODALITIES])
        for row, predicted in zip(test.to_dict("records"), pred):
            if row[TARGET] == predicted:
                continue
            pattern = "other"
            true_idx = CLASS_ORDER.index(row[TARGET])
            pred_idx = CLASS_ORDER.index(predicted)
            if {row[TARGET], predicted} == {"Low", "Moderate"}:
                pattern = "Low/Moderate confusion"
            elif {row[TARGET], predicted} == {"Moderate", "High"}:
                pattern = "Moderate/High confusion"
            elif {row[TARGET], predicted} == {"High", "Severe"}:
                pattern = "High/Severe confusion"
            elif predicted == "Severe" and row[TARGET] != "Severe":
                pattern = "false Severe prediction"
            elif row[TARGET] == "Severe" and predicted != "Severe":
                pattern = "missed Severe case"
            if pd.isna(row["speech_score"]) or pd.isna(row["face_score"]):
                pattern += " with missing modality"
            if row["scenario_type"] in {"low_profile_high_text", "high_dass_neutral_face", "mood_deterioration_missing_speech"}:
                pattern += " and contradictory evidence"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            rows.append(
                {
                    "model": model_name,
                    "synthetic_participant_id": row["synthetic_participant_id"],
                    "scenario_type": row["scenario_type"],
                    "true_class": row[TARGET],
                    "predicted_class": predicted,
                    "error_distance": pred_idx - true_idx,
                    "error_pattern": pattern,
                    **{modality: row[modality] for modality in MODALITIES},
                }
            )
    examples = pd.DataFrame(rows).head(8)
    lines = [
        "# Fusion Error Patterns",
        "",
        "Errors were reviewed for the fixed weighted rule and the selected learned model on the frozen synthetic test split.",
        "",
        "## Pattern Counts",
    ]
    for pattern, count in sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {pattern}: {count}")
    lines.extend(["", "## Example Synthetic Cases"])
    for _, row in examples.iterrows():
        scores = ", ".join(f"{m}={row[m]:.3f}" if pd.notna(row[m]) else f"{m}=missing" for m in MODALITIES)
        lines.append(f"- {row['model']}: {row['true_class']} predicted as {row['predicted_class']} ({row['scenario_type']}); {scores}.")
    lines.append("")
    lines.append("These are anonymized synthetic participant IDs, not real students or clinical cases.")
    return pd.DataFrame(rows), "\n".join(lines) + "\n"


def write_dataset_artifacts(df: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, manifest: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "fusion_dataset.csv", index=False)
    train.to_csv(DATA_DIR / "fusion_train.csv", index=False)
    validation.to_csv(DATA_DIR / "fusion_validation.csv", index=False)
    test.to_csv(DATA_DIR / "fusion_test.csv", index=False)
    write_json(MANIFEST_PATH, manifest)
    summary = {
        "dataset_version": "controlled_fusion_v2",
        "row_count": int(len(df)),
        "modality_count": len(MODALITIES),
        "modalities": MODALITIES,
        "behavioral_score": "excluded; no valid implemented evidence beyond synthetic engineering readiness",
        "score_range": "[0, 1]",
        "generation_seed": SEED,
        "target_generation": "Hidden latent psychological-risk variable with modality-specific signal, nonlinear effects, interactions, and noise.",
        "not_clinical_validation": True,
    }
    write_json(REPORT_DIR / "fusion_dataset_summary.json", summary)
    class_distribution = (
        pd.concat([train.assign(split="train"), validation.assign(split="validation"), test.assign(split="test")])
        .groupby(["split", TARGET], observed=False)
        .size()
        .reset_index(name="count")
    )
    class_distribution.to_csv(REPORT_DIR / "fusion_class_distribution.csv", index=False)
    df[MODALITIES + [TARGET_SCORE]].corr().to_csv(REPORT_DIR / "fusion_correlation_matrix.csv")

    generation_md = f"""# Controlled Synthetic Fusion Dataset v2

This dataset contains `{len(df)}` synthetic participants generated with seed `{SEED}`. Scores are normalized to `[0, 1]`.

The original profile, text, speech, and face datasets are independently collected and not participant-aligned. No source-dataset rows were merged. The target is a synthetic common fusion target produced from a hidden latent psychological-risk process plus interaction effects, nonlinear effects, and noise.

Behavioral score was excluded because the repository contains only synthetic behavioral engineering/readiness evidence, not a validated implemented behavioral modality.

Contradictory scenario types are included: low profile with high text evidence, high DASS with neutral facial evidence, mood deterioration with missing speech, and moderate evidence across several modalities.

The class thresholds used by the generator are Low `<0.30`, Moderate `0.30-0.52`, High `0.52-0.72`, and Severe `>=0.72`. These are synthetic research labels, not clinical ground truth.
"""
    (REPORT_DIR / "fusion_dataset_generation.md").write_text(generation_md, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = class_distribution.pivot(index=TARGET, columns="split", values="count").reindex(CLASS_ORDER)
    ordered.plot(kind="bar", ax=ax, color=["#457b9d", "#e9c46a", "#2a9d8f"])
    ax.set_xlabel("Synthetic target risk class")
    ax.set_ylabel("Count")
    ax.set_title("Controlled Fusion Class Distribution")
    save_plot(fig, REPORT_DIR / "fusion_class_distribution.png")

    corr = df[MODALITIES + [TARGET_SCORE]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="vlag" if "vlag" in plt.colormaps() else "coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(im, ax=ax)
    ax.set_title("Fusion Modality Correlation Heatmap")
    save_plot(fig, REPORT_DIR / "fusion_modality_correlation_heatmap.png")


def write_repository_audit() -> None:
    artifacts = [
        ("generated/reports/profile_baseline/v2/profile_metrics_test.json", "used", "latest profile v2 held-out metrics; self-reported depression target only"),
        ("generated/reports/text_baseline/v1/text_metrics_test.json", "used", "latest text baseline probabilities/metrics inform text signal strength"),
        ("generated/reports/speech_baseline/v2/speech_metrics_test.json", "used", "latest validated speech v2 acted-emotion evidence"),
        ("generated/reports/face_baseline/v1/face_metrics_test.json", "used", "latest face baseline metrics; weak supporting evidence"),
        ("generated/reports/dass21/scoring_validation_v1.json", "used", "deterministic questionnaire scoring evidence"),
        ("generated/preprocessing/mood/v1/mood_readiness_report.json", "used if present", "mood is deterministic/trend contextual score, not trained classifier"),
        ("generated/preprocessing/behavioral/v1/behavioral_readiness_report.json", "excluded", "behavioral evidence is engineering/synthetic only"),
        ("backend/app/ml/fusion/synthetic.py", "inspected", "existing v1 synthetic generator uses 0-100 scores"),
        ("backend/app/ml/fusion/evaluation.py", "inspected", "existing v1 evaluation incomplete for requested v2 artifact names"),
        ("backend/scripts/run_synthetic_fusion_evaluation.py", "inspected", "v1 runner retained, not overwritten"),
        ("ml-research/configs/fusion.synthetic_late_fusion.v1.json", "inspected", "prior fusion configuration retained as historical evidence"),
        ("backend/app/models/weights.csv", "inspected", "runtime weighted sum exists but includes behavioral and no missing renormalization"),
        ("backend/app/models/risk_assessment.py", "inspected", "runtime rule endpoint not a trained real fusion model"),
        ("docs/thesis/chapter4_metrics_summary.csv", "modified", "versioned v2 rows appended"),
        ("docs/thesis/chapter4_evidence_inventory.csv", "modified", "versioned v2 evidence entries appended"),
        ("generated/reports/fusion_evaluation/synthetic_fusion_v1_seed20260716_r2/metrics.json", "inspected", "prior synthetic v1 evidence preserved but superseded by controlled v2 for Chapter 4"),
    ]
    lines = [
        "# Fusion Repository Audit v2",
        "",
        "The audit confirms that original modality datasets are not participant-aligned. No source rows were merged as real participants.",
        "",
        "| Artifact | Decision | Rationale | Exists |",
        "| --- | --- | --- | --- |",
    ]
    for rel, decision, rationale in artifacts:
        exists = (REPO_ROOT / rel).exists()
        lines.append(f"| `{rel}` | {decision} | {rationale} | {exists} |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- Latest validated profile evidence is profile baseline v2.",
            "- Latest validated speech evidence is speech baseline v2.",
            "- Text and face use their latest available v1 baselines.",
            "- DASS-21 and mood are deterministic/contextual scores, not trained classifiers.",
            "- Behavioral score is excluded from v2 because valid implemented behavioral evidence is not available.",
            "- Existing runtime fusion is a weighted score rule and did not provide the requested controlled v2 evaluation.",
            "- Existing synthetic fusion v1 is historical evidence and was not overwritten.",
        ]
    )
    (REPORT_DIR / "fusion_repository_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(selected: ModelResult, model_comparison: pd.DataFrame, individual: pd.DataFrame, missing: pd.DataFrame, test: pd.DataFrame) -> None:
    cm = np.asarray(selected.test_metrics["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASS_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(4), CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Selected Fusion Model Confusion Matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    save_plot(fig, REPORT_DIR / "fusion_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = model_comparison.sort_values("test_macro_f1")
    ax.barh(plot_df["model"], plot_df["test_macro_f1"], color="#2a9d8f")
    ax.set_xlabel("Frozen test macro F1")
    ax.set_title("Fusion Model Comparison")
    save_plot(fig, REPORT_DIR / "fusion_model_comparison.png")

    prob = probabilities_for_estimator(selected.estimator, test[MODALITIES])
    y_bin = label_binarize(test[TARGET], classes=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], prob[:, idx])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Multiclass ROC Curve")
    ax.legend()
    save_plot(fig, REPORT_DIR / "fusion_roc_curve.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(CLASS_ORDER):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], prob[:, idx])
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curve")
    ax.legend()
    save_plot(fig, REPORT_DIR / "fusion_precision_recall_curve.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = individual.sort_values("test_macro_f1")
    ax.barh(plot["comparison"], plot["test_macro_f1"], color="#457b9d")
    ax.set_xlabel("Frozen test macro F1")
    ax.set_title("Fusion versus Individual Modalities")
    save_plot(fig, REPORT_DIR / "fusion_vs_individual_modalities.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    subset = missing[missing["model"] == selected.model_name]
    ax.plot(subset["scenario"], subset["macro_f1"], marker="o", color="#e76f51", label=selected.model_name)
    ax.tick_params(axis="x", rotation=40)
    ax.set_ylabel("Macro F1")
    ax.set_title("Missing-Modality Robustness")
    ax.legend()
    save_plot(fig, REPORT_DIR / "fusion_missing_modality_robustness.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, "Profile\nDASS\nMood"),
        (0.02, 0.18, "Text\nSpeech\nFace"),
        (0.34, 0.38, "Normalized\nmodality scores\n[0, 1]"),
        (0.62, 0.38, "Late-fusion\nrule or learned\nmodel"),
        (0.84, 0.38, "Risk class\nLow to Severe"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 0.18, 0.22, fill=False, edgecolor="#264653", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)
    for start, end in [((0.20, 0.69), (0.34, 0.49)), ((0.20, 0.29), (0.34, 0.49)), ((0.52, 0.49), (0.62, 0.49)), ((0.80, 0.49), (0.84, 0.49))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("Controlled Late-Fusion Architecture")
    save_plot(fig, REPORT_DIR / "fusion_late_fusion_architecture.png")


def write_summaries(
    selected: ModelResult,
    weighted: ModelResult,
    model_comparison: pd.DataFrame,
    individual: pd.DataFrame,
    missing: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    error_patterns_text: str,
) -> None:
    best = model_comparison.iloc[0]
    fixed = weighted.test_metrics
    learned = selected.test_metrics
    eval_summary = f"""# Fusion Evaluation Summary v2

The controlled v2 experiment used synthetic participant-level data because the source modality datasets are not participant-aligned. Results are technical feasibility evidence only and are not clinical validation.

Selected learned model: `{selected.model_name}`, selected by validation macro F1 without test-set access.

Frozen test macro F1:

- Fixed weighted rule: {fixed['macro_f1']:.4f}
- Selected learned model: {learned['macro_f1']:.4f}
- Best comparison row by validation: {best['model']}

Severe-class recall on frozen test:

- Fixed weighted rule: {fixed['per_class']['Severe']['recall']:.4f}
- Selected learned model: {learned['per_class']['Severe']['recall']:.4f}

The selected learned model improved macro F1 over the fixed rule within this controlled synthetic experiment. No clinical or real-world superiority claim is supported.
"""
    (REPORT_DIR / "fusion_evaluation_summary.md").write_text(eval_summary, encoding="utf-8")
    (REPORT_DIR / "fusion_error_patterns.md").write_text(error_patterns_text, encoding="utf-8")

    thesis = f"""# 4.4 Controlled Multimodal Fusion Evaluation

The multimodal fusion experiment was implemented as a controlled synthetic participant-level feasibility evaluation. The original profile, text, speech, facial, DASS-21, and mood evidence sources were not participant-aligned and did not share a real-world suicide-risk target. For this reason, the experiment does not merge source-dataset rows as if they represented the same individuals. The results demonstrate technical feasibility only and do not constitute clinical validation.

## 4.4.1 Fusion Experimental Configuration

Table 4.7 Modalities Used in the Fusion Framework lists six evaluated modalities: profile_score, dass_score, mood_score, text_score, speech_score, and face_score. Behavioral score was excluded because only synthetic engineering readiness evidence existed. Table 4.8 Fusion Experimental Configuration describes the seed 42 synthetic generator, 70/15/15 stratified split, normalized [0, 1] scores, and class order Low, Moderate, High, Severe.

Figure 4.16 Controlled Late-Fusion Architecture: `{REPORT_DIR / 'fusion_late_fusion_architecture.png'}`

Figure 4.17 Fusion Dataset Class Distribution: `{REPORT_DIR / 'fusion_class_distribution.png'}`

## 4.4.2 Comparison of Fusion Models

Table 4.9 Comparison of Fusion Models is available at `{REPORT_DIR / 'fusion_model_comparison.csv'}`. The selected learned model was `{selected.model_name}` with validation macro F1 {selected.validation_metrics['macro_f1']:.4f}. On the frozen test split it achieved accuracy {learned['accuracy']:.4f}, balanced accuracy {learned['balanced_accuracy']:.4f}, macro F1 {learned['macro_f1']:.4f}, and severe-class recall {learned['per_class']['Severe']['recall']:.4f}.

Figure 4.18 Comparison of Fusion Models: `{REPORT_DIR / 'fusion_model_comparison.png'}`

## 4.4.3 Comparison with the Fixed Weighted Rule

The fixed weighted late-fusion rule used validation-selected research weights and thresholds with missing-modality renormalization. It achieved frozen test macro F1 {fixed['macro_f1']:.4f}, balanced accuracy {fixed['balanced_accuracy']:.4f}, and severe-class recall {fixed['per_class']['Severe']['recall']:.4f}. These thresholds are research classification thresholds, not clinical intervention thresholds.

Table 4.10 Fusion and Individual Modality Comparison is available at `{REPORT_DIR / 'fusion_vs_individual_modalities.csv'}`.

Figure 4.22 Fusion versus Individual Modalities: `{REPORT_DIR / 'fusion_vs_individual_modalities.png'}`

## 4.4.4 Confusion Matrix Analysis

The selected learned model confusion matrix is saved as `{REPORT_DIR / 'fusion_confusion_matrix.png'}`. Per-class metrics are saved at `{REPORT_DIR / 'fusion_per_class_metrics.csv'}`. The most important interpretation is whether severe synthetic cases are missed or confused with adjacent high-risk cases, not overall accuracy alone.

Figure 4.19 Confusion Matrix of the Selected Fusion Model: `{REPORT_DIR / 'fusion_confusion_matrix.png'}`

Figure 4.20 Multiclass ROC Curve: `{REPORT_DIR / 'fusion_roc_curve.png'}`

## 4.4.5 Missing-Modality Robustness

Table 4.11 Missing-Modality Robustness Results is available at `{REPORT_DIR / 'fusion_missing_modality_results.csv'}`. The selected learned model retained macro F1 {missing[(missing['scenario'] == 'All modalities available') & (missing['model'] == selected.model_name)]['macro_f1'].iloc[0]:.4f} with all modalities available. Robustness decreased under larger dropout scenarios, especially when direct text and speech evidence were removed.

Figure 4.21 Missing-Modality Robustness: `{REPORT_DIR / 'fusion_missing_modality_robustness.png'}`

## 4.4.6 Interpretation of Fusion Results

Within the controlled synthetic experiment, multimodal late fusion improved over several single-modality comparisons and provided a reproducible framework for combining modality-level evidence. The result should be interpreted as technical feasibility evidence. It does not prove real-world suicide-risk prediction, clinical validity, intervention effectiveness, or deployment readiness. External and clinical validation require participant-aligned multimodal data collected from the intended undergraduate population with an ethically approved shared outcome definition.
"""
    THESIS_SECTION.parent.mkdir(parents=True, exist_ok=True)
    THESIS_SECTION.write_text(thesis, encoding="utf-8")


def update_config_files(selected_weights: dict[str, float], selected_thresholds: dict[str, float]) -> None:
    weight_payload = {
        "config_version": "2.0.0",
        "experiment": "controlled_synthetic_late_fusion_v2",
        "status": "selected_on_validation_only",
        "research_boundary": "technical feasibility evidence only; not clinical validation",
        "initial_weights": INITIAL_WEIGHTS,
        "selected_weights": selected_weights,
        "selection_policy": "Small interpretable validation-only adjustment around initial research weights.",
        "missing_modality_strategy": "Use only available modality scores and renormalize by the sum of available weights.",
    }
    threshold_payload = {
        "config_version": "2.0.0",
        "experiment": "controlled_synthetic_late_fusion_v2",
        "status": "selected_on_validation_only",
        "research_boundary": "research classification thresholds only; not clinical intervention thresholds",
        "initial_thresholds": INITIAL_THRESHOLDS,
        "selected_thresholds": selected_thresholds,
        "selection_policy": "Nearby thresholds compared on validation macro F1 and balanced accuracy only.",
    }
    write_json(WEIGHT_CONFIG_PATH, weight_payload)
    write_json(THRESHOLD_CONFIG_PATH, threshold_payload)


def package_model(selected: ModelResult, selected_weights: dict[str, float], selected_thresholds: dict[str, float]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected.estimator, MODEL_DIR / "model.joblib")
    write_json(MODEL_DIR / "fusion_weights.json", {"weights": selected_weights})
    write_json(MODEL_DIR / "fusion_thresholds.json", {"thresholds": selected_thresholds})
    write_json(
        MODEL_DIR / "feature_schema.json",
        {
            "modality_order": MODALITIES,
            "feature_columns": FEATURE_COLUMNS,
            "class_order": CLASS_ORDER,
            "score_range": [0, 1],
            "missing_modality_strategy": "median imputation plus availability indicators for learned model; weight renormalization for fixed rule",
        },
    )
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    metadata = {
        "model_name": selected.model_name,
        "version": "2.0.0",
        "random_seed": SEED,
        "selection_metric": "validation_macro_f1",
        "test_set_used_for_selection": False,
        "dataset_fingerprint": sha256_file(DATA_DIR / "fusion_dataset.csv"),
        "split_manifest_hash": sha256_file(MANIFEST_PATH),
        "code_version": commit or "unavailable",
        "metric_definitions": ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "one-vs-rest ROC-AUC", "one-vs-rest PR-AUC"],
        "training_configuration": "Scikit-learn pipeline using modality-level scores and availability indicators only.",
    }
    write_json(MODEL_DIR / "metadata.json", metadata)
    (MODEL_DIR / "README.md").write_text(
        "# Controlled Late-Fusion v2\n\nResearch-only synthetic technical feasibility model. Not clinical validation, not deployment ready, and not a suicide-risk diagnostic system.\n",
        encoding="utf-8",
    )


def update_thesis_csvs(weighted: ModelResult, selected: ModelResult, missing: pd.DataFrame) -> None:
    metrics_rows = [
        {
            "Modality": "Fusion v2",
            "Dataset": f"controlled synthetic participant-level fusion dataset v2; {N_SYNTHETIC} rows",
            "Selected Model/Method": "Fixed weighted fusion",
            "Accuracy": weighted.test_metrics["accuracy"],
            "Precision": weighted.test_metrics["macro_precision"],
            "Recall": weighted.test_metrics["macro_recall"],
            "F1": weighted.test_metrics["macro_f1"],
            "ROC-AUC": weighted.test_metrics["roc_auc_macro_ovr"],
            "Artifact Path": str(MODEL_DIR / "fusion_weights.json"),
            "Evaluation Evidence": str(REPORT_DIR / "fusion_metrics_test.json"),
            "Status": "Controlled synthetic technical feasibility only",
        },
        {
            "Modality": "Fusion v2",
            "Dataset": "controlled synthetic participant-level fusion dataset v2; frozen test split",
            "Selected Model/Method": selected.model_name,
            "Accuracy": selected.test_metrics["accuracy"],
            "Precision": selected.test_metrics["macro_precision"],
            "Recall": selected.test_metrics["macro_recall"],
            "F1": selected.test_metrics["macro_f1"],
            "ROC-AUC": selected.test_metrics["roc_auc_macro_ovr"],
            "Artifact Path": str(MODEL_DIR / "model.joblib"),
            "Evaluation Evidence": str(REPORT_DIR / "fusion_model_comparison.csv"),
            "Status": "Controlled synthetic technical feasibility only",
        },
        {
            "Modality": "Fusion v2 robustness",
            "Dataset": "controlled synthetic participant-level fusion dataset v2; frozen test split",
            "Selected Model/Method": "Missing-modality robustness summary",
            "Accuracy": "",
            "Precision": "",
            "Recall": "",
            "F1": missing["macro_f1"].mean(),
            "ROC-AUC": "",
            "Artifact Path": str(REPORT_DIR / "fusion_missing_modality_results.csv"),
            "Evaluation Evidence": str(REPORT_DIR / "fusion_missing_modality_robustness.png"),
            "Status": "Controlled synthetic robustness analysis only",
        },
    ]
    existing = pd.read_csv(THESIS_METRICS)
    existing = existing[~existing["Selected Model/Method"].isin([row["Selected Model/Method"] for row in metrics_rows])]
    pd.concat([existing, pd.DataFrame(metrics_rows)], ignore_index=True).to_csv(THESIS_METRICS, index=False)

    inventory_rows = [
        {
            "Area": "Fusion",
            "Feature or Artifact": "Controlled synthetic fusion v2",
            "Status": "Complete research artifact",
            "Evidence Path": str(REPORT_DIR / "fusion_evaluation_summary.md"),
            "Key Finding": f"Selected learned model {selected.model_name} test macro F1 {selected.test_metrics['macro_f1']:.4f}; technical feasibility only",
            "Thesis Use": "Use in Section 4.4 with explicit no-clinical-validation boundary",
        },
        {
            "Area": "Fusion",
            "Feature or Artifact": "Fusion v2 missing-modality robustness",
            "Status": "Complete research artifact",
            "Evidence Path": str(REPORT_DIR / "fusion_missing_modality_results.csv"),
            "Key Finding": "Robustness evaluated across 12 incomplete-input scenarios",
            "Thesis Use": "Use in Table 4.11 and Figure 4.21",
        },
    ]
    inventory = pd.read_csv(THESIS_INVENTORY)
    inventory = inventory[~inventory["Feature or Artifact"].isin([row["Feature or Artifact"] for row in inventory_rows])]
    pd.concat([inventory, pd.DataFrame(inventory_rows)], ignore_index=True).to_csv(THESIS_INVENTORY, index=False)


def write_hashes() -> None:
    roots = [DATA_DIR, REPORT_DIR, MODEL_DIR]
    hashes = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    hashes[str(MANIFEST_PATH.relative_to(REPO_ROOT))] = sha256_file(MANIFEST_PATH)
    for path in [
        WEIGHT_CONFIG_PATH,
        THRESHOLD_CONFIG_PATH,
        BACKEND_ROOT / "app" / "ml" / "fusion" / "weighted_late_fusion.py",
        BACKEND_ROOT / "scripts" / "run_controlled_fusion_v2.py",
        BACKEND_ROOT / "tests" / "test_fusion_controlled_v2.py",
        THESIS_SECTION,
        THESIS_METRICS,
        THESIS_INVENTORY,
    ]:
        if path.exists():
            hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    write_json(REPORT_DIR / "fusion_artifact_hashes.json", hashes)


def write_final_artifacts() -> dict[str, Any]:
    df = generate_controlled_dataset()
    train, validation, test, manifest = split_dataset(df)
    write_dataset_artifacts(df, train, validation, test, manifest)
    write_repository_audit()

    selected_weights, selected_thresholds, weighted_result, weight_sens_val, threshold_sens = select_weighted_rule(train, validation, test)
    update_config_files(selected_weights, selected_thresholds)

    learned_results = train_learned_models(train, validation, test)
    selected = select_learned_model(learned_results)
    all_results = [weighted_result] + learned_results
    comparison = model_comparison_rows(all_results)
    per_class = per_class_rows(all_results)
    classification = classification_report_rows(selected)
    individual = evaluate_modality_combinations(validation, test, weighted_result, selected)
    missing = missing_modality_robustness(test, weighted_result, selected)
    weight_sens = run_test_sensitivity(test, selected_weights, weight_sens_val)
    errors, error_text = error_analysis(test, weighted_result, selected)

    comparison.to_csv(REPORT_DIR / "fusion_model_comparison.csv", index=False)
    per_class.to_csv(REPORT_DIR / "fusion_per_class_metrics.csv", index=False)
    classification.to_csv(REPORT_DIR / "fusion_classification_report.csv", index=False)
    individual.to_csv(REPORT_DIR / "fusion_vs_individual_modalities.csv", index=False)
    missing.to_csv(REPORT_DIR / "fusion_missing_modality_results.csv", index=False)
    weight_sens.to_csv(REPORT_DIR / "fusion_weight_sensitivity.csv", index=False)
    threshold_sens.to_csv(REPORT_DIR / "fusion_threshold_sensitivity.csv", index=False)
    errors.to_csv(REPORT_DIR / "fusion_error_analysis.csv", index=False)

    metrics_payload = {
        "research_boundary": "controlled synthetic technical feasibility evidence only",
        "test_set_used_for_selection": False,
        "class_order": CLASS_ORDER,
        "selected_learned_model": selected.model_name,
        "fixed_weighted_rule": weighted_result.test_metrics,
        "selected_learned_model_metrics": selected.test_metrics,
        "all_models": {result.model_name: {"validation": result.validation_metrics, "test": result.test_metrics} for result in all_results},
    }
    write_json(REPORT_DIR / "fusion_metrics_test.json", metrics_payload)

    write_figures(selected, comparison, individual, missing, test)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    short = weight_sens.sort_values("validation_macro_f1", ascending=False).head(12).reset_index(drop=True)
    ax.plot(range(len(short)), short["validation_macro_f1"], marker="o", label="validation")
    ax.plot(range(len(short)), short["test_macro_f1_after_finalization"], marker="s", label="test after finalization")
    ax.set_xlabel("Weight/threshold variant rank")
    ax.set_ylabel("Macro F1")
    ax.set_title("Weight Sensitivity")
    ax.legend()
    save_plot(fig, REPORT_DIR / "fusion_weight_sensitivity.png")

    package_model(selected, selected_weights, selected_thresholds)
    write_summaries(selected, weighted_result, comparison, individual, missing, weight_sens, threshold_sens, error_text)
    update_thesis_csvs(weighted_result, selected, missing)
    write_hashes()
    return {
        "selected_model": selected.model_name,
        "weighted_macro_f1": weighted_result.test_metrics["macro_f1"],
        "selected_macro_f1": selected.test_metrics["macro_f1"],
        "train_distribution": train[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict(),
        "validation_distribution": validation[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict(),
        "test_distribution": test[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict(),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = write_final_artifacts()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
