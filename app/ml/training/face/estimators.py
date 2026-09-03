"""Bounded classical estimators for the Phase 3I Face baseline."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from app.ml.training.face.schemas import FaceCandidateSpec


def logistic_regression_candidate_specs(config: Mapping[str, Any] | None = None) -> list[FaceCandidateSpec]:
    payload = dict(config or {})
    specs = []
    for c_value, class_weight in product(payload.get("C", [0.1]), payload.get("class_weight", [None])):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "solver": "saga",
            "multi_class": "auto",
            "max_iter": int(payload.get("max_iter", 150)),
            "random_state": int(payload.get("random_state", 43107)),
            "n_jobs": 1,
        }
        specs.append(FaceCandidateSpec("logistic_regression", f"logistic_regression_C={c_value}_class_weight={class_weight or 'none'}", params))
    return specs


def linear_svm_candidate_specs(config: Mapping[str, Any] | None = None) -> list[FaceCandidateSpec]:
    payload = dict(config or {})
    specs = []
    for c_value, class_weight in product(payload.get("C", [0.1]), payload.get("class_weight", [None])):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "max_iter": int(payload.get("max_iter", 1000)),
            "random_state": int(payload.get("random_state", 43107)),
        }
        specs.append(FaceCandidateSpec("linear_svm", f"linear_svm_C={c_value}_class_weight={class_weight or 'none'}", params))
    return specs


def random_forest_candidate_specs(config: Mapping[str, Any] | None = None) -> list[FaceCandidateSpec]:
    payload = dict(config or {})
    specs = []
    for n_estimators, max_depth, class_weight in product(
        payload.get("n_estimators", [50]),
        payload.get("max_depth", [8]),
        payload.get("class_weight", [None]),
    ):
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(payload.get("min_samples_leaf", 2)),
            "class_weight": class_weight,
            "random_state": int(payload.get("random_state", 43107)),
            "n_jobs": int(payload.get("n_jobs", 1)),
        }
        specs.append(
            FaceCandidateSpec(
                "random_forest",
                f"random_forest_n={n_estimators}_depth={max_depth or 'none'}_class_weight={class_weight or 'none'}",
                params,
                scale_features=False,
            )
        )
    return specs


def face_candidate_specs(config: Mapping[str, Any], *, candidate: str = "all", feature_set: str = "flattened_pixels") -> list[FaceCandidateSpec]:
    hyper = dict(config.get("hyperparameter_search") or {})
    selected = candidate or "all"
    specs: list[FaceCandidateSpec] = []
    if selected in ("all", "logistic_regression"):
        specs.extend(logistic_regression_candidate_specs(hyper.get("logistic_regression")))
    if selected in ("all", "linear_svm"):
        specs.extend(linear_svm_candidate_specs(hyper.get("linear_svm")))
    if selected in ("all", "random_forest"):
        specs.extend(random_forest_candidate_specs(hyper.get("random_forest")))
    if not specs:
        raise ValueError(f"No Face candidate specs generated for candidate={candidate}")
    max_count = int(config.get("max_candidate_count", 12))
    if len(specs) > max_count:
        raise ValueError("bounded Face candidate search exceeded max_candidate_count")
    return [FaceCandidateSpec(spec.estimator_type, spec.name, spec.hyperparameters, feature_set, spec.scale_features) for spec in specs]


def create_face_estimator(spec: FaceCandidateSpec):
    if spec.estimator_type == "logistic_regression":
        return LogisticRegression(**spec.hyperparameters)
    if spec.estimator_type == "linear_svm":
        return LinearSVC(**spec.hyperparameters)
    if spec.estimator_type == "random_forest":
        return RandomForestClassifier(**spec.hyperparameters)
    raise ValueError(f"Unsupported Face estimator type: {spec.estimator_type}")

