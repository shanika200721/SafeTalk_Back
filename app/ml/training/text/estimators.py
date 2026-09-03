"""Candidate estimators for Text TF-IDF baselines."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from app.ml.training.text.schemas import CandidateSpec


def logistic_regression_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    c_values = payload.get("C", [0.5, 1.0, 2.0])
    class_weights = payload.get("class_weight", [None, "balanced"])
    specs: list[CandidateSpec] = []
    for c_value, class_weight in product(c_values, class_weights):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "solver": str(payload.get("solver", "liblinear")),
            "max_iter": int(payload.get("max_iter", 300)),
            "random_state": int(payload.get("random_state", 42)),
        }
        name = f"logistic_regression_C={params['C']}_class_weight={class_weight or 'none'}"
        specs.append(CandidateSpec("logistic_regression", name, params))
    return specs


def linear_svm_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    c_values = payload.get("C", [0.5, 1.0, 2.0])
    class_weights = payload.get("class_weight", [None, "balanced"])
    specs: list[CandidateSpec] = []
    for c_value, class_weight in product(c_values, class_weights):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "max_iter": int(payload.get("max_iter", 3000)),
            "random_state": int(payload.get("random_state", 42)),
        }
        name = f"linear_svm_C={params['C']}_class_weight={class_weight or 'none'}"
        specs.append(CandidateSpec("linear_svm", name, params))
    return specs


def create_text_estimator(spec: CandidateSpec):
    if spec.estimator_type == "logistic_regression":
        return LogisticRegression(**spec.hyperparameters)
    if spec.estimator_type == "linear_svm":
        return LinearSVC(**spec.hyperparameters)
    raise ValueError(f"Unsupported Text estimator type: {spec.estimator_type}")


def text_candidate_specs(config: Mapping[str, Any], *, candidate: str = "all") -> list[CandidateSpec]:
    hyper = dict(config.get("hyperparameter_search") or {})
    selected = candidate or "all"
    specs: list[CandidateSpec] = []
    if selected in ("all", "logistic_regression") and "logistic_regression" in hyper:
        specs.extend(logistic_regression_candidate_specs(hyper.get("logistic_regression")))
    if selected in ("all", "linear_svm") and "linear_svm" in hyper:
        specs.extend(linear_svm_candidate_specs(hyper.get("linear_svm")))
    if selected == "logistic_regression" and not specs:
        specs.extend(logistic_regression_candidate_specs())
    if selected == "linear_svm" and not specs:
        specs.extend(linear_svm_candidate_specs())
    if not specs:
        specs.extend(logistic_regression_candidate_specs(hyper.get("logistic_regression")))
        specs.extend(linear_svm_candidate_specs(hyper.get("linear_svm")))
    if len(specs) > int(config.get("max_candidate_count", 24)):
        raise ValueError("bounded Text candidate search exceeded max_candidate_count")
    return specs

