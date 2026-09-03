"""Bounded candidate estimators for Speech emotion classification."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from app.ml.training.speech.schemas import CandidateSpec


def logistic_regression_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    specs: list[CandidateSpec] = []
    for c_value, class_weight in product(payload.get("C", [0.1, 1.0, 10.0]), payload.get("class_weight", [None, "balanced"])):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "solver": str(payload.get("solver", "lbfgs")),
            "multi_class": str(payload.get("multi_class", "multinomial")),
            "max_iter": int(payload.get("max_iter", 300)),
            "random_state": int(payload.get("random_state", 42)),
        }
        specs.append(CandidateSpec("logistic_regression", f"logistic_regression_C={params['C']}_class_weight={class_weight or 'none'}", params))
    return specs


def random_forest_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    specs: list[CandidateSpec] = []
    for n_value, depth, leaf, class_weight in product(
        payload.get("n_estimators", [100]),
        payload.get("max_depth", [8, 12]),
        payload.get("min_samples_leaf", [1, 3]),
        payload.get("class_weight", [None, "balanced"]),
    ):
        params = {
            "n_estimators": int(n_value),
            "max_depth": None if depth is None else int(depth),
            "min_samples_leaf": int(leaf),
            "class_weight": class_weight,
            "random_state": int(payload.get("random_state", 42)),
            "n_jobs": int(payload.get("n_jobs", 1)),
        }
        name = f"random_forest_n={params['n_estimators']}_depth={params['max_depth'] or 'none'}_leaf={params['min_samples_leaf']}_class_weight={class_weight or 'none'}"
        specs.append(CandidateSpec("random_forest", name, params))
    return specs


def svm_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    specs: list[CandidateSpec] = []
    kernels = payload.get("kernel", ["linear"])
    for kernel in kernels:
        if kernel == "linear":
            for c_value, class_weight in product(payload.get("C", [0.1, 1.0]), payload.get("class_weight", [None, "balanced"])):
                params = {
                    "C": float(c_value),
                    "class_weight": class_weight,
                    "max_iter": int(payload.get("max_iter", 3000)),
                    "random_state": int(payload.get("random_state", 42)),
                }
                specs.append(CandidateSpec("linear_svm", f"linear_svm_C={params['C']}_class_weight={class_weight or 'none'}", params))
        elif kernel == "rbf":
            for c_value, gamma, class_weight in product(payload.get("C", [1.0]), payload.get("gamma", ["scale"]), payload.get("class_weight", [None])):
                params = {
                    "C": float(c_value),
                    "gamma": gamma,
                    "kernel": "rbf",
                    "class_weight": class_weight,
                    "probability": bool(payload.get("probability", False)),
                    "max_iter": int(payload.get("max_iter", 2000)),
                    "random_state": int(payload.get("random_state", 42)),
                }
                specs.append(CandidateSpec("rbf_svm", f"rbf_svm_C={params['C']}_gamma={gamma}_class_weight={class_weight or 'none'}", params))
        else:
            raise ValueError(f"unsupported Speech SVM kernel: {kernel}")
    return specs


def create_speech_estimator(spec: CandidateSpec):
    if spec.estimator_type == "logistic_regression":
        return LogisticRegression(**spec.hyperparameters)
    if spec.estimator_type == "random_forest":
        return RandomForestClassifier(**spec.hyperparameters)
    if spec.estimator_type == "linear_svm":
        return LinearSVC(**spec.hyperparameters)
    if spec.estimator_type == "rbf_svm":
        return SVC(**spec.hyperparameters)
    raise ValueError(f"Unsupported Speech estimator type: {spec.estimator_type}")


def speech_candidate_specs(config: Mapping[str, Any], *, candidate: str = "all") -> list[CandidateSpec]:
    hyper = dict(config.get("hyperparameter_search") or {})
    selected = candidate or "all"
    specs: list[CandidateSpec] = []
    if selected in ("all", "logistic_regression"):
        specs.extend(logistic_regression_candidate_specs(hyper.get("logistic_regression")))
    if selected in ("all", "random_forest"):
        specs.extend(random_forest_candidate_specs(hyper.get("random_forest")))
    if selected in ("all", "svm", "linear_svm", "rbf_svm"):
        specs.extend(svm_candidate_specs(hyper.get("svm")))
    if not specs:
        raise ValueError(f"No Speech candidate specs generated for candidate={candidate}")
    if len(specs) > int(config.get("max_candidate_count", 64)):
        raise ValueError("bounded Speech candidate search exceeded max_candidate_count")
    return specs

