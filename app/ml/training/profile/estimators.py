"""Candidate estimators for the Profile depression baseline."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.ml.training.profile.schemas import CandidateSpec


def logistic_regression_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    c_values = payload.get("C", [0.1, 1.0, 10.0])
    class_weights = payload.get("class_weight", [None, "balanced"])
    max_iter = int(payload.get("max_iter", 500))
    specs: list[CandidateSpec] = []
    for c_value, class_weight in product(c_values, class_weights):
        params = {
            "C": float(c_value),
            "class_weight": class_weight,
            "solver": "liblinear",
            "max_iter": max_iter,
            "random_state": int(payload.get("random_state", 42)),
        }
        name = f"logistic_regression_C={params['C']}_class_weight={class_weight or 'none'}"
        specs.append(CandidateSpec("logistic_regression", name, params))
    return specs


def random_forest_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    payload = dict(config or {})
    n_estimators = payload.get("n_estimators", [50, 100])
    max_depths = payload.get("max_depth", [2, 4])
    min_samples_leaf = payload.get("min_samples_leaf", [1, 3])
    class_weights = payload.get("class_weight", [None, "balanced"])
    specs: list[CandidateSpec] = []
    for n_value, depth, leaf, class_weight in product(n_estimators, max_depths, min_samples_leaf, class_weights):
        params = {
            "n_estimators": int(n_value),
            "max_depth": None if depth is None else int(depth),
            "min_samples_leaf": int(leaf),
            "class_weight": class_weight,
            "random_state": int(payload.get("random_state", 42)),
            "n_jobs": 1,
        }
        name = (
            f"random_forest_n={params['n_estimators']}_depth={params['max_depth'] or 'none'}_"
            f"leaf={params['min_samples_leaf']}_class_weight={class_weight or 'none'}"
        )
        specs.append(CandidateSpec("random_forest", name, params))
    return specs


def create_profile_estimator(spec: CandidateSpec):
    if spec.estimator_type == "logistic_regression":
        return LogisticRegression(**spec.hyperparameters)
    if spec.estimator_type == "random_forest":
        return RandomForestClassifier(**spec.hyperparameters)
    raise ValueError(f"Unsupported Profile estimator type: {spec.estimator_type}")


def profile_candidate_specs(config: Mapping[str, Any], *, candidate: str = "all") -> list[CandidateSpec]:
    hyper = dict(config.get("hyperparameter_search") or {})
    selected = candidate or "all"
    specs: list[CandidateSpec] = []
    if selected in ("all", "logistic_regression"):
        specs.extend(logistic_regression_candidate_specs(hyper.get("logistic_regression")))
    if selected in ("all", "random_forest"):
        specs.extend(random_forest_candidate_specs(hyper.get("random_forest")))
    if not specs:
        raise ValueError(f"No candidate specs generated for candidate={candidate}")
    if len(specs) > int(config.get("max_candidate_count", 32)):
        raise ValueError("bounded Profile candidate search exceeded max_candidate_count")
    return specs
