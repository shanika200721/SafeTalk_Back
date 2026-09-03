"""Configurable weighted late-fusion scoring for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]


@dataclass(frozen=True)
class ThresholdConfig:
    """Four-class threshold boundaries for normalized risk scores."""

    low_moderate: float = 0.30
    moderate_high: float = 0.50
    high_severe: float = 0.70

    def as_tuple(self) -> tuple[float, float, float]:
        values = (self.low_moderate, self.moderate_high, self.high_severe)
        if not (0.0 <= values[0] <= values[1] <= values[2] <= 1.0):
            raise ValueError("Thresholds must be ordered within [0, 1].")
        return values


def validate_weights(weights: Mapping[str, float], expected_total: float = 1.0, tolerance: float = 1e-8) -> None:
    """Validate that all weights are non-negative and sum to the expected total."""

    if not weights:
        raise ValueError("At least one modality weight is required.")
    invalid = {name: value for name, value in weights.items() if not np.isfinite(value) or value < 0}
    if invalid:
        raise ValueError(f"Invalid non-finite or negative weights: {invalid}")
    total = float(sum(weights.values()))
    if abs(total - expected_total) > tolerance:
        raise ValueError(f"Weights must sum to {expected_total}; observed {total}.")


def validate_scores(scores: Mapping[str, float | int | None], modalities: Iterable[str]) -> None:
    """Validate available normalized modality scores."""

    for modality in modalities:
        value = scores.get(modality)
        if value is None or pd.isna(value):
            continue
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{modality} must be normalized to [0, 1]; observed {value}.")


def weighted_score(
    scores: Mapping[str, float | int | None],
    weights: Mapping[str, float],
    *,
    min_available_modalities: int = 1,
) -> float:
    """Compute a normalized weighted score using only available modalities.

    Missing values are excluded from both the numerator and the denominator.
    They are never silently converted to zero.
    """

    validate_weights(weights)
    validate_scores(scores, weights.keys())
    numerator = 0.0
    denominator = 0.0
    available = 0
    for modality, weight in weights.items():
        value = scores.get(modality)
        if value is None or pd.isna(value):
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
        available += 1
    if available < min_available_modalities or denominator <= 0:
        raise ValueError("No valid modality scores are available for weighted fusion.")
    return float(numerator / denominator)


def classify_score(score: float, thresholds: ThresholdConfig | Mapping[str, float] | Iterable[float]) -> str:
    """Map a normalized score into the configured four-class risk category."""

    if isinstance(thresholds, ThresholdConfig):
        low_moderate, moderate_high, high_severe = thresholds.as_tuple()
    elif isinstance(thresholds, Mapping):
        low_moderate = float(thresholds["low_moderate"])
        moderate_high = float(thresholds["moderate_high"])
        high_severe = float(thresholds["high_severe"])
        ThresholdConfig(low_moderate, moderate_high, high_severe).as_tuple()
    else:
        low_moderate, moderate_high, high_severe = tuple(float(value) for value in thresholds)
        ThresholdConfig(low_moderate, moderate_high, high_severe).as_tuple()
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(f"Score must be normalized to [0, 1]; observed {score}.")
    if score < low_moderate:
        return "Low"
    if score < moderate_high:
        return "Moderate"
    if score < high_severe:
        return "High"
    return "Severe"


class WeightedLateFusion:
    """Small estimator-like wrapper around the weighted late-fusion rule."""

    def __init__(self, weights: Mapping[str, float], thresholds: Mapping[str, float] | Iterable[float]) -> None:
        self.weights = dict(weights)
        validate_weights(self.weights)
        if isinstance(thresholds, Mapping):
            self.thresholds = ThresholdConfig(
                float(thresholds["low_moderate"]),
                float(thresholds["moderate_high"]),
                float(thresholds["high_severe"]),
            )
        else:
            self.thresholds = ThresholdConfig(*tuple(float(value) for value in thresholds))
        self.thresholds.as_tuple()

    def score_row(self, row: Mapping[str, float | int | None]) -> float:
        return weighted_score(row, self.weights)

    def predict_row(self, row: Mapping[str, float | int | None]) -> str:
        return classify_score(self.score_row(row), self.thresholds)

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.score_row(row) for row in frame[self.weights.keys()].to_dict("records")])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.score_frame(frame)
        return np.asarray([classify_score(score, self.thresholds) for score in scores], dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return smooth class-membership proxies for plotting/AUC diagnostics."""

        scores = self.score_frame(frame)
        centers = np.asarray([0.15, 0.40, 0.60, 0.85])
        sigma = 0.16
        distances = ((scores[:, None] - centers[None, :]) ** 2) / (2 * sigma**2)
        probabilities = np.exp(-distances)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
