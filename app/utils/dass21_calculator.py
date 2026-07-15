"""Deprecated compatibility wrapper for DASS-21 scoring.

New code should import from app.ml.preprocessing.dass21. This class remains so
existing routes and seed scripts keep the same field names and database shape.
"""

from __future__ import annotations

from app.ml.preprocessing.dass21.constants import (
    DASS21_ITEM_MAPPING,
    DASS21_SEVERITY_THRESHOLDS,
    DASS21_SCORING_VERSION,
)
from app.ml.preprocessing.dass21.scoring import (
    convert_frontend_payload,
    score_dass21,
    to_legacy_api_dict,
)


class DASS21Calculator:
    """Calculate DASS-21 scores from 21 ordered 0-3 responses."""

    SCORING_VERSION = DASS21_SCORING_VERSION
    DEPRESSION_ITEMS = tuple(int(item[1:]) - 1 for item in DASS21_ITEM_MAPPING["depression"])
    ANXIETY_ITEMS = tuple(int(item[1:]) - 1 for item in DASS21_ITEM_MAPPING["anxiety"])
    STRESS_ITEMS = tuple(int(item[1:]) - 1 for item in DASS21_ITEM_MAPPING["stress"])
    DEPRESSION_SEVERITY = DASS21_SEVERITY_THRESHOLDS["depression"]
    ANXIETY_SEVERITY = DASS21_SEVERITY_THRESHOLDS["anxiety"]
    STRESS_SEVERITY = DASS21_SEVERITY_THRESHOLDS["stress"]

    @staticmethod
    def calculate(responses: list[int]) -> dict:
        typed_responses = convert_frontend_payload({"responses": responses})
        result = score_dass21(typed_responses)
        return to_legacy_api_dict(result, include_responses=list(responses))

    @staticmethod
    def _get_severity(score: int, severity_dict: dict) -> str:
        for severity, (minimum, maximum) in severity_dict.items():
            if minimum <= score <= maximum:
                return severity.replace("_", " ").title()
        return "Extremely Severe"

    @staticmethod
    def calculate_dass21_risk_score(dass21_data: dict) -> float:
        """
        Backward-compatible engineering normalization for legacy aggregation.

        This is not a DASS-21 clinical output and does not independently
        calculate suicide risk, alerts, diagnosis, or treatment recommendations.
        """

        total_score = dass21_data.get("total_dass21_score", 0)
        risk_score = (total_score / 126) * 100
        return min(100, round(risk_score, 2))
