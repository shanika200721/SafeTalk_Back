"""Typed schemas for authoritative DASS-21 scoring."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.dass21.constants import (
    DASS21_SCORING_VERSION,
    QUESTIONNAIRE_VERSION,
    RESPONSE_SCALE_0_3,
    SEVERITY_LEVELS,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class DASS21BaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return self.dict(exclude_none=True)


class DASS21Responses(DASS21BaseModel):
    responses: Dict[str, int]
    questionnaire_version: str = QUESTIONNAIRE_VERSION
    response_scale: str = RESPONSE_SCALE_0_3
    completed_at: Optional[datetime] = None

    @validator("completed_at")
    def validate_completed_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        return _timezone_aware(value, "completed_at")


class DASS21SubscaleScore(DASS21BaseModel):
    raw_score: int
    multiplied_score: int
    severity: str
    normalized_score_0_100: float

    @validator("severity")
    def validate_severity(cls, value: str) -> str:
        if value not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}")
        return value

    @validator("normalized_score_0_100")
    def validate_normalized_score(cls, value: float) -> float:
        if value < 0 or value > 100:
            raise ValueError("normalized_score_0_100 must be between 0 and 100")
        return round(value, 2)


class DASS21ScoreResult(DASS21BaseModel):
    scoring_version: str = DASS21_SCORING_VERSION
    questionnaire_version: str = QUESTIONNAIRE_VERSION
    depression: DASS21SubscaleScore
    anxiety: DASS21SubscaleScore
    stress: DASS21SubscaleScore
    answered_item_count: int
    missing_item_count: int
    is_complete: bool
    validation_warnings: List[str] = Field(default_factory=list)
    scored_at: datetime = Field(default_factory=utc_now)

    @validator("scored_at")
    def validate_scored_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "scored_at")

    @root_validator
    def validate_counts(cls, values):
        answered = values.get("answered_item_count")
        missing = values.get("missing_item_count")
        is_complete = values.get("is_complete")
        if answered is not None and answered < 0:
            raise ValueError("answered_item_count must be non-negative")
        if missing is not None and missing < 0:
            raise ValueError("missing_item_count must be non-negative")
        if answered is not None and missing is not None and answered + missing != 21:
            raise ValueError("answered_item_count + missing_item_count must equal 21")
        if is_complete is not None and missing is not None and is_complete != (missing == 0):
            raise ValueError("is_complete must reflect missing_item_count")
        return values
