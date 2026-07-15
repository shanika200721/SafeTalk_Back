"""Typed schemas for Daily Mood preprocessing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.mood.constants import (
    MOOD_FEATURE_SCHEMA_VERSION,
    MOOD_MAPPING_VERSION,
    MOOD_PREPROCESSING_VERSION,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class MoodBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class MoodFieldRole(str, Enum):
    FEATURE = "feature"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"
    EXCLUDED = "excluded"


class MoodFieldMapping(MoodBaseModel):
    source_field: str
    canonical_field: str
    role: MoodFieldRole
    expected_type: str
    valid_range: Optional[List[float]] = None
    valid_categories: Optional[List[str]] = None
    missing_value_strategy: str
    aggregation_rule: str
    production_field_equivalent: Optional[str] = None
    notes: str

    @validator("source_field", "canonical_field", "expected_type", "missing_value_strategy", "aggregation_rule", "notes")
    def validate_non_blank(cls, value: str, field) -> str:
        if not str(value).strip():
            raise ValueError(f"{field.name} cannot be blank")
        return str(value).strip()


class MoodMappingConfig(MoodBaseModel):
    mapping_version: str = MOOD_MAPPING_VERSION
    dataset_name: str
    dataset_version: str
    source_columns: List[str]
    fields: List[MoodFieldMapping]
    notes: Optional[str] = None

    @root_validator
    def validate_mapping_integrity(cls, values):
        source_columns = values.get("source_columns") or []
        mapped = [field.source_field for field in values.get("fields") or []]
        missing = [column for column in source_columns if column not in mapped]
        extra = [column for column in mapped if column not in source_columns]
        if missing or extra:
            raise ValueError(f"Mapping fields must match source columns; missing={missing}, extra={extra}")
        canonical_by_role = {field.role: [] for field in values.get("fields") or []}
        for field in values.get("fields") or []:
            canonical_by_role.setdefault(field.role, []).append(field.canonical_field)
        if len(canonical_by_role.get(MoodFieldRole.IDENTIFIER, [])) != 1:
            raise ValueError("Mood mapping must contain exactly one identifier")
        if len(canonical_by_role.get(MoodFieldRole.TIMESTAMP, [])) != 1:
            raise ValueError("Mood mapping must contain exactly one timestamp")
        if "mood_value" not in [field.canonical_field for field in values.get("fields") or []]:
            raise ValueError("Mood mapping must include canonical mood_value")
        return values


class MoodSourceRecord(MoodBaseModel):
    student_id: str
    checkin_timestamp: datetime
    mood_value: int
    crying_episode_count: Optional[float] = None
    physical_symptom_count: Optional[float] = None
    source_record_id: Optional[str] = None

    @validator("checkin_timestamp")
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "checkin_timestamp")


class MoodCanonicalRecord(MoodBaseModel):
    record_id: str
    participant_key: str
    timestamp: datetime
    mood_value: float
    crying_episode_count: Optional[float] = None
    physical_symptom_count: Optional[float] = None
    notes_present: Optional[bool] = None
    time_of_day: str
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("timestamp")
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "timestamp")


class MoodFeatureRecord(MoodBaseModel):
    record_id: str
    participant_key: str
    feature_timestamp: datetime
    feature_values: Dict[str, Any]
    history_length: int
    completeness: float
    warnings: List[str] = Field(default_factory=list)

    @validator("feature_timestamp")
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "feature_timestamp")


class MoodPreprocessingReport(MoodBaseModel):
    preprocessing_version: str = MOOD_PREPROCESSING_VERSION
    feature_schema_version: str = MOOD_FEATURE_SCHEMA_VERSION
    mapping_version: str = MOOD_MAPPING_VERSION
    source_record_count: int
    output_record_count: int
    participant_count: int
    date_range: Dict[str, Optional[str]]
    missing_value_summary: Dict[str, int]
    duplicate_summary: Dict[str, Any]
    temporal_order_violations: int
    feature_columns: List[str]
    excluded_columns: List[str]
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

    @validator("source_record_count", "output_record_count", "participant_count", "temporal_order_violations")
    def validate_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value
