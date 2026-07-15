"""Typed schemas for behavioral preprocessing."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_MAPPING_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    EVENT_TYPES,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite_optional(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return value
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


class BehavioralBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class BehavioralFieldRole(str, Enum):
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    SESSION = "session"
    EVENT_TYPE = "event_type"
    FEATURE = "feature"
    CONTEXT = "context"
    METADATA = "metadata"
    EXCLUDED = "excluded"


class BehavioralFieldMapping(BehavioralBaseModel):
    source_field: str
    canonical_field: str
    role: BehavioralFieldRole
    expected_type: str
    missing_value_strategy: str
    aggregation_rule: str
    notes: str

    @validator("source_field", "canonical_field", "expected_type", "missing_value_strategy", "aggregation_rule", "notes")
    def validate_non_blank(cls, value: str, field) -> str:
        if not str(value).strip():
            raise ValueError(f"{field.name} cannot be blank")
        return str(value).strip()


class BehavioralMappingConfig(BehavioralBaseModel):
    mapping_version: str = BEHAVIORAL_MAPPING_VERSION
    dataset_name: str
    dataset_version: str
    source_columns: List[str]
    fields: List[BehavioralFieldMapping]
    notes: Optional[str] = None

    @root_validator
    def validate_mapping_integrity(cls, values):
        source_columns = values.get("source_columns") or []
        mapped = [field.source_field for field in values.get("fields") or []]
        missing = [column for column in source_columns if column not in mapped]
        extra = [column for column in mapped if column not in source_columns]
        if missing or extra:
            raise ValueError(f"Mapping fields must match source columns; missing={missing}, extra={extra}")
        roles = [field.role for field in values.get("fields") or []]
        if roles.count(BehavioralFieldRole.IDENTIFIER) != 1:
            raise ValueError("Behavioral mapping must contain exactly one identifier field")
        if roles.count(BehavioralFieldRole.TIMESTAMP) > 1:
            raise ValueError("Behavioral mapping may contain at most one timestamp field")
        return values


class BehavioralEvent(BehavioralBaseModel):
    event_id: str
    participant_key: str
    event_timestamp: datetime
    session_id: str
    event_type: str
    page_or_context: Optional[str] = None
    response_latency_ms: Optional[float] = None
    key_dwell_time_ms: Optional[float] = None
    key_flight_time_ms: Optional[float] = None
    typing_speed_cpm: Optional[float] = None
    backspace_count: Optional[float] = None
    correction_count: Optional[float] = None
    mouse_distance_px: Optional[float] = None
    mouse_speed_px_per_second: Optional[float] = None
    click_count: Optional[float] = None
    hesitation_count: Optional[float] = None
    session_duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("event_timestamp")
    def validate_event_timestamp(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "event_timestamp")

    @validator("event_type")
    def validate_event_type(cls, value: str) -> str:
        if value not in EVENT_TYPES:
            raise ValueError(f"Unsupported behavioral event_type: {value}")
        return value

    @validator(
        "response_latency_ms",
        "key_dwell_time_ms",
        "key_flight_time_ms",
        "typing_speed_cpm",
        "backspace_count",
        "correction_count",
        "mouse_distance_px",
        "mouse_speed_px_per_second",
        "click_count",
        "hesitation_count",
        "session_duration_seconds",
    )
    def validate_optional_numeric(cls, value: Optional[float], field) -> Optional[float]:
        value = _finite_optional(value, field.name)
        if value is not None and value < 0:
            raise ValueError(f"{field.name} cannot be negative")
        return value


class BehavioralSessionRecord(BehavioralBaseModel):
    session_id: str
    participant_key: str
    session_start: datetime
    session_end: datetime
    event_count: int
    typing_event_count: int
    mouse_event_count: int
    prompt_response_count: int
    session_features: Dict[str, Any]
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("session_start", "session_end")
    def validate_session_timestamps(cls, value: datetime, field) -> datetime:
        return _timezone_aware(value, field.name)


class BehavioralBaseline(BehavioralBaseModel):
    participant_key: str
    baseline_start: Optional[datetime] = None
    baseline_end: Optional[datetime] = None
    observation_count: int
    feature_means: Dict[str, float] = Field(default_factory=dict)
    feature_standard_deviations: Dict[str, float] = Field(default_factory=dict)
    feature_medians: Dict[str, float] = Field(default_factory=dict)
    feature_iqrs: Dict[str, float] = Field(default_factory=dict)
    completeness: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class BehavioralFeatureRecord(BehavioralBaseModel):
    record_id: str
    participant_key: str
    feature_timestamp: datetime
    session_id: str
    raw_features: Dict[str, Any]
    baseline_deviation_features: Dict[str, Any] = Field(default_factory=dict)
    completeness: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @validator("feature_timestamp")
    def validate_feature_timestamp(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "feature_timestamp")


class BehavioralPreprocessingReport(BehavioralBaseModel):
    preprocessing_version: str = BEHAVIORAL_PREPROCESSING_VERSION
    feature_schema_version: str = BEHAVIORAL_FEATURE_SCHEMA_VERSION
    mapping_version: str = BEHAVIORAL_MAPPING_VERSION
    source_type: str
    source_record_count: int
    output_event_count: int
    output_session_count: int
    participant_count: int
    date_range: Dict[str, Optional[str]]
    missing_value_summary: Dict[str, int]
    invalid_event_count: int
    duplicate_event_count: int
    observations_per_participant_summary: Dict[str, Any]
    baseline_eligible_participant_count: int
    feature_columns: List[str]
    unavailable_features: List[str]
    privacy_warnings: List[str]
    readiness_status: str
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

    @validator(
        "source_record_count",
        "output_event_count",
        "output_session_count",
        "participant_count",
        "invalid_event_count",
        "duplicate_event_count",
        "baseline_eligible_participant_count",
    )
    def validate_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

