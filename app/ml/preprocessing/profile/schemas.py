"""Typed schemas for Student Profile preprocessing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.profile.constants import (
    PROFILE_FEATURE_SCHEMA_VERSION,
    PROFILE_MAPPING_VERSION,
    PROFILE_PREPROCESSING_VERSION,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class ProfileBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class ProfileFieldRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    SENSITIVE_CONTEXT = "sensitive_context"
    METADATA = "metadata"
    EXCLUDED = "excluded"


class ProfileFieldMapping(ProfileBaseModel):
    source_column_name: str
    canonical_feature_name: str
    role: ProfileFieldRole
    expected_type: str
    allowed_categories: Optional[List[str]] = None
    missing_value_strategy: str
    encoding_strategy: str
    production_field_equivalent: Optional[str] = None
    notes: str
    include_by_default: bool = False
    leakage_candidate: bool = False

    @validator("source_column_name", "canonical_feature_name", "expected_type", "missing_value_strategy", "encoding_strategy", "notes")
    def validate_non_blank(cls, value: str, field) -> str:
        if not str(value).strip():
            raise ValueError(f"{field.name} cannot be blank")
        return str(value).strip()


class ProfileMappingConfig(ProfileBaseModel):
    mapping_version: str = PROFILE_MAPPING_VERSION
    dataset_name: str
    dataset_version: str
    target_column: str
    source_columns: List[str]
    fields: List[ProfileFieldMapping]
    default_excluded_columns: List[str] = Field(default_factory=list)
    sensitive_context_columns: List[str] = Field(default_factory=list)
    leakage_candidate_columns: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @root_validator
    def validate_mapping_integrity(cls, values):
        source_columns = values.get("source_columns") or []
        mapped = [field.source_column_name for field in values.get("fields") or []]
        missing = [column for column in source_columns if column not in mapped]
        extra = [column for column in mapped if column not in source_columns]
        if missing or extra:
            raise ValueError(f"Mapping fields must match source columns; missing={missing}, extra={extra}")

        target_column = values.get("target_column")
        target_fields = [field for field in values.get("fields") or [] if field.role == ProfileFieldRole.TARGET]
        if len(target_fields) != 1 or target_fields[0].source_column_name != target_column:
            raise ValueError("Exactly one mapped target field must match target_column")
        return values


class ProfileSourceRecord(ProfileBaseModel):
    source_row_id: int
    timestamp: Optional[str] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    course: Optional[str] = None
    year_of_study: Optional[str] = None
    cgpa: Optional[str] = None
    marital_status: Optional[str] = None
    depression_label: str
    anxiety_label: Optional[str] = None
    panic_attack_label: Optional[str] = None
    treatment_label: Optional[str] = None


class ProfileCanonicalRecord(ProfileBaseModel):
    record_id: str
    numeric_features: Dict[str, Optional[float]] = Field(default_factory=dict)
    categorical_features: Dict[str, Optional[str]] = Field(default_factory=dict)
    target_label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_warnings: List[str] = Field(default_factory=list)

    @root_validator
    def validate_feature_boundaries(cls, values):
        target_label = values.get("target_label")
        for bucket in ("numeric_features", "categorical_features"):
            if target_label in (values.get(bucket) or {}):
                raise ValueError("target label must not be included in feature values")
        return values


class ProfileFeatureRecord(ProfileBaseModel):
    record_id: str
    feature_values: Dict[str, Any]
    target_label: str
    split_group_id: Optional[str] = None

    @root_validator
    def validate_no_target_feature(cls, values):
        feature_values = values.get("feature_values") or {}
        if "target_depression" in feature_values or "Do you have Depression?" in feature_values:
            raise ValueError("target must not be included in feature values")
        if "source_timestamp" in feature_values or "Timestamp" in feature_values:
            raise ValueError("metadata must not be included in feature values")
        return values


class ProfilePreprocessingReport(ProfileBaseModel):
    preprocessing_version: str = PROFILE_PREPROCESSING_VERSION
    feature_schema_version: str = PROFILE_FEATURE_SCHEMA_VERSION
    mapping_version: str = PROFILE_MAPPING_VERSION
    source_fingerprint: str
    source_row_count: int
    output_row_count: int
    excluded_row_count: int
    missing_value_summary: Dict[str, int]
    category_normalization_summary: Dict[str, Dict[str, Any]]
    target_distribution: Dict[str, int]
    feature_columns: List[str]
    excluded_columns: List[str]
    sensitive_context_columns: List[str]
    leakage_checks: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

    @validator("source_row_count", "output_row_count", "excluded_row_count")
    def validate_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value
