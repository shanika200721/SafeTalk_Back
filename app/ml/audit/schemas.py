"""Typed, privacy-safe dataset audit schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.common import paths
from app.ml.common.schemas import Modality


DATASET_AUDIT_VERSION = "1.0.0"

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AuditSeverity(_StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class _SafeBaseModel(BaseModel):
    class Config:
        use_enum_values = False
        json_encoders = {
            Path: str,
            datetime: lambda value: value.astimezone(timezone.utc).isoformat(),
        }
        extra = "forbid"

    def to_safe_dict(self) -> Dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        try:
            return str(value.relative_to(paths.get_repository_root())).replace("\\", "/")
        except ValueError:
            return value.name
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value


def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be blank")
    return value.strip()


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_percentage(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return value
    if value < 0 or value > 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return float(value)


def _validate_non_negative(value: Optional[float | int], field_name: str) -> Optional[float | int]:
    if value is None:
        return value
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_relative_report_path(value: str, field_name: str) -> str:
    value = _non_blank(str(value).replace("\\", "/"), field_name)
    if value.startswith("/") or value.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(value):
        raise ValueError(f"{field_name} must be repository-relative")
    parts = [part for part in value.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError(f"{field_name} cannot contain traversal")
    return "/".join(parts)


class AuditIssue(_SafeBaseModel):
    code: str
    severity: AuditSeverity
    message: str
    field_name: Optional[str] = None
    count: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None

    @validator("code", "message")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("field_name", "recommendation")
    def validate_optional_strings(cls, value: Optional[str], field) -> Optional[str]:
        if value is None:
            return value
        return _non_blank(value, field.name)

    @validator("count")
    def validate_count(cls, value: Optional[int]) -> Optional[int]:
        return _validate_non_negative(value, "count")


class ClassDistributionItem(_SafeBaseModel):
    label: str
    count: int
    percentage: float

    @validator("label")
    def validate_label(cls, value: str) -> str:
        return _non_blank(value, "label")

    @validator("count")
    def validate_count(cls, value: int) -> int:
        return _validate_non_negative(value, "count")

    @validator("percentage")
    def validate_percentage(cls, value: float) -> float:
        return _validate_percentage(value, "percentage")


class LengthSummary(_SafeBaseModel):
    minimum: float = 0
    maximum: float = 0
    mean: float = 0
    median: float = 0
    percentile_25: float = 0
    percentile_75: float = 0
    percentile_95: float = 0

    @validator("*")
    def validate_non_negative(cls, value: float, field) -> float:
        return float(_validate_non_negative(value, field.name))


class ColumnAudit(_SafeBaseModel):
    column_name: str
    inferred_dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    duplicate_count: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    standard_deviation: Optional[float] = None
    most_common_values: List[Dict[str, Any]] = Field(default_factory=list)
    possible_identifier: bool = False
    possible_sensitive_field: bool = False
    possible_target_leakage: bool = False
    issues: List[AuditIssue] = Field(default_factory=list)

    @validator("column_name", "inferred_dtype")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("non_null_count", "null_count", "unique_count", "duplicate_count")
    def validate_counts(cls, value: Optional[int], field) -> Optional[int]:
        return _validate_non_negative(value, field.name)

    @validator("null_percentage")
    def validate_null_percentage(cls, value: float) -> float:
        return _validate_percentage(value, "null_percentage")


class TabularAuditResult(_SafeBaseModel):
    row_count: int
    column_count: int
    columns: List[ColumnAudit]
    duplicate_row_count: int
    class_distribution: Dict[str, List[ClassDistributionItem]] = Field(default_factory=dict)
    invalid_range_counts: Dict[str, int] = Field(default_factory=dict)
    possible_identifier_columns: List[str] = Field(default_factory=list)
    possible_sensitive_columns: List[str] = Field(default_factory=list)
    possible_leakage_columns: List[str] = Field(default_factory=list)
    issues: List[AuditIssue] = Field(default_factory=list)

    @validator("row_count", "column_count", "duplicate_row_count")
    def validate_counts(cls, value: int, field) -> int:
        return _validate_non_negative(value, field.name)


class TextAuditResult(_SafeBaseModel):
    record_count: int
    label_distribution: Dict[str, List[ClassDistributionItem]] = Field(default_factory=dict)
    missing_text_count: int
    exact_duplicate_text_count: int
    duplicate_text_conflicting_labels_count: int
    near_duplicate_candidate_count: int
    character_length_summary: LengthSummary
    word_count_summary: LengthSummary
    url_occurrence_count: int
    email_occurrence_count: int
    username_occurrence_count: int
    phone_occurrence_count: int
    possible_person_name_occurrence_count: Optional[int] = None
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    issues: List[AuditIssue] = Field(default_factory=list)

    @validator(
        "record_count",
        "missing_text_count",
        "exact_duplicate_text_count",
        "duplicate_text_conflicting_labels_count",
        "near_duplicate_candidate_count",
        "url_occurrence_count",
        "email_occurrence_count",
        "username_occurrence_count",
        "phone_occurrence_count",
        "possible_person_name_occurrence_count",
    )
    def validate_counts(cls, value: Optional[int], field) -> Optional[int]:
        return _validate_non_negative(value, field.name)


class AudioAuditResult(_SafeBaseModel):
    file_count: int
    readable_file_count: int
    unreadable_file_count: int
    total_duration_seconds: float
    duration_summary: LengthSummary
    sample_rate_distribution: Dict[str, int] = Field(default_factory=dict)
    channel_distribution: Dict[str, int] = Field(default_factory=dict)
    format_distribution: Dict[str, int] = Field(default_factory=dict)
    empty_file_count: int
    corrupt_file_count: int
    duplicate_hash_group_count: int
    label_distribution: Dict[str, List[ClassDistributionItem]] = Field(default_factory=dict)
    issues: List[AuditIssue] = Field(default_factory=list)

    @validator(
        "file_count",
        "readable_file_count",
        "unreadable_file_count",
        "empty_file_count",
        "corrupt_file_count",
        "duplicate_hash_group_count",
    )
    def validate_counts(cls, value: int, field) -> int:
        return _validate_non_negative(value, field.name)

    @validator("total_duration_seconds")
    def validate_duration(cls, value: float) -> float:
        return float(_validate_non_negative(value, "total_duration_seconds"))


class ImageAuditResult(_SafeBaseModel):
    file_count: int
    readable_file_count: int
    unreadable_file_count: int
    width_summary: LengthSummary
    height_summary: LengthSummary
    color_mode_distribution: Dict[str, int] = Field(default_factory=dict)
    format_distribution: Dict[str, int] = Field(default_factory=dict)
    corrupt_file_count: int
    duplicate_hash_group_count: int
    label_distribution: Dict[str, List[ClassDistributionItem]] = Field(default_factory=dict)
    issues: List[AuditIssue] = Field(default_factory=list)

    @validator(
        "file_count",
        "readable_file_count",
        "unreadable_file_count",
        "corrupt_file_count",
        "duplicate_hash_group_count",
    )
    def validate_counts(cls, value: int, field) -> int:
        return _validate_non_negative(value, field.name)


class DatasetAuditReport(_SafeBaseModel):
    audit_version: str = DATASET_AUDIT_VERSION
    dataset_name: str
    dataset_version: str
    modality: Modality
    source_relative_path: str
    source_fingerprint_hash: str
    config_hash: Optional[str] = None
    audit_started_at: datetime
    audit_completed_at: datetime
    source_type: str
    tabular_result: Optional[TabularAuditResult] = None
    text_result: Optional[TextAuditResult] = None
    audio_result: Optional[AudioAuditResult] = None
    image_result: Optional[ImageAuditResult] = None
    issues: List[AuditIssue] = Field(default_factory=list)
    summary_status: str
    notes: Optional[str] = None

    @validator("audit_version", "dataset_name", "dataset_version", "source_type", "summary_status")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("source_relative_path")
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_relative_report_path(value, "source_relative_path")

    @validator("audit_started_at", "audit_completed_at")
    def validate_timestamps(cls, value: datetime, field) -> datetime:
        return _timezone_aware(value, field.name)

    @root_validator
    def validate_modality_results(cls, values):
        populated = [
            name
            for name in ("tabular_result", "text_result", "audio_result", "image_result")
            if values.get(name) is not None
        ]
        if len(populated) > 1:
            raise ValueError("Only one modality-specific audit result may be populated")
        started = values.get("audit_started_at")
        completed = values.get("audit_completed_at")
        if started is not None and completed is not None and completed < started:
            raise ValueError("audit_completed_at cannot be before audit_started_at")
        return values
