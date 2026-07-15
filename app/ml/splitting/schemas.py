"""Typed schemas for Phase 3A split manifests and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator


_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SplitStrategy(_StringEnum):
    RANDOM_STRATIFIED = "random_stratified"
    GROUPED_STRATIFIED = "grouped_stratified"
    PREDEFINED_PRESERVED = "predefined_preserved"
    PREDEFINED_REPAIRED = "predefined_repaired"
    CORPUS_GROUPED = "corpus_grouped"
    PARTICIPANT_GROUPED = "participant_grouped"


class SplitRecord(BaseModel):
    record_id: str
    split: str
    label: str
    group_id: Optional[str] = None
    source_name: Optional[str] = None
    source_split: Optional[str] = None
    duplicate_group_id: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        extra = "forbid"

    @validator("record_id", "split", "label")
    def non_blank(cls, value: str, field) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} cannot be blank")
        return value.strip()


class SplitValidationSummary(BaseModel):
    train_count: int
    validation_count: int
    test_count: int
    total_count: int
    train_distribution: Dict[str, int] = Field(default_factory=dict)
    validation_distribution: Dict[str, int] = Field(default_factory=dict)
    test_distribution: Dict[str, int] = Field(default_factory=dict)
    group_overlap_count: int = 0
    duplicate_overlap_count: int = 0
    source_overlap_count: int = 0
    missing_record_count: int = 0
    unexpected_record_count: int = 0
    deterministic_replay_passed: bool = False
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @validator(
        "train_count",
        "validation_count",
        "test_count",
        "total_count",
        "group_overlap_count",
        "duplicate_overlap_count",
        "source_overlap_count",
        "missing_record_count",
        "unexpected_record_count",
    )
    def non_negative(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value


class ModalitySplitManifest(BaseModel):
    manifest_version: str
    split_design_version: str
    modality: str
    dataset_name: str
    dataset_version: str
    preprocessing_version: str
    feature_schema_version: str
    source_fingerprint: str
    preprocessing_artifact_hash: str
    config_hash: str
    random_seed: int
    split_strategy: SplitStrategy
    train_ids: List[str]
    validation_ids: List[str]
    test_ids: List[str]
    excluded_ids: Dict[str, str] = Field(default_factory=dict)
    grouping_column: Optional[str] = None
    stratify_column: str
    source_split_policy: Optional[str] = None
    duplicate_policy: str
    created_at: datetime
    validation_summary: SplitValidationSummary
    notes: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    @validator(
        "manifest_version",
        "split_design_version",
        "modality",
        "dataset_name",
        "dataset_version",
        "preprocessing_version",
        "feature_schema_version",
        "source_fingerprint",
        "preprocessing_artifact_hash",
        "config_hash",
        "stratify_column",
        "duplicate_policy",
    )
    def non_blank(cls, value: str, field) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} cannot be blank")
        return value.strip()

    @validator("train_ids", "validation_ids", "test_ids", pre=True, always=True)
    def unique_ids_in_split(cls, values, field) -> List[str]:
        ids = [str(value).strip() for value in (values or [])]
        if any(not value for value in ids):
            raise ValueError(f"{field.name} cannot contain blank IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{field.name} contains duplicate IDs")
        return ids

    @validator("created_at")
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @root_validator
    def no_split_overlap(cls, values):
        train = set(values.get("train_ids") or [])
        validation = set(values.get("validation_ids") or [])
        test = set(values.get("test_ids") or [])
        if train & validation or train & test or validation & test:
            raise ValueError("record IDs cannot overlap across splits")
        return values

    def to_safe_dict(self) -> Dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


class SplitDesignReport(BaseModel):
    modality: str
    strategy: str
    source_count: int
    included_count: int
    excluded_count: int
    split_counts: Dict[str, int]
    label_distributions: Dict[str, Dict[str, int]]
    grouping_summary: Dict[str, Any] = Field(default_factory=dict)
    duplicate_handling: Dict[str, Any] = Field(default_factory=dict)
    leakage_checks: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    generated_at: datetime

    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    @validator("source_count", "included_count", "excluded_count")
    def non_negative(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @validator("generated_at")
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    def to_safe_dict(self) -> Dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


def validate_relative_path(value: str) -> str:
    cleaned = str(value).replace("\\", "/").strip()
    if not cleaned:
        raise ValueError("path cannot be blank")
    if cleaned.startswith("/") or cleaned.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(cleaned):
        raise ValueError("paths must be repository-relative")
    if any(part == ".." for part in cleaned.split("/")):
        raise ValueError("paths cannot contain traversal")
    return cleaned


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("split reports cannot contain NaN or infinity")
        return value
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value
