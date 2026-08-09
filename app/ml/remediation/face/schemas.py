"""Typed schemas for facial duplicate remediation artifacts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.remediation.face.constants import (
    FACE_DEDUPLICATED_VIEW_VERSION,
    FACE_DUPLICATE_POLICY_VERSION,
    FACE_REMEDIATION_VERSION,
    FACE_REVISED_SPLIT_VERSION,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_RE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value


def validate_repo_relative_path(value: str) -> str:
    cleaned = str(value).replace("\\", "/").strip()
    if not cleaned:
        raise ValueError("path cannot be blank")
    if _ABSOLUTE_RE.match(cleaned):
        raise ValueError("path must be repository-relative")
    if any(part == ".." for part in cleaned.split("/")):
        raise ValueError("path cannot contain traversal")
    return cleaned


class FaceRemediationModel(BaseModel):
    class Config:
        extra = "forbid"
        use_enum_values = True
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


class FaceRemediationAction(str, Enum):
    KEEP = "keep"
    EXCLUDE_DUPLICATE = "exclude_duplicate"
    QUARANTINE_CROSS_LABEL = "quarantine_cross_label"
    EXCLUDE_INVALID = "exclude_invalid"


class FaceDuplicateRecord(FaceRemediationModel):
    record_id: str
    image_hash: str
    canonical_label: str
    original_split: str
    relative_path: str
    group_id: str
    readable: bool = True

    @validator("record_id", "canonical_label", "original_split", "group_id")
    def non_blank(cls, value: str, field) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field.name} cannot be blank")
        return value

    @validator("image_hash")
    def sha256(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("image_hash must be SHA-256")
        return value

    @validator("relative_path")
    def relative_path_only(cls, value: str) -> str:
        return validate_repo_relative_path(value)


class FaceDuplicateGroup(FaceRemediationModel):
    group_id: str
    image_hash: str
    record_ids: List[str]
    labels: List[str]
    original_splits: List[str]
    same_label: bool
    cross_label: bool
    cross_split: bool
    selected_representative_id: Optional[str] = None
    quarantined: bool = False
    decision: str = "pending"
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)

    @validator("record_ids", "labels", "original_splits", pre=True)
    def sorted_unique_strings(cls, value) -> list[str]:
        return sorted({str(item).strip() for item in (value or []) if str(item).strip()})

    @validator("image_hash")
    def sha256(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("image_hash must be SHA-256")
        return value


class FaceRemediationDecision(FaceRemediationModel):
    record_id: str
    action: FaceRemediationAction
    representative_id: Optional[str] = None
    group_id: Optional[str] = None
    reason: str
    policy_version: str = FACE_DUPLICATE_POLICY_VERSION

    @validator("record_id", "reason", "policy_version")
    def non_blank(cls, value: str, field) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field.name} cannot be blank")
        return value


class FaceDeduplicatedViewReport(FaceRemediationModel):
    remediation_version: str = FACE_REMEDIATION_VERSION
    view_version: str = FACE_DEDUPLICATED_VIEW_VERSION
    policy_version: str = FACE_DUPLICATE_POLICY_VERSION
    source_fingerprint: str
    canonical_manifest_hash: str
    duplicate_policy_hash: str
    source_record_count: int
    retained_record_count: int
    excluded_same_label_duplicate_count: int
    quarantined_cross_label_record_count: int
    duplicate_group_count: int
    same_label_group_count: int
    cross_split_group_count: int
    cross_label_group_count: int
    records_in_duplicate_groups: int
    retained_label_distribution: Dict[str, int]
    excluded_label_distribution: Dict[str, int]
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("generated_at")
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @validator("source_fingerprint", "canonical_manifest_hash", "duplicate_policy_hash")
    def sha256_fields(cls, value: str, field) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError(f"{field.name} must be SHA-256")
        return value

    @validator(
        "source_record_count",
        "retained_record_count",
        "excluded_same_label_duplicate_count",
        "quarantined_cross_label_record_count",
        "duplicate_group_count",
        "same_label_group_count",
        "cross_split_group_count",
        "cross_label_group_count",
        "records_in_duplicate_groups",
    )
    def non_negative_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value


class FaceRevisedSplitManifest(FaceRemediationModel):
    split_version: str = FACE_REVISED_SPLIT_VERSION
    remediation_version: str = FACE_REMEDIATION_VERSION
    source_fingerprint: str
    canonical_manifest_hash: str
    duplicate_policy_hash: str
    deduplicated_view_hash: str
    random_seed: int
    strategy: str
    train_ids: List[str]
    validation_ids: List[str]
    test_ids: List[str]
    excluded_ids: Dict[str, str]
    quarantined_ids: Dict[str, str]
    label_distributions: Dict[str, Dict[str, int]]
    duplicate_overlap_count: int
    image_hash_overlap_count: int
    record_overlap_count: int = 0
    original_split_overlap_summary: Dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)
    warnings: List[str] = Field(default_factory=list)

    @validator("source_fingerprint", "canonical_manifest_hash", "duplicate_policy_hash", "deduplicated_view_hash")
    def sha256_fields(cls, value: str, field) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError(f"{field.name} must be SHA-256")
        return value

    @validator("created_at")
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @validator("train_ids", "validation_ids", "test_ids", pre=True)
    def unique_ids(cls, value, field) -> list[str]:
        ids = [str(item).strip() for item in (value or [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{field.name} contains duplicate IDs")
        if any(not item for item in ids):
            raise ValueError(f"{field.name} contains blank IDs")
        return ids

    @root_validator
    def no_id_overlap(cls, values):
        train = set(values.get("train_ids") or [])
        validation = set(values.get("validation_ids") or [])
        test = set(values.get("test_ids") or [])
        if train & validation or train & test or validation & test:
            raise ValueError("records cannot overlap across revised splits")
        return values

