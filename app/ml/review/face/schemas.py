"""Typed schemas for Phase 3H face review artifacts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.review.face.constants import (
    FACE_ALLOWED_REVIEW_DECISIONS,
    FACE_FINAL_ACTIONS,
    FACE_RECONCILIATION_POLICY_VERSION,
    FACE_REVIEW_CONFIDENCE_LEVELS,
    FACE_REVIEW_DECISION_SCHEMA_VERSION,
    FACE_REVIEW_ITEM_TYPES,
    FACE_REVIEW_WORKFLOW_VERSION,
)

_ABSOLUTE_RE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IDENTITY_FIELDS = {
    "name",
    "person_name",
    "participant_name",
    "email",
    "identity",
    "subject_identity",
    "face_embedding",
    "embedding",
    "biometric_template",
    "race",
    "ethnicity",
    "health_condition",
    "suicide_risk",
}


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


def validate_timezone_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def validate_repo_relative_path(value: str) -> str:
    cleaned = str(value).replace("\\", "/").strip()
    if not cleaned:
        raise ValueError("path cannot be blank")
    if _ABSOLUTE_RE.match(cleaned):
        raise ValueError("path must be repository-relative")
    if any(part == ".." for part in cleaned.split("/")):
        raise ValueError("path cannot contain traversal")
    return cleaned


def validate_no_forbidden_fields(payload: dict[str, Any]) -> None:
    keys = {str(key).strip().lower() for key in payload}
    forbidden = sorted(keys & _FORBIDDEN_IDENTITY_FIELDS)
    if forbidden:
        raise ValueError(f"forbidden identity or biometric fields present: {forbidden}")


class FaceReviewModel(BaseModel):
    class Config:
        extra = "forbid"
        use_enum_values = True
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


class FaceReviewItemType(str, Enum):
    CROSS_LABEL_CONFLICT = "cross_label_conflict"
    PERCEPTUAL_DUPLICATE_CANDIDATE = "perceptual_duplicate_candidate"


class FaceReviewStatus(str, Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    UNRESOLVED = "unresolved"


class FaceReviewConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FaceReviewItem(FaceReviewModel):
    review_item_id: str
    item_type: FaceReviewItemType
    group_id: str
    record_ids: List[str]
    current_labels: Dict[str, str]
    original_splits: Dict[str, str]
    safe_image_references: Dict[str, str]
    image_hashes: Dict[str, str]
    perceptual_distance: Optional[int] = None
    review_status: str = "pending"
    required_reviewers: int
    policy_version: str
    created_at: datetime = Field(default_factory=utc_now)

    @validator("review_item_id", "group_id", "policy_version", "review_status")
    def non_blank(cls, value: str, field) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field.name} cannot be blank")
        return value

    @validator("item_type")
    def allowed_item_type(cls, value: str) -> str:
        if str(value) not in FACE_REVIEW_ITEM_TYPES:
            raise ValueError("invalid review item type")
        return value

    @validator("record_ids", pre=True)
    def unique_record_ids(cls, value) -> list[str]:
        ids = [str(item).strip() for item in (value or []) if str(item).strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("record_ids cannot contain duplicates")
        if not ids:
            raise ValueError("record_ids cannot be empty")
        return ids

    @validator("safe_image_references")
    def relative_references(cls, value: dict[str, str]) -> dict[str, str]:
        return {str(key): validate_repo_relative_path(str(item)) for key, item in value.items()}

    @validator("image_hashes")
    def sha256_values(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned = {}
        for key, item in value.items():
            digest = str(item).lower()
            if not _SHA256_RE.match(digest):
                raise ValueError("image_hashes must be SHA-256 values")
            cleaned[str(key)] = digest
        return cleaned

    @validator("created_at")
    def aware_created_at(cls, value: datetime) -> datetime:
        return validate_timezone_aware(value)

    @root_validator
    def records_have_metadata(cls, values):
        record_ids = set(values.get("record_ids") or [])
        for field in ("current_labels", "original_splits", "safe_image_references", "image_hashes"):
            missing = sorted(record_ids - set((values.get(field) or {}).keys()))
            if missing:
                raise ValueError(f"{field} missing records: {missing}")
        return values


class FaceReviewerDecision(FaceReviewModel):
    review_item_id: str
    reviewer_alias: str
    decision: str
    reason_code: str
    confidence: FaceReviewConfidence
    notes: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=utc_now)
    workflow_version: str = FACE_REVIEW_WORKFLOW_VERSION

    @validator("review_item_id", "decision", "reason_code", "workflow_version")
    def non_blank(cls, value: str, field) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field.name} cannot be blank")
        return value

    @validator("reviewer_alias")
    def alias_not_email(cls, value: str) -> str:
        alias = str(value).strip()
        if not alias:
            raise ValueError("reviewer_alias cannot be blank")
        if _EMAIL_RE.match(alias):
            raise ValueError("reviewer_alias must not be an email address")
        return alias

    @validator("decision")
    def allowed_decision(cls, value: str) -> str:
        if value not in FACE_ALLOWED_REVIEW_DECISIONS:
            raise ValueError("invalid review decision")
        return value

    @validator("confidence")
    def allowed_confidence(cls, value: str) -> str:
        if str(value) not in FACE_REVIEW_CONFIDENCE_LEVELS:
            raise ValueError("invalid confidence value")
        return value

    @validator("reviewed_at", pre=True)
    def parse_timestamp(cls, value):
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)

    @validator("reviewed_at")
    def aware_reviewed_at(cls, value: datetime) -> datetime:
        return validate_timezone_aware(value)


class FaceReconciledDecision(FaceReviewModel):
    review_item_id: str
    final_status: str
    final_action: str
    retained_record_ids: List[str] = Field(default_factory=list)
    excluded_record_ids: List[str] = Field(default_factory=list)
    quarantined_record_ids: List[str] = Field(default_factory=list)
    label_change_recommended: bool = False
    consensus_reached: bool = False
    reconciliation_reason: str
    reconciled_at: datetime = Field(default_factory=utc_now)
    policy_version: str = FACE_RECONCILIATION_POLICY_VERSION

    @validator("final_action")
    def allowed_final_action(cls, value: str) -> str:
        if value not in FACE_FINAL_ACTIONS:
            raise ValueError("invalid final action")
        return value

    @validator("reconciled_at")
    def aware_reconciled_at(cls, value: datetime) -> datetime:
        return validate_timezone_aware(value)


class FaceReviewSummary(FaceReviewModel):
    workflow_version: str = FACE_REVIEW_WORKFLOW_VERSION
    decision_schema_version: str = FACE_REVIEW_DECISION_SCHEMA_VERSION
    source_fingerprint: str
    total_review_items: int
    reviewed_items: int
    pending_items: int
    consensus_items: int
    disagreement_items: int
    unresolved_items: int
    restored_record_count: int
    retained_quarantine_count: int
    excluded_record_count: int
    review_completion_percentage: float
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("source_fingerprint")
    def source_sha(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("source_fingerprint must be SHA-256")
        return value

    @validator("generated_at")
    def aware_generated_at(cls, value: datetime) -> datetime:
        return validate_timezone_aware(value)

