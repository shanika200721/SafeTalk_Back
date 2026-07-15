"""Typed schemas for text preprocessing and reporting."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.text.constants import (
    TEXT_FEATURE_SCHEMA_VERSION,
    TEXT_LABEL_MAPPING_VERSION,
    TEXT_PREPROCESSING_VERSION,
    TEXT_PRIVACY_RULESET_VERSION,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class TextBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class DuplicateType(str, Enum):
    EXACT = "exact"
    NEAR_CANDIDATE = "near_candidate"


class LabelMappingEntry(TextBaseModel):
    original_label: str
    canonical_label: str
    description: str
    retained: bool
    merged: bool = False
    excluded: bool = False
    mapping_justification: str

    @validator("original_label", "canonical_label", "description", "mapping_justification")
    def non_blank(cls, value: str, field) -> str:
        if not str(value).strip():
            raise ValueError(f"{field.name} cannot be blank")
        return str(value).strip()

    @root_validator
    def validate_status(cls, values):
        if values.get("merged") and values.get("original_label") == values.get("canonical_label"):
            raise ValueError("merged labels must change canonical_label")
        if values.get("retained") and values.get("excluded"):
            raise ValueError("label cannot be both retained and excluded")
        return values


class TextLabelMappingConfig(TextBaseModel):
    mapping_version: str = TEXT_LABEL_MAPPING_VERSION
    dataset_name: str
    dataset_version: str
    entries: List[LabelMappingEntry]
    notes: Optional[str] = None

    @root_validator
    def validate_entries(cls, values):
        labels = [entry.original_label for entry in values.get("entries") or []]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"duplicate label mapping entries: {duplicates}")
        return values


class SourceRole(str, Enum):
    AUTHORITATIVE_RAW = "authoritative_raw"
    EXCLUDED_DERIVED = "excluded_derived"
    REFERENCE_TEST = "reference_test"


class TextSourceSelectionItem(TextBaseModel):
    filename: str
    role: SourceRole
    include_in_canonical: bool
    reason: str
    text_column: str = "text"
    label_column: str = "status"


class TextSourceSelectionConfig(TextBaseModel):
    selection_version: str = "1.0.0"
    dataset_name: str
    dataset_version: str
    authoritative_source_file: str
    duplicate_check_policy: str
    conflict_policy: str = "quarantine"
    sources: List[TextSourceSelectionItem]
    notes: Optional[str] = None

    @root_validator
    def validate_selection(cls, values):
        items = values.get("sources") or []
        canonical = [item for item in items if item.include_in_canonical]
        if len(canonical) != 1:
            raise ValueError("exactly one source may be included in canonical preprocessing")
        if canonical[0].filename != values.get("authoritative_source_file"):
            raise ValueError("canonical source must match authoritative_source_file")
        return values


class TextSourceRecord(TextBaseModel):
    record_id: str
    raw_text: str
    original_label: str
    source_file: str
    source_row_index: int
    group_id: Optional[str] = None
    original_id: Optional[str] = None


class TextPrivacySummary(TextBaseModel):
    url_count: int = 0
    email_count: int = 0
    phone_count: int = 0
    username_count: int = 0
    ip_address_count: int = 0
    community_count: int = 0
    possible_person_identifier_count: int = 0
    replacement_rules_version: str = TEXT_PRIVACY_RULESET_VERSION

    def add(self, other: "TextPrivacySummary") -> "TextPrivacySummary":
        return TextPrivacySummary(
            url_count=self.url_count + other.url_count,
            email_count=self.email_count + other.email_count,
            phone_count=self.phone_count + other.phone_count,
            username_count=self.username_count + other.username_count,
            ip_address_count=self.ip_address_count + other.ip_address_count,
            community_count=self.community_count + other.community_count,
            possible_person_identifier_count=self.possible_person_identifier_count + other.possible_person_identifier_count,
            replacement_rules_version=self.replacement_rules_version,
        )


class TextCanonicalRecord(TextBaseModel):
    record_id: str
    normalized_text: str
    canonical_label: str
    source_name: str
    text_hash: str
    privacy_replacement_counts: TextPrivacySummary
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("text_hash")
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError("text_hash must be valid SHA-256")
        return value


class TextDuplicateGroup(TextBaseModel):
    duplicate_hash: str
    record_ids: List[str]
    labels: List[str]
    source_files: List[str]
    conflict: bool
    duplicate_type: DuplicateType

    @validator("duplicate_hash")
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError("duplicate_hash must be valid SHA-256")
        return value


class TextPreprocessingReport(TextBaseModel):
    preprocessing_version: str = TEXT_PREPROCESSING_VERSION
    feature_schema_version: str = TEXT_FEATURE_SCHEMA_VERSION
    label_mapping_version: str = TEXT_LABEL_MAPPING_VERSION
    privacy_ruleset_version: str = TEXT_PRIVACY_RULESET_VERSION
    source_fingerprint: str
    source_record_count: int
    output_record_count: int
    excluded_record_count: int
    empty_text_count: int
    missing_text_count: int
    exact_duplicate_group_count: int
    conflicting_duplicate_group_count: int
    near_duplicate_candidate_count: int
    label_distribution_before: Dict[str, int]
    label_distribution_after: Dict[str, int]
    privacy_replacement_summary: Dict[str, int]
    language_summary: Dict[str, int]
    length_summary: Dict[str, Any]
    leakage_checks: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("source_fingerprint")
    def validate_source_fingerprint(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("source_fingerprint must be valid SHA-256")
        return value

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")
