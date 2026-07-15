"""Typed schemas for facial emotion preprocessing."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.face.constants import (
    FACE_FEATURE_SCHEMA_VERSION,
    FACE_IMAGE_POLICY_VERSION,
    FACE_LABEL_MAPPING_VERSION,
    FACE_PREPROCESSING_VERSION,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH_RE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be blank")
    return value.strip()


def _safe_relative_path(value: str, field_name: str) -> str:
    value = _non_blank(str(value).replace("\\", "/"), field_name)
    if _ABSOLUTE_PATH_RE.match(value):
        raise ValueError(f"{field_name} must not be absolute")
    if any(part == ".." for part in value.split("/")):
        raise ValueError(f"{field_name} must not contain traversal")
    return value


class FaceBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class FaceLabelMappingEntry(FaceBaseModel):
    original_label: str
    canonical_label: str
    retained: bool = True
    merged: bool = False
    excluded: bool = False
    notes: str

    @validator("original_label", "canonical_label", "notes")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @root_validator
    def validate_status(cls, values):
        if values.get("merged"):
            raise ValueError("facial label mapping must not silently merge classes")
        if values.get("retained") and values.get("excluded"):
            raise ValueError("label cannot be both retained and excluded")
        return values


class FaceLabelMappingConfig(FaceBaseModel):
    mapping_version: str = FACE_LABEL_MAPPING_VERSION
    canonical_labels: List[str]
    entries: List[FaceLabelMappingEntry]
    notes: Optional[str] = None

    @root_validator
    def validate_entries(cls, values):
        originals = [entry.original_label for entry in values.get("entries") or []]
        duplicates = sorted({label for label in originals if originals.count(label) > 1})
        if duplicates:
            raise ValueError(f"duplicate face label mapping entries: {duplicates}")
        for entry in values.get("entries") or []:
            if entry.retained and not entry.excluded and entry.canonical_label not in values.get("canonical_labels", []):
                raise ValueError(f"canonical label is not declared: {entry.canonical_label}")
        return values


class FaceSourceStructureConfig(FaceBaseModel):
    structure_version: str = "1.0.0"
    dataset_name: str = "facial-emotion"
    dataset_root: str
    predefined_split_folders: List[str]
    class_folder_depth: int = 1
    supported_image_extensions: List[str]
    subject_id_available: bool = False
    filename_parsing_rule: str
    duplicate_policy: str
    corruption_policy: str
    inclusion_exclusion_notes: str
    license_note: str

    @validator("dataset_root", "filename_parsing_rule", "duplicate_policy", "corruption_policy", "inclusion_exclusion_notes", "license_note")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("class_folder_depth")
    def validate_depth(cls, value: int) -> int:
        if value < 1:
            raise ValueError("class_folder_depth must be positive")
        return value


class FaceSourceRecord(FaceBaseModel):
    source_file: str
    source_split: str
    original_label: str
    subject_id: Optional[str] = None
    original_id: Optional[str] = None

    @validator("source_file")
    def validate_source_file(cls, value: str) -> str:
        return _safe_relative_path(value, "source_file")

    @validator("source_split", "original_label")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)


class FaceImageMetadata(FaceBaseModel):
    width: int = 0
    height: int = 0
    color_mode: str = "unknown"
    file_format: str = "unknown"
    file_size_bytes: int
    readable: bool
    image_hash: str
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("width", "height")
    def validate_dimensions(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @validator("file_size_bytes")
    def validate_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("file_size_bytes must be non-negative")
        return value

    @validator("image_hash")
    def validate_hash(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("image_hash must be valid SHA-256")
        return value

    @root_validator
    def validate_readable_image(cls, values):
        if values.get("readable") and (values.get("width", 0) <= 0 or values.get("height", 0) <= 0):
            raise ValueError("readable images must have positive width and height")
        return values


class FaceCanonicalRecord(FaceBaseModel):
    record_id: str
    source_split: str
    original_label: str
    canonical_emotion_label: str
    image_relative_path: str
    image_hash: str
    metadata: FaceImageMetadata
    safe_subject_key: Optional[str] = None
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("image_relative_path")
    def validate_image_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value, "image_relative_path")

    @validator("image_hash")
    def validate_hash(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("image_hash must be valid SHA-256")
        return value


class FaceFeatureRecord(FaceBaseModel):
    record_id: str
    canonical_emotion_label: str
    feature_values: Dict[str, float]
    feature_extraction_warnings: List[str] = Field(default_factory=list)

    @validator("feature_values")
    def validate_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        for name, raw in value.items():
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ValueError(f"feature value must be finite: {name}")
            value[name] = numeric
        return value


class FaceDuplicateGroup(FaceBaseModel):
    image_hash: str
    record_ids: List[str]
    source_splits: List[str]
    labels: List[str]
    cross_split: bool
    cross_label: bool

    @validator("image_hash")
    def validate_hash(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("image_hash must be valid SHA-256")
        return value


class FacePreprocessingReport(FaceBaseModel):
    preprocessing_version: str = FACE_PREPROCESSING_VERSION
    feature_schema_version: str = FACE_FEATURE_SCHEMA_VERSION
    label_mapping_version: str = FACE_LABEL_MAPPING_VERSION
    image_policy_version: str = FACE_IMAGE_POLICY_VERSION
    source_fingerprint: str
    source_file_count: int
    readable_file_count: int
    unreadable_file_count: int
    output_record_count: int
    excluded_record_count: int
    split_distribution: Dict[str, int]
    label_distribution: Dict[str, int]
    width_summary: Dict[str, float]
    height_summary: Dict[str, float]
    color_mode_distribution: Dict[str, int]
    format_distribution: Dict[str, int]
    duplicate_group_count: int
    cross_split_duplicate_count: int
    cross_label_duplicate_count: int
    corrupt_file_count: int
    zero_byte_file_count: int
    subject_count: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("source_fingerprint")
    def validate_source_fingerprint(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("source_fingerprint must be valid SHA-256")
        return value

    @validator(
        "source_file_count",
        "readable_file_count",
        "unreadable_file_count",
        "output_record_count",
        "excluded_record_count",
        "duplicate_group_count",
        "cross_split_duplicate_count",
        "cross_label_duplicate_count",
        "corrupt_file_count",
        "zero_byte_file_count",
        "subject_count",
    )
    def validate_counts(cls, value: Optional[int], field) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

