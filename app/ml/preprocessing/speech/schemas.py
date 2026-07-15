"""Typed schemas for speech emotion preprocessing."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.preprocessing.speech.constants import (
    SPEECH_CORPUS_MAPPING_VERSION,
    SPEECH_FEATURE_SCHEMA_VERSION,
    SPEECH_LABEL_MAPPING_VERSION,
    SPEECH_PREPROCESSING_VERSION,
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


class SpeechBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> dict:
        return json.loads(self.json(exclude_none=True))


class SpeechLabelMappingEntry(SpeechBaseModel):
    corpus_name: str
    original_label: str
    canonical_label: str
    retained: bool = True
    merged: bool = False
    excluded: bool = False
    notes: str

    @validator("corpus_name", "original_label", "canonical_label", "notes")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @root_validator
    def validate_status(cls, values):
        if values.get("retained") and values.get("excluded"):
            raise ValueError("label cannot be both retained and excluded")
        return values


class SpeechLabelMappingConfig(SpeechBaseModel):
    mapping_version: str = SPEECH_LABEL_MAPPING_VERSION
    canonical_labels: List[str]
    entries: List[SpeechLabelMappingEntry]
    notes: Optional[str] = None

    @root_validator
    def validate_entries(cls, values):
        seen = set()
        duplicates = []
        for entry in values.get("entries") or []:
            key = (entry.corpus_name, entry.original_label)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            if entry.retained and not entry.excluded and entry.canonical_label not in values.get("canonical_labels", []):
                raise ValueError(f"canonical label is not declared: {entry.canonical_label}")
        if duplicates:
            raise ValueError(f"duplicate speech label mapping entries: {duplicates}")
        return values


class SpeechCorpusConfig(SpeechBaseModel):
    corpus_name: str
    source_path: str
    filename_parser: str
    speaker_id_rule: str
    emotion_label_rule: str
    gender_rule: Optional[str] = None
    expected_file_format: str = "wav"
    expected_sample_rate: Optional[int] = None
    license_note: str
    included: bool = True
    notes: Optional[str] = None

    @validator("corpus_name", "filename_parser", "speaker_id_rule", "emotion_label_rule", "expected_file_format", "license_note")
    def validate_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("source_path")
    def validate_source_path(cls, value: str) -> str:
        value = _non_blank(str(value), "source_path")
        if any(part == ".." for part in value.replace("\\", "/").split("/")):
            raise ValueError("source_path must not contain traversal")
        return value

    @validator("expected_sample_rate")
    def validate_sample_rate(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("expected_sample_rate must be positive")
        return value


class SpeechCorpusMappingConfig(SpeechBaseModel):
    mapping_version: str = SPEECH_CORPUS_MAPPING_VERSION
    corpora: List[SpeechCorpusConfig]
    notes: Optional[str] = None

    @root_validator
    def validate_corpora(cls, values):
        names = [item.corpus_name for item in values.get("corpora") or []]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate speech corpora: {duplicates}")
        return values


class SpeechSourceRecord(SpeechBaseModel):
    source_file: str
    corpus_name: str
    speaker_id: str
    original_emotion_label: str
    gender: Optional[str] = None
    intensity: Optional[str] = None
    statement_id: Optional[str] = None
    repetition_id: Optional[str] = None

    @validator("source_file")
    def validate_source_file(cls, value: str) -> str:
        return _safe_relative_path(value, "source_file")

    @validator("corpus_name", "speaker_id", "original_emotion_label")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)


class SpeechAudioMetadata(SpeechBaseModel):
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channel_count: int = 0
    sample_width: int = 0
    frame_count: int = 0
    file_format: str
    file_size_bytes: int
    readable: bool
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("duration_seconds")
    def validate_duration(cls, value: float) -> float:
        if value < 0 or not math.isfinite(float(value)):
            raise ValueError("duration_seconds must be finite and non-negative")
        return float(value)

    @validator("sample_rate", "channel_count", "sample_width", "frame_count", "file_size_bytes")
    def validate_non_negative(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @root_validator
    def validate_readable_audio(cls, values):
        if values.get("readable"):
            for name in ("duration_seconds", "sample_rate", "channel_count", "sample_width", "frame_count"):
                if values.get(name, 0) <= 0:
                    raise ValueError(f"{name} must be positive for readable audio")
        return values


class SpeechCanonicalRecord(SpeechBaseModel):
    record_id: str
    corpus_name: str
    safe_speaker_key: str
    canonical_emotion_label: str
    audio_relative_path: str
    original_audio_hash: str
    metadata: SpeechAudioMetadata
    validation_warnings: List[str] = Field(default_factory=list)

    @validator("audio_relative_path")
    def validate_audio_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value, "audio_relative_path")

    @validator("original_audio_hash")
    def validate_hash(cls, value: str) -> str:
        value = str(value).lower()
        if not _SHA256_RE.match(value):
            raise ValueError("original_audio_hash must be valid SHA-256")
        return value


class SpeechFeatureRecord(SpeechBaseModel):
    record_id: str
    safe_speaker_key: str
    corpus_name: str
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


class SpeechPreprocessingReport(SpeechBaseModel):
    preprocessing_version: str = SPEECH_PREPROCESSING_VERSION
    feature_schema_version: str = SPEECH_FEATURE_SCHEMA_VERSION
    label_mapping_version: str = SPEECH_LABEL_MAPPING_VERSION
    corpus_mapping_version: str = SPEECH_CORPUS_MAPPING_VERSION
    source_fingerprints: Dict[str, str]
    source_file_count: int
    readable_file_count: int
    unreadable_file_count: int
    output_record_count: int
    excluded_record_count: int
    corpus_distribution: Dict[str, int]
    label_distribution: Dict[str, int]
    speaker_count: int
    sample_rate_distribution: Dict[str, int]
    channel_distribution: Dict[str, int]
    duration_summary: Dict[str, float]
    duplicate_summary: Dict[str, Any]
    feature_missing_summary: Dict[str, int]
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)

    @validator("source_fingerprints")
    def validate_source_fingerprints(cls, value: Dict[str, str]) -> Dict[str, str]:
        for corpus, digest in value.items():
            if not _SHA256_RE.match(str(digest).lower()):
                raise ValueError(f"source fingerprint for {corpus} must be valid SHA-256")
            value[corpus] = str(digest).lower()
        return value

    @validator("source_file_count", "readable_file_count", "unreadable_file_count", "output_record_count", "excluded_record_count", "speaker_count")
    def validate_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")
