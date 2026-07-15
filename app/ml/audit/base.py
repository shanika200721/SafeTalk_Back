"""Shared read-only audit behavior and modality dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from app.ml.common import paths
from app.ml.common.fingerprinting import dataset_config_hash, fingerprint_dataset, verify_dataset_fingerprint
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, Modality, SupportedFileFormat
from app.ml.audit.schemas import AuditIssue, AuditSeverity, DatasetAuditReport


DEFAULT_MAX_RECORDS = 100_000
DEFAULT_MAX_FILES = 2_000
DEFAULT_SAMPLE_SEED = 42

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_USERNAME_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SENSITIVE_FIELD_HINTS = {
    "name",
    "email",
    "phone",
    "address",
    "gender",
    "religion",
    "orientation",
    "race",
    "marital",
    "married",
    "age",
    "course",
    "cgpa",
    "major",
}


@dataclass(frozen=True)
class AuditOptions:
    max_records: int = DEFAULT_MAX_RECORDS
    max_files: int = DEFAULT_MAX_FILES
    sample_seed: int = DEFAULT_SAMPLE_SEED
    summary_only: bool = False
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    expected_numeric_ranges: Mapping[str, tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    folder_label_depth: int = 1
    predefined_split_folder_names: tuple[str, ...] = ("train", "test", "validation", "val")
    minimum_audio_duration: Optional[float] = None
    maximum_audio_duration: Optional[float] = None


@dataclass(frozen=True)
class AuditContext:
    dataset_config: DatasetConfig
    source_path: Path
    fingerprint: DatasetFingerprint
    options: AuditOptions
    started_at: datetime


def coerce_audit_options(options: Optional[Mapping[str, Any] | AuditOptions]) -> AuditOptions:
    if options is None:
        return AuditOptions()
    if isinstance(options, AuditOptions):
        return options
    payload = dict(options)
    ranges = payload.get("expected_numeric_ranges") or {}
    normalized_ranges = {}
    for column, bounds in ranges.items():
        if isinstance(bounds, Mapping):
            normalized_ranges[str(column)] = (bounds.get("minimum"), bounds.get("maximum"))
        else:
            minimum, maximum = list(bounds)[:2]
            normalized_ranges[str(column)] = (minimum, maximum)
    payload["expected_numeric_ranges"] = normalized_ranges
    split_names = payload.get("predefined_split_folder_names")
    if split_names is not None:
        payload["predefined_split_folder_names"] = tuple(str(item) for item in split_names)
    return AuditOptions(**payload)


def detect_source_type(dataset_config: DatasetConfig) -> str:
    source = dataset_config.validate_source_exists()
    if source.is_dir():
        return "directory"
    if source.is_file():
        return "file"
    raise ValueError(f"Unsupported dataset source type: {source}")


def validate_fingerprint_matches_source(dataset_config: DatasetConfig, fingerprint: DatasetFingerprint) -> None:
    if not verify_dataset_fingerprint(fingerprint, dataset_config):
        raise ValueError("Dataset fingerprint mismatch: source has changed or config does not match fingerprint")


def create_audit_context(
    dataset_config: DatasetConfig,
    fingerprint: Optional[DatasetFingerprint] = None,
    options: Optional[Mapping[str, Any] | AuditOptions] = None,
) -> AuditContext:
    source_path = dataset_config.validate_source_exists()
    resolved_options = coerce_audit_options(options)
    if fingerprint is None:
        fingerprint = fingerprint_dataset(dataset_config)
    validate_fingerprint_matches_source(dataset_config, fingerprint)
    return AuditContext(
        dataset_config=dataset_config,
        source_path=source_path,
        fingerprint=fingerprint,
        options=resolved_options,
        started_at=datetime.now(timezone.utc),
    )


def summarize_audit_status(issues: list[AuditIssue]) -> str:
    severities = {issue.severity for issue in issues}
    if AuditSeverity.CRITICAL in severities:
        return "critical"
    if AuditSeverity.ERROR in severities:
        return "error"
    if AuditSeverity.WARNING in severities:
        return "warning"
    return "ok"


def _value_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:12]


def contains_sensitive_pattern(value: Any) -> bool:
    text = str(value)
    phone_match = any(len(re.sub(r"\D", "", match.group(0))) >= 8 for match in _PHONE_RE.finditer(text))
    return bool(_EMAIL_RE.search(text) or phone_match or _USERNAME_RE.search(text))


def is_sensitive_field_name(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SENSITIVE_FIELD_HINTS)


def safe_label_value(value: Any, *, field_name: Optional[str] = None, force_hash: bool = False) -> str:
    text = "" if value is None else str(value)
    if not text:
        return "<blank>"
    sensitive = force_hash or contains_sensitive_pattern(text) or (field_name is not None and is_sensitive_field_name(field_name))
    if sensitive:
        return f"<redacted:{_value_hash(text)}>"
    if len(text) > 48:
        return f"<length:{len(text)} hash:{_value_hash(text)}>"
    return text


def safe_value_summary(value: Any, *, field_name: Optional[str] = None, count: Optional[int] = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"value": safe_label_value(value, field_name=field_name)}
    if count is not None:
        summary["count"] = int(count)
    return summary


def issue(
    code: str,
    severity: AuditSeverity,
    message: str,
    *,
    field_name: Optional[str] = None,
    count: Optional[int] = None,
    details: Optional[dict[str, Any]] = None,
    recommendation: Optional[str] = None,
) -> AuditIssue:
    return AuditIssue(
        code=code,
        severity=severity,
        message=message,
        field_name=field_name,
        count=count,
        details=details or {},
        recommendation=recommendation,
    )


def build_report(
    context: AuditContext,
    *,
    modality_result_name: str,
    modality_result: Any,
    issues: list[AuditIssue],
    notes: Optional[str] = None,
) -> DatasetAuditReport:
    completed = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "dataset_name": context.dataset_config.dataset_name,
        "dataset_version": context.dataset_config.dataset_version,
        "modality": context.dataset_config.modality,
        "source_relative_path": context.fingerprint.source_relative_path,
        "source_fingerprint_hash": context.fingerprint.combined_sha256,
        "config_hash": context.fingerprint.config_hash or dataset_config_hash(context.dataset_config),
        "audit_started_at": context.started_at,
        "audit_completed_at": completed,
        "source_type": context.fingerprint.source_type,
        modality_result_name: modality_result,
        "issues": sorted(issues, key=lambda item: (severity_rank(item.severity), item.code, item.field_name or "")),
        "summary_status": summarize_audit_status(issues),
        "notes": notes,
    }
    return DatasetAuditReport(**payload)


def severity_rank(severity: AuditSeverity) -> int:
    order = {
        AuditSeverity.CRITICAL: 0,
        AuditSeverity.ERROR: 1,
        AuditSeverity.WARNING: 2,
        AuditSeverity.INFO: 3,
    }
    return order[severity]


def audit_dataset(
    dataset_config: DatasetConfig,
    fingerprint: Optional[DatasetFingerprint] = None,
    options: Optional[Mapping[str, Any] | AuditOptions] = None,
) -> DatasetAuditReport:
    """Run a read-only audit for the source described by ``dataset_config``."""
    context = create_audit_context(dataset_config, fingerprint=fingerprint, options=options)
    modality = dataset_config.modality
    file_format = dataset_config.file_format

    if modality == Modality.TEXT:
        from app.ml.audit.text import audit_text_dataset

        return audit_text_dataset(context)
    if modality == Modality.VOICE or file_format in {SupportedFileFormat.WAV, SupportedFileFormat.MP3, SupportedFileFormat.FLAC}:
        from app.ml.audit.audio import audit_audio_dataset

        return audit_audio_dataset(context)
    if modality == Modality.FACE or file_format in {SupportedFileFormat.JPG, SupportedFileFormat.JPEG, SupportedFileFormat.PNG}:
        from app.ml.audit.image import audit_image_dataset

        return audit_image_dataset(context)
    if file_format in {
        SupportedFileFormat.CSV,
        SupportedFileFormat.TSV,
        SupportedFileFormat.JSON,
        SupportedFileFormat.JSONL,
        SupportedFileFormat.XLSX,
    }:
        from app.ml.audit.tabular import audit_tabular_dataset

        return audit_tabular_dataset(context)
    if file_format == SupportedFileFormat.FOLDER:
        if modality == Modality.FACE:
            from app.ml.audit.image import audit_image_dataset

            return audit_image_dataset(context)
        if modality == Modality.VOICE:
            from app.ml.audit.audio import audit_audio_dataset

            return audit_audio_dataset(context)
    raise ValueError(f"Unsupported audit modality/file format: {modality.value}/{file_format.value}")


__all__ = [
    "AuditContext",
    "AuditOptions",
    "audit_dataset",
    "contains_sensitive_pattern",
    "create_audit_context",
    "detect_source_type",
    "safe_label_value",
    "safe_value_summary",
    "summarize_audit_status",
    "validate_fingerprint_matches_source",
]
