"""Typed schemas for Phase 2 validation results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.common import paths


_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ValidationSeverity(_StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(_StringEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class ModalityReadiness(_StringEnum):
    READY_FOR_SPLIT_DESIGN = "ready_for_split_design"
    READY_WITH_RESTRICTIONS = "ready_with_restrictions"
    BLOCKED_PENDING_DATA = "blocked_pending_data"
    BLOCKED_PENDING_LEAKAGE_RESOLUTION = "blocked_pending_leakage_resolution"
    SCORING_ONLY_NOT_ML = "scoring_only_not_ml"
    ENGINEERING_TESTS_ONLY = "engineering_tests_only"


class _SafeBaseModel(BaseModel):
    class Config:
        use_enum_values = True
        extra = "forbid"
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}

    def to_safe_dict(self) -> Dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_relative_path(value: str) -> str:
    cleaned = str(value).replace("\\", "/").strip()
    if not cleaned:
        raise ValueError("artifact path cannot be blank")
    if cleaned.startswith("/") or cleaned.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(cleaned):
        raise ValueError(f"artifact path must be repository-relative: {value}")
    parts = [part for part in cleaned.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError(f"artifact path cannot contain traversal: {value}")
    return "/".join(parts)


def repo_relative(path_like: Any) -> str:
    candidate = paths.get_repository_root() / str(path_like) if not hasattr(path_like, "is_absolute") else path_like
    try:
        path = candidate if candidate.is_absolute() else paths.get_repository_root() / candidate
        return _validate_relative_path(path.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix())
    except Exception:
        return _validate_relative_path(str(path_like))


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("validation reports cannot contain NaN or infinity")
        return value
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value


class ValidationCheckResult(_SafeBaseModel):
    check_name: str
    modality: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    artifact_paths: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)

    @validator("check_name", "modality", "message")
    def validate_non_blank(cls, value: str, field) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.name} cannot be blank")
        return value.strip()

    @validator("artifact_paths", pre=True, always=True)
    def validate_artifact_paths(cls, values) -> List[str]:
        return [_validate_relative_path(value) for value in (values or [])]

    @validator("checked_at")
    def validate_checked_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "checked_at")

    @root_validator
    def validate_blocker_recommendation(cls, values):
        status = values.get("status")
        severity = values.get("severity")
        recommendation = values.get("recommendation")
        if (str(status) == ValidationStatus.BLOCKED.value or str(severity) == ValidationSeverity.CRITICAL.value) and not recommendation:
            raise ValueError("every blocker or critical finding must include a recommendation")
        return values


class ModalityValidationResult(_SafeBaseModel):
    modality: str
    dataset_name: str
    dataset_version: Optional[str] = None
    preprocessing_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    source_fingerprint: Optional[str] = None
    source_fingerprint_verified: bool = False
    audit_status: ValidationStatus = ValidationStatus.NOT_APPLICABLE
    preprocessing_status: ValidationStatus = ValidationStatus.NOT_APPLICABLE
    row_or_file_count: int = 0
    output_record_count: int = 0
    feature_count: int = 0
    target_columns: List[str] = Field(default_factory=list)
    grouping_columns: List[str] = Field(default_factory=list)
    sensitive_columns: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)
    leakage_findings: List[str] = Field(default_factory=list)
    duplicate_findings: List[str] = Field(default_factory=list)
    missing_value_findings: List[str] = Field(default_factory=list)
    identifier_findings: List[str] = Field(default_factory=list)
    privacy_findings: List[str] = Field(default_factory=list)
    split_readiness: ValidationStatus = ValidationStatus.NOT_APPLICABLE
    model_training_readiness: ValidationStatus = ValidationStatus.NOT_APPLICABLE
    readiness_classification: ModalityReadiness
    checks: List[ValidationCheckResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)

    @validator("row_or_file_count", "output_record_count", "feature_count")
    def validate_non_negative_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value

    @root_validator
    def validate_blockers_have_recommendations(cls, values):
        blockers = values.get("blockers") or []
        checks = values.get("checks") or []
        if blockers:
            critical_checks = [
                check
                for check in checks
                if str(check.status) == ValidationStatus.BLOCKED.value or str(check.severity) == ValidationSeverity.CRITICAL.value
            ]
            if not critical_checks:
                raise ValueError("blockers require at least one blocked or critical check with a recommendation")
        return values


class CrossModalityValidationReport(_SafeBaseModel):
    validation_version: str
    readiness_policy_version: str
    generated_at: datetime = Field(default_factory=utc_now)
    modalities: List[ModalityValidationResult]
    total_checks: int = 0
    passed_checks: int = 0
    warning_checks: int = 0
    failed_checks: int = 0
    blocked_checks: int = 0
    global_findings: List[str] = Field(default_factory=list)
    global_blockers: List[str] = Field(default_factory=list)
    global_recommendations: List[str] = Field(default_factory=list)
    phase2_completion_status: str
    next_phase_recommendation: str

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

    @validator("total_checks", "passed_checks", "warning_checks", "failed_checks", "blocked_checks")
    def validate_counts(cls, value: int, field) -> int:
        if value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value


class ArtifactInventoryItem(_SafeBaseModel):
    modality: str
    artifact_type: str
    relative_path: str
    exists: bool
    size: int = 0
    sha256: Optional[str] = None
    version: Optional[str] = None
    generated_status: str = "unknown"
    source_generated_classification: str = "unknown"
    expected_classification: str = "expected"

    @validator("relative_path")
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @validator("size")
    def validate_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("size must be non-negative")
        return value
