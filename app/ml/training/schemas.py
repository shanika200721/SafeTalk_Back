"""Typed schemas for Phase 3B training, evaluation, and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.common import paths
from app.ml.training.constants import (
    ARTIFACT_MANIFEST_VERSION,
    CLINICAL_DISCLAIMER,
    EVALUATION_SCHEMA_VERSION,
    MODEL_CARD_VERSION,
    PRODUCTION_IDENTIFIER_COLUMNS,
    TRAINING_FRAMEWORK_VERSION,
)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TrainingTask(_StringEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class TrainingStatus(_StringEnum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelCandidateStatus(_StringEnum):
    CANDIDATE = "candidate"
    REGISTERED = "registered"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ThresholdStrategy(_StringEnum):
    DEFAULT = "default"
    MAX_F1 = "max_f1"
    RECALL_PRIORITY = "recall_priority"
    PRECISION_PRIORITY = "precision_priority"
    COST_SENSITIVE = "cost_sensitive"
    FIXED = "fixed"


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


def _dedupe(values: List[str] | None, field_name: str) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for raw in values or []:
        value = _non_blank(str(raw), field_name)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_relative_path(value: str, field_name: str, *, allow_model_root: bool = True) -> str:
    cleaned = _non_blank(str(value).replace("\\", "/"), field_name)
    if cleaned.startswith("/") or cleaned.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(cleaned):
        raise ValueError(f"{field_name} must be repository-relative or under MODEL_ROOT")
    if any(part == ".." for part in cleaned.split("/") if part):
        raise ValueError(f"{field_name} cannot contain traversal")
    if allow_model_root:
        return cleaned
    if cleaned.startswith("ml_models/"):
        raise ValueError(f"{field_name} must not point under MODEL_ROOT")
    return cleaned


def _validate_hash(value: str, field_name: str) -> str:
    value = _non_blank(value, field_name).lower()
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return value


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value


class SafeTrainingModel(BaseModel):
    class Config:
        use_enum_values = False
        json_encoders = {datetime: lambda value: value.astimezone(timezone.utc).isoformat()}
        extra = "forbid"

    def to_safe_dict(self) -> Dict[str, Any]:
        return _safe_serialize(self.dict(exclude_none=True))


class TrainingConfig(SafeTrainingModel):
    experiment_name: str
    experiment_version: str
    model_name: str
    model_version: str
    modality: str
    task: TrainingTask
    framework: str
    estimator_type: str
    dataset_name: str
    dataset_version: str
    preprocessing_version: str
    feature_schema_version: str
    split_manifest_path: str
    random_seed: int
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    class_weight_policy: Optional[str] = None
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.DEFAULT
    target_column: str
    feature_columns: List[str]
    excluded_columns: List[str] = Field(default_factory=list)
    sensitive_columns: List[str] = Field(default_factory=list)
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    artifact_subdirectory: str
    notes: Optional[str] = None

    @validator(
        "experiment_name",
        "experiment_version",
        "model_name",
        "model_version",
        "modality",
        "framework",
        "estimator_type",
        "dataset_name",
        "dataset_version",
        "preprocessing_version",
        "feature_schema_version",
        "target_column",
        "primary_metric",
    )
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("split_manifest_path", "artifact_subdirectory")
    def validate_paths(cls, value: str, field) -> str:
        return _safe_relative_path(value, field.name)

    @validator("random_seed")
    def validate_random_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed must be non-negative")
        return value

    @validator("feature_columns", "excluded_columns", "sensitive_columns", "secondary_metrics", pre=True, always=True)
    def validate_lists(cls, values, field) -> List[str]:
        return _dedupe(values or [], field.name)

    @root_validator
    def validate_training_config(cls, values):
        target = values.get("target_column")
        features = set(values.get("feature_columns") or [])
        excluded = set(values.get("excluded_columns") or [])
        sensitive = values.get("sensitive_columns")
        if not features:
            raise ValueError("feature_columns must not be empty")
        if target in features:
            raise ValueError("target must not appear among feature columns")
        overlap = excluded & features
        if overlap:
            raise ValueError(f"excluded columns must not appear among features: {sorted(overlap)}")
        if sensitive is None:
            raise ValueError("sensitive columns must be explicitly documented")
        forbidden = PRODUCTION_IDENTIFIER_COLUMNS & (features | {target})
        if forbidden:
            raise ValueError(f"production identifiers are not allowed in training schemas: {sorted(forbidden)}")
        return values


class DatasetSplitReference(SafeTrainingModel):
    train_ids: List[str]
    validation_ids: List[str]
    test_ids: List[str]
    manifest_hash: str
    source_fingerprint: str
    preprocessing_artifact_hash: str

    @validator("train_ids", "validation_ids", "test_ids", pre=True, always=True)
    def validate_ids(cls, values, field) -> List[str]:
        return _dedupe([str(value) for value in (values or [])], field.name)

    @validator("manifest_hash", "source_fingerprint", "preprocessing_artifact_hash")
    def validate_hashes(cls, value: str, field) -> str:
        return _validate_hash(value, field.name)


class MetricSet(SafeTrainingModel):
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_macro: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    per_class_metrics: Dict[str, Dict[str, float | int | None]] = Field(default_factory=dict)
    support: Dict[str, int] = Field(default_factory=dict)
    threshold: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    false_negative_count: Optional[int] = None
    false_positive_count: Optional[int] = None


class TrainingRunResult(SafeTrainingModel):
    run_id: str
    status: TrainingStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    training_framework_version: str = TRAINING_FRAMEWORK_VERSION
    config_hash: str
    dataset_reference: DatasetSplitReference
    model_artifact_path: Optional[str] = None
    preprocessor_artifact_path: Optional[str] = None
    metrics_path: Optional[str] = None
    model_card_path: Optional[str] = None
    artifact_manifest_path: Optional[str] = None
    train_metrics: Optional[MetricSet] = None
    validation_metrics: Optional[MetricSet] = None
    test_metrics: Optional[MetricSet] = None
    selected_thresholds: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None

    @validator("started_at", "completed_at")
    def validate_times(cls, value: Optional[datetime], field):
        if value is None:
            return value
        return _timezone_aware(value, field.name)

    @validator("config_hash")
    def validate_config_hash(cls, value: str) -> str:
        return _validate_hash(value, "config_hash")

    @validator("model_artifact_path", "preprocessor_artifact_path", "metrics_path", "model_card_path", "artifact_manifest_path")
    def validate_optional_paths(cls, value: Optional[str], field) -> Optional[str]:
        if value is None:
            return value
        return _safe_relative_path(value, field.name)

    @root_validator(pre=True)
    def reject_activation_fields(cls, values):
        if "activate" in values or "is_active" in values:
            raise ValueError("model activation must not be part of TrainingRunResult")
        return values


class ArtifactManifest(SafeTrainingModel):
    manifest_version: str = ARTIFACT_MANIFEST_VERSION
    run_id: str
    model_name: str
    model_version: str
    modality: str
    files: List[str]
    file_hashes: Dict[str, str]
    dataset_version: str
    preprocessing_version: str
    feature_schema_version: str
    split_manifest_hash: str
    config_hash: str
    created_at: datetime

    @validator("manifest_version")
    def validate_manifest_version(cls, value: str) -> str:
        if value != ARTIFACT_MANIFEST_VERSION:
            raise ValueError("unsupported artifact manifest version")
        return value

    @validator("created_at")
    def validate_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "created_at")

    @validator("files", pre=True, always=True)
    def validate_files(cls, values) -> List[str]:
        return [_safe_relative_path(str(value), "files") for value in values or []]

    @validator("split_manifest_hash", "config_hash")
    def validate_hashes(cls, value: str, field) -> str:
        return _validate_hash(value, field.name)

    @validator("file_hashes")
    def validate_file_hashes(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            raise ValueError("file_hashes cannot be empty")
        return {_safe_relative_path(key, "file_hashes"): _validate_hash(hash_value, "file_hash") for key, hash_value in value.items()}


class ModelCard(SafeTrainingModel):
    model_card_version: str = MODEL_CARD_VERSION
    model_name: str
    model_version: str
    modality: str
    intended_use: str
    prohibited_use: str
    dataset_summary: str
    preprocessing_summary: str
    split_summary: str
    model_description: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    threshold_policy: str
    fairness_considerations: str
    privacy_considerations: str
    limitations: List[str]
    ethical_warnings: List[str]
    human_oversight_requirement: str
    clinical_disclaimer: str
    created_at: datetime

    @validator("created_at")
    def validate_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "created_at")

    @validator("clinical_disclaimer")
    def validate_disclaimer(cls, value: str) -> str:
        if CLINICAL_DISCLAIMER not in value:
            raise ValueError("model cards must include the required clinical disclaimer")
        return value

    @validator("model_card_version")
    def validate_card_version(cls, value: str) -> str:
        if value != MODEL_CARD_VERSION:
            raise ValueError("unsupported model card version")
        return value


class EvaluationReport(SafeTrainingModel):
    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION
    train_metrics: MetricSet
    validation_metrics: MetricSet
    test_metrics: MetricSet
    selected_thresholds: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
