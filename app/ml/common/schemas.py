"""Typed Phase 2 ML configuration schemas.

These schemas intentionally describe datasets, preprocessing, split manifests,
and feature contracts only. They do not load datasets, calculate hashes, run
preprocessing, or train models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.common import paths


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Modality(_StringEnum):
    PROFILE = "profile"
    DASS21 = "dass21"
    MOOD = "mood"
    TEXT = "text"
    VOICE = "voice"
    FACE = "face"
    BEHAVIORAL = "behavioral"
    FUSION = "fusion"


class SupportedFileFormat(_StringEnum):
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"
    XLSX = "xlsx"
    TXT = "txt"
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    FOLDER = "folder"


class MissingValuePolicy(_StringEnum):
    ERROR = "error"
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    IMPUTE = "impute"
    PRESERVE = "preserve"


class DuplicatePolicy(_StringEnum):
    ERROR = "error"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    REMOVE_EXACT = "remove_exact"
    REPORT_ONLY = "report_only"


class NormalizationMethod(_StringEnum):
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    PER_SAMPLE = "per_sample"


class CategoricalEncoding(_StringEnum):
    NONE = "none"
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    TARGET = "target"
    LABEL = "label"


class OutputFormat(_StringEnum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    NPZ = "npz"


def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be blank")
    return value.strip()


def _deduplicate_non_blank(values: Iterable[Any], field_name: str) -> List[str]:
    seen = set()
    result: List[str] = []
    for raw in values or []:
        value = str(raw).strip()
        if not value:
            raise ValueError(f"{field_name} cannot contain blank column names")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _path_has_traversal(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve(strict=False)
    return (paths.get_repository_root() / path).resolve(strict=False)


def _timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


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
            return str(value.relative_to(paths.get_repository_root()))
        except ValueError:
            return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    return value


class DatasetConfig(_SafeBaseModel):
    dataset_name: str
    dataset_version: str
    modality: Modality
    source_path: Path
    file_format: SupportedFileFormat
    label_columns: List[str] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)
    identifier_columns: List[str] = Field(default_factory=list)
    sensitive_columns: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)
    expected_columns: List[str] = Field(default_factory=list)
    missing_value_policy: MissingValuePolicy
    duplicate_policy: DuplicatePolicy
    notes: Optional[str] = None
    source_description: Optional[str] = None
    is_raw_source: bool = True
    validation_context: Optional[str] = Field(default=None, exclude=True)

    @validator("dataset_name", "dataset_version")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator(
        "label_columns",
        "feature_columns",
        "identifier_columns",
        "sensitive_columns",
        "excluded_columns",
        "expected_columns",
        pre=True,
        always=True,
    )
    def validate_column_lists(cls, values, field) -> List[str]:
        return _deduplicate_non_blank(values or [], field.name)

    @validator("source_path")
    def validate_source_path_syntax(cls, value: Path) -> Path:
        if _path_has_traversal(value):
            raise ValueError("source_path cannot contain path traversal")
        return value

    @root_validator
    def validate_source_policy_and_columns(cls, values):
        source_path: Path = values.get("source_path")
        if source_path is not None:
            resolved = _resolve_project_path(source_path)
            blocked_roots = (
                paths.get_generated_root(),
                paths.get_model_root(),
                paths.get_ml_research_root() / "reports",
            )
            for blocked_root in blocked_roots:
                if paths.is_path_inside(blocked_root, resolved):
                    raise ValueError(f"source_path cannot point inside {blocked_root}")

            is_raw_source = values.get("is_raw_source", True)
            context = values.get("validation_context")
            if (
                is_raw_source
                and not paths.is_path_inside(paths.get_raw_dataset_root(), resolved)
                and context != "test"
            ):
                raise ValueError("raw source_path must be inside Final Dataset/ unless validation_context='test'")

        labels = set(values.get("label_columns") or [])
        excluded = set(values.get("excluded_columns") or [])
        overlap = labels & excluded
        if overlap:
            raise ValueError(f"label columns cannot overlap excluded columns: {sorted(overlap)}")

        expected = values.get("expected_columns") or []
        expected_seen = set(expected)
        for column in (
            (values.get("label_columns") or [])
            + (values.get("feature_columns") or [])
            + (values.get("identifier_columns") or [])
        ):
            if column not in expected_seen:
                expected.append(column)
                expected_seen.add(column)
        values["expected_columns"] = expected
        return values

    def resolved_source_path(self) -> Path:
        return _resolve_project_path(self.source_path)

    def validate_source_exists(self) -> Path:
        resolved = self.resolved_source_path()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {resolved}")
        return resolved

    def all_declared_columns(self) -> List[str]:
        return _deduplicate_non_blank(
            self.label_columns
            + self.feature_columns
            + self.identifier_columns
            + self.sensitive_columns
            + self.excluded_columns
            + self.expected_columns,
            "declared_columns",
        )

    def ml_feature_columns(self) -> List[str]:
        blocked = set(self.label_columns) | set(self.identifier_columns) | set(self.sensitive_columns) | set(self.excluded_columns)
        return [column for column in self.feature_columns if column not in blocked]


def _validate_sha256(value: str, field_name: str) -> str:
    value = _non_blank(value, field_name).lower()
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field_name} must be exactly 64 hexadecimal characters")
    return value


def _validate_relative_report_path(value: str, field_name: str) -> str:
    value = _non_blank(str(value).replace("\\", "/"), field_name)
    if value.startswith("/") or value.startswith("//") or _WINDOWS_ABSOLUTE_RE.match(value):
        raise ValueError(f"{field_name} must be relative")
    parts = [part for part in value.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError(f"{field_name} cannot contain traversal")
    return "/".join(parts)


class FileFingerprint(_SafeBaseModel):
    relative_path: str
    sha256: str
    size_bytes: int
    extension: str
    modified_time_utc: Optional[datetime] = None
    media_type: Optional[str] = None

    @validator("relative_path")
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_report_path(value, "relative_path")

    @validator("sha256")
    def validate_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "sha256")

    @validator("size_bytes")
    def validate_size_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("size_bytes must be non-negative")
        return value

    @validator("extension")
    def validate_extension(cls, value: str) -> str:
        cleaned = _non_blank(value.lower().lstrip("."), "extension")
        if "/" in cleaned or "\\" in cleaned:
            raise ValueError("extension must not contain path separators")
        return cleaned

    @validator("modified_time_utc")
    def validate_modified_time(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        return _timezone_aware(value, "modified_time_utc")


class SkippedFingerprintFile(_SafeBaseModel):
    relative_path: str
    reason: str

    @validator("relative_path")
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_report_path(value, "relative_path")

    @validator("reason")
    def validate_reason(cls, value: str) -> str:
        return _non_blank(value, "reason")


class DatasetFingerprint(_SafeBaseModel):
    dataset_name: str
    dataset_version: str
    modality: Modality
    source_relative_path: str
    source_type: str
    file_count: int
    total_bytes: int
    combined_sha256: str
    files: List[FileFingerprint]
    skipped_files: List[SkippedFingerprintFile] = Field(default_factory=list)
    generated_at: datetime
    fingerprint_version: str
    config_hash: Optional[str] = None
    notes: Optional[str] = None
    allow_empty: bool = Field(default=False, exclude=True)

    @validator("dataset_name", "dataset_version", "fingerprint_version")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("source_relative_path")
    def validate_source_relative_path(cls, value: str) -> str:
        return _validate_relative_report_path(value, "source_relative_path")

    @validator("source_type")
    def validate_source_type(cls, value: str) -> str:
        value = _non_blank(value, "source_type")
        if value not in {"file", "directory"}:
            raise ValueError("source_type must be 'file' or 'directory'")
        return value

    @validator("file_count")
    def validate_file_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("file_count must be non-negative")
        return value

    @validator("total_bytes")
    def validate_total_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("total_bytes must be non-negative")
        return value

    @validator("combined_sha256")
    def validate_combined_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "combined_sha256")

    @validator("config_hash")
    def validate_config_hash(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return _validate_sha256(value, "config_hash")

    @validator("generated_at")
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "generated_at")

    @root_validator
    def validate_fingerprint_integrity(cls, values):
        files: List[FileFingerprint] = values.get("files") or []
        file_count = values.get("file_count")
        total_bytes = values.get("total_bytes")
        allow_empty = values.get("allow_empty", False)

        if not files and not allow_empty:
            raise ValueError("Dataset fingerprint must contain at least one file")
        if file_count != len(files):
            raise ValueError("file_count must equal the number of included files")
        summed_bytes = sum(file.size_bytes for file in files)
        if total_bytes != summed_bytes:
            raise ValueError("total_bytes must equal the sum of included file sizes")

        relative_paths = [file.relative_path for file in files]
        duplicates = sorted({relative_path for relative_path in relative_paths if relative_paths.count(relative_path) > 1})
        if duplicates:
            raise ValueError(f"file relative paths must be unique: {duplicates}")
        return values

    def file_map(self) -> Dict[str, FileFingerprint]:
        return {file.relative_path: file for file in self.files}

    def duplicate_hash_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for file in self.files:
            groups.setdefault(file.sha256, []).append(file.relative_path)
        return {sha256: sorted(relative_paths) for sha256, relative_paths in groups.items() if len(relative_paths) >= 2}

    def total_megabytes(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def verify_current_source(self, source_root: Path) -> bool:
        from app.ml.common.fingerprinting import verify_fingerprint_against_path

        return verify_fingerprint_against_path(self, source_root)


class PreprocessingConfig(_SafeBaseModel):
    preprocessing_name: str
    preprocessing_version: str
    dataset_name: str
    dataset_version: str
    modality: Modality
    random_seed: int
    test_size: float
    validation_size: float
    stratify_column: Optional[str] = None
    group_column: Optional[str] = None
    normalization_method: NormalizationMethod
    categorical_encoding: CategoricalEncoding
    text_cleaning_options: Dict[str, Any] = Field(default_factory=dict)
    audio_options: Dict[str, Any] = Field(default_factory=dict)
    image_options: Dict[str, Any] = Field(default_factory=dict)
    output_format: OutputFormat
    feature_schema_version: str
    output_subdirectory: Path
    notes: Optional[str] = None

    @validator("preprocessing_name", "preprocessing_version", "dataset_name", "dataset_version", "feature_schema_version")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("random_seed")
    def validate_random_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed must be non-negative")
        return value

    @validator("test_size")
    def validate_test_size(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("test_size must be greater than 0 and less than 1")
        return value

    @validator("validation_size")
    def validate_validation_size(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("validation_size must be 0 or greater and less than 1")
        return value

    @validator("stratify_column", "group_column")
    def validate_optional_column_names(cls, value: Optional[str], field) -> Optional[str]:
        if value is None:
            return value
        return _non_blank(value, field.name)

    @validator("output_subdirectory")
    def validate_output_subdirectory(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError("output_subdirectory must be relative")
        if _path_has_traversal(value):
            raise ValueError("output_subdirectory cannot contain path traversal")
        resolved = (paths.get_generated_preprocessing_root() / value).resolve(strict=False)
        if paths.is_path_inside(paths.get_raw_dataset_root(), resolved):
            raise ValueError("output_subdirectory cannot resolve inside Final Dataset/")
        return value

    @root_validator
    def validate_splits_and_modality_options(cls, values):
        test_size = values.get("test_size")
        validation_size = values.get("validation_size")
        if test_size is not None and validation_size is not None and test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")

        modality = values.get("modality")
        if modality is not None:
            if values.get("text_cleaning_options") and modality != Modality.TEXT:
                raise ValueError("text_cleaning_options are only meaningful for text modality")
            if values.get("audio_options") and modality != Modality.VOICE:
                raise ValueError("audio_options are only meaningful for voice modality")
            if values.get("image_options") and modality != Modality.FACE:
                raise ValueError("image_options are only meaningful for face modality")
        return values

    @property
    def train_size(self) -> float:
        return round(1.0 - self.test_size - self.validation_size, 10)

    def split_percentages(self) -> Dict[str, float]:
        return {
            "train": self.train_size,
            "validation": self.validation_size,
            "test": self.test_size,
        }

    def resolved_output_path(self) -> Path:
        return paths.resolve_generated_path(Path("preprocessing") / self.output_subdirectory)

    def config_identity(self) -> str:
        return (
            f"{self.dataset_name}:{self.dataset_version}:"
            f"{self.preprocessing_name}:{self.preprocessing_version}:"
            f"{self.feature_schema_version}"
        )


class SplitManifest(_SafeBaseModel):
    dataset_name: str
    dataset_version: str
    preprocessing_name: str
    preprocessing_version: str
    feature_schema_version: str
    modality: Modality
    random_seed: int
    train_ids: List[str]
    validation_ids: List[str] = Field(default_factory=list)
    test_ids: List[str]
    split_created_at: datetime
    source_hash: str
    config_hash: str
    grouping_column: Optional[str] = None
    stratify_column: Optional[str] = None
    notes: Optional[str] = None

    @validator("dataset_name", "dataset_version", "preprocessing_name", "preprocessing_version", "feature_schema_version", "source_hash", "config_hash")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("random_seed")
    def validate_random_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed must be non-negative")
        return value

    @validator("train_ids", "validation_ids", "test_ids", pre=True, always=True)
    def coerce_ids(cls, values, field) -> List[str]:
        result = []
        seen = set()
        for raw in values or []:
            value = str(raw)
            if not value:
                raise ValueError(f"{field.name} cannot contain blank IDs")
            if value in seen:
                raise ValueError(f"{field.name} contains duplicate ID: {value}")
            seen.add(value)
            result.append(value)
        return result

    @validator("split_created_at")
    def validate_split_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "split_created_at")

    @root_validator
    def validate_split_integrity(cls, values):
        if not values.get("train_ids"):
            raise ValueError("train_ids must not be empty")
        if not values.get("test_ids"):
            raise ValueError("test_ids must not be empty")
        manifest = cls.construct(**values)
        manifest.assert_no_overlap()
        return values

    def total_records(self) -> int:
        return len(self.train_ids) + len(self.validation_ids) + len(self.test_ids)

    def split_counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train_ids),
            "validation": len(self.validation_ids),
            "test": len(self.test_ids),
        }

    def assert_no_overlap(self) -> None:
        splits = {
            "train": set(self.train_ids),
            "validation": set(self.validation_ids),
            "test": set(self.test_ids),
        }
        pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
        for left, right in pairs:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(f"split IDs overlap between {left} and {right}: {sorted(overlap)}")

    def contains_id(self, record_id: Any) -> bool:
        value = str(record_id)
        return value in self.train_ids or value in self.validation_ids or value in self.test_ids


class FeatureDefinition(_SafeBaseModel):
    name: str
    dtype: str
    description: str
    source_columns: List[str]
    nullable: bool
    category_values: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    preprocessing_step: Optional[str] = None

    @validator("name", "dtype", "description")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("source_columns", pre=True, always=True)
    def validate_source_columns(cls, values) -> List[str]:
        return _deduplicate_non_blank(values or [], "source_columns")

    @validator("category_values", pre=True, always=True)
    def validate_category_values(cls, values) -> Optional[List[str]]:
        if values is None:
            return None
        return _deduplicate_non_blank(values, "category_values")

    @root_validator
    def validate_numeric_range(cls, values):
        minimum = values.get("minimum")
        maximum = values.get("maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum cannot exceed maximum")
        return values


class FeatureSchema(_SafeBaseModel):
    schema_name: str
    feature_schema_version: str
    dataset_name: str
    dataset_version: str
    preprocessing_version: str
    modality: Modality
    features: List[FeatureDefinition]
    target_columns: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)
    created_at: datetime
    schema_hash: Optional[str] = None
    notes: Optional[str] = None

    @validator("schema_name", "feature_schema_version", "dataset_name", "dataset_version", "preprocessing_version")
    def validate_required_strings(cls, value: str, field) -> str:
        return _non_blank(value, field.name)

    @validator("target_columns", "excluded_columns", pre=True, always=True)
    def validate_column_lists(cls, values, field) -> List[str]:
        return _deduplicate_non_blank(values or [], field.name)

    @validator("created_at")
    def validate_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, "created_at")

    @root_validator
    def validate_feature_contract(cls, values):
        features = values.get("features") or []
        names = [feature.name for feature in features]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"feature names must be unique: {duplicates}")

        feature_names = set(names)
        target_overlap = feature_names & set(values.get("target_columns") or [])
        if target_overlap:
            raise ValueError(f"target columns cannot appear as model features: {sorted(target_overlap)}")

        excluded_overlap = feature_names & set(values.get("excluded_columns") or [])
        if excluded_overlap:
            raise ValueError(f"excluded columns cannot appear as model features: {sorted(excluded_overlap)}")
        return values

    def feature_names(self) -> List[str]:
        return [feature.name for feature in self.features]

    def required_source_columns(self) -> List[str]:
        columns: List[str] = []
        seen = set()
        for feature in self.features:
            for column in feature.source_columns:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
        return columns

    def validate_dataframe_columns(self, columns: Iterable[str]) -> None:
        available = {str(column) for column in columns}
        missing = [column for column in self.required_source_columns() if column not in available]
        if missing:
            raise ValueError(f"Missing required source columns: {missing}")


class DatasetReference(_SafeBaseModel):
    """Compatibility reference for source datasets."""

    name: str
    relative_path: str
    modality: Modality
    version: Optional[str] = None


class ModelArtifactMetadata(_SafeBaseModel):
    """Minimal metadata expected beside a saved model artifact."""

    model_name: str
    modality: Modality
    version: str
    framework: str
    artifact_relative_path: str
    preprocessing_relative_path: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
