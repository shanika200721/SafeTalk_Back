"""Deterministic canonical preprocessing for the Student Profile dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import paths
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, FeatureDefinition, FeatureSchema, Modality, PreprocessingConfig
from app.ml.preprocessing.profile.constants import (
    CANONICAL_COLUMNS,
    CGPA_ALLOWED_VALUES,
    DEFAULT_EXCLUDED_SOURCE_COLUMNS,
    PROFILE_FEATURE_SCHEMA_VERSION,
    PROFILE_MAPPING_VERSION,
    PROFILE_PREPROCESSING_VERSION,
    RECORD_ID_PREFIX,
    SOURCE_COLUMNS,
    TARGET_COLUMN,
    TARGET_CANONICAL_COLUMN,
    TREATMENT_COLUMN,
)
from app.ml.preprocessing.profile.mapping import mapping_by_source, selected_feature_source_columns
from app.ml.preprocessing.profile.schemas import (
    ProfileCanonicalRecord,
    ProfileFieldRole,
    ProfileMappingConfig,
    ProfilePreprocessingReport,
)
from app.ml.preprocessing.profile.validation import (
    detect_profile_identifiers,
    detect_profile_leakage,
    validate_profile_categories,
    validate_profile_mapping,
    validate_profile_missing_values,
    validate_profile_numeric_ranges,
    validate_profile_source_columns,
    validate_profile_target_values,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def normalize_binary_label(value: Any, *, field_name: str = "binary label") -> str:
    text = _clean_text(value)
    if text is None:
        raise ValueError(f"{field_name} is missing")
    normalized = text.lower()
    if normalized in {"yes", "y"}:
        return "yes"
    if normalized in {"no", "n"}:
        return "no"
    raise ValueError(f"{field_name} contains unrecognized value: {text}")


def parse_age(value: Any) -> float | None:
    if pd.isna(value) or _clean_text(value) is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Age must be numeric: {value}") from exc
    if not math.isfinite(numeric):
        raise ValueError("Age must not be infinite")
    if numeric < 10 or numeric > 120:
        raise ValueError(f"Age is outside expected human range: {numeric}")
    if not numeric.is_integer():
        raise ValueError(f"Age must be a whole number: {numeric}")
    return numeric


def parse_cgpa(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text.replace("–", "-")
    normalized = " - ".join(part.strip() for part in normalized.split("-"))
    if normalized not in CGPA_ALLOWED_VALUES:
        raise ValueError(f"CGPA range is not recognized: {text}")
    return normalized


def normalize_year_of_study(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text.lower()
    normalized = normalized.replace("year", "year ").replace("  ", " ").strip()
    if normalized not in {"year 1", "year 2", "year 3", "year 4"}:
        raise ValueError(f"Year of study is not recognized: {text}")
    return normalized


def generate_record_id(source_row_index: int, source_fingerprint: str) -> str:
    if source_row_index < 0:
        raise ValueError("source_row_index must be non-negative")
    if not source_fingerprint or len(source_fingerprint) < 12:
        raise ValueError("source_fingerprint must be available for deterministic record IDs")
    row_number = source_row_index + 1
    digest = hashlib.sha256(f"{RECORD_ID_PREFIX}:{source_fingerprint}:{row_number}".encode("utf-8")).hexdigest()[:10]
    return f"{RECORD_ID_PREFIX}-{row_number:06d}-{digest}"


def load_profile_source(dataset_config: DatasetConfig) -> pd.DataFrame:
    source_path = dataset_config.validate_source_exists()
    return pd.read_csv(source_path)


def normalize_profile_categories(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(lambda value: _clean_text(value) if not pd.isna(value) else value)
    return normalized


def _normalize_source_column(source_column: str, value: Any) -> Any:
    if source_column == "Age":
        return parse_age(value)
    if source_column == "What is your CGPA?":
        return parse_cgpa(value)
    if source_column == "Your current year of Study":
        return normalize_year_of_study(value)
    if source_column in {
        TARGET_COLUMN,
        "Do you have Anxiety?",
        "Do you have Panic attack?",
        "Marital status",
        TREATMENT_COLUMN,
    }:
        return normalize_binary_label(value, field_name=source_column)
    text = _clean_text(value)
    return text.lower() if source_column == "Choose your gender" and text is not None else text


def canonicalize_profile_dataframe(
    df: pd.DataFrame,
    mapping_config: ProfileMappingConfig,
    *,
    source_fingerprint: str,
) -> pd.DataFrame:
    validate_profile_source_columns(df.columns, mapping_config.source_columns)
    validate_profile_target_values(df, mapping_config.target_column)
    validate_profile_numeric_ranges(df)

    canonical_rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        output: dict[str, Any] = {
            "record_id": generate_record_id(int(index), source_fingerprint),
        }
        for source_column in mapping_config.source_columns:
            canonical_name = CANONICAL_COLUMNS[source_column]
            output[canonical_name] = _normalize_source_column(source_column, row[source_column])
        canonical_rows.append(output)
    return pd.DataFrame(canonical_rows)


def build_profile_feature_table(
    canonical_df: pd.DataFrame,
    mapping_config: ProfileMappingConfig,
    *,
    include_sensitive_context: bool = False,
    exclude_treatment_seeking: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    feature_source_columns = selected_feature_source_columns(
        mapping_config,
        include_sensitive_context=include_sensitive_context,
        exclude_treatment_seeking=exclude_treatment_seeking,
    )
    feature_columns = [CANONICAL_COLUMNS[column] for column in feature_source_columns]
    excluded_columns = [
        CANONICAL_COLUMNS[column]
        for column in mapping_config.source_columns
        if CANONICAL_COLUMNS[column] not in feature_columns and CANONICAL_COLUMNS[column] != TARGET_CANONICAL_COLUMN
    ]
    sensitive_columns = [
        CANONICAL_COLUMNS[column]
        for column in mapping_config.sensitive_context_columns
        if CANONICAL_COLUMNS[column] in feature_columns
    ]
    leakage = detect_profile_leakage(feature_columns, TARGET_CANONICAL_COLUMN)
    if leakage["has_leakage"]:
        raise ValueError(f"Refusing to build feature table with leakage columns: {leakage['leakage_columns']}")

    output_columns = ["record_id"] + feature_columns + [TARGET_CANONICAL_COLUMN]
    feature_df = canonical_df.loc[:, output_columns].copy()
    validate_preprocessed_profile(feature_df, feature_columns)
    return feature_df, feature_columns, excluded_columns, sensitive_columns


def _category_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    safe: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        if column == "record_id" or pd.api.types.is_numeric_dtype(df[column]):
            continue
        unique_count = int(df[column].nunique(dropna=True))
        item: dict[str, Any] = {"unique_count": unique_count, "missing_count": int(df[column].isna().sum())}
        if column in {"year_of_study", "cgpa_band", "target_depression", "self_reported_anxiety", "self_reported_panic_attack", "sought_specialist_treatment", "marital_status", "gender"}:
            item["canonical_values"] = sorted(str(value) for value in df[column].dropna().unique())
        safe[column] = item
    return safe


def _build_feature_schema(
    mapping_config: ProfileMappingConfig,
    feature_columns: list[str],
    excluded_columns: list[str],
    *,
    include_sensitive_context: bool,
) -> FeatureSchema:
    fields = mapping_by_source(mapping_config)
    features: list[FeatureDefinition] = []
    by_canonical = {field.canonical_feature_name: field for field in fields.values()}
    descriptions = {
        "age": "Student age, preserved as nullable numeric context without imputation.",
        "gender": "Self-reported gender; optional sensitive-context feature.",
        "course": "Self-reported course/program; optional sensitive-context feature.",
        "year_of_study": "Canonical year-of-study category.",
        "cgpa_band": "Self-reported CGPA range preserved as ordered categorical text.",
        "marital_status": "Self-reported marital-status binary label; optional sensitive-context feature.",
        "self_reported_anxiety": "Self-reported anxiety binary label.",
        "self_reported_panic_attack": "Self-reported panic-attack binary label.",
        "sought_specialist_treatment": "Treatment-seeking binary label; leakage candidate and excluded by default.",
    }
    for feature_name in feature_columns:
        field = by_canonical[feature_name]
        dtype = "float" if feature_name == "age" else "category"
        allowed = field.allowed_categories
        features.append(
            FeatureDefinition(
                name=feature_name,
                dtype=dtype,
                description=descriptions.get(feature_name, field.notes),
                source_columns=[field.source_column_name],
                nullable=feature_name == "age",
                category_values=allowed,
                minimum=10 if feature_name == "age" else None,
                maximum=120 if feature_name == "age" else None,
                preprocessing_step=(
                    "canonicalize_profile_dataframe; sensitive_context_included="
                    f"{include_sensitive_context}; no encoder/scaler fitting"
                ),
            )
        )
    return FeatureSchema(
        schema_name="student-profile-canonical-features",
        feature_schema_version=PROFILE_FEATURE_SCHEMA_VERSION,
        dataset_name=mapping_config.dataset_name,
        dataset_version=mapping_config.dataset_version,
        preprocessing_version=PROFILE_PREPROCESSING_VERSION,
        modality=Modality.PROFILE,
        features=features,
        target_columns=[TARGET_CANONICAL_COLUMN],
        excluded_columns=excluded_columns + [TARGET_CANONICAL_COLUMN],
        created_at=_utc_now(),
        notes=(
            "Feature groups: baseline features are included by default; sensitive contextual features are optional; "
            "Timestamp, target, and treatment-seeking leakage candidate are excluded in the default baseline. "
            "No train/validation/test split, encoder, scaler, or model artifact is created."
        ),
    )


def _build_report(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mapping_config: ProfileMappingConfig,
    feature_columns: list[str],
    excluded_columns: list[str],
    sensitive_columns: list[str],
    source_fingerprint: str,
) -> ProfilePreprocessingReport:
    missing = validate_profile_missing_values(source_df)
    categories = validate_profile_categories(source_df)
    leakage = detect_profile_leakage(feature_columns, TARGET_CANONICAL_COLUMN)
    identifiers = detect_profile_identifiers(source_df.columns)
    target_distribution = {
        str(label): int(count)
        for label, count in feature_df[TARGET_CANONICAL_COLUMN].value_counts(dropna=False).sort_index().items()
    }
    warnings = list(missing["warnings"])
    if categories["unexpected_categories"]:
        warnings.append(f"Unexpected categories reported without silent removal: {categories['unexpected_categories']}")
    warnings.extend(
        [
            "Only 101 records are available; later model metrics will be unstable.",
            "Target is self-reported depression, not suicidal ideation or suicide-risk ground truth.",
            "Dataset should not be treated as Sri Lankan-wide representative.",
            "Sensitive attributes require fairness review before predictive use.",
            "Treatment-seeking is excluded by default as a possible post-outcome/leakage feature.",
        ]
    )
    return ProfilePreprocessingReport(
        preprocessing_version=PROFILE_PREPROCESSING_VERSION,
        feature_schema_version=PROFILE_FEATURE_SCHEMA_VERSION,
        mapping_version=PROFILE_MAPPING_VERSION,
        source_fingerprint=source_fingerprint,
        source_row_count=int(len(source_df)),
        output_row_count=int(len(feature_df)),
        excluded_row_count=int(len(source_df) - len(feature_df)),
        missing_value_summary=missing["missing_value_summary"],
        category_normalization_summary=_category_summary(canonical_df),
        target_distribution=target_distribution,
        feature_columns=feature_columns,
        excluded_columns=excluded_columns,
        sensitive_context_columns=sensitive_columns,
        leakage_checks={
            "feature_leakage": leakage,
            "identifier_detection": identifiers,
            "default_excluded_source_columns": list(DEFAULT_EXCLUDED_SOURCE_COLUMNS),
            "treatment_seeking_decision": "excluded_by_default_possible_post_outcome_leakage",
        },
        warnings=warnings,
    )


def validate_preprocessed_profile(feature_df: pd.DataFrame, feature_columns: list[str]) -> None:
    if TARGET_CANONICAL_COLUMN in feature_columns:
        raise ValueError("Target must never be included in feature columns")
    blocked = {"source_timestamp", "Timestamp", TARGET_COLUMN}
    overlap = blocked & set(feature_columns)
    if overlap:
        raise ValueError(f"Metadata or raw target leaked into features: {sorted(overlap)}")
    for column in feature_columns:
        if column not in feature_df.columns:
            raise ValueError(f"Feature column missing from canonical table: {column}")
        if pd.api.types.is_numeric_dtype(feature_df[column]):
            values = feature_df[column].dropna().tolist()
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"Numeric feature contains infinity: {column}")
    if TARGET_CANONICAL_COLUMN not in feature_df.columns:
        raise ValueError("Target column missing from canonical table")


def _write_json(payload: dict[str, Any], output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _write_csv(df: pd.DataFrame, output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_path


def _record_manifest(feature_df: pd.DataFrame, source_fingerprint: str) -> dict[str, Any]:
    return {
        "dataset": RECORD_ID_PREFIX,
        "source_fingerprint": source_fingerprint,
        "record_count": int(len(feature_df)),
        "record_id_strategy": "student-profile-v1-<1-based-row-position>-<hash(dataset-version,fingerprint,row-position)>",
        "record_ids": feature_df["record_id"].tolist(),
    }


def preprocess_profile_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    mapping_config: ProfileMappingConfig,
    fingerprint: DatasetFingerprint,
    *,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    include_sensitive_context: bool = False,
    exclude_treatment_seeking: bool = True,
) -> dict[str, Any]:
    validate_profile_mapping(mapping_config)
    source_df = load_profile_source(dataset_config)
    validate_profile_source_columns(source_df.columns, mapping_config.source_columns)
    validate_profile_target_values(source_df, mapping_config.target_column)
    validate_profile_numeric_ranges(source_df)
    category_validation = validate_profile_categories(source_df)

    canonical_df = canonicalize_profile_dataframe(
        normalize_profile_categories(source_df),
        mapping_config,
        source_fingerprint=fingerprint.combined_sha256,
    )
    feature_df, feature_columns, excluded_columns, sensitive_columns = build_profile_feature_table(
        canonical_df,
        mapping_config,
        include_sensitive_context=include_sensitive_context,
        exclude_treatment_seeking=exclude_treatment_seeking,
    )
    report = _build_report(
        source_df,
        canonical_df,
        feature_df,
        mapping_config,
        feature_columns,
        excluded_columns,
        sensitive_columns,
        fingerprint.combined_sha256,
    )
    feature_schema = _build_feature_schema(
        mapping_config,
        feature_columns,
        excluded_columns,
        include_sensitive_context=include_sensitive_context,
    )

    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Profile preprocessing outputs must be under generated/")
        outputs = {
            "canonical_csv": str(_write_csv(feature_df, resolved_output_dir / "canonical_profile.csv", overwrite=overwrite)),
            "feature_schema_json": str(
                _write_json(feature_schema.to_safe_dict(), resolved_output_dir / "profile_feature_schema.json", overwrite=overwrite)
            ),
            "report_json": str(
                _write_json(report.to_safe_dict(), resolved_output_dir / "profile_preprocessing_report.json", overwrite=overwrite)
            ),
            "record_manifest_json": str(
                _write_json(_record_manifest(feature_df, fingerprint.combined_sha256), resolved_output_dir / "profile_record_manifest.json", overwrite=overwrite)
            ),
        }
        from app.ml.preprocessing.profile.reporting import create_profile_preprocessing_markdown

        md_path = resolved_output_dir / "profile_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_profile_preprocessing_markdown(report, feature_schema), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)

    return {
        "valid": True,
        "validate_only": validate_only,
        "source_rows": int(len(source_df)),
        "output_rows": int(len(feature_df)),
        "excluded_rows": int(len(source_df) - len(feature_df)),
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "sensitive_context_columns": sensitive_columns,
        "target_distribution": report.target_distribution,
        "category_validation": category_validation,
        "report": report,
        "feature_schema": feature_schema,
        "outputs": outputs,
    }
