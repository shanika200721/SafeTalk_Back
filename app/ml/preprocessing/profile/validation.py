"""Validation helpers for Student Profile preprocessing."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import pandas as pd

from app.ml.preprocessing.profile.constants import (
    CGPA_ALLOWED_VALUES,
    DEFAULT_EXCLUDED_SOURCE_COLUMNS,
    LEAKAGE_CANDIDATE_COLUMNS,
    SOURCE_COLUMNS,
    TARGET_ALLOWED_VALUES,
    TARGET_COLUMN,
    TREATMENT_COLUMN,
    YEAR_ALLOWED_VALUES,
)
from app.ml.preprocessing.profile.schemas import ProfileFieldRole, ProfileMappingConfig


_LABEL_DERIVED_RE = re.compile(r"(depression|target|label|outcome|diagnos)", re.I)
_IDENTIFIER_RE = re.compile(r"(^id$|_id$|student_id|participant|user|email|phone|name|address)", re.I)


def validate_profile_source_columns(columns: Iterable[str], required_columns: Iterable[str] = SOURCE_COLUMNS) -> dict[str, Any]:
    available = [str(column) for column in columns]
    available_set = set(available)
    required = [str(column) for column in required_columns]
    missing = [column for column in required if column not in available_set]
    unexpected = [column for column in available if column not in set(required)]
    duplicates = sorted({column for column in available if available.count(column) > 1})
    if missing:
        raise ValueError(f"Missing required Student Profile source columns: {missing}")
    if duplicates:
        raise ValueError(f"Duplicate Student Profile source columns: {duplicates}")
    return {"valid": True, "missing_columns": [], "unexpected_columns": unexpected, "column_count": len(available)}


def _normalized_non_null_values(series: pd.Series) -> list[str]:
    return [str(value).strip().lower() for value in series.dropna().tolist() if str(value).strip()]


def validate_profile_target_values(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Target column is missing: {target_column}")
    values = _normalized_non_null_values(df[target_column])
    unknown = sorted({value for value in values if value not in TARGET_ALLOWED_VALUES})
    missing_count = int(df[target_column].isna().sum())
    if missing_count:
        raise ValueError(f"Target column contains missing values: {missing_count}")
    if unknown:
        raise ValueError(f"Target column contains unrecognized values: {unknown}")
    return {"valid": True, "allowed_values": list(TARGET_ALLOWED_VALUES), "distribution": dict(df[target_column].str.strip().str.lower().value_counts())}


def validate_profile_numeric_ranges(df: pd.DataFrame) -> dict[str, Any]:
    issues: dict[str, Any] = {}
    if "Age" in df.columns:
        numeric = pd.to_numeric(df["Age"], errors="coerce")
        invalid_text_count = int((df["Age"].notna() & numeric.isna()).sum())
        out_of_range_count = int(((numeric < 10) | (numeric > 120)).sum())
        infinity_count = int(sum(math.isinf(float(value)) for value in numeric.dropna()))
        if invalid_text_count or out_of_range_count or infinity_count:
            issues["Age"] = {
                "invalid_text_count": invalid_text_count,
                "out_of_range_count": out_of_range_count,
                "infinity_count": infinity_count,
            }
    if issues:
        raise ValueError(f"Invalid Student Profile numeric ranges: {issues}")
    return {"valid": True, "issues": issues}


def validate_profile_categories(df: pd.DataFrame) -> dict[str, Any]:
    expected = {
        "Your current year of Study": set(YEAR_ALLOWED_VALUES),
        "What is your CGPA?": set(CGPA_ALLOWED_VALUES),
        TARGET_COLUMN: set(TARGET_ALLOWED_VALUES),
        "Do you have Anxiety?": set(TARGET_ALLOWED_VALUES),
        "Do you have Panic attack?": set(TARGET_ALLOWED_VALUES),
        TREATMENT_COLUMN: set(TARGET_ALLOWED_VALUES),
        "Marital status": set(TARGET_ALLOWED_VALUES),
    }
    unexpected: dict[str, list[str]] = {}
    for column, allowed in expected.items():
        if column not in df.columns:
            continue
        observed = {str(value).strip().lower() for value in df[column].dropna().tolist() if str(value).strip()}
        allowed_lower = {value.lower() for value in allowed}
        extra = sorted(observed - allowed_lower)
        if extra:
            unexpected[column] = extra
    return {"valid": not unexpected, "unexpected_categories": unexpected}


def validate_profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    summary = {column: int(count) for column, count in df.isna().sum().items() if int(count) > 0}
    age_missing = summary.get("Age", 0)
    warnings = []
    if age_missing:
        warnings.append(f"Age has {age_missing} missing value(s); canonical output preserves missing values.")
    return {"missing_value_summary": summary, "age_missing_count": age_missing, "warnings": warnings}


def detect_profile_leakage(feature_columns: Iterable[str], target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    features = [str(column) for column in feature_columns]
    leakage = []
    if target_column in features or "target_depression" in features:
        leakage.append({"column": target_column, "reason": "target column included as feature"})
    for column in features:
        if column == "Timestamp" or column == "source_timestamp":
            leakage.append({"column": column, "reason": "timestamp metadata included as feature"})
        elif column == TREATMENT_COLUMN or column == "sought_specialist_treatment":
            leakage.append({"column": column, "reason": "post-outcome treatment-seeking candidate"})
        elif column != target_column and _LABEL_DERIVED_RE.search(column):
            leakage.append({"column": column, "reason": "label-derived column name"})
    return {"has_leakage": bool(leakage), "leakage_columns": leakage}


def detect_profile_identifiers(columns: Iterable[str]) -> dict[str, Any]:
    candidates = [str(column) for column in columns if _IDENTIFIER_RE.search(str(column))]
    return {"identifier_candidates": candidates, "auto_identifier_selected": None}


def validate_profile_mapping(mapping_config: ProfileMappingConfig, columns: Iterable[str] | None = None) -> dict[str, Any]:
    source_columns = list(columns) if columns is not None else mapping_config.source_columns
    validate_profile_source_columns(source_columns, mapping_config.source_columns)
    fields = mapping_config.fields
    target_fields = [field for field in fields if field.role == ProfileFieldRole.TARGET]
    if len(target_fields) != 1:
        raise ValueError("Profile mapping must contain exactly one target field")
    if target_fields[0].source_column_name != TARGET_COLUMN:
        raise ValueError(f"Profile mapping target must be {TARGET_COLUMN}")

    feature_sources = [field.source_column_name for field in fields if field.role in {ProfileFieldRole.FEATURE, ProfileFieldRole.SENSITIVE_CONTEXT}]
    leakage = detect_profile_leakage(feature_sources, mapping_config.target_column)
    if leakage["has_leakage"]:
        raise ValueError(f"Profile mapping includes leakage as feature: {leakage['leakage_columns']}")

    default_exclusions = set(mapping_config.default_excluded_columns)
    required_exclusions = set(DEFAULT_EXCLUDED_SOURCE_COLUMNS)
    if not required_exclusions <= default_exclusions:
        raise ValueError(f"Profile mapping must exclude by default: {sorted(required_exclusions - default_exclusions)}")

    return {
        "valid": True,
        "mapping_version": mapping_config.mapping_version,
        "feature_column_count": len(feature_sources),
        "excluded_columns": list(mapping_config.default_excluded_columns),
        "leakage_candidate_columns": list(LEAKAGE_CANDIDATE_COLUMNS),
    }
