"""Validation helpers for text preprocessing."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from app.ml.preprocessing.text.constants import ENGINEERED_LEAKAGE_CANDIDATES, PRIVACY_PLACEHOLDERS
from app.ml.preprocessing.text.mapping import label_mapping_dict
from app.ml.preprocessing.text.normalization import text_contains_only_placeholders
from app.ml.preprocessing.text.privacy import EMAIL_RE, IP_RE, PHONE_RE, URL_RE, USERNAME_RE
from app.ml.preprocessing.text.schemas import TextLabelMappingConfig, TextSourceSelectionConfig


def validate_text_source_columns(columns: Iterable[str], text_column: str, label_column: str) -> dict[str, object]:
    available = [str(column) for column in columns]
    missing = [column for column in (text_column, label_column) if column not in available]
    if missing:
        raise ValueError(f"Missing required text source columns: {missing}")
    duplicates = sorted({column for column in available if available.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate text source columns: {duplicates}")
    return {"valid": True, "column_count": len(available), "missing_columns": []}


def detect_empty_text(df: pd.DataFrame, text_column: str) -> dict[str, int]:
    missing = int(df[text_column].isna().sum())
    blank = int(df[text_column].fillna("").astype(str).str.strip().eq("").sum())
    return {"missing_text_count": missing, "empty_text_count": blank}


def validate_text_values(df: pd.DataFrame, text_column: str) -> dict[str, int]:
    summary = detect_empty_text(df, text_column)
    if summary["empty_text_count"]:
        raise ValueError(f"Text source contains blank text rows: {summary['empty_text_count']}")
    return summary


def validate_label_values(df: pd.DataFrame, label_column: str, mapping_config: TextLabelMappingConfig) -> dict[str, object]:
    entries = label_mapping_dict(mapping_config)
    labels = sorted(df[label_column].dropna().astype(str).str.strip().unique().tolist())
    unknown = [label for label in labels if label not in entries]
    if unknown:
        raise ValueError(f"Unknown text labels: {unknown}")
    return {"valid": True, "labels": labels}


def validate_text_mapping(mapping_config: TextLabelMappingConfig) -> dict[str, object]:
    entries = mapping_config.entries
    merged = [entry.original_label for entry in entries if entry.merged]
    return {"valid": True, "mapping_version": mapping_config.mapping_version, "merged_labels": merged}


def detect_privacy_pattern_leakage(texts: Iterable[str]) -> dict[str, int]:
    joined = "\n".join(str(text) for text in texts)
    phone_matches = PHONE_RE.findall(joined)
    phone_count = sum(1 for value in phone_matches if str(value).strip().startswith("+") or len(re.sub(r"\D", "", str(value))) >= 10)
    return {
        "url_count": len(URL_RE.findall(joined)),
        "email_count": len(EMAIL_RE.findall(joined)),
        "phone_count": phone_count,
        "username_count": len(USERNAME_RE.findall(joined)),
        "ip_address_count": len(IP_RE.findall(joined)),
    }


def detect_target_leakage_columns(columns: Iterable[str], label_column: str) -> list[str]:
    label = label_column.lower()
    patterns = ("label", "target", "class", "status", "suicid", "depress", "anxiety", "stress")
    return [str(column) for column in columns if str(column).lower() != label and any(token in str(column).lower() for token in patterns)]


def detect_engineered_feature_leakage(columns: Iterable[str]) -> list[str]:
    available = {str(column) for column in columns}
    return [column for column in ENGINEERED_LEAKAGE_CANDIDATES if column in available]


def validate_source_selection(config: TextSourceSelectionConfig) -> dict[str, object]:
    canonical = [source.filename for source in config.sources if source.include_in_canonical]
    if canonical != [config.authoritative_source_file]:
        raise ValueError("Source selection must include only the authoritative raw source")
    derived_included = [source.filename for source in config.sources if source.role.value == "excluded_derived" and source.include_in_canonical]
    if derived_included:
        raise ValueError(f"Derived files cannot be canonical sources: {derived_included}")
    return {"valid": True, "authoritative_source_file": config.authoritative_source_file}


def validate_predefined_test_overlap(raw_df: pd.DataFrame, test_df: pd.DataFrame, *, text_column: str = "comparison_text") -> dict[str, int]:
    if text_column not in raw_df.columns or text_column not in test_df.columns:
        raise ValueError(f"Overlap check requires column: {text_column}")
    overlap = set(raw_df[text_column].astype(str)) & set(test_df[text_column].astype(str))
    return {"exact_overlap_count": len(overlap)}


def placeholder_only_count(texts: Iterable[str]) -> int:
    return sum(1 for text in texts if text_contains_only_placeholders(str(text)))


def contains_privacy_placeholder(text: str) -> int:
    return sum(str(text).count(token) for token in PRIVACY_PLACEHOLDERS)
