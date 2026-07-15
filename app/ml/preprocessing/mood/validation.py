"""Validation helpers for Daily Mood preprocessing."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from app.ml.preprocessing.mood.constants import (
    CANONICAL_MOOD,
    CANONICAL_STUDENT_ID,
    CANONICAL_TIMESTAMP,
    DEFAULT_SOURCE_COLUMNS,
    MOOD_MAX,
    MOOD_MIN,
)
from app.ml.preprocessing.mood.schemas import MoodFieldRole, MoodMappingConfig


_DIRECT_IDENTIFIER_RE = re.compile(r"(@|email|full[_ -]?name|phone|address)", re.I)


def validate_mood_source_columns(columns: Iterable[str], required_columns: Iterable[str] = DEFAULT_SOURCE_COLUMNS) -> dict[str, Any]:
    available = [str(column) for column in columns]
    available_set = set(available)
    required = [str(column) for column in required_columns]
    missing = [column for column in required if column not in available_set]
    unexpected = [column for column in available if column not in set(required)]
    duplicates = sorted({column for column in available if available.count(column) > 1})
    if missing:
        raise ValueError(f"Missing required Daily Mood source columns: {missing}")
    if duplicates:
        raise ValueError(f"Duplicate Daily Mood source columns: {duplicates}")
    return {"valid": True, "missing_columns": [], "unexpected_columns": unexpected, "column_count": len(available)}


def validate_mood_value_range(
    df: pd.DataFrame,
    mood_column: str = "Mood",
    *,
    minimum: int = MOOD_MIN,
    maximum: int = MOOD_MAX,
) -> dict[str, Any]:
    if mood_column not in df.columns:
        raise ValueError(f"Mood column is missing: {mood_column}")
    numeric = pd.to_numeric(df[mood_column], errors="coerce")
    invalid_text_count = int((df[mood_column].notna() & numeric.isna()).sum())
    missing_count = int(numeric.isna().sum())
    out_of_range_count = int(((numeric < minimum) | (numeric > maximum)).sum())
    infinity_count = int(sum(math.isinf(float(value)) for value in numeric.dropna()))
    if invalid_text_count or missing_count or out_of_range_count or infinity_count:
        raise ValueError(
            "Invalid Daily Mood mood values: "
            f"invalid_text_count={invalid_text_count}, missing_count={missing_count}, "
            f"out_of_range_count={out_of_range_count}, infinity_count={infinity_count}"
        )
    return {"valid": True, "minimum": minimum, "maximum": maximum, "missing_count": missing_count}


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str = "Date",
    *,
    now: datetime | None = None,
    assume_timezone: timezone = timezone.utc,
) -> dict[str, Any]:
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column is missing: {timestamp_column}")
    parsed = pd.to_datetime(df[timestamp_column], errors="coerce", utc=False)
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        invalid_examples = df.loc[parsed.isna(), timestamp_column].astype(str).head(5).tolist()
        raise ValueError(f"Invalid Daily Mood timestamps: count={invalid_count}, examples={invalid_examples}")

    normalized = pd.to_datetime(df[timestamp_column], errors="raise", utc=True)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.tzinfo.utcoffset(current) is None:
        current = current.replace(tzinfo=assume_timezone).astimezone(timezone.utc)
    future_count = int((normalized > pd.Timestamp(current.astimezone(timezone.utc))).sum())
    if future_count:
        raise ValueError(f"Daily Mood timestamps contain future observations: {future_count}")
    return {
        "valid": True,
        "timezone_policy": "normalized_to_utc",
        "min_timestamp": normalized.min().isoformat() if len(normalized) else None,
        "max_timestamp": normalized.max().isoformat() if len(normalized) else None,
    }


def validate_participant_keys(df: pd.DataFrame, participant_column: str = "ParticipantID") -> dict[str, Any]:
    if participant_column not in df.columns:
        raise ValueError(f"Participant identifier column is missing: {participant_column}")
    series = df[participant_column]
    missing_count = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
    if missing_count:
        raise ValueError(f"Daily Mood participant identifiers are missing: {missing_count}")
    direct_candidates = [str(value) for value in series.dropna().astype(str).unique() if _DIRECT_IDENTIFIER_RE.search(str(value))]
    if direct_candidates:
        raise ValueError("Daily Mood participant identifiers appear to contain direct identifiers")
    return {"valid": True, "participant_count": int(series.astype(str).nunique())}


def detect_duplicate_checkins(
    df: pd.DataFrame,
    participant_column: str = "ParticipantID",
    timestamp_column: str = "Date",
) -> dict[str, Any]:
    key_frame = df[[participant_column, timestamp_column]].copy()
    duplicates = key_frame.duplicated(subset=[participant_column, timestamp_column], keep=False)
    duplicate_rows = df.loc[duplicates].index.astype(int).tolist()
    return {
        "duplicate_count": int(duplicates.sum()),
        "duplicate_rows": duplicate_rows,
        "deterministic_rule": "records are sorted by participant, timestamp, then original source order; duplicates are reported, not combined",
    }


def detect_multiple_checkins_per_period(
    df: pd.DataFrame,
    participant_column: str = "ParticipantID",
    timestamp_column: str = "Date",
    *,
    period: str = "day",
) -> dict[str, Any]:
    timestamps = pd.to_datetime(df[timestamp_column], utc=True)
    periods = timestamps.dt.floor("D") if period == "day" else timestamps
    grouped = pd.DataFrame({"participant": df[participant_column].astype(str), "period": periods}).value_counts()
    multiples = grouped[grouped > 1]
    return {
        "period": period,
        "multiple_period_count": int(len(multiples)),
        "multiple_record_count": int(multiples.sum()) if len(multiples) else 0,
        "allowed_by_current_production_schema": True,
    }


def detect_future_leakage(feature_df: pd.DataFrame) -> dict[str, Any]:
    required = {"participant_key", "feature_timestamp"}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"Cannot check temporal leakage; missing columns: {sorted(missing)}")
    violations = 0
    for _, group in feature_df.groupby("participant_key", sort=True):
        timestamps = pd.to_datetime(group["feature_timestamp"], utc=True)
        violations += int((timestamps.diff().dt.total_seconds().dropna() < 0).sum())
    return {"valid": violations == 0, "temporal_order_violations": violations}


def detect_temporal_gaps(
    df: pd.DataFrame,
    participant_column: str = "participant_key",
    timestamp_column: str = "timestamp",
    *,
    large_gap_days: int = 7,
) -> dict[str, Any]:
    gap_counts: dict[str, int] = {}
    max_gap_days = 0.0
    for participant, group in df.groupby(participant_column, sort=True):
        timestamps = pd.to_datetime(group[timestamp_column], utc=True).sort_values()
        gaps = timestamps.diff().dt.total_seconds().div(86400).dropna()
        gap_counts[str(participant)] = int((gaps > large_gap_days).sum())
        if not gaps.empty:
            max_gap_days = max(max_gap_days, float(gaps.max()))
    return {"large_gap_days": large_gap_days, "large_gap_counts": gap_counts, "max_gap_days": max_gap_days}


def validate_mood_mapping(mapping_config: MoodMappingConfig, columns: Iterable[str] | None = None) -> dict[str, Any]:
    source_columns = list(columns) if columns is not None else mapping_config.source_columns
    validate_mood_source_columns(source_columns, mapping_config.source_columns)
    roles = [field.role for field in mapping_config.fields]
    if roles.count(MoodFieldRole.IDENTIFIER) != 1:
        raise ValueError("Mood mapping must contain exactly one identifier field")
    if roles.count(MoodFieldRole.TIMESTAMP) != 1:
        raise ValueError("Mood mapping must contain exactly one timestamp field")
    canonical = {field.canonical_field for field in mapping_config.fields}
    required = {CANONICAL_STUDENT_ID, CANONICAL_TIMESTAMP, CANONICAL_MOOD}
    missing = required - canonical
    if missing:
        raise ValueError(f"Mood mapping is missing required canonical fields: {sorted(missing)}")
    return {
        "valid": True,
        "mapping_version": mapping_config.mapping_version,
        "feature_column_count": len([field for field in mapping_config.fields if field.role == MoodFieldRole.FEATURE]),
    }
