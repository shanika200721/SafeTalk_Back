"""Validation helpers for privacy-preserving behavioral preprocessing."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from app.ml.preprocessing.behavioral.constants import (
    EVENT_TYPES,
    FEATURE_COLUMNS,
    OPTIONAL_CANONICAL_COLUMNS,
    REQUIRED_CANONICAL_COLUMNS,
    SENSITIVE_PAYLOAD_FIELDS,
)
from app.ml.preprocessing.behavioral.schemas import BehavioralFieldRole, BehavioralMappingConfig


_DIRECT_IDENTIFIER_RE = re.compile(r"(@|email|full[_ -]?name|phone|address|student[_ -]?number|university[_ -]?id)", re.I)
_LONG_ID_RE = re.compile(r"^\d{6,}$")
_URL_WITH_QUERY_RE = re.compile(r"https?://\S*[?&]\S+", re.I)


def _column_list(columns: Iterable[str]) -> list[str]:
    return [str(column) for column in columns]


def validate_behavioral_source_columns(columns: Iterable[str], required_columns: Iterable[str] | None = None) -> dict[str, Any]:
    available = _column_list(columns)
    duplicates = sorted({column for column in available if available.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate behavioral source columns: {duplicates}")
    required = list(required_columns or [])
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Missing required behavioral source columns: {missing}")
    return {"valid": True, "column_count": len(available), "missing_columns": missing, "available_columns": available}


def validate_event_types(df: pd.DataFrame, event_type_column: str = "event_type") -> dict[str, Any]:
    if event_type_column not in df.columns:
        raise ValueError(f"Behavioral event type column is missing: {event_type_column}")
    invalid = sorted(set(df[event_type_column].dropna().astype(str)) - set(EVENT_TYPES))
    if invalid:
        raise ValueError(f"Unsupported behavioral event types: {invalid}")
    return {"valid": True, "allowed_event_types": list(EVENT_TYPES)}


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str = "event_timestamp",
    *,
    now: datetime | None = None,
    assume_timezone: timezone = timezone.utc,
) -> dict[str, Any]:
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column is missing: {timestamp_column}")
    parsed = pd.to_datetime(df[timestamp_column], errors="coerce", utc=False)
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        examples = df.loc[parsed.isna(), timestamp_column].astype(str).head(5).tolist()
        raise ValueError(f"Invalid behavioral timestamps: count={invalid_count}, examples={examples}")
    normalized = pd.to_datetime(df[timestamp_column], errors="raise", utc=True)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.tzinfo.utcoffset(current) is None:
        current = current.replace(tzinfo=assume_timezone).astimezone(timezone.utc)
    future_count = int((normalized > pd.Timestamp(current.astimezone(timezone.utc))).sum())
    if future_count:
        raise ValueError(f"Behavioral timestamps contain future observations: {future_count}")
    return {
        "valid": True,
        "timezone_policy": "normalized_to_utc",
        "min_timestamp": normalized.min().isoformat() if len(normalized) else None,
        "max_timestamp": normalized.max().isoformat() if len(normalized) else None,
    }


def validate_durations(df: pd.DataFrame, duration_columns: Iterable[str] | None = None) -> dict[str, Any]:
    columns = list(duration_columns or [column for column in OPTIONAL_CANONICAL_COLUMNS if any(token in column for token in ("time", "latency", "duration", "speed", "distance"))])
    negative: dict[str, int] = {}
    nonfinite: dict[str, int] = {}
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        negative[column] = int((numeric.dropna() < 0).sum())
        nonfinite[column] = int(sum(not math.isfinite(float(value)) for value in numeric.dropna()))
    bad_negative = {column: count for column, count in negative.items() if count}
    bad_nonfinite = {column: count for column, count in nonfinite.items() if count}
    if bad_negative or bad_nonfinite:
        raise ValueError(f"Invalid behavioral durations/latencies: negative={bad_negative}, nonfinite={bad_nonfinite}")
    return {"valid": True, "checked_columns": columns}


def validate_participant_keys(df: pd.DataFrame, participant_column: str = "participant_key", *, require_safe: bool = False) -> dict[str, Any]:
    if participant_column not in df.columns:
        raise ValueError(f"Participant identifier column is missing: {participant_column}")
    series = df[participant_column].astype(str)
    missing_count = int(series.isna().sum() + (series.str.strip() == "").sum())
    if missing_count:
        raise ValueError(f"Behavioral participant identifiers are missing: {missing_count}")
    direct = [value for value in series.dropna().unique() if _DIRECT_IDENTIFIER_RE.search(value) or _LONG_ID_RE.match(value)]
    if require_safe:
        direct.extend([value for value in series.dropna().unique() if not value.startswith("behavioral-v1-participant-")])
    if direct:
        raise ValueError("Behavioral participant identifiers appear to expose direct identity")
    return {"valid": True, "participant_count": int(series.nunique())}


def validate_session_ids(df: pd.DataFrame, session_column: str = "session_id") -> dict[str, Any]:
    if session_column not in df.columns:
        raise ValueError(f"Session identifier column is missing: {session_column}")
    series = df[session_column].astype(str)
    missing_count = int((series.str.strip() == "").sum())
    malformed = [value for value in series.unique() if any(token in value.lower() for token in [" ", "@", "password"])]
    if missing_count or malformed:
        raise ValueError(f"Invalid behavioral session IDs: missing={missing_count}, malformed={malformed[:5]}")
    return {"valid": True, "session_count": int(series.nunique())}


def detect_duplicate_events(df: pd.DataFrame, subset: Iterable[str] | None = None) -> dict[str, Any]:
    columns = [column for column in (subset or ["participant_key", "event_timestamp", "session_id", "event_type"]) if column in df.columns]
    if not columns:
        return {"duplicate_count": 0, "duplicate_rows": [], "deduplication_policy": "report_only"}
    duplicates = df.duplicated(subset=columns, keep=False)
    return {
        "duplicate_count": int(duplicates.sum()),
        "duplicate_rows": df.loc[duplicates].index.astype(int).tolist(),
        "deduplication_policy": "report_only; duplicates are not merged or deleted",
    }


def detect_impossible_values(df: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "typing_speed_cpm": (0, 1200),
        "key_dwell_time_ms": (0, 5000),
        "key_flight_time_ms": (0, 60000),
        "response_latency_ms": (0, 3600000),
        "mouse_speed_px_per_second": (0, 20000),
        "mouse_distance_px": (0, 10000000),
        "session_duration_seconds": (0, 86400),
    }
    impossible: dict[str, int] = {}
    for column, (minimum, maximum) in checks.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        count = int(((numeric < minimum) | (numeric > maximum)).sum())
        if count:
            impossible[column] = count
    if impossible:
        raise ValueError(f"Impossible behavioral values detected: {impossible}")
    return {"valid": True, "checked_columns": [column for column in checks if column in df.columns]}


def detect_raw_keystroke_content(df: pd.DataFrame) -> dict[str, Any]:
    lower_columns = {str(column).lower(): str(column) for column in df.columns}
    sensitive = sorted(original for lower, original in lower_columns.items() if lower in SENSITIVE_PAYLOAD_FIELDS or "typed" in lower or "keystroke" in lower)
    if sensitive:
        raise ValueError(f"Raw keystroke or content payload fields are prohibited: {sensitive}")
    return {"valid": True}


def detect_sensitive_payload_fields(df: pd.DataFrame) -> dict[str, Any]:
    lower_columns = {str(column).lower(): str(column) for column in df.columns}
    blocked = []
    for lower, original in lower_columns.items():
        if any(token in lower for token in ("clipboard", "screen", "screenshot", "password", "url")):
            blocked.append(original)
    if blocked:
        raise ValueError(f"Sensitive behavioral payload fields are prohibited: {sorted(blocked)}")
    searchable_columns = [column for column in df.columns if str(column).lower() in {"event_type", "page_or_context", "input_type", "field_type"}]
    for column in searchable_columns:
        values = df[column].dropna().astype(str).str.lower()
        if values.str.contains("password", regex=False).any():
            raise ValueError("Password-field behavioral telemetry is prohibited")
        if values.str.contains(_URL_WITH_QUERY_RE).any():
            raise ValueError("Exact URLs containing query data are prohibited in behavioral telemetry")
    return {"valid": True}


def detect_sparse_participants(
    df: pd.DataFrame,
    participant_column: str = "participant_key",
    *,
    minimum_events: int = 20,
) -> dict[str, Any]:
    if participant_column not in df.columns:
        raise ValueError(f"Participant column is missing: {participant_column}")
    counts = df[participant_column].astype(str).value_counts().sort_index()
    sparse = {participant: int(count) for participant, count in counts.items() if int(count) < minimum_events}
    return {"minimum_events": minimum_events, "sparse_participant_count": len(sparse), "sparse_participants": sparse}


def assess_baseline_eligibility(
    events_or_sessions: pd.DataFrame,
    participant_column: str = "participant_key",
    timestamp_column: str = "event_timestamp",
    session_column: str = "session_id",
    *,
    minimum_sessions: int = 3,
    minimum_days: int = 3,
    minimum_events: int = 20,
) -> dict[str, Any]:
    if events_or_sessions.empty:
        return {"eligible_participant_count": 0, "participants": {}}
    missing = {participant_column, timestamp_column, session_column} - set(events_or_sessions.columns)
    if missing:
        raise ValueError(f"Cannot assess baseline eligibility; missing columns: {sorted(missing)}")
    df = events_or_sessions.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    participants: dict[str, dict[str, Any]] = {}
    for participant, group in df.groupby(participant_column, sort=True):
        session_count = int(group[session_column].astype(str).nunique())
        day_count = int(group[timestamp_column].dt.floor("D").nunique())
        event_count = int(len(group))
        eligible = session_count >= minimum_sessions and day_count >= minimum_days and event_count >= minimum_events
        participants[str(participant)] = {
            "eligible": eligible,
            "session_count": session_count,
            "distinct_day_count": day_count,
            "event_count": event_count,
            "reason": "eligible" if eligible else "insufficient_history",
        }
    return {
        "minimum_sessions": minimum_sessions,
        "minimum_days": minimum_days,
        "minimum_events": minimum_events,
        "eligible_participant_count": sum(1 for value in participants.values() if value["eligible"]),
        "participants": participants,
    }


def validate_behavioral_mapping(mapping_config: BehavioralMappingConfig, columns: Iterable[str] | None = None) -> dict[str, Any]:
    source_columns = list(columns) if columns is not None else mapping_config.source_columns
    validate_behavioral_source_columns(source_columns)
    available = set(source_columns)
    missing = [field.source_field for field in mapping_config.fields if field.source_field not in available]
    if missing:
        raise ValueError(f"Behavioral mapping source fields missing from source columns: {missing}")
    roles = [field.role for field in mapping_config.fields]
    if roles.count(BehavioralFieldRole.IDENTIFIER) != 1:
        raise ValueError("Behavioral mapping must contain exactly one identifier field")
    canonical = {field.canonical_field for field in mapping_config.fields}
    required = {"participant_key"}
    if not required <= canonical:
        raise ValueError(f"Behavioral mapping is missing required canonical fields: {sorted(required - canonical)}")
    return {
        "valid": True,
        "mapping_version": mapping_config.mapping_version,
        "feature_column_count": len([field for field in mapping_config.fields if field.role == BehavioralFieldRole.FEATURE]),
    }


def validate_preprocessed_columns(df: pd.DataFrame) -> dict[str, Any]:
    missing = set(REQUIRED_CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Canonical behavioral events missing required columns: {sorted(missing)}")
    validate_participant_keys(df, "participant_key", require_safe=True)
    validate_session_ids(df, "session_id")
    validate_event_types(df, "event_type")
    validate_durations(df)
    detect_sensitive_payload_fields(df)
    for column in df.columns:
        if column in FEATURE_COLUMNS or column in OPTIONAL_CANONICAL_COLUMNS:
            numeric = pd.to_numeric(df[column], errors="coerce").dropna()
            if any(not math.isfinite(float(value)) for value in numeric):
                raise ValueError(f"Preprocessed behavioral output contains NaN or infinity in {column}")
    return {"valid": True}

