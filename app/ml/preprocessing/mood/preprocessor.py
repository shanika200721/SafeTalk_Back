"""Deterministic canonical preprocessing for Daily Mood datasets."""

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
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, FeatureSchema, PreprocessingConfig
from app.ml.preprocessing.mood.constants import (
    CANONICAL_MOOD,
    CANONICAL_STUDENT_ID,
    CANONICAL_TIMESTAMP,
    FEATURE_COLUMNS,
    MOOD_FEATURE_SCHEMA_VERSION,
    MOOD_MAPPING_VERSION,
    MOOD_MAX,
    MOOD_MIN,
    MOOD_PREPROCESSING_VERSION,
    RECORD_ID_PREFIX,
    SAFE_PARTICIPANT_KEY_PREFIX,
)
from app.ml.preprocessing.mood.features import build_mood_feature_schema, build_mood_feature_table
from app.ml.preprocessing.mood.mapping import mapping_by_source
from app.ml.preprocessing.mood.schemas import MoodFieldRole, MoodMappingConfig, MoodPreprocessingReport
from app.ml.preprocessing.mood.validation import (
    detect_duplicate_checkins,
    detect_future_leakage,
    detect_multiple_checkins_per_period,
    detect_temporal_gaps,
    validate_mood_mapping,
    validate_mood_source_columns,
    validate_mood_value_range,
    validate_participant_keys,
    validate_timestamps,
)


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def normalize_mood_values(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Mood value is missing or non-numeric: {value}")
    numeric = float(numeric)
    if not math.isfinite(numeric) or numeric < MOOD_MIN or numeric > MOOD_MAX:
        raise ValueError(f"Mood value outside confirmed {MOOD_MIN}-{MOOD_MAX} range: {value}")
    return numeric


def normalize_timestamp(value: Any, *, assume_timezone: timezone = timezone.utc) -> datetime:
    timestamp = pd.to_datetime(value, errors="raise", utc=False)
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp.to_pydatetime()) is None:
        timestamp = timestamp.tz_localize(assume_timezone)
    return timestamp.tz_convert("UTC").to_pydatetime()


def generate_mood_record_id(participant_key: str, timestamp: datetime, source_position: int, source_fingerprint: str) -> str:
    if source_position < 0:
        raise ValueError("source_position must be non-negative")
    if not source_fingerprint or len(source_fingerprint) < 12:
        raise ValueError("source_fingerprint must be available for deterministic record IDs")
    stamp = timestamp.astimezone(timezone.utc).isoformat()
    digest = hashlib.sha256(f"{RECORD_ID_PREFIX}:{source_fingerprint}:{participant_key}:{stamp}:{source_position}".encode("utf-8")).hexdigest()[:12]
    return f"{RECORD_ID_PREFIX}-{source_position + 1:06d}-{digest}"


def create_safe_participant_key(raw_identifier: Any, *, salt: str = SAFE_PARTICIPANT_KEY_PREFIX) -> str:
    text = _clean_text(raw_identifier)
    if text is None:
        raise ValueError("Participant identifier is missing")
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{salt}-{digest}"


def _physical_symptom_count(value: Any) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    if text.lower() in {"none", "no", "0", "false"}:
        return 0.0
    return 1.0


def _numeric_optional(value: Any) -> float | None:
    if pd.isna(value) or _clean_text(value) is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Numeric mood source value must not be infinite")
    return numeric


def _time_of_day(timestamp: datetime) -> str:
    hour = timestamp.astimezone(timezone.utc).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def load_mood_source(dataset_config: DatasetConfig) -> pd.DataFrame:
    source_path = dataset_config.validate_source_exists()
    return pd.read_csv(source_path)


def canonicalize_mood_dataframe(
    df: pd.DataFrame,
    mapping_config: MoodMappingConfig,
    *,
    source_fingerprint: str,
    max_participants: int | None = None,
    max_records: int | None = None,
) -> pd.DataFrame:
    validate_mood_source_columns(df.columns, mapping_config.source_columns)
    fields = mapping_by_source(mapping_config)
    identifier_column = next(field.source_field for field in fields.values() if field.role == MoodFieldRole.IDENTIFIER)
    timestamp_column = next(field.source_field for field in fields.values() if field.role == MoodFieldRole.TIMESTAMP)
    mood_column = next(field.source_field for field in fields.values() if field.canonical_field == CANONICAL_MOOD)

    validate_participant_keys(df, identifier_column)
    validate_timestamps(df, timestamp_column)
    validate_mood_value_range(df, mood_column)

    working = df.copy()
    if max_participants is not None:
        keep = sorted(working[identifier_column].astype(str).unique())[:max_participants]
        working = working[working[identifier_column].astype(str).isin(keep)]
    if max_records is not None:
        working = working.head(max_records)

    rows: list[dict[str, Any]] = []
    for source_position, row in working.reset_index(drop=True).iterrows():
        raw_participant = row[identifier_column]
        participant_key = create_safe_participant_key(raw_participant)
        timestamp = normalize_timestamp(row[timestamp_column])
        output = {
            "record_id": generate_mood_record_id(participant_key, timestamp, int(source_position), source_fingerprint),
            "participant_key": participant_key,
            "timestamp": timestamp,
            "mood_value": normalize_mood_values(row[mood_column]),
            "source_record_id": None,
            "validation_warnings": "",
            "time_of_day": _time_of_day(timestamp),
        }
        for field in fields.values():
            if field.canonical_field in {CANONICAL_STUDENT_ID, CANONICAL_TIMESTAMP, CANONICAL_MOOD}:
                continue
            if field.source_field not in row:
                continue
            if field.canonical_field == "physical_symptom_count":
                output[field.canonical_field] = _physical_symptom_count(row[field.source_field])
            elif field.expected_type in {"number", "integer", "float"}:
                output[field.canonical_field] = _numeric_optional(row[field.source_field])
            elif field.canonical_field == "notes_present":
                output[field.canonical_field] = _clean_text(row[field.source_field]) is not None
            else:
                output[field.canonical_field] = _clean_text(row[field.source_field])
        rows.append(output)

    canonical = pd.DataFrame(rows)
    canonical = canonical.sort_values(["participant_key", "timestamp", "record_id"], kind="mergesort").reset_index(drop=True)
    validate_canonical_mood(canonical)
    return canonical


def validate_canonical_mood(canonical_df: pd.DataFrame) -> None:
    required = {"record_id", "participant_key", "timestamp", "mood_value"}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"Canonical mood data missing columns: {sorted(missing)}")
    if any(canonical_df["participant_key"].astype(str).str.contains("@", regex=False)):
        raise ValueError("Canonical mood participant keys must not expose raw email-like identifiers")
    mood = pd.to_numeric(canonical_df["mood_value"], errors="coerce")
    if mood.isna().any() or bool(((mood < MOOD_MIN) | (mood > MOOD_MAX)).any()):
        raise ValueError("Canonical mood values are outside confirmed range")
    for column in canonical_df.columns:
        if pd.api.types.is_numeric_dtype(canonical_df[column]):
            values = canonical_df[column].dropna().tolist()
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"Canonical mood numeric output contains infinity: {column}")
    for _, group in canonical_df.groupby("participant_key", sort=True):
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        if bool((timestamps.diff().dt.total_seconds().dropna() < 0).any()):
            raise ValueError("Canonical mood records are not deterministically sorted")


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in df.isna().sum().items() if int(count) > 0}


def _build_report(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mapping_config: MoodMappingConfig,
    duplicate_summary: dict[str, Any],
) -> MoodPreprocessingReport:
    timestamps = pd.to_datetime(canonical_df["timestamp"], utc=True) if len(canonical_df) else pd.Series([], dtype="datetime64[ns, UTC]")
    leakage = detect_future_leakage(feature_df)
    gaps = detect_temporal_gaps(canonical_df) if len(canonical_df) else {"large_gap_counts": {}, "max_gap_days": 0}
    warnings = [
        "Mood self-reports are subjective and cannot diagnose suicide risk.",
        "Irregular check-in frequency can bias temporal trend features.",
        "Missing check-ins are preserved; they do not necessarily indicate deterioration.",
        "Synthetic or generated mood records cannot validate predictive performance.",
        "Rolling features require sufficient longitudinal history.",
        "The experimental 0-100 trend score is non-clinical and creates no alerts.",
        f"Temporal gap summary: {gaps}",
    ]
    return MoodPreprocessingReport(
        preprocessing_version=MOOD_PREPROCESSING_VERSION,
        feature_schema_version=MOOD_FEATURE_SCHEMA_VERSION,
        mapping_version=MOOD_MAPPING_VERSION,
        source_record_count=int(len(source_df)),
        output_record_count=int(len(feature_df)),
        participant_count=int(canonical_df["participant_key"].nunique()) if len(canonical_df) else 0,
        date_range={
            "min": timestamps.min().isoformat() if len(timestamps) else None,
            "max": timestamps.max().isoformat() if len(timestamps) else None,
        },
        missing_value_summary=_missing_summary(source_df),
        duplicate_summary=duplicate_summary,
        temporal_order_violations=int(leakage["temporal_order_violations"]),
        feature_columns=list(FEATURE_COLUMNS),
        excluded_columns=[CANONICAL_STUDENT_ID, "raw_identifier", "names", "emails", "free_text_notes"],
        warnings=warnings,
    )


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


def _record_manifest(canonical_df: pd.DataFrame, source_fingerprint: str, *, synthetic: bool) -> dict[str, Any]:
    return {
        "dataset": RECORD_ID_PREFIX,
        "source_fingerprint": source_fingerprint,
        "synthetic": synthetic,
        "record_count": int(len(canonical_df)),
        "record_id_strategy": "daily-mood-v1-<sorted-source-position>-<hash(dataset-version,fingerprint,participant,timestamp,position)>",
        "record_ids": canonical_df["record_id"].tolist(),
    }


def preprocess_mood_dataframe(
    source_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None,
    mapping_config: MoodMappingConfig,
    *,
    source_fingerprint: str,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    synthetic: bool = False,
    max_participants: int | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    validate_mood_mapping(mapping_config, source_df.columns)
    duplicate_summary = detect_duplicate_checkins(source_df, mapping_config.source_columns[0], mapping_config.source_columns[1])
    period_summary = detect_multiple_checkins_per_period(source_df, mapping_config.source_columns[0], mapping_config.source_columns[1])
    canonical_df = canonicalize_mood_dataframe(
        source_df,
        mapping_config,
        source_fingerprint=source_fingerprint,
        max_participants=max_participants,
        max_records=max_records,
    )
    feature_df = build_mood_feature_table(canonical_df)
    report = _build_report(source_df, canonical_df, feature_df, mapping_config, {**duplicate_summary, "multiple_checkins_per_day": period_summary})
    feature_schema = build_mood_feature_schema(dataset_name=mapping_config.dataset_name, dataset_version=mapping_config.dataset_version)

    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Mood preprocessing outputs must be under generated/")
        outputs = {
            "canonical_csv": str(_write_csv(canonical_df, resolved_output_dir / "canonical_mood.csv", overwrite=overwrite)),
            "features_csv": str(_write_csv(feature_df, resolved_output_dir / "mood_features.csv", overwrite=overwrite)),
            "feature_schema_json": str(_write_json(feature_schema.to_safe_dict(), resolved_output_dir / "mood_feature_schema.json", overwrite=overwrite)),
            "report_json": str(_write_json(report.to_safe_dict(), resolved_output_dir / "mood_preprocessing_report.json", overwrite=overwrite)),
            "record_manifest_json": str(
                _write_json(_record_manifest(canonical_df, source_fingerprint, synthetic=synthetic), resolved_output_dir / "mood_record_manifest.json", overwrite=overwrite)
            ),
        }
        from app.ml.preprocessing.mood.reporting import create_mood_preprocessing_markdown

        md_path = resolved_output_dir / "mood_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_mood_preprocessing_markdown(report, feature_schema), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)

    return {
        "valid": True,
        "validate_only": validate_only,
        "synthetic": synthetic,
        "source_rows": int(len(source_df)),
        "output_rows": int(len(feature_df)),
        "participant_count": int(canonical_df["participant_key"].nunique()) if len(canonical_df) else 0,
        "date_range": report.date_range,
        "duplicate_count": int(duplicate_summary["duplicate_count"]),
        "missing_value_summary": report.missing_value_summary,
        "feature_columns": list(FEATURE_COLUMNS),
        "report": report,
        "feature_schema": feature_schema,
        "outputs": outputs,
    }


def preprocess_mood_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    mapping_config: MoodMappingConfig,
    fingerprint: DatasetFingerprint,
    *,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_participants: int | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    source_df = load_mood_source(dataset_config)
    return preprocess_mood_dataframe(
        source_df,
        preprocessing_config,
        mapping_config,
        source_fingerprint=fingerprint.combined_sha256,
        output_dir=output_dir,
        overwrite=overwrite,
        validate_only=validate_only,
        synthetic=False,
        max_participants=max_participants,
        max_records=max_records,
    )


def synthetic_mood_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ParticipantID": "SYN001", "Date": "2025-03-01T08:00:00Z", "Mood": 4, "CryingEpisodes": 0, "PhysicalPain": "none"},
            {"ParticipantID": "SYN001", "Date": "2025-03-02T08:00:00Z", "Mood": 3, "CryingEpisodes": 0, "PhysicalPain": "none"},
            {"ParticipantID": "SYN001", "Date": "2025-03-04T08:00:00Z", "Mood": 1, "CryingEpisodes": 2, "PhysicalPain": "headache"},
            {"ParticipantID": "SYN002", "Date": "2025-03-01T18:00:00Z", "Mood": 5, "CryingEpisodes": 0, "PhysicalPain": "none"},
            {"ParticipantID": "SYN002", "Date": "2025-03-03T18:00:00Z", "Mood": 2, "CryingEpisodes": 1, "PhysicalPain": "fatigue"},
        ]
    )


def synthetic_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
