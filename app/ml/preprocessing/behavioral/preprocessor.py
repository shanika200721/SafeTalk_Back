"""Deterministic behavioral preprocessing without model training."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import paths
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, PreprocessingConfig
from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_MAPPING_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    DEFAULT_MINIMUM_BASELINE_DAYS,
    DEFAULT_MINIMUM_BASELINE_EVENTS,
    DEFAULT_MINIMUM_BASELINE_SESSIONS,
    FEATURE_COLUMNS,
    READINESS_ENGINEERING_ONLY,
    READINESS_UNSUITABLE,
    RECORD_ID_PREFIX,
    SAFE_PARTICIPANT_KEY_PREFIX,
    SESSION_ID_PREFIX,
    SOURCE_STATUS_NO_BEHAVIORAL_DATA,
    SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET,
    SOURCE_STATUS_REAL_OFFLINE_DATASET,
    SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY,
)
from app.ml.preprocessing.behavioral.features import (
    build_behavioral_feature_schema,
    build_behavioral_feature_table,
    calculate_session_features,
)
from app.ml.preprocessing.behavioral.mapping import mapping_by_source
from app.ml.preprocessing.behavioral.schemas import BehavioralFieldRole, BehavioralMappingConfig, BehavioralPreprocessingReport
from app.ml.preprocessing.behavioral.validation import (
    assess_baseline_eligibility,
    detect_duplicate_events,
    detect_impossible_values,
    detect_raw_keystroke_content,
    detect_sensitive_payload_fields,
    detect_sparse_participants,
    validate_behavioral_mapping,
    validate_behavioral_source_columns,
    validate_durations,
    validate_event_types,
    validate_participant_keys,
    validate_preprocessed_columns,
    validate_session_ids,
    validate_timestamps,
)


def normalize_behavioral_timestamp(value: Any, *, assume_timezone: timezone = timezone.utc) -> datetime:
    timestamp = pd.to_datetime(value, errors="raise", utc=False)
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp.to_pydatetime()) is None:
        timestamp = timestamp.tz_localize(assume_timezone)
    return timestamp.tz_convert("UTC").to_pydatetime()


def generate_safe_participant_key(raw_identifier: Any, *, salt: str = SAFE_PARTICIPANT_KEY_PREFIX) -> str:
    text = "" if pd.isna(raw_identifier) else " ".join(str(raw_identifier).strip().split())
    if not text:
        raise ValueError("Participant identifier is missing")
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{salt}-{digest}"


def derive_session_id(participant_key: str, timestamp: datetime, source_session_id: Any | None = None) -> str:
    if source_session_id is not None and not pd.isna(source_session_id) and str(source_session_id).strip():
        raw = " ".join(str(source_session_id).strip().split())
        digest = hashlib.sha256(f"{SESSION_ID_PREFIX}:{participant_key}:{raw}".encode("utf-8")).hexdigest()[:16]
        return f"{SESSION_ID_PREFIX}-{digest}"
    bucket = timestamp.astimezone(timezone.utc).strftime("%Y%m%d%H")
    digest = hashlib.sha256(f"{SESSION_ID_PREFIX}:{participant_key}:{bucket}".encode("utf-8")).hexdigest()[:16]
    return f"{SESSION_ID_PREFIX}-{digest}"


def generate_behavioral_event_id(participant_key: str, session_id: str, timestamp: datetime, source_position: int, source_fingerprint: str) -> str:
    if source_position < 0:
        raise ValueError("source_position must be non-negative")
    if not source_fingerprint or len(source_fingerprint) < 12:
        raise ValueError("source_fingerprint must be available for deterministic behavioral event IDs")
    stamp = timestamp.astimezone(timezone.utc).isoformat()
    digest = hashlib.sha256(f"{RECORD_ID_PREFIX}:{source_fingerprint}:{participant_key}:{session_id}:{stamp}:{source_position}".encode("utf-8")).hexdigest()[:12]
    return f"{RECORD_ID_PREFIX}-{source_position + 1:06d}-{digest}"


def load_behavioral_source(dataset_config: DatasetConfig) -> pd.DataFrame:
    source_path = dataset_config.validate_source_exists()
    if source_path.suffix.lower() == ".jsonl":
        return pd.read_json(source_path, lines=True)
    if source_path.suffix.lower() == ".json":
        return pd.read_json(source_path)
    return pd.read_csv(source_path)


def _numeric_optional(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    numeric = float(value)
    if not pd.notna(numeric):
        return None
    return numeric


def _event_type_from_row(row: pd.Series, explicit: str | None) -> str:
    if explicit and explicit in row and not pd.isna(row[explicit]) and str(row[explicit]).strip():
        return str(row[explicit]).strip()
    if any(column in row and not pd.isna(row[column]) for column in ("key_dwell_time_ms", "key_flight_time_ms", "typing_speed_cpm", "TypingSpeed_cpm")):
        return "typing_timing"
    if any(column in row and not pd.isna(row[column]) for column in ("mouse_distance_px", "mouse_speed_px_per_second", "MouseSpeed_pxs")):
        return "mouse_aggregate"
    if any(column in row and not pd.isna(row[column]) for column in ("response_latency_ms", "ResponseTime_sec")):
        return "prompt_response"
    return "page_view"


def canonicalize_behavioral_events(
    df: pd.DataFrame,
    mapping_config: BehavioralMappingConfig,
    *,
    source_fingerprint: str,
    max_participants: int | None = None,
    max_records: int | None = None,
) -> pd.DataFrame:
    detect_raw_keystroke_content(df)
    detect_sensitive_payload_fields(df)
    validate_behavioral_mapping(mapping_config, df.columns)
    fields = mapping_by_source(mapping_config)
    identifier_column = next(field.source_field for field in fields.values() if field.role == BehavioralFieldRole.IDENTIFIER)
    timestamp_fields = [field.source_field for field in fields.values() if field.role == BehavioralFieldRole.TIMESTAMP]
    session_fields = [field.source_field for field in fields.values() if field.role == BehavioralFieldRole.SESSION]
    event_type_fields = [field.source_field for field in fields.values() if field.role == BehavioralFieldRole.EVENT_TYPE]
    timestamp_column = timestamp_fields[0] if timestamp_fields else None
    session_column = session_fields[0] if session_fields else None
    event_type_column = event_type_fields[0] if event_type_fields else None

    validate_participant_keys(df, identifier_column)
    if timestamp_column:
        validate_timestamps(df, timestamp_column)
    validate_durations(df, [field.source_field for field in fields.values() if field.role == BehavioralFieldRole.FEATURE and field.source_field in df.columns])
    detect_impossible_values(df.rename(columns={field.source_field: field.canonical_field for field in fields.values()}))

    working = df.copy()
    if max_participants is not None:
        keep = sorted(working[identifier_column].astype(str).unique())[:max_participants]
        working = working[working[identifier_column].astype(str).isin(keep)]
    if max_records is not None:
        working = working.head(max_records)

    rows: list[dict[str, Any]] = []
    for source_position, row in working.reset_index(drop=True).iterrows():
        participant_key = generate_safe_participant_key(row[identifier_column])
        timestamp = normalize_behavioral_timestamp(row[timestamp_column]) if timestamp_column else datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=int(source_position))
        session_id = derive_session_id(participant_key, timestamp, row[session_column] if session_column and session_column in row else None)
        event_type = _event_type_from_row(row, event_type_column)
        output: dict[str, Any] = {
            "event_id": generate_behavioral_event_id(participant_key, session_id, timestamp, int(source_position), source_fingerprint),
            "participant_key": participant_key,
            "event_timestamp": timestamp,
            "session_id": session_id,
            "event_type": event_type,
            "page_or_context": row["page_or_context"] if "page_or_context" in row and not pd.isna(row["page_or_context"]) else None,
            "validation_warnings": "",
        }
        source_to_canonical = {field.source_field: field.canonical_field for field in fields.values()}
        legacy_aliases = {
            "TypingSpeed_cpm": "typing_speed_cpm",
            "KeystrokeVar_ms": "key_flight_time_ms",
            "MouseSpeed_pxs": "mouse_speed_px_per_second",
            "HesitationPauses": "hesitation_count",
            "SessionDuration_min": "session_duration_seconds",
            "ResponseTime_sec": "response_latency_ms",
        }
        for source, canonical in {**source_to_canonical, **legacy_aliases}.items():
            if canonical in {"participant_key", "event_timestamp", "session_id", "event_type", "page_or_context"} or source not in row:
                continue
            value = _numeric_optional(row[source])
            if source == "SessionDuration_min" and value is not None:
                value *= 60.0
            if source == "ResponseTime_sec" and value is not None:
                value *= 1000.0
            output[canonical] = value
        rows.append(output)

    canonical = pd.DataFrame(rows)
    canonical = canonical.sort_values(["participant_key", "session_id", "event_timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    validate_preprocessed_behavioral(canonical)
    return canonical


def aggregate_behavioral_sessions(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["session_id", "participant_key", "session_start", "session_end", "event_count", "typing_event_count", "mouse_event_count", "prompt_response_count", "session_features", "validation_warnings"])
    events = events_df.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    events = events.sort_values(["participant_key", "session_id", "event_timestamp", "event_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    previous_end_by_participant: dict[str, pd.Timestamp] = {}
    for (participant, session_id), group in events.groupby(["participant_key", "session_id"], sort=True):
        group = group.sort_values(["event_timestamp", "event_id"], kind="mergesort")
        features = calculate_session_features(group, previous_end_by_participant.get(str(participant)))
        start = group["event_timestamp"].min()
        end = group["event_timestamp"].max()
        rows.append(
            {
                "session_id": session_id,
                "participant_key": participant,
                "session_start": start.isoformat(),
                "session_end": end.isoformat(),
                "event_count": int(len(group)),
                "typing_event_count": int(group["event_type"].eq("typing_timing").sum()),
                "mouse_event_count": int(group["event_type"].eq("mouse_aggregate").sum()),
                "prompt_response_count": int(group["event_type"].eq("prompt_response").sum()),
                "session_features": features,
                "validation_warnings": [],
            }
        )
        previous_end_by_participant[str(participant)] = end
    return pd.DataFrame(rows).sort_values(["participant_key", "session_start", "session_id"], kind="mergesort").reset_index(drop=True)


def assess_participant_baseline_eligibility(
    events_df: pd.DataFrame,
    *,
    minimum_sessions: int = DEFAULT_MINIMUM_BASELINE_SESSIONS,
    minimum_days: int = DEFAULT_MINIMUM_BASELINE_DAYS,
    minimum_events: int = DEFAULT_MINIMUM_BASELINE_EVENTS,
) -> dict[str, Any]:
    return assess_baseline_eligibility(events_df, minimum_sessions=minimum_sessions, minimum_days=minimum_days, minimum_events=minimum_events)


def calculate_baseline_statistics(prior_feature_rows: pd.DataFrame, feature_columns: list[str] | None = None) -> dict[str, Any]:
    columns = feature_columns or list(FEATURE_COLUMNS)
    if prior_feature_rows.empty:
        return {"observation_count": 0, "feature_means": {}, "feature_standard_deviations": {}, "feature_medians": {}, "feature_iqrs": {}, "warnings": ["insufficient_history"]}
    stats = {"observation_count": int(len(prior_feature_rows)), "feature_means": {}, "feature_standard_deviations": {}, "feature_medians": {}, "feature_iqrs": {}, "warnings": []}
    for column in columns:
        if column not in prior_feature_rows.columns:
            continue
        values = pd.to_numeric(prior_feature_rows[column], errors="coerce").dropna()
        if values.empty:
            continue
        stats["feature_means"][column] = round(float(values.mean()), 6)
        stats["feature_standard_deviations"][column] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
        stats["feature_medians"][column] = round(float(values.median()), 6)
        stats["feature_iqrs"][column] = round(float(values.quantile(0.75) - values.quantile(0.25)), 6)
    return stats


def calculate_baseline_deviation_features(feature_df: pd.DataFrame, *, minimum_prior_observations: int = 3) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df.copy()
    rows: list[dict[str, Any]] = []
    working = feature_df.copy()
    working["feature_timestamp"] = pd.to_datetime(working["feature_timestamp"], utc=True)
    working = working.sort_values(["participant_key", "feature_timestamp", "session_id"], kind="mergesort")
    for _, group in working.groupby("participant_key", sort=True):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            prior = group.iloc[:idx]
            output = row.to_dict()
            if len(prior) < minimum_prior_observations:
                output["baseline_status"] = "insufficient_history"
            else:
                stats = calculate_baseline_statistics(prior)
                for column in FEATURE_COLUMNS:
                    value = row.get(column)
                    mean = stats["feature_means"].get(column)
                    std = stats["feature_standard_deviations"].get(column)
                    if value is not None and pd.notna(value) and mean is not None:
                        output[f"{column}_deviation_from_prior_mean"] = round(float(value) - float(mean), 6)
                        if std and std > 0:
                            output[f"{column}_prior_z"] = round((float(value) - float(mean)) / float(std), 6)
                output["baseline_status"] = "prior_only"
            rows.append(output)
    return pd.DataFrame(rows)


def validate_preprocessed_behavioral(canonical_df: pd.DataFrame) -> None:
    validate_preprocessed_columns(canonical_df)
    duplicates = detect_duplicate_events(canonical_df, subset=["event_id"])
    if duplicates["duplicate_count"]:
        raise ValueError("Canonical behavioral event IDs must be unique")


def _missing_summary(df: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in df.isna().sum().items() if int(count) > 0}


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.items()})
    return records


def _write_json(payload: dict[str, Any], output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return output_path


def _write_csv(df: pd.DataFrame, output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda value: isinstance(value, (dict, list))).any():
            safe[column] = safe[column].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
    safe.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_path


def build_behavioral_report(
    source_df: pd.DataFrame,
    events_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    source_type: str,
    minimum_baseline_sessions: int,
    minimum_baseline_days: int,
) -> BehavioralPreprocessingReport:
    timestamps = pd.to_datetime(events_df["event_timestamp"], utc=True) if len(events_df) else pd.Series([], dtype="datetime64[ns, UTC]")
    duplicates = detect_duplicate_events(events_df) if len(events_df) else {"duplicate_count": 0}
    sparse = detect_sparse_participants(events_df, minimum_events=DEFAULT_MINIMUM_BASELINE_EVENTS) if len(events_df) else {"sparse_participants": {}}
    eligibility = assess_participant_baseline_eligibility(events_df, minimum_sessions=minimum_baseline_sessions, minimum_days=minimum_baseline_days) if len(events_df) else {"eligible_participant_count": 0}
    observations = events_df["participant_key"].value_counts() if len(events_df) else pd.Series(dtype=int)
    unavailable = [column for column in FEATURE_COLUMNS if column not in feature_df.columns or feature_df[column].isna().all()] if len(feature_df) else list(FEATURE_COLUMNS)
    readiness = READINESS_UNSUITABLE if source_type in {SOURCE_STATUS_REAL_OFFLINE_DATASET, SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET} else READINESS_ENGINEERING_ONLY
    return BehavioralPreprocessingReport(
        preprocessing_version=BEHAVIORAL_PREPROCESSING_VERSION,
        feature_schema_version=BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        mapping_version=BEHAVIORAL_MAPPING_VERSION,
        source_type=source_type,
        source_record_count=int(len(source_df)),
        output_event_count=int(len(events_df)),
        output_session_count=int(len(sessions_df)),
        participant_count=int(events_df["participant_key"].nunique()) if len(events_df) else 0,
        date_range={"min": timestamps.min().isoformat() if len(timestamps) else None, "max": timestamps.max().isoformat() if len(timestamps) else None},
        missing_value_summary=_missing_summary(source_df),
        invalid_event_count=0,
        duplicate_event_count=int(duplicates["duplicate_count"]),
        observations_per_participant_summary={
            "minimum": int(observations.min()) if len(observations) else 0,
            "median": float(observations.median()) if len(observations) else 0,
            "maximum": int(observations.max()) if len(observations) else 0,
            "sparse_participant_count": len(sparse.get("sparse_participants", {})),
        },
        baseline_eligible_participant_count=int(eligibility["eligible_participant_count"]),
        feature_columns=list(FEATURE_COLUMNS),
        unavailable_features=unavailable,
        privacy_warnings=[
            "Timing-only telemetry is permitted; typed characters, clipboard content, screen content, and password-field telemetry are prohibited.",
            "Behavioral features are indirect and must not trigger autonomous crisis decisions.",
            "Participant keys in generated outputs are hashed and are not predictive features.",
        ],
        readiness_status=readiness,
    )


def preprocess_behavioral_dataframe(
    source_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None,
    mapping_config: BehavioralMappingConfig,
    *,
    source_fingerprint: str,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    source_type: str = SOURCE_STATUS_REAL_OFFLINE_DATASET,
    max_participants: int | None = None,
    max_records: int | None = None,
    minimum_baseline_sessions: int = DEFAULT_MINIMUM_BASELINE_SESSIONS,
    minimum_baseline_days: int = DEFAULT_MINIMUM_BASELINE_DAYS,
) -> dict[str, Any]:
    del preprocessing_config
    events_df = canonicalize_behavioral_events(source_df, mapping_config, source_fingerprint=source_fingerprint, max_participants=max_participants, max_records=max_records)
    sessions_df = aggregate_behavioral_sessions(events_df)
    feature_df = build_behavioral_feature_table(sessions_df)
    feature_with_baseline_df = calculate_baseline_deviation_features(feature_df)
    report = build_behavioral_report(
        source_df,
        events_df,
        sessions_df,
        feature_df,
        source_type=source_type,
        minimum_baseline_sessions=minimum_baseline_sessions,
        minimum_baseline_days=minimum_baseline_days,
    )
    feature_schema = build_behavioral_feature_schema()
    eligibility = assess_participant_baseline_eligibility(events_df, minimum_sessions=minimum_baseline_sessions, minimum_days=minimum_baseline_days)
    privacy_validation = {"valid": True, "policy": "timing_only_no_content", "checked_at": datetime.now(timezone.utc).isoformat()}
    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Behavioral preprocessing outputs must be under generated/")
        from app.ml.preprocessing.behavioral.reporting import create_behavioral_preprocessing_markdown

        outputs = {
            "events_csv": str(_write_csv(events_df, resolved_output_dir / "behavioral_events_canonical.csv", overwrite=overwrite)),
            "sessions_csv": str(_write_csv(sessions_df, resolved_output_dir / "behavioral_sessions.csv", overwrite=overwrite)),
            "features_csv": str(_write_csv(feature_with_baseline_df, resolved_output_dir / "behavioral_features.csv", overwrite=overwrite)),
            "feature_schema_json": str(_write_json(feature_schema, resolved_output_dir / "behavioral_feature_schema.json", overwrite=overwrite)),
            "report_json": str(_write_json(report.to_safe_dict(), resolved_output_dir / "behavioral_preprocessing_report.json", overwrite=overwrite)),
            "record_manifest_json": str(_write_json({"record_count": int(len(events_df)), "record_ids": events_df["event_id"].tolist(), "source_type": source_type}, resolved_output_dir / "behavioral_record_manifest.json", overwrite=overwrite)),
            "baseline_eligibility_json": str(_write_json(eligibility, resolved_output_dir / "behavioral_baseline_eligibility.json", overwrite=overwrite)),
            "privacy_validation_json": str(_write_json(privacy_validation, resolved_output_dir / "behavioral_privacy_validation.json", overwrite=overwrite)),
        }
        md_path = resolved_output_dir / "behavioral_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_behavioral_preprocessing_markdown(report, feature_schema), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)
    return {
        "valid": True,
        "validate_only": validate_only,
        "source_type": source_type,
        "source_rows": int(len(source_df)),
        "output_events": int(len(events_df)),
        "output_sessions": int(len(sessions_df)),
        "participant_count": int(events_df["participant_key"].nunique()) if len(events_df) else 0,
        "date_range": report.date_range,
        "feature_columns": list(FEATURE_COLUMNS),
        "baseline_eligible_participant_count": int(report.baseline_eligible_participant_count),
        "report": report,
        "feature_schema": feature_schema,
        "outputs": outputs,
    }


def preprocess_behavioral_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    mapping_config: BehavioralMappingConfig,
    fingerprint: DatasetFingerprint,
    *,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_participants: int | None = None,
    max_records: int | None = None,
    minimum_baseline_sessions: int = DEFAULT_MINIMUM_BASELINE_SESSIONS,
    minimum_baseline_days: int = DEFAULT_MINIMUM_BASELINE_DAYS,
) -> dict[str, Any]:
    source_df = load_behavioral_source(dataset_config)
    return preprocess_behavioral_dataframe(
        source_df,
        preprocessing_config,
        mapping_config,
        source_fingerprint=fingerprint.combined_sha256,
        output_dir=output_dir,
        overwrite=overwrite,
        validate_only=validate_only,
        source_type=SOURCE_STATUS_REAL_OFFLINE_DATASET,
        max_participants=max_participants,
        max_records=max_records,
        minimum_baseline_sessions=minimum_baseline_sessions,
        minimum_baseline_days=minimum_baseline_days,
    )


def write_schema_only_outputs(output_dir: Path, *, overwrite: bool = False, source_status: str = SOURCE_STATUS_NO_BEHAVIORAL_DATA) -> dict[str, str]:
    resolved = output_dir.resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if not paths.is_path_inside(paths.get_generated_root(), resolved):
        raise ValueError("Behavioral schema-only outputs must be under generated/")
    schema = build_behavioral_feature_schema()
    readiness = {
        "source_status": source_status,
        "readiness_status": READINESS_ENGINEERING_ONLY,
        "model_training_blocked": True,
        "reason": "No usable real offline behavioral dataset is available; schema and synthetic engineering fixtures only.",
        "preprocessing_version": BEHAVIORAL_PREPROCESSING_VERSION,
        "feature_schema_version": BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        "mapping_version": BEHAVIORAL_MAPPING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "feature_schema_json": str(_write_json(schema, resolved / "behavioral_feature_schema.json", overwrite=overwrite)),
        "readiness_report_json": str(_write_json(readiness, resolved / "behavioral_readiness_report.json", overwrite=overwrite)),
    }


def classify_behavioral_source_status(files: list[Path], *, final_dataset_empty: bool = False) -> str:
    if final_dataset_empty and not files:
        return SOURCE_STATUS_NO_BEHAVIORAL_DATA
    if not files:
        return SOURCE_STATUS_NO_BEHAVIORAL_DATA
    real_like = False
    partial = False
    synthetic = False
    for path in files:
        lower = str(path).lower()
        if "synthetic" in lower or "weight detection" in lower or "projext_plan-deltelater" in lower:
            synthetic = True
        if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue
        try:
            sample = pd.read_csv(path, nrows=5) if path.suffix.lower() == ".csv" else pd.read_json(path, lines=path.suffix.lower() == ".jsonl").head(5)
        except Exception:
            continue
        columns = {str(column).lower() for column in sample.columns}
        has_id = bool(columns & {"participantid", "participant_id", "student_id", "user_id"})
        has_time = bool(columns & {"timestamp", "event_timestamp", "session_start", "session_end", "date"})
        has_behavior = bool(columns & {"typingspeed_cpm", "typing_speed_cpm", "response_latency_ms", "responsetime_sec", "mouse_speed_px_per_second", "mousespeed_pxs", "hesitationpauses"})
        if has_id and has_time and has_behavior:
            real_like = True
        elif has_id and has_behavior:
            partial = True
    if real_like and not synthetic:
        return SOURCE_STATUS_REAL_OFFLINE_DATASET
    if real_like or partial:
        return SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY if synthetic and not real_like else SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET
    return SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY if synthetic else SOURCE_STATUS_NO_BEHAVIORAL_DATA


def discover_behavioral_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = []
    for pattern in ("*.csv", "*.json", "*.jsonl", "*.log"):
        candidates.extend(root.rglob(pattern))
    behavioral_tokens = ("behavior", "typing", "mouse", "telemetry", "session", "hesitation", "response")
    return sorted(path for path in candidates if any(token in path.name.lower() for token in behavioral_tokens))

