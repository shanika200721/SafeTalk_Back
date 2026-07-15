"""Explainable behavioral feature engineering without model training."""

from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import numpy as np
import pandas as pd

from app.ml.common.schemas import FeatureDefinition, FeatureSchema, Modality
from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    FEATURE_COLUMNS,
)


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 6)


def _safe_std(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return None
    return round(float(values.std(ddof=1)), 6)


def _safe_sum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.sum()), 6)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def calculate_session_features(session_events: pd.DataFrame, previous_session_end: pd.Timestamp | None = None) -> dict[str, Any]:
    if session_events.empty:
        raise ValueError("Cannot calculate behavioral features for an empty session")
    df = session_events.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df = df.sort_values(["event_timestamp", "event_id"], kind="mergesort")
    event_count = int(len(df))
    start = df["event_timestamp"].min()
    end = df["event_timestamp"].max()
    observed_span = max(0.0, float((end - start).total_seconds()))
    supplied_duration = _safe_mean(df["session_duration_seconds"]) if "session_duration_seconds" in df.columns else None
    session_duration = supplied_duration if supplied_duration is not None else observed_span

    typing_mask = df["event_type"].eq("typing_timing")
    mouse_mask = df["event_type"].eq("mouse_aggregate")
    response_mask = df["event_type"].eq("prompt_response")
    page_transitions = int(df["page_or_context"].fillna("").astype(str).replace("", np.nan).dropna().nunique() - 1) if "page_or_context" in df.columns else 0
    page_transitions = max(0, page_transitions)
    pauses = pd.to_numeric(df.get("key_flight_time_ms", pd.Series(dtype=float)), errors="coerce")
    long_pauses = int((pauses > 2000).sum())
    key_event_count = int(typing_mask.sum())
    idle_time = round(float(max(0.0, pauses[pauses > 2000].sum() / 1000.0)), 6) if not pauses.dropna().empty else 0.0
    active_time = round(float(max(0.0, session_duration - idle_time)), 6) if session_duration is not None else None

    features = {
        "key_event_count": key_event_count,
        "typing_duration_seconds": active_time if key_event_count else None,
        "typing_speed_cpm": _safe_mean(df.loc[typing_mask, "typing_speed_cpm"]) if "typing_speed_cpm" in df else None,
        "dwell_time_mean": _safe_mean(df.loc[typing_mask, "key_dwell_time_ms"]) if "key_dwell_time_ms" in df else None,
        "dwell_time_std": _safe_std(df.loc[typing_mask, "key_dwell_time_ms"]) if "key_dwell_time_ms" in df else None,
        "flight_time_mean": _safe_mean(df.loc[typing_mask, "key_flight_time_ms"]) if "key_flight_time_ms" in df else None,
        "flight_time_std": _safe_std(df.loc[typing_mask, "key_flight_time_ms"]) if "key_flight_time_ms" in df else None,
        "backspace_rate": _ratio(_safe_sum(df.loc[typing_mask, "backspace_count"]) if "backspace_count" in df else None, key_event_count),
        "correction_rate": _ratio(_safe_sum(df.loc[typing_mask, "correction_count"]) if "correction_count" in df else None, key_event_count),
        "pause_count": long_pauses,
        "long_pause_ratio": _ratio(long_pauses, key_event_count),
        "mouse_event_count": int(mouse_mask.sum()),
        "path_distance": _safe_sum(df.loc[mouse_mask, "mouse_distance_px"]) if "mouse_distance_px" in df else None,
        "mean_speed": _safe_mean(df.loc[mouse_mask, "mouse_speed_px_per_second"]) if "mouse_speed_px_per_second" in df else None,
        "speed_variability": _safe_std(df.loc[mouse_mask, "mouse_speed_px_per_second"]) if "mouse_speed_px_per_second" in df else None,
        "click_count": _safe_sum(df.loc[mouse_mask, "click_count"]) if "click_count" in df else None,
        "hesitation_count": _safe_sum(df["hesitation_count"]) if "hesitation_count" in df else None,
        "idle_ratio": _ratio(idle_time, session_duration),
        "prompt_response_count": int(response_mask.sum()),
        "response_latency_mean": _safe_mean(df.loc[response_mask, "response_latency_ms"]) if "response_latency_ms" in df else None,
        "response_latency_std": _safe_std(df.loc[response_mask, "response_latency_ms"]) if "response_latency_ms" in df else None,
        "skipped_prompt_count": int(response_mask.sum() - pd.to_numeric(df.loc[response_mask, "response_latency_ms"], errors="coerce").notna().sum()) if "response_latency_ms" in df else 0,
        "session_duration": round(float(session_duration), 6) if session_duration is not None else None,
        "event_count": event_count,
        "active_time": active_time,
        "idle_time": idle_time,
        "page_transition_count": page_transitions,
        "sessions_per_day": None,
        "time_since_previous_session": round(float((start - previous_session_end).total_seconds()), 6) if previous_session_end is not None else None,
    }
    return {column: features.get(column) for column in FEATURE_COLUMNS}


def build_behavioral_feature_table(sessions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if sessions_df.empty:
        return pd.DataFrame(columns=["record_id", "participant_key", "feature_timestamp", "session_id", *FEATURE_COLUMNS, "data_completeness", "warnings"])
    working = sessions_df.copy()
    working["session_start"] = pd.to_datetime(working["session_start"], utc=True)
    working["session_end"] = pd.to_datetime(working["session_end"], utc=True)
    working = working.sort_values(["participant_key", "session_start", "session_id"], kind="mergesort")
    for participant, group in working.groupby("participant_key", sort=True):
        daily_counts = group["session_start"].dt.floor("D").value_counts().to_dict()
        previous_end = None
        for _, row in group.iterrows():
            features = dict(row["session_features"])
            features["sessions_per_day"] = int(daily_counts[row["session_start"].floor("D")])
            features["time_since_previous_session"] = round(float((row["session_start"] - previous_end).total_seconds()), 6) if previous_end is not None else None
            available = sum(1 for column in FEATURE_COLUMNS if features.get(column) is not None)
            output = {
                "record_id": f"{row['session_id']}-features",
                "participant_key": participant,
                "feature_timestamp": row["session_end"].isoformat(),
                "session_id": row["session_id"],
                **{column: features.get(column) for column in FEATURE_COLUMNS},
                "data_completeness": round(float(available / len(FEATURE_COLUMNS)), 6),
                "warnings": ";".join(row.get("validation_warnings", [])) if isinstance(row.get("validation_warnings"), list) else "",
            }
            rows.append(output)
            previous_end = row["session_end"]
    feature_df = pd.DataFrame(rows)
    validate_behavioral_features(feature_df)
    return feature_df


def validate_behavioral_features(feature_df: pd.DataFrame) -> None:
    blocked = {"risk_level", "risk_class", "anomaly_score", "clinical_label", "suicide_risk"}
    if blocked & set(feature_df.columns):
        raise ValueError("Behavioral features must not include risk classes, clinical labels, or anomaly scores")
    for column in FEATURE_COLUMNS:
        if column not in feature_df.columns:
            raise ValueError(f"Missing behavioral feature column: {column}")
        numeric = pd.to_numeric(feature_df[column], errors="coerce").dropna()
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"Behavioral feature contains infinity: {column}")


def build_behavioral_feature_schema(*, dataset_name: str = "behavioral-telemetry", dataset_version: str = "v1") -> dict[str, Any]:
    metadata: dict[str, tuple[str, list[str], str, float | None, float | None, bool, str]] = {
        "key_event_count": ("integer", ["event_type"], "count", 0, None, False, "timing"),
        "typing_duration_seconds": ("float", ["event_timestamp", "key_flight_time_ms"], "seconds", 0, None, True, "timing"),
        "typing_speed_cpm": ("float", ["typing_speed_cpm"], "characters/minute aggregate", 0, 1200, True, "timing"),
        "dwell_time_mean": ("float", ["key_dwell_time_ms"], "milliseconds", 0, 5000, True, "timing"),
        "dwell_time_std": ("float", ["key_dwell_time_ms"], "milliseconds", 0, None, True, "timing"),
        "flight_time_mean": ("float", ["key_flight_time_ms"], "milliseconds", 0, 60000, True, "timing"),
        "flight_time_std": ("float", ["key_flight_time_ms"], "milliseconds", 0, None, True, "timing"),
        "backspace_rate": ("float", ["backspace_count"], "count/event", 0, None, True, "aggregate"),
        "correction_rate": ("float", ["correction_count"], "count/event", 0, None, True, "aggregate"),
        "pause_count": ("integer", ["key_flight_time_ms"], "count", 0, None, False, "timing"),
        "long_pause_ratio": ("float", ["key_flight_time_ms"], "ratio", 0, 1, True, "timing"),
        "mouse_event_count": ("integer", ["event_type"], "count", 0, None, False, "interaction_aggregate"),
        "path_distance": ("float", ["mouse_distance_px"], "pixels", 0, None, True, "interaction_aggregate"),
        "mean_speed": ("float", ["mouse_speed_px_per_second"], "pixels/second", 0, 20000, True, "interaction_aggregate"),
        "speed_variability": ("float", ["mouse_speed_px_per_second"], "pixels/second", 0, None, True, "interaction_aggregate"),
        "click_count": ("float", ["click_count"], "count", 0, None, True, "interaction_aggregate"),
        "hesitation_count": ("float", ["hesitation_count"], "count", 0, None, True, "interaction_aggregate"),
        "idle_ratio": ("float", ["key_flight_time_ms", "session_duration_seconds"], "ratio", 0, 1, True, "timing"),
        "prompt_response_count": ("integer", ["event_type"], "count", 0, None, False, "timing"),
        "response_latency_mean": ("float", ["response_latency_ms"], "milliseconds", 0, 3600000, True, "timing"),
        "response_latency_std": ("float", ["response_latency_ms"], "milliseconds", 0, None, True, "timing"),
        "skipped_prompt_count": ("integer", ["response_latency_ms"], "count", 0, None, False, "timing"),
        "session_duration": ("float", ["event_timestamp", "session_duration_seconds"], "seconds", 0, 86400, True, "session_aggregate"),
        "event_count": ("integer", ["event_type"], "count", 0, None, False, "session_aggregate"),
        "active_time": ("float", ["event_timestamp", "key_flight_time_ms"], "seconds", 0, None, True, "session_aggregate"),
        "idle_time": ("float", ["key_flight_time_ms"], "seconds", 0, None, False, "session_aggregate"),
        "page_transition_count": ("integer", ["page_or_context"], "count", 0, None, False, "context"),
        "sessions_per_day": ("integer", ["session_id", "event_timestamp"], "count/day", 1, None, True, "session_aggregate"),
        "time_since_previous_session": ("float", ["session_id", "event_timestamp"], "seconds", 0, None, True, "prior_sessions_only"),
    }
    features = []
    for name in FEATURE_COLUMNS:
        dtype, source_fields, units, minimum, maximum, nullable, privacy_classification = metadata[name]
        features.append(
            {
                "name": name,
                "dtype": dtype,
                "source_event_fields": source_fields,
                "aggregation_window": "within_session; participant prior sessions only where noted",
                "units": units,
                "minimum": minimum,
                "maximum": maximum,
                "nullable": nullable,
                "baseline_requirement": "participant-specific baselines require configurable minimum history",
                "leakage_safety_rule": "uses only events in the current session and prior sessions; never future observations",
                "privacy_classification": privacy_classification,
                "non_clinical_status": "engineering feature only; not a clinical label, risk class, or anomaly score",
            }
        )
    return {
        "schema_name": "behavioral-session-features",
        "feature_schema_version": BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "preprocessing_version": BEHAVIORAL_PREPROCESSING_VERSION,
        "modality": Modality.BEHAVIORAL.value,
        "features": features,
        "target_columns": [],
        "excluded_columns": ["participant_key", "session_id", "feature_timestamp", "page_or_context", "engineering_scenario"],
        "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "notes": "Behavioral features are timing and aggregate interaction telemetry only. They do not create clinical labels, anomaly scores, alerts, or treatment recommendations.",
    }


def supported_feature_names(source_columns: set[str]) -> list[str]:
    schema = build_behavioral_feature_schema()
    return [
        feature["name"]
        for feature in schema["features"]
        if any(field in source_columns or field == "event_type" for field in feature["source_event_fields"])
    ]

