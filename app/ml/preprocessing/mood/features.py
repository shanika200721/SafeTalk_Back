"""Leakage-safe temporal features for Daily Mood preprocessing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.ml.common.schemas import FeatureDefinition, FeatureSchema, Modality
from app.ml.preprocessing.mood.constants import (
    FEATURE_COLUMNS,
    LOW_MOOD_THRESHOLD,
    MOOD_FEATURE_SCHEMA_VERSION,
    MOOD_MAX,
    MOOD_MIN,
    MOOD_PREPROCESSING_VERSION,
    SUDDEN_DETERIORATION_DROP,
)


def _slope(values: pd.Series) -> float:
    clean = values.dropna().astype(float).to_numpy()
    if len(clean) < 2:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean, 1)[0])


def _consecutive_low(values: pd.Series) -> int:
    count = 0
    for value in reversed(values.dropna().astype(float).tolist()):
        if value <= LOW_MOOD_THRESHOLD:
            count += 1
        else:
            break
    return count


def _checkins_last_days(group: pd.DataFrame, timestamp: pd.Timestamp, days: int = 7) -> int:
    start = timestamp - pd.Timedelta(days=days)
    return int(((group["timestamp"] >= start) & (group["timestamp"] <= timestamp)).sum())


def _missing_day_ratio(group: pd.DataFrame, timestamp: pd.Timestamp, days: int = 7) -> float:
    start = timestamp - pd.Timedelta(days=days - 1)
    observed_days = group.loc[(group["timestamp"] >= start) & (group["timestamp"] <= timestamp), "timestamp"].dt.floor("D").nunique()
    return round(float(max(0, days - observed_days) / days), 6)


def compute_mood_trend_score(row: pd.Series | dict[str, Any]) -> float:
    """Experimental non-clinical trend score. Higher means more deterioration in recent mood signals."""
    get = row.get if isinstance(row, dict) else row.get
    current = get("current_mood")
    change = get("mood_change_from_previous")
    low_ratio = get("low_mood_ratio_last_7_observations")
    missing_ratio = get("missing_day_ratio_last_7_days")
    sudden = get("sudden_deterioration_flag")
    slope7 = get("slope_last_7_observations")

    current_component = 0.0 if pd.isna(current) else ((MOOD_MAX - float(current)) / (MOOD_MAX - MOOD_MIN)) * 35.0
    deterioration_component = 0.0 if pd.isna(change) else max(0.0, -float(change)) / (MOOD_MAX - MOOD_MIN) * 20.0
    low_ratio_component = 0.0 if pd.isna(low_ratio) else float(low_ratio) * 20.0
    missing_component = 0.0 if pd.isna(missing_ratio) else float(missing_ratio) * 10.0
    sudden_component = 10.0 if bool(sudden) else 0.0
    slope_component = 0.0 if pd.isna(slope7) else max(0.0, -float(slope7)) / (MOOD_MAX - MOOD_MIN) * 5.0
    score = current_component + deterioration_component + low_ratio_component + missing_component + sudden_component + slope_component
    return round(float(min(100.0, max(0.0, score))), 6)


def build_mood_feature_table(canonical_df: pd.DataFrame) -> pd.DataFrame:
    required = {"record_id", "participant_key", "timestamp", "mood_value"}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"Cannot build mood features; missing columns: {sorted(missing)}")

    df = canonical_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["participant_key", "timestamp", "record_id"], kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for _, group in df.groupby("participant_key", sort=True):
        group = group.sort_values(["timestamp", "record_id"], kind="mergesort").reset_index(drop=True)
        mood = group["mood_value"].astype(float)
        previous = mood.shift(1)
        rolling3 = mood.rolling(window=3, min_periods=1)
        rolling7 = mood.rolling(window=7, min_periods=1)
        crying = group["crying_episode_count"] if "crying_episode_count" in group else pd.Series([np.nan] * len(group))
        physical = group["physical_symptom_count"] if "physical_symptom_count" in group else pd.Series([np.nan] * len(group))

        for idx, source_row in group.iterrows():
            timestamp = source_row["timestamp"]
            history = group.iloc[: idx + 1]
            last7 = mood.iloc[max(0, idx - 6) : idx + 1]
            feature_row: dict[str, Any] = {
                "record_id": source_row["record_id"],
                "participant_key": source_row["participant_key"],
                "feature_timestamp": timestamp.isoformat(),
                "current_mood": float(mood.iloc[idx]),
                "previous_mood": np.nan if pd.isna(previous.iloc[idx]) else float(previous.iloc[idx]),
                "mood_change_from_previous": np.nan if pd.isna(previous.iloc[idx]) else float(mood.iloc[idx] - previous.iloc[idx]),
                "rolling_mean_3_observations": float(rolling3.mean().iloc[idx]),
                "rolling_mean_7_observations": float(rolling7.mean().iloc[idx]),
                "rolling_std_3_observations": np.nan if idx < 1 else float(rolling3.std(ddof=1).iloc[idx]),
                "rolling_std_7_observations": np.nan if idx < 1 else float(rolling7.std(ddof=1).iloc[idx]),
                "slope_last_3_observations": _slope(mood.iloc[max(0, idx - 2) : idx + 1]),
                "slope_last_7_observations": _slope(last7),
                "consecutive_low_mood_count": _consecutive_low(mood.iloc[: idx + 1]),
                "low_mood_ratio_last_7_observations": round(float((last7 <= LOW_MOOD_THRESHOLD).sum() / len(last7)), 6),
                "days_since_previous_checkin": np.nan if idx == 0 else round(float((timestamp - group.loc[idx - 1, "timestamp"]).total_seconds() / 86400), 6),
                "checkins_last_7_days": _checkins_last_days(history, timestamp, 7),
                "missing_day_ratio_last_7_days": _missing_day_ratio(history, timestamp, 7),
                "sudden_deterioration_flag": bool(idx > 0 and (mood.iloc[idx - 1] - mood.iloc[idx]) >= SUDDEN_DETERIORATION_DROP),
                "crying_episode_trend": _slope(crying.iloc[max(0, idx - 2) : idx + 1]),
                "physical_symptom_trend": _slope(physical.iloc[max(0, idx - 2) : idx + 1]),
                "history_length": int(idx + 1),
            }
            available = sum(0 if pd.isna(feature_row[name]) else 1 for name in FEATURE_COLUMNS if name not in {"mood_trend_score_0_100", "data_completeness"})
            denominator = len(FEATURE_COLUMNS) - 2
            feature_row["data_completeness"] = round(float(available / denominator), 6)
            feature_row["mood_trend_score_0_100"] = compute_mood_trend_score(feature_row)
            rows.append(feature_row)

    feature_df = pd.DataFrame(rows)
    validate_mood_features(feature_df)
    return feature_df


def validate_mood_features(feature_df: pd.DataFrame) -> None:
    for column in FEATURE_COLUMNS:
        if column not in feature_df.columns:
            raise ValueError(f"Missing mood feature column: {column}")
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "sudden_deterioration_flag"]
    for column in numeric_columns:
        values = pd.to_numeric(feature_df[column], errors="coerce").dropna().tolist()
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Mood feature contains infinity: {column}")
    scores = pd.to_numeric(feature_df["mood_trend_score_0_100"], errors="coerce")
    if bool(((scores < 0) | (scores > 100)).any()):
        raise ValueError("mood_trend_score_0_100 must be bounded 0-100")
    blocked = {"risk_level", "alert", "recommendation", "suicide_risk"}
    if blocked & set(feature_df.columns):
        raise ValueError("Mood features must not include risk classification, alerts, or recommendations")


def build_mood_feature_schema(*, dataset_name: str, dataset_version: str) -> FeatureSchema:
    definitions = []
    metadata: dict[str, tuple[str, bool, float | None, float | None, str]] = {
        "current_mood": ("float", False, 1, 5, "current observation only"),
        "previous_mood": ("float", True, 1, 5, "previous observation only"),
        "mood_change_from_previous": ("float", True, -4, 4, "current minus previous observation"),
        "rolling_mean_3_observations": ("float", False, 1, 5, "current and prior two observations"),
        "rolling_mean_7_observations": ("float", False, 1, 5, "current and prior six observations"),
        "rolling_std_3_observations": ("float", True, 0, 4, "current and prior two observations"),
        "rolling_std_7_observations": ("float", True, 0, 4, "current and prior six observations"),
        "slope_last_3_observations": ("float", True, None, None, "current and prior two observations"),
        "slope_last_7_observations": ("float", True, None, None, "current and prior six observations"),
        "consecutive_low_mood_count": ("integer", False, 0, None, "all past observations through current"),
        "low_mood_ratio_last_7_observations": ("float", False, 0, 1, "current and prior six observations"),
        "days_since_previous_checkin": ("float", True, 0, None, "current and immediately previous timestamp"),
        "checkins_last_7_days": ("integer", False, 1, 7, "current and prior 7 calendar days"),
        "missing_day_ratio_last_7_days": ("float", False, 0, 1, "current and prior 7 calendar days"),
        "sudden_deterioration_flag": ("boolean", False, None, None, "current and immediately previous mood"),
        "crying_episode_trend": ("float", True, None, None, "current and prior two observations if source field exists"),
        "physical_symptom_trend": ("float", True, None, None, "current and prior two observations if source field exists"),
        "history_length": ("integer", False, 1, None, "count of observations through current row"),
        "data_completeness": ("float", False, 0, 1, "available engineered values for this row"),
        "mood_trend_score_0_100": ("float", False, 0, 100, "experimental non-clinical trend formula"),
    }
    for name in FEATURE_COLUMNS:
        dtype, nullable, minimum, maximum, window = metadata[name]
        source_columns = ["Mood"]
        if "crying" in name:
            source_columns = ["CryingEpisodes"]
        elif "physical" in name:
            source_columns = ["PhysicalPain"]
        elif name in {"days_since_previous_checkin", "checkins_last_7_days", "missing_day_ratio_last_7_days"}:
            source_columns = ["Date"]
        definitions.append(
            FeatureDefinition(
                name=name,
                dtype=dtype,
                description=f"{name}; leakage-safe temporal window: {window}.",
                source_columns=source_columns,
                nullable=nullable,
                minimum=minimum,
                maximum=maximum,
                preprocessing_step="build_mood_feature_table; no centered windows, no backfill, no interpolation",
            )
        )
    return FeatureSchema(
        schema_name="daily-mood-temporal-features",
        feature_schema_version=MOOD_FEATURE_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        preprocessing_version=MOOD_PREPROCESSING_VERSION,
        modality=Modality.MOOD,
        features=definitions,
        target_columns=[],
        excluded_columns=["student_id", "source_record_id", "raw notes", "names", "emails"],
        created_at=pd.Timestamp.utcnow().to_pydatetime(),
        notes=(
            "Daily Mood features are temporal engineering features only. "
            "The mood_trend_score_0_100 is experimental and non-clinical; it creates no alerts, risk levels, or recommendations."
        ),
    )
