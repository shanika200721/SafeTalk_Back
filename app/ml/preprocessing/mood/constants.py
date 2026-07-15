"""Constants for Daily Mood preprocessing."""

from __future__ import annotations

MOOD_PREPROCESSING_VERSION = "1.0.0"
MOOD_FEATURE_SCHEMA_VERSION = "1.0.0"
MOOD_MAPPING_VERSION = "1.0.0"

DATASET_NAME = "daily-mood"
DATASET_VERSION = "v1"
RECORD_ID_PREFIX = "daily-mood-v1"

CANONICAL_STUDENT_ID = "student_id"
CANONICAL_TIMESTAMP = "checkin_timestamp"
CANONICAL_MOOD = "mood_value"

DEFAULT_SOURCE_COLUMNS = (
    "ParticipantID",
    "Date",
    "Mood",
    "CryingEpisodes",
    "PhysicalPain",
)

CANONICAL_COLUMNS = {
    "ParticipantID": CANONICAL_STUDENT_ID,
    "Date": CANONICAL_TIMESTAMP,
    "Mood": CANONICAL_MOOD,
    "CryingEpisodes": "crying_episode_count",
    "PhysicalPain": "physical_symptom_count",
}

PRODUCTION_DAILY_CHECKIN_FIELDS = (
    "id",
    "user_id",
    "mood",
    "mood_description",
    "sleep_hours",
    "exercise_minutes",
    "social_interaction",
    "stress_level",
    "anxiety_level",
    "negative_thoughts",
    "substance_use_today",
    "self_harm_thoughts",
    "notes",
    "created_at",
)

MAPPED_REQUIRED_CANONICAL_FIELDS = (
    CANONICAL_STUDENT_ID,
    CANONICAL_TIMESTAMP,
    CANONICAL_MOOD,
)

MOOD_MIN = 1
MOOD_MAX = 5
LOW_MOOD_THRESHOLD = 2
SUDDEN_DETERIORATION_DROP = 2
SAFE_PARTICIPANT_KEY_PREFIX = "mood-participant-v1"

FEATURE_COLUMNS = (
    "current_mood",
    "previous_mood",
    "mood_change_from_previous",
    "rolling_mean_3_observations",
    "rolling_mean_7_observations",
    "rolling_std_3_observations",
    "rolling_std_7_observations",
    "slope_last_3_observations",
    "slope_last_7_observations",
    "consecutive_low_mood_count",
    "low_mood_ratio_last_7_observations",
    "days_since_previous_checkin",
    "checkins_last_7_days",
    "missing_day_ratio_last_7_days",
    "sudden_deterioration_flag",
    "crying_episode_trend",
    "physical_symptom_trend",
    "history_length",
    "data_completeness",
    "mood_trend_score_0_100",
)
