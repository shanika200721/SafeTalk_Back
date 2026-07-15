"""Constants for privacy-preserving behavioral preprocessing."""

from __future__ import annotations

BEHAVIORAL_PREPROCESSING_VERSION = "1.0.0"
BEHAVIORAL_FEATURE_SCHEMA_VERSION = "1.0.0"
BEHAVIORAL_MAPPING_VERSION = "1.0.0"
BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION = "1.0.0"

DATASET_NAME = "behavioral-telemetry"
DATASET_VERSION = "v1"

RECORD_ID_PREFIX = "behavioral-v1-event"
SESSION_ID_PREFIX = "behavioral-v1-session"
SAFE_PARTICIPANT_KEY_PREFIX = "behavioral-v1-participant"

SOURCE_STATUS_REAL_OFFLINE_DATASET = "real_offline_dataset"
SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET = "partial_behavioral_dataset"
SOURCE_STATUS_PRODUCTION_SCHEMA_ONLY = "production_schema_only"
SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY = "synthetic_engineering_data_only"
SOURCE_STATUS_NO_BEHAVIORAL_DATA = "no_behavioral_data"

READINESS_ENGINEERING_ONLY = "engineering_tests_only"
READINESS_UNSUITABLE = "unsuitable_for_model_training"
READINESS_SUITABLE_WITH_RESTRICTIONS = "suitable_with_restrictions"
READINESS_SUITABLE = "suitable"

EVENT_TYPES = (
    "typing_timing",
    "mouse_aggregate",
    "prompt_response",
    "session_start",
    "session_end",
    "page_view",
)

REQUIRED_CANONICAL_COLUMNS = (
    "event_id",
    "participant_key",
    "event_timestamp",
    "session_id",
    "event_type",
)

OPTIONAL_CANONICAL_COLUMNS = (
    "page_or_context",
    "response_latency_ms",
    "key_dwell_time_ms",
    "key_flight_time_ms",
    "typing_speed_cpm",
    "backspace_count",
    "correction_count",
    "mouse_distance_px",
    "mouse_speed_px_per_second",
    "click_count",
    "hesitation_count",
    "session_duration_seconds",
)

FEATURE_COLUMNS = (
    "key_event_count",
    "typing_duration_seconds",
    "typing_speed_cpm",
    "dwell_time_mean",
    "dwell_time_std",
    "flight_time_mean",
    "flight_time_std",
    "backspace_rate",
    "correction_rate",
    "pause_count",
    "long_pause_ratio",
    "mouse_event_count",
    "path_distance",
    "mean_speed",
    "speed_variability",
    "click_count",
    "hesitation_count",
    "idle_ratio",
    "prompt_response_count",
    "response_latency_mean",
    "response_latency_std",
    "skipped_prompt_count",
    "session_duration",
    "event_count",
    "active_time",
    "idle_time",
    "page_transition_count",
    "sessions_per_day",
    "time_since_previous_session",
)

IDENTIFIER_FIELD_CANDIDATES = ("participant_id", "student_id", "user_id", "ParticipantID")
TIMESTAMP_FIELD_CANDIDATES = ("event_timestamp", "timestamp", "created_at", "Date", "session_start")
SESSION_FIELD_CANDIDATES = ("session_id", "session", "SessionID")

SENSITIVE_PAYLOAD_FIELDS = {
    "key",
    "key_value",
    "key_code",
    "typed_text",
    "text",
    "message",
    "input_value",
    "raw_keystrokes",
    "clipboard",
    "clipboard_content",
    "screen",
    "screen_content",
    "screenshot",
    "password",
}

PROHIBITED_OUTPUT_COLUMNS = {
    "risk_class",
    "risk_level",
    "clinical_label",
    "suicide_risk",
    "anomaly_score",
    "treatment_recommendation",
}

DEFAULT_MINIMUM_BASELINE_SESSIONS = 3
DEFAULT_MINIMUM_BASELINE_DAYS = 3
DEFAULT_MINIMUM_BASELINE_EVENTS = 20
DEFAULT_MINIMUM_NON_MISSING_PER_FEATURE = 3

