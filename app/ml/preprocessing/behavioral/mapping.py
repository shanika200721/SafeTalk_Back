"""Field mapping helpers for behavioral telemetry sources."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic.v1 import ValidationError

from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_MAPPING_VERSION,
    DATASET_NAME,
    DATASET_VERSION,
)
from app.ml.preprocessing.behavioral.schemas import (
    BehavioralFieldMapping,
    BehavioralFieldRole,
    BehavioralMappingConfig,
)


def _field(source: str, canonical: str, role: BehavioralFieldRole, expected_type: str, aggregation_rule: str, notes: str) -> BehavioralFieldMapping:
    return BehavioralFieldMapping(
        source_field=source,
        canonical_field=canonical,
        role=role,
        expected_type=expected_type,
        missing_value_strategy="preserve; do not impute or infer missing events",
        aggregation_rule=aggregation_rule,
        notes=notes,
    )


def default_behavioral_mapping_config() -> BehavioralMappingConfig:
    source_columns = [
        "participant_id",
        "event_timestamp",
        "session_id",
        "event_type",
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
    ]
    return BehavioralMappingConfig(
        mapping_version=BEHAVIORAL_MAPPING_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        source_columns=source_columns,
        fields=[
            _field("participant_id", "participant_key", BehavioralFieldRole.IDENTIFIER, "string", "hashed; never predictive", "Raw participant identifiers are hashed in generated outputs."),
            _field("event_timestamp", "event_timestamp", BehavioralFieldRole.TIMESTAMP, "datetime", "UTC-normalized", "Timezone-aware or consistently normalized timestamp."),
            _field("session_id", "session_id", BehavioralFieldRole.SESSION, "string", "preserve or deterministic derivation", "Missing session IDs are derived only from participant/date bucket."),
            _field("event_type", "event_type", BehavioralFieldRole.EVENT_TYPE, "category", "validate", "Allowed non-content behavioral event types only."),
            _field("page_or_context", "page_or_context", BehavioralFieldRole.CONTEXT, "string", "sanitize", "Must not contain exact URLs with personal data."),
            _field("response_latency_ms", "response_latency_ms", BehavioralFieldRole.FEATURE, "number", "mean/std/count", "Prompt response timing only."),
            _field("key_dwell_time_ms", "key_dwell_time_ms", BehavioralFieldRole.FEATURE, "number", "mean/std", "Timing only; no key values."),
            _field("key_flight_time_ms", "key_flight_time_ms", BehavioralFieldRole.FEATURE, "number", "mean/std/pause", "Timing only; no typed characters."),
            _field("typing_speed_cpm", "typing_speed_cpm", BehavioralFieldRole.FEATURE, "number", "mean", "Characters-per-minute aggregate; no content."),
            _field("backspace_count", "backspace_count", BehavioralFieldRole.FEATURE, "number", "sum/rate", "Aggregate correction proxy."),
            _field("correction_count", "correction_count", BehavioralFieldRole.FEATURE, "number", "sum/rate", "Aggregate correction proxy."),
            _field("mouse_distance_px", "mouse_distance_px", BehavioralFieldRole.FEATURE, "number", "sum", "Aggregate path distance only."),
            _field("mouse_speed_px_per_second", "mouse_speed_px_per_second", BehavioralFieldRole.FEATURE, "number", "mean/std", "Aggregate mouse speed only."),
            _field("click_count", "click_count", BehavioralFieldRole.FEATURE, "number", "sum", "Aggregate click count."),
            _field("hesitation_count", "hesitation_count", BehavioralFieldRole.FEATURE, "number", "sum", "Aggregate hesitation count."),
            _field("session_duration_seconds", "session_duration_seconds", BehavioralFieldRole.FEATURE, "number", "max/session", "Session duration supplied by source, if present."),
        ],
        notes="Canonical v1 mapping for timing-only behavioral telemetry. Raw typed content, clipboard, screen content, and password fields are prohibited.",
    )


def load_behavioral_mapping_config(path: str | Path | None) -> BehavioralMappingConfig:
    if path is None:
        return default_behavioral_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Behavioral mapping config must be a JSON object")
    try:
        return BehavioralMappingConfig.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse behavioral mapping config: {exc}") from exc


def mapping_by_source(mapping_config: BehavioralMappingConfig) -> dict[str, BehavioralFieldMapping]:
    return {field.source_field: field for field in mapping_config.fields}

