"""Field mapping helpers for offline Daily Mood preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic.v1 import ValidationError

from app.ml.preprocessing.mood.constants import (
    CANONICAL_COLUMNS,
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_SOURCE_COLUMNS,
    MOOD_MAPPING_VERSION,
)
from app.ml.preprocessing.mood.schemas import MoodFieldMapping, MoodFieldRole, MoodMappingConfig


def _field(
    source_field: str,
    canonical_field: str,
    role: MoodFieldRole,
    expected_type: str,
    missing_strategy: str,
    aggregation_rule: str,
    notes: str,
    *,
    valid_range: list[float] | None = None,
    valid_categories: list[str] | None = None,
    production_field_equivalent: str | None = None,
) -> MoodFieldMapping:
    return MoodFieldMapping(
        source_field=source_field,
        canonical_field=canonical_field,
        role=role,
        expected_type=expected_type,
        valid_range=valid_range,
        valid_categories=valid_categories,
        missing_value_strategy=missing_strategy,
        aggregation_rule=aggregation_rule,
        production_field_equivalent=production_field_equivalent,
        notes=notes,
    )


def default_mood_mapping_config() -> MoodMappingConfig:
    fields = [
        _field(
            "ParticipantID",
            CANONICAL_COLUMNS["ParticipantID"],
            MoodFieldRole.IDENTIFIER,
            "string",
            "critical failure if missing; hash before generated outputs",
            "no aggregation; participant-specific temporal order",
            "Project-generated participant identifier. Production equivalent is users.id / daily_checkins.user_id.",
            production_field_equivalent="user_id",
        ),
        _field(
            "Date",
            CANONICAL_COLUMNS["Date"],
            MoodFieldRole.TIMESTAMP,
            "date or datetime",
            "critical failure if missing or invalid",
            "sort deterministically; do not combine records silently",
            "Project-generated date field. Production equivalent is daily_checkins.created_at.",
            production_field_equivalent="created_at",
        ),
        _field(
            "Mood",
            CANONICAL_COLUMNS["Mood"],
            MoodFieldRole.FEATURE,
            "integer",
            "critical failure if missing or outside confirmed range",
            "current and prior observations only",
            "Self-reported mood on confirmed 1-5 application scale; higher means better mood.",
            valid_range=[1, 5],
            production_field_equivalent="mood",
        ),
        _field(
            "CryingEpisodes",
            CANONICAL_COLUMNS["CryingEpisodes"],
            MoodFieldRole.FEATURE,
            "number",
            "preserve missing; do not impute",
            "trend features use current and previous observations only",
            "Confirmed only in project-generated mood CSV, not production daily_checkins.",
            valid_range=[0, 100],
            production_field_equivalent=None,
        ),
        _field(
            "PhysicalPain",
            CANONICAL_COLUMNS["PhysicalPain"],
            MoodFieldRole.FEATURE,
            "categorical symptom text",
            "none/blank maps to 0 symptoms; non-empty symptom text maps to 1",
            "trend features use current and previous observations only",
            "Confirmed only in project-generated mood CSV; production has no structured physical symptom field.",
            production_field_equivalent=None,
        ),
    ]
    return MoodMappingConfig(
        mapping_version=MOOD_MAPPING_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        source_columns=list(DEFAULT_SOURCE_COLUMNS),
        fields=fields,
        notes=(
            "Mapping covers the project-generated daily_mood.csv schema and documents production equivalents. "
            "Production PostgreSQL is not queried by preprocessing."
        ),
    )


def load_mood_mapping_config(path: str | Path | None) -> MoodMappingConfig:
    if path is None:
        return default_mood_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Mood field mapping config must be a JSON object")
    try:
        return MoodMappingConfig.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse mood mapping config: {exc}") from exc


def mapping_by_source(mapping_config: MoodMappingConfig) -> dict[str, MoodFieldMapping]:
    return {field.source_field: field for field in mapping_config.fields}


def canonical_name_for_source(source_field: str, mapping_config: MoodMappingConfig | None = None) -> str:
    mapping_config = mapping_config or default_mood_mapping_config()
    fields = mapping_by_source(mapping_config)
    if source_field not in fields:
        raise KeyError(f"Unknown mood source field: {source_field}")
    return fields[source_field].canonical_field


def feature_source_columns(mapping_config: MoodMappingConfig) -> list[str]:
    return [field.source_field for field in mapping_config.fields if field.role == MoodFieldRole.FEATURE]


def inspect_mood_dataset_columns(columns: Iterable[str]) -> dict:
    column_list = [str(column) for column in columns]
    available = set(column_list)
    return {
        "column_count": len(column_list),
        "source_columns": column_list,
        "missing_required_columns": [column for column in DEFAULT_SOURCE_COLUMNS if column not in available],
        "unexpected_columns": [column for column in column_list if column not in DEFAULT_SOURCE_COLUMNS],
        "baseline_feature_columns": feature_source_columns(default_mood_mapping_config()),
        "identifier_column": "ParticipantID",
        "timestamp_column": "Date",
        "target_column": None,
    }
