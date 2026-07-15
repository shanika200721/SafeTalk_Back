"""Header-level mapping helpers for the audited DASS-42 source."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.ml.preprocessing.dass21.constants import (
    DASS21_EXPECTED_ITEMS,
    DASS21_ITEM_MAPPING,
    DASS21_ITEM_MAPPING_VERSION,
    DASS42_TO_DASS21_SOURCE_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    METADATA_COLUMNS,
    POSITION_COLUMN_PATTERN,
    RESPONSE_COLUMN_PATTERN,
    TIMING_COLUMN_PATTERN,
)
from app.ml.preprocessing.dass21.scoring import validate_dass21_responses


def _sorted_question_columns(columns: Iterable[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)

    def key(column: str) -> int:
        return int(re.search(r"\d+", column).group(0))

    return sorted([str(column) for column in columns if regex.match(str(column))], key=key)


def identify_response_columns(columns: Iterable[str]) -> list[str]:
    return _sorted_question_columns(columns, RESPONSE_COLUMN_PATTERN)


def identify_timing_columns(columns: Iterable[str]) -> list[str]:
    return _sorted_question_columns(columns, TIMING_COLUMN_PATTERN)


def identify_position_columns(columns: Iterable[str]) -> list[str]:
    return _sorted_question_columns(columns, POSITION_COLUMN_PATTERN)


def identify_demographic_columns(columns: Iterable[str]) -> list[str]:
    available = {str(column) for column in columns}
    return [column for column in DEMOGRAPHIC_COLUMNS if column in available]


def identify_metadata_columns(columns: Iterable[str]) -> list[str]:
    available = {str(column) for column in columns}
    return [column for column in METADATA_COLUMNS if column in available]


def inspect_dass_dataset_columns(columns: Iterable[str]) -> dict[str, Any]:
    column_list = [str(column) for column in columns]
    response_columns = identify_response_columns(column_list)
    return {
        "column_count": len(column_list),
        "questionnaire_source": "DASS-42" if len(response_columns) == 42 else "DASS-21" if len(response_columns) == 21 else "unknown",
        "response_columns": response_columns,
        "timing_columns": identify_timing_columns(column_list),
        "position_columns": identify_position_columns(column_list),
        "demographic_columns": identify_demographic_columns(column_list),
        "metadata_columns": identify_metadata_columns(column_list),
    }


def default_mapping_config() -> dict[str, Any]:
    return {
        "mapping_version": DASS21_ITEM_MAPPING_VERSION,
        "source_questionnaire_version": "DASS-42",
        "target_questionnaire_version": "DASS-21",
        "source_response_scale": "1-4",
        "target_response_scale": "0-3",
        "response_transformation": "subtract_1_from_1_4",
        "timing_column_exclusion_pattern": TIMING_COLUMN_PATTERN,
        "position_column_exclusion_pattern": POSITION_COLUMN_PATTERN,
        "metadata_exclusions": list(METADATA_COLUMNS),
        "demographic_exclusions": list(DEMOGRAPHIC_COLUMNS),
        "selected_dass21_items": [
            {
                "target_question_id": target,
                "source_column": source,
                "subscale": subscale,
            }
            for target, source in DASS42_TO_DASS21_SOURCE_COLUMNS.items()
            for subscale, items in DASS21_ITEM_MAPPING.items()
            if target in items
        ],
    }


def load_mapping_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("DASS-21 item mapping config must be a JSON object")
    return payload


def _mapping_items(mapping_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = mapping_config.get("selected_dass21_items")
    if not isinstance(items, list):
        raise ValueError("Mapping config must include selected_dass21_items")
    return items


def validate_dataset_item_mapping(columns: Iterable[str], mapping_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    mapping_config = mapping_config or default_mapping_config()
    items = _mapping_items(mapping_config)
    available = {str(column) for column in columns}

    target_ids = [str(item.get("target_question_id")) for item in items]
    source_columns = [str(item.get("source_column")) for item in items]
    duplicates = sorted({item for item in target_ids if target_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate DASS-21 target question IDs in mapping: {duplicates}")

    if set(target_ids) != set(DASS21_EXPECTED_ITEMS):
        missing = sorted(set(DASS21_EXPECTED_ITEMS) - set(target_ids), key=lambda item: int(item[1:]))
        extra = sorted(set(target_ids) - set(DASS21_EXPECTED_ITEMS))
        raise ValueError(f"Mapping must contain exactly 21 DASS-21 targets; missing={missing}, extra={extra}")

    subscale_counts = {
        subscale: sum(1 for item in items if item.get("subscale") == subscale)
        for subscale in DASS21_ITEM_MAPPING
    }
    invalid_counts = {subscale: count for subscale, count in subscale_counts.items() if count != 7}
    if invalid_counts:
        raise ValueError(f"Each DASS-21 subscale must contain 7 mapped items: {invalid_counts}")

    missing_columns = [column for column in source_columns if column not in available]
    if missing_columns:
        raise ValueError(f"Missing expected DASS source response columns: {missing_columns}")

    response_columns = identify_response_columns(columns)
    timing_columns = identify_timing_columns(columns)
    return {
        "mapping_success": True,
        "mapping_version": mapping_config.get("mapping_version", DASS21_ITEM_MAPPING_VERSION),
        "response_column_count": len(response_columns),
        "excluded_timing_column_count": len(timing_columns),
        "selected_item_count": len(items),
        "subscale_item_counts": subscale_counts,
        "questionnaire_source": "DASS-42" if len(response_columns) == 42 else "DASS-21" if len(response_columns) == 21 else "unknown",
    }


def map_dataset_row_to_dass21_responses(
    row: Mapping[str, Any],
    mapping_config: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Map one bounded sample row to DASS-21 0-3 responses without persisting it."""

    mapping_config = mapping_config or default_mapping_config()
    responses: dict[str, int] = {}
    for item in _mapping_items(mapping_config):
        target = str(item["target_question_id"])
        source_column = str(item["source_column"])
        if source_column not in row:
            raise ValueError(f"Missing source column in row: {source_column}")
        try:
            source_value = int(row[source_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Source response {source_column} must be an integer 1-4") from exc
        if source_value < 1 or source_value > 4:
            raise ValueError(f"Source response {source_column} must be between 1 and 4")
        responses[target] = source_value - 1

    return validate_dass21_responses(responses)
