"""Versioned text label mapping helpers."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.preprocessing.text.constants import CONFIRMED_LABELS, DATASET_NAME, DATASET_VERSION, TEXT_LABEL_MAPPING_VERSION
from app.ml.preprocessing.text.schemas import LabelMappingEntry, TextLabelMappingConfig


def default_text_label_mapping_config() -> TextLabelMappingConfig:
    descriptions = {
        "Anxiety": "Posts labeled as anxiety-related in the source dataset.",
        "Depression": "Posts labeled as depression-related in the source dataset.",
        "Normal": "Posts labeled as normal/non-condition in the source dataset.",
        "Suicidal": "Posts labeled as suicidal in the source dataset.",
    }
    return TextLabelMappingConfig(
        mapping_version=TEXT_LABEL_MAPPING_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        entries=[
            LabelMappingEntry(
                original_label=label,
                canonical_label=label.lower(),
                description=descriptions[label],
                retained=True,
                merged=False,
                excluded=False,
                mapping_justification="Retained without merging because labels are confirmed distinct source classes.",
            )
            for label in CONFIRMED_LABELS
        ],
        notes="No label merging is performed in v1.",
    )


def load_text_label_mapping_config(path: str | Path | None) -> TextLabelMappingConfig:
    if path is None:
        return default_text_label_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Text label mapping config must be a JSON object")
    return TextLabelMappingConfig.parse_obj(payload)


def label_mapping_dict(config: TextLabelMappingConfig) -> dict[str, LabelMappingEntry]:
    return {entry.original_label: entry for entry in config.entries}


def normalize_label(label: object, config: TextLabelMappingConfig) -> str:
    value = "" if label is None else str(label).strip()
    entries = label_mapping_dict(config)
    if value not in entries:
        raise ValueError(f"Unknown text label: {value!r}")
    entry = entries[value]
    if not entry.retained or entry.excluded:
        raise ValueError(f"Text label is excluded by mapping: {value!r}")
    return entry.canonical_label
