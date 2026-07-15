"""Field mapping helpers for the offline Student Profile dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic.v1 import ValidationError

from app.ml.preprocessing.profile.constants import (
    BASELINE_FEATURE_SOURCE_COLUMNS,
    CANONICAL_COLUMNS,
    CGPA_ALLOWED_VALUES,
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_EXCLUDED_SOURCE_COLUMNS,
    GENDER_ALLOWED_VALUES,
    LEAKAGE_CANDIDATE_COLUMNS,
    METADATA_COLUMNS,
    OPTIONAL_SENSITIVE_CONTEXT_SOURCE_COLUMNS,
    PROFILE_MAPPING_VERSION,
    SOURCE_COLUMNS,
    TARGET_ALLOWED_VALUES,
    TARGET_COLUMN,
    TREATMENT_COLUMN,
    YEAR_ALLOWED_VALUES,
)
from app.ml.preprocessing.profile.schemas import (
    ProfileFieldMapping,
    ProfileFieldRole,
    ProfileMappingConfig,
)


def _field(
    source_column: str,
    role: ProfileFieldRole,
    expected_type: str,
    missing_strategy: str,
    encoding_strategy: str,
    notes: str,
    *,
    allowed_categories: list[str] | None = None,
    production_field_equivalent: str | None = None,
    include_by_default: bool = False,
    leakage_candidate: bool = False,
) -> ProfileFieldMapping:
    return ProfileFieldMapping(
        source_column_name=source_column,
        canonical_feature_name=CANONICAL_COLUMNS[source_column],
        role=role,
        expected_type=expected_type,
        allowed_categories=allowed_categories,
        missing_value_strategy=missing_strategy,
        encoding_strategy=encoding_strategy,
        production_field_equivalent=production_field_equivalent,
        notes=notes,
        include_by_default=include_by_default,
        leakage_candidate=leakage_candidate,
    )


def default_profile_mapping_config() -> ProfileMappingConfig:
    """Return the reviewed v1 mapping without loading participant rows."""

    fields = [
        _field(
            "Timestamp",
            ProfileFieldRole.METADATA,
            "datetime string",
            "not applicable; excluded from ML features",
            "excluded",
            "Collection timestamp is metadata and a leakage/timing candidate.",
            leakage_candidate=True,
        ),
        _field(
            "Choose your gender",
            ProfileFieldRole.SENSITIVE_CONTEXT,
            "categorical",
            "preserve missing if encountered",
            "categorical canonical label; encoder fitting deferred to training split",
            "Sensitive contextual attribute for fairness/utility comparisons; not a direct identifier.",
            allowed_categories=list(GENDER_ALLOWED_VALUES),
            include_by_default=False,
        ),
        _field(
            "Age",
            ProfileFieldRole.SENSITIVE_CONTEXT,
            "integer",
            "preserve missing; the known single missing Age is not imputed in canonical output",
            "numeric passthrough; imputation deferred to training-only artifacts",
            "Sensitive contextual attribute; not a direct identifier.",
            include_by_default=False,
        ),
        _field(
            "What is your course?",
            ProfileFieldRole.SENSITIVE_CONTEXT,
            "categorical",
            "preserve missing if encountered",
            "categorical canonical label; rare-category handling deferred",
            "Course is sensitive/contextual and high-cardinality; production has no direct profile assessment equivalent.",
            production_field_equivalent="department",
            include_by_default=False,
        ),
        _field(
            "Your current year of Study",
            ProfileFieldRole.FEATURE,
            "ordered categorical",
            "preserve missing if encountered",
            "ordered categorical canonical label; no ordinal integers fitted here",
            "Closest offline equivalent to production user.year_of_study.",
            allowed_categories=list(YEAR_ALLOWED_VALUES),
            production_field_equivalent="year_of_study",
            include_by_default=True,
        ),
        _field(
            "What is your CGPA?",
            ProfileFieldRole.SENSITIVE_CONTEXT,
            "ordered categorical range",
            "preserve missing if encountered",
            "ordered categorical range; do not invent midpoints",
            "Closest offline equivalent to production gpa, but stored as ranges rather than numeric GPA.",
            allowed_categories=list(CGPA_ALLOWED_VALUES),
            production_field_equivalent="gpa",
            include_by_default=False,
        ),
        _field(
            "Marital status",
            ProfileFieldRole.SENSITIVE_CONTEXT,
            "binary categorical",
            "preserve missing if encountered",
            "binary canonical label",
            "Sensitive contextual attribute with no production profile assessment equivalent.",
            allowed_categories=list(TARGET_ALLOWED_VALUES),
            include_by_default=False,
        ),
        _field(
            TARGET_COLUMN,
            ProfileFieldRole.TARGET,
            "binary categorical",
            "critical failure if missing or unrecognized",
            "binary canonical label",
            "Target label for this research dataset; never a feature.",
            allowed_categories=list(TARGET_ALLOWED_VALUES),
        ),
        _field(
            "Do you have Anxiety?",
            ProfileFieldRole.FEATURE,
            "binary categorical",
            "preserve missing if encountered",
            "binary canonical label",
            "Self-reported anxiety is a candidate baseline feature, not the depression target.",
            allowed_categories=list(TARGET_ALLOWED_VALUES),
            include_by_default=True,
        ),
        _field(
            "Do you have Panic attack?",
            ProfileFieldRole.FEATURE,
            "binary categorical",
            "preserve missing if encountered",
            "binary canonical label",
            "Self-reported panic attack is a candidate baseline feature, not the depression target.",
            allowed_categories=list(TARGET_ALLOWED_VALUES),
            include_by_default=True,
        ),
        _field(
            TREATMENT_COLUMN,
            ProfileFieldRole.EXCLUDED,
            "binary categorical",
            "preserve missing if encountered",
            "excluded by default; evaluate only in explicit sensitivity analysis",
            "Treatment-seeking may be post-outcome and can encode the mental-health outcome pathway.",
            allowed_categories=list(TARGET_ALLOWED_VALUES),
            include_by_default=False,
            leakage_candidate=True,
        ),
    ]
    return ProfileMappingConfig(
        mapping_version=PROFILE_MAPPING_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        target_column=TARGET_COLUMN,
        source_columns=list(SOURCE_COLUMNS),
        fields=fields,
        default_excluded_columns=list(DEFAULT_EXCLUDED_SOURCE_COLUMNS),
        sensitive_context_columns=list(OPTIONAL_SENSITIVE_CONTEXT_SOURCE_COLUMNS),
        leakage_candidate_columns=list(LEAKAGE_CANDIDATE_COLUMNS),
        notes=(
            "Default baseline excludes metadata, target leakage candidates, and optional sensitive-context fields. "
            "Sensitive-context fields are preserved for explicit fairness/utility experiments."
        ),
    )


def load_profile_mapping_config(path: str | Path | None) -> ProfileMappingConfig:
    if path is None:
        return default_profile_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Profile field mapping config must be a JSON object")
    try:
        return ProfileMappingConfig.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse profile mapping config: {exc}") from exc


def mapping_by_source(mapping_config: ProfileMappingConfig) -> dict[str, ProfileFieldMapping]:
    return {field.source_column_name: field for field in mapping_config.fields}


def canonical_name_for_source(source_column: str, mapping_config: ProfileMappingConfig | None = None) -> str:
    mapping_config = mapping_config or default_profile_mapping_config()
    fields = mapping_by_source(mapping_config)
    if source_column not in fields:
        raise KeyError(f"Unknown profile source column: {source_column}")
    return fields[source_column].canonical_feature_name


def selected_feature_source_columns(
    mapping_config: ProfileMappingConfig,
    *,
    include_sensitive_context: bool = False,
    exclude_treatment_seeking: bool = True,
) -> list[str]:
    selected: list[str] = []
    for field in mapping_config.fields:
        if field.role == ProfileFieldRole.FEATURE and field.include_by_default:
            selected.append(field.source_column_name)
        elif include_sensitive_context and field.role == ProfileFieldRole.SENSITIVE_CONTEXT:
            selected.append(field.source_column_name)

    if exclude_treatment_seeking:
        selected = [column for column in selected if column != TREATMENT_COLUMN]
    elif TREATMENT_COLUMN in mapping_by_source(mapping_config):
        selected.append(TREATMENT_COLUMN)

    blocked = set(METADATA_COLUMNS) | {mapping_config.target_column}
    return [column for column in selected if column not in blocked]


def inspect_profile_dataset_columns(columns: Iterable[str]) -> dict[str, Any]:
    column_list = [str(column) for column in columns]
    available = set(column_list)
    return {
        "column_count": len(column_list),
        "source_columns": column_list,
        "missing_required_columns": [column for column in SOURCE_COLUMNS if column not in available],
        "unexpected_columns": [column for column in column_list if column not in SOURCE_COLUMNS],
        "baseline_feature_columns": list(BASELINE_FEATURE_SOURCE_COLUMNS),
        "sensitive_context_columns": list(OPTIONAL_SENSITIVE_CONTEXT_SOURCE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "leakage_candidate_columns": list(LEAKAGE_CANDIDATE_COLUMNS),
        "target_column": TARGET_COLUMN,
    }
