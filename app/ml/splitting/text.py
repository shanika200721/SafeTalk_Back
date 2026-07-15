"""Text split design for Phase 3A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.splitting.common import (
    assign_duplicate_groups,
    build_split_manifest,
    compute_split_artifact_hash,
    grouped_stratified_split,
    label_distribution,
    load_json,
    make_validation_summary,
    write_assignments_csv,
)
from app.ml.splitting.reporting import save_split_outputs
from app.ml.splitting.schemas import SplitDesignReport, SplitStrategy


TEXT_LIMITATIONS = [
    "Missing user IDs prevent complete author-level leakage control.",
    "Labels are dataset annotations and not clinical diagnoses.",
    "Platform/domain mismatch remains across source material.",
]


def _read_conflict_exclusions(conflict_manifest_path: str | Path | None) -> dict[str, str]:
    if not conflict_manifest_path:
        return {}
    path = Path(conflict_manifest_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "record_id" not in df.columns:
        return {}
    return {str(record_id): "conflicting_duplicate_quarantine" for record_id in df["record_id"].dropna().astype(str).tolist()}


def create_text_split(
    *,
    input_path: str | Path,
    config_path: str | Path,
    preprocessing_report_path: str | Path,
    feature_schema_path: str | Path,
    duplicate_manifest_path: str | Path,
    conflict_manifest_path: str | Path | None = None,
    source_overlap_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    feature_schema = load_json(feature_schema_path)
    preprocessing_report = load_json(preprocessing_report_path)
    duplicate_manifest = load_json(duplicate_manifest_path)
    source_overlap_report = load_json(source_overlap_report_path) if source_overlap_report_path and Path(source_overlap_report_path).exists() else {}
    random_seed = int(seed if seed is not None else config["random_seed"])

    df = pd.read_csv(input_path, low_memory=False)
    conflict_exclusions = _read_conflict_exclusions(conflict_manifest_path)
    included = df[~df["record_id"].astype(str).isin(conflict_exclusions)].copy()
    duplicate_map = assign_duplicate_groups(duplicate_manifest)
    included["duplicate_group_id"] = included["record_id"].astype(str).map(duplicate_map)
    included["split_group_id"] = included["duplicate_group_id"].fillna(included["text_hash"].astype(str))

    assignments = grouped_stratified_split(
        included,
        record_id_column="record_id",
        label_column=str(config.get("stratify_column") or "canonical_label"),
        group_column="split_group_id",
        duplicate_column="duplicate_group_id",
        source_column="source_name",
        train_proportion=float(config["train_proportion"]),
        validation_proportion=float(config["validation_proportion"]),
        test_proportion=float(config["test_proportion"]),
        seed=random_seed,
        retry_limit=int(config.get("retry_limit", 25)),
        minimum_class_count_per_split=int(config.get("minimum_class_count_per_split", 1)),
    )
    overlap_count = int(source_overlap_report.get("exact_overlap_count", 0) or 0)
    warnings = [
        "author-level leakage cannot be fully eliminated because Unique_ID is missing for a large subset",
        "reference overlap report contains aggregate count only, so record-level train exclusion cannot be fully enforced",
    ]
    validation_summary = make_validation_summary(
        assignments,
        expected_record_ids=df["record_id"].astype(str).tolist(),
        excluded_ids=conflict_exclusions,
        source_overlap_count=0,
        warnings=warnings,
    )
    config_hash = hash_json_data(config)
    manifest = build_split_manifest(
        modality="text",
        dataset_name=feature_schema.get("dataset_name", "mental-health-text"),
        dataset_version=feature_schema.get("dataset_version", config.get("dataset_version", "v1")),
        preprocessing_version=feature_schema.get("preprocessing_version", config.get("preprocessing_version", "1.0.0")),
        feature_schema_version=feature_schema.get("feature_schema_version", config.get("feature_schema_version", "1.0.0")),
        source_fingerprint=preprocessing_report.get("source_fingerprint", ""),
        preprocessing_artifact_hash=sha256_file(input_path, allow_outside_project=True),
        config_hash=config_hash,
        random_seed=random_seed,
        split_strategy=SplitStrategy.GROUPED_STRATIFIED,
        assignments=assignments,
        excluded_ids=conflict_exclusions,
        grouping_column="text_hash",
        stratify_column=str(config.get("stratify_column") or "canonical_label"),
        source_split_policy="reference overlap is aggregate-only; predefined reference test set is not used as final test split",
        duplicate_policy=str(config.get("duplicate_policy", "exact_duplicate_groups_isolated")),
        validation_summary=validation_summary,
        notes=TEXT_LIMITATIONS + list(config.get("notes", [])),
    )
    manifest_hash = compute_split_artifact_hash(manifest)
    distributions = label_distribution(assignments)
    duplicate_group_count = sum(1 for group in duplicate_manifest.get("exact_duplicate_groups", []) if not group.get("conflict"))
    report = SplitDesignReport(
        modality="text",
        strategy=SplitStrategy.GROUPED_STRATIFIED.value,
        source_count=len(df) + len(conflict_exclusions),
        included_count=len(assignments),
        excluded_count=len(conflict_exclusions),
        split_counts={
            "train": manifest.validation_summary.train_count,
            "validation": manifest.validation_summary.validation_count,
            "test": manifest.validation_summary.test_count,
        },
        label_distributions=distributions,
        grouping_summary={
            "grouping_column": "text_hash plus duplicate-group component",
            "unique_group_count": int(included["split_group_id"].nunique()),
            "known_unique_id_policy": "Unique_ID was not preserved in canonical output; deterministic text-hash grouping is used",
        },
        duplicate_handling={
            "policy": manifest.duplicate_policy,
            "duplicate_group_count": duplicate_group_count,
            "conflict_excluded_count": len(conflict_exclusions),
            "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
        },
        leakage_checks={
            "text_hash_overlap_count": manifest.validation_summary.group_overlap_count,
            "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
            "reference_overlap_count_reported": overlap_count,
            "raw_text_in_reports": False,
        },
        warnings=warnings,
        limitations=TEXT_LIMITATIONS,
        generated_at=manifest.created_at,
    )
    reference_policy = {
        "exact_overlap_count_reported": overlap_count,
        "policy": manifest.source_split_policy,
        "record_level_overlap_ids_available": False,
        "training_exclusion_status": "not_fully_enforceable_from_current_overlap_report",
    }
    duplicate_report = {
        "duplicate_group_count": duplicate_group_count,
        "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
        "text_hash_overlap_count": manifest.validation_summary.group_overlap_count,
        "conflict_excluded_count": len(conflict_exclusions),
    }
    grouping_summary = report.grouping_summary | {"missing_unique_id_count_documented": 9600}
    paths = {}
    if output_dir is not None and not validate_only:
        paths = save_split_outputs(
            modality="text",
            output_dir=output_dir,
            manifest=manifest,
            assignments_csv_writer=lambda path, overwrite=False: write_assignments_csv(assignments, path, overwrite=overwrite),
            report=report,
            exclusions=conflict_exclusions,
            manifest_hash=manifest_hash,
            extra_json={
                "text_grouping_summary.json": grouping_summary,
                "text_duplicate_isolation_report.json": duplicate_report,
                "text_reference_overlap_policy.json": reference_policy,
            },
            overwrite=overwrite,
        )
    return {
        "manifest": manifest,
        "assignments": assignments,
        "report": report,
        "exclusions": conflict_exclusions,
        "manifest_hash": manifest_hash,
        "paths": paths,
        "reference_policy": reference_policy,
    }
