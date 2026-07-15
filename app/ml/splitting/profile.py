"""Profile split design for Phase 3A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.splitting.common import (
    build_split_manifest,
    compute_split_artifact_hash,
    label_distribution,
    load_json,
    make_validation_summary,
    stratified_split,
    write_assignments_csv,
)
from app.ml.splitting.reporting import save_split_outputs
from app.ml.splitting.schemas import SplitDesignReport, SplitStrategy


DEFAULT_PROFILE_NOTES = [
    "Profile target is self-reported depression, not suicide-risk ground truth.",
    "Sensitive context columns are not used for stratification.",
    "Metrics will be unstable because the dataset has only 101 records.",
]


def create_profile_split(
    *,
    input_path: str | Path,
    config_path: str | Path,
    preprocessing_report_path: str | Path,
    feature_schema_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    feature_schema = load_json(feature_schema_path)
    preprocessing_report = load_json(preprocessing_report_path)
    random_seed = int(seed if seed is not None else config["random_seed"])

    df = pd.read_csv(input_path)
    label_column = str(config.get("stratify_column") or "target_depression")
    assignments = stratified_split(
        df,
        record_id_column="record_id",
        label_column=label_column,
        train_proportion=float(config["train_proportion"]),
        validation_proportion=float(config["validation_proportion"]),
        test_proportion=float(config["test_proportion"]),
        seed=random_seed,
        minimum_class_count_per_split=int(config.get("minimum_class_count_per_split", 1)),
    )
    expected_ids = df["record_id"].astype(str).tolist()
    warnings = [
        "small dataset: validation and test estimates are high variance",
        "target is self-reported depression rather than clinical diagnosis or suicide-risk label",
    ]
    validation_summary = make_validation_summary(
        assignments,
        expected_record_ids=expected_ids,
        excluded_ids={},
        warnings=warnings,
    )
    config_hash = hash_json_data(config)
    manifest = build_split_manifest(
        modality="profile",
        dataset_name=feature_schema.get("dataset_name", "student-profile"),
        dataset_version=feature_schema.get("dataset_version", config.get("dataset_version", "v1")),
        preprocessing_version=feature_schema.get("preprocessing_version", config.get("preprocessing_version", "1.0.0")),
        feature_schema_version=feature_schema.get("feature_schema_version", config.get("feature_schema_version", "1.0.0")),
        source_fingerprint=preprocessing_report.get("source_fingerprint", ""),
        preprocessing_artifact_hash=sha256_file(input_path, allow_outside_project=True),
        config_hash=config_hash,
        random_seed=random_seed,
        split_strategy=SplitStrategy.RANDOM_STRATIFIED,
        assignments=assignments,
        excluded_ids={},
        grouping_column=None,
        stratify_column=label_column,
        source_split_policy=None,
        duplicate_policy=str(config.get("duplicate_policy", "no_duplicates_reported")),
        validation_summary=validation_summary,
        notes=DEFAULT_PROFILE_NOTES + list(config.get("notes", [])),
    )
    manifest_hash = compute_split_artifact_hash(manifest)
    report = SplitDesignReport(
        modality="profile",
        strategy=SplitStrategy.RANDOM_STRATIFIED.value,
        source_count=len(df),
        included_count=len(assignments),
        excluded_count=0,
        split_counts={
            "train": manifest.validation_summary.train_count,
            "validation": manifest.validation_summary.validation_count,
            "test": manifest.validation_summary.test_count,
        },
        label_distributions=label_distribution(assignments),
        grouping_summary={"grouping_column": None, "grouped_entity_overlap_count": 0},
        duplicate_handling={"policy": manifest.duplicate_policy, "duplicate_overlap_count": 0},
        leakage_checks={
            "id_overlap_count": 0,
            "group_overlap_count": manifest.validation_summary.group_overlap_count,
            "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
            "sensitive_attribute_stratification": False,
        },
        warnings=warnings,
        limitations=DEFAULT_PROFILE_NOTES,
        generated_at=manifest.created_at,
    )

    paths = {}
    if output_dir is not None and not validate_only:
        paths = save_split_outputs(
            modality="profile",
            output_dir=output_dir,
            manifest=manifest,
            assignments_csv_writer=lambda path, overwrite=False: write_assignments_csv(assignments, path, overwrite=overwrite),
            report=report,
            exclusions={},
            manifest_hash=manifest_hash,
            overwrite=overwrite,
        )
    return {
        "manifest": manifest,
        "assignments": assignments,
        "report": report,
        "exclusions": {},
        "manifest_hash": manifest_hash,
        "paths": paths,
    }
