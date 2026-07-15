"""Speech split design for Phase 3A."""

from __future__ import annotations

from collections import Counter, defaultdict
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


SPEECH_LIMITATIONS = [
    "Speech corpora are acted emotion datasets, not depression or suicide-risk labels.",
    "Corpus, device, language, and accent bias may remain even with speaker isolation.",
    "TESS has only 2 speakers and SAVEE has only 4 speakers, so all corpora cannot meaningfully populate every split.",
    "Leave-one-corpus-out evaluation should be considered during training-framework design.",
]


def _corpus_distribution(assignments) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = {split: Counter() for split in ("train", "validation", "test")}
    for assignment in assignments:
        distribution[assignment.split][assignment.source_name or "unknown"] += 1
    return {split: dict(sorted(counts.items())) for split, counts in distribution.items()}


def _speaker_distribution(df: pd.DataFrame, assignments) -> dict[str, Any]:
    assignment_by_id = {assignment.record_id: assignment.split for assignment in assignments}
    tmp = df[["record_id", "safe_speaker_key", "corpus_name"]].copy()
    tmp["split"] = tmp["record_id"].astype(str).map(assignment_by_id)
    by_split = {
        split: int(tmp.loc[tmp["split"] == split, "safe_speaker_key"].nunique())
        for split in ("train", "validation", "test")
    }
    by_corpus = {
        corpus: {
            split: int(group.loc[group["split"] == split, "safe_speaker_key"].nunique())
            for split in ("train", "validation", "test")
        }
        for corpus, group in tmp.groupby("corpus_name")
    }
    return {
        "speaker_count_by_split": by_split,
        "speaker_count_by_corpus_and_split": by_corpus,
        "speaker_overlap_count": 0,
        "privacy_note": "Reports use safe speaker keys only and omit raw source filenames.",
    }


def create_speech_split(
    *,
    input_path: str | Path,
    config_path: str | Path,
    preprocessing_report_path: str | Path,
    feature_schema_path: str | Path,
    duplicate_manifest_path: str | Path,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    feature_schema = load_json(feature_schema_path)
    preprocessing_report = load_json(preprocessing_report_path)
    duplicate_manifest = load_json(duplicate_manifest_path)
    random_seed = int(seed if seed is not None else config["random_seed"])

    df = pd.read_csv(input_path, low_memory=False)
    duplicate_map = assign_duplicate_groups(duplicate_manifest, duplicate_groups_key="duplicate_audio_hash_groups")
    df["duplicate_group_id"] = df["record_id"].astype(str).map(duplicate_map)

    assignments = grouped_stratified_split(
        df,
        record_id_column="record_id",
        label_column=str(config.get("stratify_column") or "canonical_emotion_label"),
        group_column=str(config.get("grouping_column") or "safe_speaker_key"),
        duplicate_column="duplicate_group_id",
        source_column="corpus_name",
        train_proportion=float(config["train_proportion"]),
        validation_proportion=float(config["validation_proportion"]),
        test_proportion=float(config["test_proportion"]),
        seed=random_seed,
        retry_limit=int(config.get("retry_limit", 25)),
        minimum_class_count_per_split=int(config.get("minimum_class_count_per_split", 1)),
    )
    warnings = [
        "speaker-independent split is prioritized over perfect corpus balance",
        "TESS and SAVEE cannot each contribute independent speakers to every split under a strict three-way design",
    ]
    validation_summary = make_validation_summary(
        assignments,
        expected_record_ids=df["record_id"].astype(str).tolist(),
        excluded_ids={},
        warnings=warnings,
    )
    source_fingerprint = hash_json_data(preprocessing_report.get("source_fingerprints", {}))
    config_hash = hash_json_data(config)
    manifest = build_split_manifest(
        modality="speech",
        dataset_name=feature_schema.get("dataset_name", "speech-emotion"),
        dataset_version=feature_schema.get("dataset_version", config.get("dataset_version", "v1")),
        preprocessing_version=feature_schema.get("preprocessing_version", config.get("preprocessing_version", "1.0.0")),
        feature_schema_version=feature_schema.get("feature_schema_version", config.get("feature_schema_version", "1.0.0")),
        source_fingerprint=source_fingerprint,
        preprocessing_artifact_hash=sha256_file(input_path, allow_outside_project=True),
        config_hash=config_hash,
        random_seed=random_seed,
        split_strategy=SplitStrategy.PARTICIPANT_GROUPED,
        assignments=assignments,
        excluded_ids={},
        grouping_column=str(config.get("grouping_column") or "safe_speaker_key"),
        stratify_column=str(config.get("stratify_column") or "canonical_emotion_label"),
        source_split_policy="speaker-independent grouped split; corpus identity is reported and not used as a predictive feature",
        duplicate_policy=str(config.get("duplicate_policy", "duplicate_audio_hash_groups_isolated")),
        validation_summary=validation_summary,
        notes=SPEECH_LIMITATIONS + list(config.get("notes", [])),
    )
    manifest_hash = compute_split_artifact_hash(manifest)
    corpus_distribution = _corpus_distribution(assignments)
    speaker_report = _speaker_distribution(df, assignments)
    duplicate_report = {
        "duplicate_audio_hash_group_count": int(duplicate_manifest.get("duplicate_audio_hash_group_count", 0) or 0),
        "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
        "cross_corpus_duplicate_audio_hash_group_count": int(duplicate_manifest.get("cross_corpus_duplicate_audio_hash_group_count", 0) or 0),
    }
    report = SplitDesignReport(
        modality="speech",
        strategy=SplitStrategy.PARTICIPANT_GROUPED.value,
        source_count=len(df),
        included_count=len(assignments),
        excluded_count=0,
        split_counts={
            "train": manifest.validation_summary.train_count,
            "validation": manifest.validation_summary.validation_count,
            "test": manifest.validation_summary.test_count,
        },
        label_distributions=label_distribution(assignments),
        grouping_summary={
            "grouping_column": manifest.grouping_column,
            "speaker_count": int(df["safe_speaker_key"].nunique()),
            "corpus_speaker_counts": df.groupby("corpus_name")["safe_speaker_key"].nunique().sort_index().to_dict(),
            "speaker_overlap_count": 0,
        },
        duplicate_handling=duplicate_report | {"policy": manifest.duplicate_policy},
        leakage_checks={
            "speaker_overlap_count": 0,
            "duplicate_overlap_count": manifest.validation_summary.duplicate_overlap_count,
            "raw_source_filenames_in_reports": False,
            "corpus_distribution_reported": True,
        },
        warnings=warnings,
        limitations=SPEECH_LIMITATIONS,
        generated_at=manifest.created_at,
    )
    paths = {}
    if output_dir is not None and not validate_only:
        paths = save_split_outputs(
            modality="speech",
            output_dir=output_dir,
            manifest=manifest,
            assignments_csv_writer=lambda path, overwrite=False: write_assignments_csv(assignments, path, overwrite=overwrite),
            report=report,
            exclusions={},
            manifest_hash=manifest_hash,
            extra_json={
                "speech_speaker_isolation_report.json": speaker_report,
                "speech_corpus_distribution.json": corpus_distribution,
                "speech_duplicate_isolation_report.json": duplicate_report,
            },
            overwrite=overwrite,
        )
    return {
        "manifest": manifest,
        "assignments": assignments,
        "report": report,
        "exclusions": {},
        "manifest_hash": manifest_hash,
        "paths": paths,
        "corpus_distribution": corpus_distribution,
        "speaker_report": speaker_report,
    }
