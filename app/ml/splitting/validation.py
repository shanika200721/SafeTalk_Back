"""Validation checks for Phase 3A split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app.ml.common.hashing import sha256_file
from app.ml.splitting.common import (
    load_json,
    validate_duplicate_isolation,
    validate_group_isolation as common_validate_group_isolation,
    validate_label_distribution,
    validate_manifest_coverage,
    validate_no_overlap,
)
from app.ml.splitting.constants import ALLOWED_SPLIT_MODALITIES, FORBIDDEN_SPLIT_MODALITIES
from app.ml.splitting.schemas import ModalitySplitManifest, SplitRecord


def validate_no_forbidden_modalities(modality: str) -> None:
    normalized = str(modality).strip()
    if normalized in FORBIDDEN_SPLIT_MODALITIES or normalized.lower() in {item.lower() for item in FORBIDDEN_SPLIT_MODALITIES}:
        raise ValueError(f"Phase 3A cannot create splits for blocked modality: {modality}")
    if normalized.lower() not in ALLOWED_SPLIT_MODALITIES:
        raise ValueError(f"Unsupported Phase 3A split modality: {modality}")


def validate_record_coverage(
    assignments: Sequence[SplitRecord],
    expected_record_ids: Iterable[str],
    excluded_ids: Mapping[str, str] | None = None,
) -> None:
    validate_manifest_coverage(assignments, expected_record_ids, excluded_ids)


def validate_class_presence(assignments: Sequence[SplitRecord], *, minimum_class_count_per_split: int = 1) -> None:
    validate_label_distribution(assignments, minimum_class_count_per_split=minimum_class_count_per_split)


def validate_class_balance(assignments: Sequence[SplitRecord], *, max_empty_splits: int = 0) -> None:
    split_names = {"train", "validation", "test"}
    present = {assignment.split for assignment in assignments}
    empty = split_names - present
    if len(empty) > max_empty_splits:
        raise ValueError(f"Unexpected empty splits: {sorted(empty)}")


def validate_group_isolation(assignments: Sequence[SplitRecord]) -> None:
    common_validate_group_isolation(assignments)


def validate_duplicate_hash_isolation(assignments: Sequence[SplitRecord]) -> None:
    validate_duplicate_isolation(assignments)


def validate_reference_overlap_policy(policy: str | None, *, strict: bool = False) -> None:
    if not policy:
        if strict:
            raise ValueError("Reference overlap policy is required in strict mode")
        return
    lowered = policy.lower()
    if "training" in lowered and "not" not in lowered and strict:
        raise ValueError("Reference overlap policy does not restrict training leakage")


def validate_speaker_isolation(assignments: Sequence[SplitRecord]) -> None:
    validate_group_isolation(assignments)


def validate_corpus_balance(corpus_distribution: Mapping[str, Mapping[str, int]], *, strict: bool = False) -> None:
    if not corpus_distribution:
        raise ValueError("Corpus distribution report is required")
    if strict:
        for split, counts in corpus_distribution.items():
            if not counts:
                raise ValueError(f"No corpus records reported for split {split}")


def validate_preprocessing_hash(path: str | Path, expected_hash: str) -> None:
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"Preprocessing artifact hash mismatch for {path}")


def validate_source_fingerprint(fingerprint_path: str | Path, expected_combined_hash: str) -> None:
    payload = load_json(fingerprint_path)
    actual = payload.get("combined_sha256")
    if actual != expected_combined_hash:
        raise ValueError("Source fingerprint mismatch")


def validate_deterministic_replay(original: ModalitySplitManifest, replay: ModalitySplitManifest) -> None:
    if original.train_ids != replay.train_ids or original.validation_ids != replay.validation_ids or original.test_ids != replay.test_ids:
        raise ValueError("Deterministic replay did not reproduce split IDs")


def validate_split_manifest(
    manifest: ModalitySplitManifest,
    assignments: Sequence[SplitRecord],
    expected_record_ids: Iterable[str],
    *,
    minimum_class_count_per_split: int = 1,
    strict: bool = False,
) -> None:
    validate_no_forbidden_modalities(manifest.modality)
    validate_no_overlap(assignments)
    validate_record_coverage(assignments, expected_record_ids, manifest.excluded_ids)
    validate_class_presence(assignments, minimum_class_count_per_split=minimum_class_count_per_split)
    if manifest.grouping_column:
        validate_group_isolation(assignments)
    validate_duplicate_hash_isolation(assignments)
    validate_reference_overlap_policy(manifest.source_split_policy, strict=strict)
