"""Locked Text split loading and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app.ml.common import hashing, paths
from app.ml.training.text.constants import (
    PROHIBITED_TEXT_FEATURES,
    TEXT_HASH_COLUMN,
    TEXT_LABELS,
    TEXT_RECORD_ID_COLUMN,
    TEXT_TARGET_COLUMN,
    TEXT_TEXT_COLUMN,
)
from app.ml.training.text.schemas import TextSplitManifest, TextTrainingBundle


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    with _resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text_canonical_data(path: str | Path) -> pd.DataFrame:
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Text canonical data not found: {data_path}")
    df = pd.read_csv(data_path)
    required = {TEXT_RECORD_ID_COLUMN, TEXT_TEXT_COLUMN, TEXT_TARGET_COLUMN, TEXT_HASH_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"canonical Text data missing required columns: {missing}")
    if df[TEXT_RECORD_ID_COLUMN].duplicated().any():
        raise ValueError("canonical Text data contains duplicate record_id values")
    if df[TEXT_TEXT_COLUMN].isna().any():
        raise ValueError("canonical Text data contains missing normalized_text")
    return df


def load_text_split_manifest(path: str | Path) -> TextSplitManifest:
    manifest_path = _resolve_project_path(path)
    payload = _read_json(manifest_path)
    for key in ("train_ids", "validation_ids", "test_ids"):
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f"Text split manifest missing {key}")
    return TextSplitManifest(
        train_ids=[str(value) for value in payload["train_ids"]],
        validation_ids=[str(value) for value in payload["validation_ids"]],
        test_ids=[str(value) for value in payload["test_ids"]],
        excluded_ids={str(key): str(value) for key, value in dict(payload.get("excluded_ids") or {}).items()},
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        preprocessing_artifact_hash=str(payload.get("preprocessing_artifact_hash", "")),
        manifest_hash=hashing.sha256_file(manifest_path),
        payload=payload,
    )


def load_text_conflict_quarantine(path: str | Path) -> pd.DataFrame:
    quarantine_path = _resolve_project_path(path)
    if not quarantine_path.exists():
        raise FileNotFoundError(f"Text conflict quarantine not found: {quarantine_path}")
    return pd.read_csv(quarantine_path)


def verify_text_integrity(
    *,
    canonical_data_path: str | Path,
    split_manifest: TextSplitManifest,
    source_fingerprint_path: str | Path,
    expected_split_manifest_hash: str | None = None,
) -> None:
    source_payload = _read_json(source_fingerprint_path)
    source_hash = source_payload.get("combined_sha256") or source_payload.get("sha256")
    if source_hash != split_manifest.source_fingerprint:
        raise ValueError("source fingerprint mismatch against locked Text split manifest")
    preprocessing_hash = hashing.sha256_file(canonical_data_path)
    if preprocessing_hash != split_manifest.preprocessing_artifact_hash:
        raise ValueError("preprocessing artifact hash mismatch against locked Text split manifest")
    if expected_split_manifest_hash and expected_split_manifest_hash != split_manifest.manifest_hash:
        raise ValueError("split manifest hash mismatch against Text training config")


def validate_text_feature_policy(features: list[str]) -> None:
    if features != [TEXT_TEXT_COLUMN]:
        raise ValueError("Text baseline may use only normalized_text as the predictive input")
    prohibited = sorted(set(features) & PROHIBITED_TEXT_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited Text feature(s) requested: {prohibited}")


def validate_text_target(df: pd.DataFrame) -> None:
    values = set(df[TEXT_TARGET_COLUMN].dropna().astype(str))
    if values != set(TEXT_LABELS):
        raise ValueError(f"Text target must contain exactly {TEXT_LABELS}, got {sorted(values)}")


def _assert_label_presence(splits: Mapping[str, pd.DataFrame]) -> None:
    for split_name, split_df in splits.items():
        labels = set(split_df[TEXT_TARGET_COLUMN].astype(str))
        missing = sorted(set(TEXT_LABELS) - labels)
        if missing:
            raise ValueError(f"Text split {split_name} is missing labels: {missing}")


def _assert_no_hash_overlap(splits: Mapping[str, pd.DataFrame]) -> None:
    hash_sets = {name: set(df[TEXT_HASH_COLUMN].astype(str)) for name, df in splits.items()}
    names = list(hash_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = hash_sets[left] & hash_sets[right]
            if overlap:
                raise ValueError(f"text_hash overlap across {left}/{right}: {len(overlap)}")


def _assert_no_duplicate_group_overlap(duplicate_manifest: Mapping[str, Any], id_to_split: Mapping[str, str]) -> None:
    for group in duplicate_manifest.get("exact_duplicate_groups", []):
        record_ids = [str(value) for value in group.get("record_ids", [])]
        present_splits = {id_to_split[record_id] for record_id in record_ids if record_id in id_to_split}
        if len(present_splits) > 1:
            raise ValueError("duplicate group crosses Text splits")


def select_text_split_rows(
    df: pd.DataFrame,
    manifest: TextSplitManifest,
    quarantine_df: pd.DataFrame,
    duplicate_manifest: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    all_ids = manifest.all_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Text split manifest contains duplicate IDs across splits")
    canonical_ids = set(df[TEXT_RECORD_ID_COLUMN].astype(str))
    manifest_ids = set(all_ids)
    missing = sorted(manifest_ids - canonical_ids)
    if missing:
        raise ValueError(f"canonical Text data is missing split records: {missing[:5]}")
    extra = sorted(canonical_ids - manifest_ids)
    if extra:
        raise ValueError(f"canonical Text data contains records outside locked split: {extra[:5]}")
    quarantine_ids = set(quarantine_df.get(TEXT_RECORD_ID_COLUMN, pd.Series(dtype=str)).dropna().astype(str))
    leaked_quarantine = sorted(quarantine_ids & manifest_ids)
    if leaked_quarantine:
        raise ValueError(f"quarantined Text records appear in split manifest: {leaked_quarantine[:5]}")

    indexed = df.assign(**{TEXT_RECORD_ID_COLUMN: df[TEXT_RECORD_ID_COLUMN].astype(str)}).set_index(TEXT_RECORD_ID_COLUMN)
    splits = {
        "train": indexed.loc[manifest.train_ids].reset_index(),
        "validation": indexed.loc[manifest.validation_ids].reset_index(),
        "test": indexed.loc[manifest.test_ids].reset_index(),
    }
    _assert_label_presence(splits)
    _assert_no_hash_overlap(splits)
    id_to_split = {record_id: name for name, ids in {"train": manifest.train_ids, "validation": manifest.validation_ids, "test": manifest.test_ids}.items() for record_id in ids}
    _assert_no_duplicate_group_overlap(duplicate_manifest, id_to_split)
    return splits


def build_text_training_bundle(
    *,
    canonical_data_path: str | Path,
    split_manifest_path: str | Path,
    source_fingerprint_path: str | Path,
    duplicate_manifest_path: str | Path,
    conflict_quarantine_path: str | Path,
    source_overlap_report_path: str | Path,
    features: list[str] | None = None,
    expected_split_manifest_hash: str | None = None,
) -> TextTrainingBundle:
    features = features or [TEXT_TEXT_COLUMN]
    validate_text_feature_policy(features)
    split_manifest = load_text_split_manifest(split_manifest_path)
    verify_text_integrity(
        canonical_data_path=canonical_data_path,
        split_manifest=split_manifest,
        source_fingerprint_path=source_fingerprint_path,
        expected_split_manifest_hash=expected_split_manifest_hash,
    )
    df = load_text_canonical_data(canonical_data_path)
    validate_text_target(df)
    quarantine = load_text_conflict_quarantine(conflict_quarantine_path)
    duplicate_manifest = _read_json(duplicate_manifest_path)
    source_overlap_report = _read_json(source_overlap_report_path)
    splits = select_text_split_rows(df, split_manifest, quarantine, duplicate_manifest)
    return TextTrainingBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        text_column=TEXT_TEXT_COLUMN,
        target=TEXT_TARGET_COLUMN,
        split_manifest=split_manifest,
        source_fingerprint=split_manifest.source_fingerprint,
        preprocessing_artifact_hash=split_manifest.preprocessing_artifact_hash,
        duplicate_manifest=duplicate_manifest,
        source_overlap_report=source_overlap_report,
    )
