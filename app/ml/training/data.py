"""Split-safe canonical data loading for Phase 3B training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
from pydantic.v1 import ValidationError

from app.ml.common import hashing, paths
from app.ml.splitting.schemas import ModalitySplitManifest
from app.ml.training.constants import (
    DEFAULT_RECORD_ID_COLUMN,
    IDENTIFIER_COLUMN_TOKENS,
    PRODUCTION_IDENTIFIER_COLUMNS,
)
from app.ml.training.schemas import DatasetSplitReference, TrainingConfig


@dataclass(frozen=True)
class TrainingDatasetBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dataset_reference: DatasetSplitReference
    record_id_column: str = DEFAULT_RECORD_ID_COLUMN


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve(strict=False)
        repo_candidate = (paths.get_repository_root() / candidate).resolve(strict=False)
        if cwd_candidate.exists() and paths.is_path_inside(paths.get_repository_root(), cwd_candidate):
            candidate = cwd_candidate
        elif repo_candidate.exists() or not str(candidate).replace("\\", "/").startswith("../"):
            candidate = repo_candidate
        else:
            candidate = cwd_candidate
    resolved = candidate.resolve(strict=False)
    if not paths.is_path_inside(paths.get_repository_root(), resolved):
        raise ValueError(f"training data path must stay inside the repository: {resolved}")
    return resolved


def _load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_split_manifest(path: str | Path) -> ModalitySplitManifest:
    resolved = _resolve_project_path(path)
    payload = _load_json_object(resolved)
    try:
        manifest = ModalitySplitManifest.parse_obj(payload)
    except ValidationError:
        raise
    if not manifest.manifest_version:
        raise ValueError("split manifest must be locked and versioned")
    return manifest


def load_canonical_dataset(path: str | Path) -> pd.DataFrame:
    resolved = _resolve_project_path(path)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    if resolved.suffix.lower() in {".json", ".jsonl"}:
        lines = resolved.suffix.lower() == ".jsonl"
        return pd.read_json(resolved, lines=lines)
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported canonical dataset format: {resolved.suffix}")


def select_records_by_split(
    records: pd.DataFrame,
    split_ids: Iterable[str],
    *,
    record_id_column: str = DEFAULT_RECORD_ID_COLUMN,
) -> pd.DataFrame:
    if record_id_column not in records.columns:
        raise ValueError(f"canonical records missing record ID column: {record_id_column}")
    order = [str(record_id) for record_id in split_ids]
    indexed = records.copy()
    indexed[record_id_column] = indexed[record_id_column].astype(str)
    duplicates = indexed[indexed[record_id_column].duplicated()][record_id_column].tolist()
    if duplicates:
        raise ValueError(f"canonical records contain duplicate record IDs: {duplicates[:5]}")
    by_id = indexed.set_index(record_id_column, drop=False)
    missing = [record_id for record_id in order if record_id not in by_id.index]
    if missing:
        raise ValueError(f"split manifest IDs missing from canonical records: {missing[:10]}")
    return by_id.loc[order].reset_index(drop=True)


def validate_record_coverage(
    records: pd.DataFrame,
    manifest: ModalitySplitManifest,
    *,
    record_id_column: str = DEFAULT_RECORD_ID_COLUMN,
) -> None:
    if record_id_column not in records.columns:
        raise ValueError(f"canonical records missing record ID column: {record_id_column}")
    record_ids = set(records[record_id_column].astype(str))
    included = set(manifest.train_ids) | set(manifest.validation_ids) | set(manifest.test_ids)
    excluded = set(manifest.excluded_ids.keys())
    missing = included - record_ids
    if missing:
        raise ValueError(f"manifest IDs missing from canonical records: {sorted(missing)[:10]}")
    accidentally_loaded = excluded & record_ids
    if accidentally_loaded:
        raise ValueError(f"excluded records were loaded: {sorted(accidentally_loaded)[:10]}")


def validate_target_column(records: pd.DataFrame, target_column: str, feature_columns: Iterable[str]) -> None:
    if target_column not in records.columns:
        raise ValueError(f"target column missing from canonical records: {target_column}")
    if target_column in set(feature_columns):
        raise ValueError("target leakage: target column appears in feature columns")


def _identifier_like(column: str) -> bool:
    lowered = column.lower()
    return lowered in PRODUCTION_IDENTIFIER_COLUMNS or any(token in lowered for token in IDENTIFIER_COLUMN_TOKENS)


def validate_feature_columns(
    records: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    target_column: Optional[str] = None,
    excluded_columns: Iterable[str] = (),
    allow_identifier_columns: bool = False,
) -> None:
    features = [str(column) for column in feature_columns]
    missing = [column for column in features if column not in records.columns]
    if missing:
        raise ValueError(f"feature columns missing from canonical records: {missing}")
    if target_column and target_column in features:
        raise ValueError("target leakage: target column appears in feature columns")
    excluded_overlap = sorted(set(excluded_columns) & set(features))
    if excluded_overlap:
        raise ValueError(f"excluded feature columns requested: {excluded_overlap}")
    identifier_features = [column for column in features if _identifier_like(column)]
    if identifier_features and not allow_identifier_columns:
        raise ValueError(f"identifier leakage in feature columns: {identifier_features}")


def validate_no_cross_split_overlap(manifest: ModalitySplitManifest) -> None:
    splits = {
        "train": set(manifest.train_ids),
        "validation": set(manifest.validation_ids),
        "test": set(manifest.test_ids),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = splits[left] & splits[right]
        if overlap:
            raise ValueError(f"record IDs overlap between {left} and {right}: {sorted(overlap)[:10]}")


def validate_source_fingerprint(manifest: ModalitySplitManifest, expected_source_fingerprint: str) -> None:
    if not manifest.source_fingerprint:
        raise ValueError("source fingerprint is mandatory")
    if manifest.source_fingerprint.lower() != expected_source_fingerprint.lower():
        raise ValueError("source fingerprint mismatch")


def validate_preprocessing_artifact_hash(manifest: ModalitySplitManifest, expected_preprocessing_artifact_hash: str) -> None:
    if not manifest.preprocessing_artifact_hash:
        raise ValueError("preprocessing artifact hash is mandatory")
    if manifest.preprocessing_artifact_hash.lower() != expected_preprocessing_artifact_hash.lower():
        raise ValueError("preprocessing artifact hash mismatch")


def build_split_reference(manifest: ModalitySplitManifest, manifest_path: str | Path) -> DatasetSplitReference:
    manifest_hash = hashing.sha256_file(_resolve_project_path(manifest_path))
    return DatasetSplitReference(
        train_ids=manifest.train_ids,
        validation_ids=manifest.validation_ids,
        test_ids=manifest.test_ids,
        manifest_hash=manifest_hash,
        source_fingerprint=manifest.source_fingerprint,
        preprocessing_artifact_hash=manifest.preprocessing_artifact_hash,
    )


def build_training_dataset_bundle(
    *,
    config: TrainingConfig,
    split_manifest_path: str | Path,
    canonical_data_path: str | Path,
    expected_source_fingerprint: Optional[str] = None,
    expected_preprocessing_artifact_hash: Optional[str] = None,
    record_id_column: str = DEFAULT_RECORD_ID_COLUMN,
) -> TrainingDatasetBundle:
    manifest = load_split_manifest(split_manifest_path)
    validate_no_cross_split_overlap(manifest)
    if expected_source_fingerprint is not None:
        validate_source_fingerprint(manifest, expected_source_fingerprint)
    if expected_preprocessing_artifact_hash is not None:
        validate_preprocessing_artifact_hash(manifest, expected_preprocessing_artifact_hash)

    records = load_canonical_dataset(canonical_data_path)
    validate_record_coverage(records, manifest, record_id_column=record_id_column)
    validate_target_column(records, config.target_column, config.feature_columns)
    validate_feature_columns(
        records,
        config.feature_columns,
        target_column=config.target_column,
        excluded_columns=config.excluded_columns,
    )

    train = select_records_by_split(records, manifest.train_ids, record_id_column=record_id_column)
    validation = select_records_by_split(records, manifest.validation_ids, record_id_column=record_id_column)
    test = select_records_by_split(records, manifest.test_ids, record_id_column=record_id_column)
    return TrainingDatasetBundle(
        train=train,
        validation=validation,
        test=test,
        dataset_reference=build_split_reference(manifest, split_manifest_path),
        record_id_column=record_id_column,
    )
