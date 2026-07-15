"""Locked split loading and leakage checks for Speech baseline training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.ml.common import hashing, paths
from app.ml.training.speech.constants import (
    DEFAULT_CORPUS_SUMMARY,
    DEFAULT_DUPLICATE_ISOLATION_REPORT,
    DEFAULT_DUPLICATE_MANIFEST,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_FINGERPRINT_DIR,
    DEFAULT_PREPROCESSING_REPORT,
    DEFAULT_SPEAKER_ISOLATION_REPORT,
    DEFAULT_SPLIT_ASSIGNMENTS,
    FEATURE_SETS,
    PROHIBITED_FEATURES,
    SPEECH_CORPUS_COLUMN,
    SPEECH_LABELS,
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_SPEAKER_KEY_COLUMN,
    SPEECH_TARGET_COLUMN,
)
from app.ml.training.speech.schemas import SpeechSplitManifest, SpeechTrainingBundle


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    with _resolve_project_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_speech_feature_data(path: str | Path) -> pd.DataFrame:
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Speech feature data not found: {data_path}")
    df = pd.read_csv(data_path)
    required = {SPEECH_RECORD_ID_COLUMN, SPEECH_SPEAKER_KEY_COLUMN, SPEECH_CORPUS_COLUMN, SPEECH_TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Speech feature data missing required columns: {missing}")
    if len(df) and df[SPEECH_RECORD_ID_COLUMN].duplicated().any():
        raise ValueError("Speech feature data contains duplicate record_id values")
    return df


def load_speech_canonical_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = _resolve_project_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Speech canonical manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {
        SPEECH_RECORD_ID_COLUMN,
        SPEECH_SPEAKER_KEY_COLUMN,
        SPEECH_CORPUS_COLUMN,
        SPEECH_TARGET_COLUMN,
        "audio_relative_path",
        "original_audio_hash",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Speech canonical manifest missing required columns: {missing}")
    if df[SPEECH_RECORD_ID_COLUMN].duplicated().any():
        raise ValueError("Speech canonical manifest contains duplicate record_id values")
    return df


def load_speech_split_manifest(path: str | Path) -> SpeechSplitManifest:
    manifest_path = _resolve_project_path(path)
    payload = _read_json(manifest_path)
    for key in ("train_ids", "validation_ids", "test_ids"):
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f"Speech split manifest missing {key}")
    return SpeechSplitManifest(
        train_ids=[str(value) for value in payload["train_ids"]],
        validation_ids=[str(value) for value in payload["validation_ids"]],
        test_ids=[str(value) for value in payload["test_ids"]],
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        preprocessing_artifact_hash=str(payload.get("preprocessing_artifact_hash", "")),
        manifest_hash=hashing.sha256_file(manifest_path),
        payload=payload,
    )


def load_speech_duplicate_manifest(path: str | Path = DEFAULT_DUPLICATE_MANIFEST) -> dict[str, Any]:
    return _read_json(path)


def load_speech_corpus_summary(path: str | Path = DEFAULT_CORPUS_SUMMARY) -> dict[str, Any]:
    return _read_json(path)


def feature_columns_from_schema(schema_path: str | Path = DEFAULT_FEATURE_SCHEMA) -> list[str]:
    schema = _read_json(schema_path)
    features = [str(item["name"]) for item in schema.get("features", []) if isinstance(item, dict) and item.get("name")]
    if not features:
        raise ValueError("Speech feature schema does not define feature names")
    return features


def resolve_speech_feature_set(config: Mapping[str, Any], *, feature_set: str | None = None) -> tuple[str, list[str]]:
    selected = feature_set or str(config.get("feature_set") or "full_acoustic")
    configured = config.get("feature_sets") or FEATURE_SETS
    if selected not in configured:
        raise ValueError(f"unknown Speech feature set: {selected}")
    return selected, [str(feature) for feature in configured[selected]]


def validate_speech_feature_policy(features: list[str]) -> None:
    if not features:
        raise ValueError("Speech training requires at least one feature")
    prohibited = sorted(set(features) & PROHIBITED_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited Speech feature(s) requested: {prohibited}")
    if SPEECH_CORPUS_COLUMN in features:
        raise ValueError("corpus name is not permitted as a primary Speech predictive feature")


def verify_speech_integrity(
    *,
    canonical_manifest_path: str | Path,
    split_manifest: SpeechSplitManifest,
    preprocessing_report_path: str | Path = DEFAULT_PREPROCESSING_REPORT,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    expected_split_manifest_hash: str | None = None,
) -> dict[str, str]:
    report = _read_json(preprocessing_report_path)
    expected_sources = {str(key): str(value) for key, value in dict(report.get("source_fingerprints") or {}).items()}
    observed_sources: dict[str, str] = {}
    fingerprint_root = _resolve_project_path(fingerprint_dir)
    for corpus in ("CREMA", "RAVDESS", "SAVEE", "TESS"):
        matches = sorted(fingerprint_root.glob(f"{corpus.lower()}*.json"))
        if not matches:
            raise FileNotFoundError(f"missing source fingerprint for {corpus} in {fingerprint_root}")
        payload = _read_json(matches[0])
        observed_sources[corpus] = str(payload.get("combined_sha256") or payload.get("sha256") or "")
    if expected_sources and observed_sources != expected_sources:
        raise ValueError("Speech source fingerprint mismatch against preprocessing report")
    if hashing.hash_json_data(expected_sources) != split_manifest.source_fingerprint:
        raise ValueError("Speech source fingerprint mismatch against locked split manifest")
    preprocessing_hash = hashing.sha256_file(canonical_manifest_path)
    if preprocessing_hash != split_manifest.preprocessing_artifact_hash:
        raise ValueError("Speech preprocessing artifact hash mismatch against locked split manifest")
    if expected_split_manifest_hash and expected_split_manifest_hash != split_manifest.manifest_hash:
        raise ValueError("Speech split manifest hash mismatch against training config")
    return observed_sources


def _assert_no_overlap(values_by_split: Mapping[str, set[str]], label: str) -> None:
    names = list(values_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = values_by_split[left] & values_by_split[right]
            if overlap:
                raise ValueError(f"{label} overlap across {left}/{right}: {len(overlap)}")


def validate_speech_training_contract(
    *,
    feature_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    split_manifest: SpeechSplitManifest,
    features: list[str],
    duplicate_manifest: Mapping[str, Any],
    speaker_isolation_report: Mapping[str, Any],
    duplicate_isolation_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_speech_feature_policy(features)
    missing_features = sorted(set(features) - set(feature_df.columns))
    if missing_features:
        raise ValueError(f"requested Speech features missing from feature data: {missing_features}")
    values = set(canonical_df[SPEECH_TARGET_COLUMN].dropna().astype(str))
    if values != set(SPEECH_LABELS):
        raise ValueError(f"Speech target must contain exactly {SPEECH_LABELS}, got {sorted(values)}")
    if len(set(split_manifest.all_ids)) != len(split_manifest.all_ids):
        raise ValueError("Speech split manifest contains duplicate IDs across splits")
    canonical_ids = set(canonical_df[SPEECH_RECORD_ID_COLUMN].astype(str))
    manifest_ids = set(split_manifest.all_ids)
    missing_canonical = sorted(manifest_ids - canonical_ids)
    if missing_canonical:
        raise ValueError(f"canonical Speech manifest is missing split records: {missing_canonical[:5]}")
    feature_ids = set(feature_df[SPEECH_RECORD_ID_COLUMN].astype(str))
    missing_feature_ids = sorted(manifest_ids - feature_ids)
    numeric = feature_df[features].apply(pd.to_numeric, errors="coerce") if len(feature_df) else pd.DataFrame(columns=features)
    bad_numeric = sorted(column for column in features if column in numeric.columns and np.isinf(numeric[column].to_numpy(dtype=float, na_value=np.nan)).any())
    missing_required_values = int(numeric.loc[feature_df[SPEECH_RECORD_ID_COLUMN].astype(str).isin(manifest_ids)].isna().any(axis=1).sum()) if len(feature_df) else len(manifest_ids)
    coverage = {
        "required_record_count": len(manifest_ids),
        "feature_row_count": len(feature_ids & manifest_ids),
        "missing_feature_count": len(missing_feature_ids),
        "missing_feature_record_ids": missing_feature_ids[:20],
        "feature_count": len(features),
        "feature_columns": features,
        "rows_with_missing_required_features": missing_required_values,
        "infinite_feature_columns": bad_numeric,
        "complete": len(missing_feature_ids) == 0 and missing_required_values == 0 and not bad_numeric,
    }
    if missing_feature_ids:
        raise ValueError(f"Speech feature data missing locked split records: {len(missing_feature_ids)}")
    if missing_required_values:
        raise ValueError(f"Speech feature data has missing values in required features: {missing_required_values} rows")
    if bad_numeric:
        raise ValueError(f"Speech feature data has infinite values in columns: {bad_numeric}")
    if int(speaker_isolation_report.get("speaker_overlap_count", 0) or 0) != 0:
        raise ValueError("Speech speaker isolation report shows overlap")
    if int(duplicate_isolation_report.get("duplicate_overlap_count", 0) or 0) != 0:
        raise ValueError("Speech duplicate isolation report shows overlap")
    id_to_split = {record_id: name for name, ids in {"train": split_manifest.train_ids, "validation": split_manifest.validation_ids, "test": split_manifest.test_ids}.items() for record_id in ids}
    duplicate_groups = duplicate_manifest.get("duplicate_audio_hash_groups") or {}
    for group_ids in duplicate_groups.values():
        splits = {id_to_split[str(record_id)] for record_id in group_ids if str(record_id) in id_to_split}
        if len(splits) > 1:
            raise ValueError("duplicate audio hash group crosses Speech splits")
    return coverage


def select_speech_split_rows(df: pd.DataFrame, manifest: SpeechSplitManifest) -> dict[str, pd.DataFrame]:
    indexed = df.assign(**{SPEECH_RECORD_ID_COLUMN: df[SPEECH_RECORD_ID_COLUMN].astype(str)}).set_index(SPEECH_RECORD_ID_COLUMN)
    splits = {
        "train": indexed.loc[manifest.train_ids].reset_index(),
        "validation": indexed.loc[manifest.validation_ids].reset_index(),
        "test": indexed.loc[manifest.test_ids].reset_index(),
    }
    speaker_sets = {name: set(split[SPEECH_SPEAKER_KEY_COLUMN].astype(str)) for name, split in splits.items()}
    _assert_no_overlap(speaker_sets, "safe speaker key")
    return splits


def inspect_speech_feature_coverage(
    *,
    features_path: str | Path,
    split_manifest_path: str | Path,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    feature_set: str = "full_acoustic",
) -> dict[str, Any]:
    split_manifest = load_speech_split_manifest(split_manifest_path)
    features = FEATURE_SETS.get(feature_set, feature_columns_from_schema(feature_schema_path))
    feature_df = load_speech_feature_data(features_path)
    feature_ids = set(feature_df[SPEECH_RECORD_ID_COLUMN].astype(str)) if len(feature_df) else set()
    missing = sorted(set(split_manifest.all_ids) - feature_ids)
    return {
        "required_record_count": len(split_manifest.all_ids),
        "feature_row_count": len(feature_ids & set(split_manifest.all_ids)),
        "missing_feature_count": len(missing),
        "missing_feature_record_ids": missing[:20],
        "feature_count": len(features),
        "feature_columns": features,
        "complete": len(missing) == 0,
    }


def build_speech_training_bundle(
    *,
    features_path: str | Path,
    canonical_manifest_path: str | Path,
    split_manifest_path: str | Path,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    preprocessing_report_path: str | Path = DEFAULT_PREPROCESSING_REPORT,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    duplicate_manifest_path: str | Path = DEFAULT_DUPLICATE_MANIFEST,
    corpus_summary_path: str | Path = DEFAULT_CORPUS_SUMMARY,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS,
    speaker_isolation_report_path: str | Path = DEFAULT_SPEAKER_ISOLATION_REPORT,
    duplicate_isolation_report_path: str | Path = DEFAULT_DUPLICATE_ISOLATION_REPORT,
    feature_set: str = "full_acoustic",
    features: list[str] | None = None,
    expected_split_manifest_hash: str | None = None,
) -> SpeechTrainingBundle:
    split_manifest = load_speech_split_manifest(split_manifest_path)
    verify_speech_integrity(
        canonical_manifest_path=canonical_manifest_path,
        split_manifest=split_manifest,
        preprocessing_report_path=preprocessing_report_path,
        fingerprint_dir=fingerprint_dir,
        expected_split_manifest_hash=expected_split_manifest_hash,
    )
    schema = _read_json(feature_schema_path)
    schema_features = feature_columns_from_schema(feature_schema_path)
    selected_features = list(features or FEATURE_SETS.get(feature_set, schema_features))
    feature_df = load_speech_feature_data(features_path)
    canonical_df = load_speech_canonical_manifest(canonical_manifest_path)
    duplicate_manifest = load_speech_duplicate_manifest(duplicate_manifest_path)
    corpus_summary = load_speech_corpus_summary(corpus_summary_path)
    speaker_isolation_report = _read_json(speaker_isolation_report_path)
    duplicate_isolation_report = _read_json(duplicate_isolation_report_path)
    coverage = validate_speech_training_contract(
        feature_df=feature_df,
        canonical_df=canonical_df,
        split_manifest=split_manifest,
        features=selected_features,
        duplicate_manifest=duplicate_manifest,
        speaker_isolation_report=speaker_isolation_report,
        duplicate_isolation_report=duplicate_isolation_report,
    )
    canonical_meta = canonical_df[[SPEECH_RECORD_ID_COLUMN, "audio_relative_path", "original_audio_hash", "duration_seconds", "sample_rate", "channel_count"]]
    merged = feature_df.merge(canonical_meta, on=SPEECH_RECORD_ID_COLUMN, how="left", suffixes=("", "_canonical"))
    if _resolve_project_path(split_assignments_path).exists():
        assignments = pd.read_csv(_resolve_project_path(split_assignments_path))
        if "split" in assignments.columns:
            split_counts = assignments["split"].value_counts().to_dict()
            if split_counts.get("train") != len(split_manifest.train_ids):
                raise ValueError("Speech split assignments mismatch train count")
    splits = select_speech_split_rows(merged, split_manifest)
    return SpeechTrainingBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        features=selected_features,
        target=SPEECH_TARGET_COLUMN,
        split_manifest=split_manifest,
        source_fingerprint=split_manifest.source_fingerprint,
        preprocessing_artifact_hash=split_manifest.preprocessing_artifact_hash,
        feature_schema=schema,
        preprocessing_report=_read_json(preprocessing_report_path),
        duplicate_manifest=duplicate_manifest,
        corpus_summary=corpus_summary,
        speaker_isolation_report=speaker_isolation_report,
        duplicate_isolation_report=duplicate_isolation_report,
        feature_coverage=coverage,
    )

