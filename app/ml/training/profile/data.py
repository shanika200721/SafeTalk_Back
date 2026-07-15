"""Data loading and leakage policy checks for Profile baseline training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app.ml.common import hashing, paths
from app.ml.training.profile.constants import (
    FEATURE_SETS,
    OUTCOME_LIKE_FEATURES,
    PROFILE_NEGATIVE_LABEL,
    PROFILE_POSITIVE_LABEL,
    PROFILE_RECORD_ID_COLUMN,
    PROFILE_TARGET_COLUMN,
    PROHIBITED_FEATURES,
    SENSITIVE_CONTEXT_FEATURES,
    SENSITIVE_CONTEXT_FEATURE_SET,
)
from app.ml.training.profile.schemas import ProfileSplitManifest, ProfileTrainingBundle


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    with _resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_profile_canonical_data(path: str | Path) -> pd.DataFrame:
    data_path = _resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Profile canonical data not found: {data_path}")
    df = pd.read_csv(data_path)
    if PROFILE_RECORD_ID_COLUMN not in df.columns:
        raise ValueError("canonical Profile data must include record_id")
    if df[PROFILE_RECORD_ID_COLUMN].duplicated().any():
        raise ValueError("canonical Profile data contains duplicate record_id values")
    return df


def load_profile_split_manifest(path: str | Path) -> ProfileSplitManifest:
    manifest_path = _resolve_project_path(path)
    payload = _read_json(manifest_path)
    manifest_hash = hashing.sha256_file(manifest_path)
    for key in ("train_ids", "validation_ids", "test_ids"):
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f"split manifest missing {key}")
    return ProfileSplitManifest(
        train_ids=[str(value) for value in payload["train_ids"]],
        validation_ids=[str(value) for value in payload["validation_ids"]],
        test_ids=[str(value) for value in payload["test_ids"]],
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        preprocessing_artifact_hash=str(payload.get("preprocessing_artifact_hash", "")),
        manifest_hash=manifest_hash,
        payload=payload,
    )


def verify_profile_integrity(
    *,
    canonical_data_path: str | Path,
    split_manifest: ProfileSplitManifest,
    source_fingerprint_path: str | Path,
    expected_split_manifest_hash: str | None = None,
) -> None:
    source_payload = _read_json(source_fingerprint_path)
    source_hash = source_payload.get("combined_sha256") or source_payload.get("sha256")
    if source_hash != split_manifest.source_fingerprint:
        raise ValueError("source fingerprint mismatch against locked split manifest")

    preprocessing_hash = hashing.sha256_file(canonical_data_path)
    if preprocessing_hash != split_manifest.preprocessing_artifact_hash:
        raise ValueError("preprocessing artifact hash mismatch against locked split manifest")

    if expected_split_manifest_hash and expected_split_manifest_hash != split_manifest.manifest_hash:
        raise ValueError("split manifest hash mismatch against training config")


def select_profile_split_rows(df: pd.DataFrame, manifest: ProfileSplitManifest) -> dict[str, pd.DataFrame]:
    all_ids = manifest.all_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("split manifest contains duplicate IDs across splits")

    canonical_ids = set(df[PROFILE_RECORD_ID_COLUMN].astype(str))
    manifest_ids = set(all_ids)
    missing = sorted(manifest_ids - canonical_ids)
    if missing:
        raise ValueError(f"canonical Profile data is missing split records: {missing[:5]}")
    extra = sorted(canonical_ids - manifest_ids)
    if extra:
        raise ValueError(f"canonical Profile data contains records outside the locked split: {extra[:5]}")

    indexed = df.assign(**{PROFILE_RECORD_ID_COLUMN: df[PROFILE_RECORD_ID_COLUMN].astype(str)}).set_index(PROFILE_RECORD_ID_COLUMN)
    return {
        "train": indexed.loc[manifest.train_ids].reset_index(),
        "validation": indexed.loc[manifest.validation_ids].reset_index(),
        "test": indexed.loc[manifest.test_ids].reset_index(),
    }


def validate_profile_target(df: pd.DataFrame, target_column: str = PROFILE_TARGET_COLUMN) -> None:
    if target_column not in df.columns:
        raise ValueError(f"target column not found: {target_column}")
    values = set(df[target_column].dropna().astype(str))
    expected = {PROFILE_NEGATIVE_LABEL, PROFILE_POSITIVE_LABEL}
    if values != expected:
        raise ValueError(f"Profile target must contain exactly {sorted(expected)}, got {sorted(values)}")


def validate_profile_features(df: pd.DataFrame, features: list[str], target_column: str = PROFILE_TARGET_COLUMN) -> None:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"requested Profile features are missing from canonical data: {missing}")
    if target_column in features:
        raise ValueError("target must not appear in feature columns")
    prohibited = sorted(set(features) & PROHIBITED_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited Profile feature(s) requested: {prohibited}")


def validate_profile_feature_policy(
    features: list[str],
    *,
    feature_set: str,
    allow_sensitive_context: bool = False,
) -> None:
    if not features:
        raise ValueError("Profile training requires at least one feature")
    prohibited = sorted(set(features) & PROHIBITED_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited Profile feature(s) requested: {prohibited}")
    if PROFILE_TARGET_COLUMN in features:
        raise ValueError("target must not appear in Profile features")
    sensitive = sorted(set(features) & SENSITIVE_CONTEXT_FEATURES)
    if sensitive and (not allow_sensitive_context or feature_set != SENSITIVE_CONTEXT_FEATURE_SET):
        raise ValueError("sensitive Profile context features require explicit exploratory mode")
    if feature_set == "minimal_contextual" and set(features) & OUTCOME_LIKE_FEATURES:
        raise ValueError("minimal contextual baseline must not include outcome-like self-report features")
    if "sought_specialist_treatment" in features:
        raise ValueError("treatment-seeking is excluded from all Profile feature sets")


def resolve_profile_feature_set(
    config: Mapping[str, Any],
    *,
    feature_set: str | None = None,
) -> tuple[str, list[str]]:
    selected = feature_set or str(config.get("feature_set") or "minimal_contextual")
    ablations = config.get("feature_sets") or FEATURE_SETS
    if selected not in ablations:
        raise ValueError(f"unknown Profile feature set: {selected}")
    return selected, [str(feature) for feature in ablations[selected]]


def build_profile_training_bundle(
    *,
    canonical_data_path: str | Path,
    split_manifest_path: str | Path,
    source_fingerprint_path: str | Path,
    features: list[str],
    feature_set: str,
    allow_sensitive_context: bool = False,
    target_column: str = PROFILE_TARGET_COLUMN,
    expected_split_manifest_hash: str | None = None,
) -> ProfileTrainingBundle:
    split_manifest = load_profile_split_manifest(split_manifest_path)
    verify_profile_integrity(
        canonical_data_path=canonical_data_path,
        split_manifest=split_manifest,
        source_fingerprint_path=source_fingerprint_path,
        expected_split_manifest_hash=expected_split_manifest_hash,
    )
    df = load_profile_canonical_data(canonical_data_path)
    validate_profile_target(df, target_column)
    validate_profile_feature_policy(
        features,
        feature_set=feature_set,
        allow_sensitive_context=allow_sensitive_context,
    )
    validate_profile_features(df, features, target_column)
    splits = select_profile_split_rows(df, split_manifest)
    return ProfileTrainingBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        features=features,
        target=target_column,
        split_manifest=split_manifest,
        source_fingerprint=split_manifest.source_fingerprint,
        preprocessing_artifact_hash=split_manifest.preprocessing_artifact_hash,
    )
