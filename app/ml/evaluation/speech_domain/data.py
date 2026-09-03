"""Corpus-aware loading for Speech domain-shift evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.ml.common import hashing, paths
from app.ml.evaluation.speech_domain.constants import CORPORA
from app.ml.evaluation.speech_domain.schemas import SpeechDomainBundle
from app.ml.training.speech.constants import (
    PROHIBITED_FEATURES,
    SPEECH_CORPUS_COLUMN,
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_SPEAKER_KEY_COLUMN,
    SPEECH_TARGET_COLUMN,
)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _feature_names_from_schema(schema: dict[str, Any]) -> list[str]:
    features = [str(item["name"]) for item in schema.get("features", []) if isinstance(item, dict) and item.get("name")]
    if not features:
        raise ValueError("Speech feature schema does not define feature names")
    prohibited = sorted(set(features) & PROHIBITED_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited metadata requested as Speech predictive feature: {prohibited}")
    return features


def load_speech_features(path: str | Path) -> pd.DataFrame:
    data_path = resolve_project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Speech feature file not found: {data_path}")
    df = pd.read_csv(data_path)
    required = {SPEECH_RECORD_ID_COLUMN, SPEECH_SPEAKER_KEY_COLUMN, SPEECH_CORPUS_COLUMN, SPEECH_TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Speech feature file missing required columns: {missing}")
    if df[SPEECH_RECORD_ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Speech feature file contains duplicate record_id values")
    return df.sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)


def load_canonical_speech_metadata(path: str | Path) -> pd.DataFrame:
    manifest_path = resolve_project_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Speech canonical manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {
        SPEECH_RECORD_ID_COLUMN,
        SPEECH_SPEAKER_KEY_COLUMN,
        SPEECH_CORPUS_COLUMN,
        SPEECH_TARGET_COLUMN,
        "original_audio_hash",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Speech canonical manifest missing required columns: {missing}")
    if df[SPEECH_RECORD_ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Speech canonical manifest contains duplicate record_id values")
    safe_columns = [
        column
        for column in [
            SPEECH_RECORD_ID_COLUMN,
            SPEECH_SPEAKER_KEY_COLUMN,
            SPEECH_CORPUS_COLUMN,
            SPEECH_TARGET_COLUMN,
            "original_audio_hash",
            "duration_seconds",
            "sample_rate",
            "channel_count",
        ]
        if column in df.columns
    ]
    return df[safe_columns].sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)


def validate_feature_coverage(records: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    missing = sorted(set(features) - set(records.columns))
    if missing:
        raise ValueError(f"feature columns missing from Speech data: {missing}")
    numeric = records[features].apply(pd.to_numeric, errors="coerce")
    bad_finite = sorted(column for column in features if not np.isfinite(numeric[column].to_numpy(dtype=float, na_value=np.nan)).all())
    if bad_finite:
        raise ValueError(f"Speech features contain NaN or infinity in columns: {bad_finite}")
    return {
        "record_count": int(len(records)),
        "feature_count": int(len(features)),
        "missing_feature_columns": missing,
        "non_finite_feature_columns": bad_finite,
        "complete": True,
    }


def validate_corpus_labels(records: pd.DataFrame, label_policy: dict[str, Any]) -> dict[str, Any]:
    corpus_policy = dict(label_policy.get("corpora") or {})
    findings: dict[str, Any] = {}
    observed_corpora = set(records[SPEECH_CORPUS_COLUMN].astype(str))
    unknown = sorted(observed_corpora - set(corpus_policy))
    if unknown:
        raise ValueError(f"Speech data contains corpora missing from label policy: {unknown}")
    for corpus in sorted(observed_corpora):
        observed = sorted(records.loc[records[SPEECH_CORPUS_COLUMN].astype(str) == corpus, SPEECH_TARGET_COLUMN].astype(str).unique())
        allowed = sorted(str(label) for label in corpus_policy[corpus].get("canonical_labels", []))
        unexpected = sorted(set(observed) - set(allowed))
        if unexpected:
            raise ValueError(f"{corpus} has labels not allowed by policy: {unexpected}")
        findings[corpus] = {"observed_labels": observed, "policy_labels": allowed, "missing_policy_labels": sorted(set(allowed) - set(observed))}
    return findings


def select_corpus_records(records: pd.DataFrame, corpora: Iterable[str]) -> pd.DataFrame:
    wanted = {str(corpus) for corpus in corpora}
    return records.loc[records[SPEECH_CORPUS_COLUMN].astype(str).isin(wanted)].sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)


def select_shared_label_records(records: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    wanted = {str(label) for label in labels}
    return records.loc[records[SPEECH_TARGET_COLUMN].astype(str).isin(wanted)].sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)


def validate_speaker_keys(records: pd.DataFrame) -> None:
    if SPEECH_SPEAKER_KEY_COLUMN not in records.columns:
        raise ValueError("Speech records missing safe speaker grouping key")
    if records[SPEECH_SPEAKER_KEY_COLUMN].isna().any() or (records[SPEECH_SPEAKER_KEY_COLUMN].astype(str).str.strip() == "").any():
        raise ValueError("Speech records contain blank safe speaker grouping keys")


def validate_no_speaker_overlap(*splits: pd.DataFrame) -> None:
    speaker_sets = [set(split[SPEECH_SPEAKER_KEY_COLUMN].astype(str)) for split in splits]
    for index, left in enumerate(speaker_sets):
        for right in speaker_sets[index + 1 :]:
            if left & right:
                raise ValueError("speaker overlap detected across domain evaluation splits")


def validate_no_duplicate_audio_overlap(*splits: pd.DataFrame) -> None:
    if not all("original_audio_hash" in split.columns for split in splits):
        return
    hash_sets = [set(split["original_audio_hash"].dropna().astype(str)) for split in splits]
    for index, left in enumerate(hash_sets):
        for right in hash_sets[index + 1 :]:
            if left & right:
                raise ValueError("duplicate audio hash overlap detected across domain evaluation splits")


def load_source_fingerprints(fingerprint_dir: str | Path) -> dict[str, str]:
    root = resolve_project_path(fingerprint_dir)
    if not root.exists():
        raise FileNotFoundError(f"Speech fingerprint directory not found: {root}")
    output: dict[str, str] = {}
    for corpus in CORPORA:
        matches = sorted(root.glob(f"{corpus.lower()}*.json"))
        if not matches:
            raise FileNotFoundError(f"missing source fingerprint for {corpus}")
        payload = load_json(matches[0])
        output[corpus] = str(payload.get("combined_sha256") or payload.get("sha256") or "")
        if not output[corpus]:
            raise ValueError(f"source fingerprint for {corpus} has no SHA-256 value")
    return output


def build_domain_evaluation_bundle(
    *,
    features_path: str | Path,
    canonical_manifest_path: str | Path,
    feature_schema_path: str | Path,
    label_policy_path: str | Path,
    fingerprint_dir: str | Path,
    max_records_per_corpus: int | None = None,
) -> SpeechDomainBundle:
    feature_df = load_speech_features(features_path)
    canonical = load_canonical_speech_metadata(canonical_manifest_path)
    schema = load_json(feature_schema_path)
    label_policy = load_json(label_policy_path)
    features = _feature_names_from_schema(schema)
    metadata = canonical.drop(columns=[col for col in canonical.columns if col != SPEECH_RECORD_ID_COLUMN and col in feature_df.columns])
    records = feature_df.merge(
        metadata,
        on=SPEECH_RECORD_ID_COLUMN,
        how="left",
    )
    if max_records_per_corpus is not None:
        if max_records_per_corpus <= 0:
            raise ValueError("max_records_per_corpus must be positive")
        records = (
            records.sort_values(SPEECH_RECORD_ID_COLUMN)
            .groupby([SPEECH_CORPUS_COLUMN, SPEECH_TARGET_COLUMN], group_keys=False)
            .apply(lambda group: group.head(max(1, max_records_per_corpus // max(1, len(group[SPEECH_TARGET_COLUMN].unique())))))
            .groupby(SPEECH_CORPUS_COLUMN, group_keys=False)
            .head(max_records_per_corpus)
            .reset_index(drop=True)
        )
    validate_feature_coverage(records, features)
    validate_corpus_labels(records, label_policy)
    validate_speaker_keys(records)
    return SpeechDomainBundle(
        records=records.sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True),
        features=features,
        feature_schema=schema,
        label_policy=label_policy,
        feature_file_hash=hashing.sha256_file(features_path),
        source_fingerprints=load_source_fingerprints(fingerprint_dir),
    )
