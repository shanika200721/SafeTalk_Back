"""Lightweight typed containers for the Text baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TextSplitManifest:
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    excluded_ids: dict[str, str]
    source_fingerprint: str
    preprocessing_artifact_hash: str
    manifest_hash: str
    payload: dict[str, Any]

    @property
    def all_ids(self) -> list[str]:
        return [*self.train_ids, *self.validation_ids, *self.test_ids]


@dataclass(frozen=True)
class TextTrainingBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    text_column: str
    target: str
    split_manifest: TextSplitManifest
    source_fingerprint: str
    preprocessing_artifact_hash: str
    duplicate_manifest: dict[str, Any]
    source_overlap_report: dict[str, Any]


@dataclass(frozen=True)
class TextVectorizerResult:
    vectorizer: Any
    feature_names: list[str]
    feature_count: int
    vocabulary_hash: str
    vectorizer_config: dict[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    estimator_type: str
    name: str
    hyperparameters: dict[str, Any]


@dataclass
class CandidateResult:
    candidate_id: str
    spec: CandidateSpec
    vectorizer_name: str
    vectorizer_config: dict[str, Any]
    estimator: Any
    vectorizer: Any
    feature_names: list[str]
    vocabulary_hash: str
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    overfitting_gap: dict[str, Any]
    selected_score: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TextRunArtifacts:
    run_id: str
    report_dir: Path
    run_dir: Path | None
    selected_candidate: CandidateResult | None
    metrics: dict[str, Any]
    artifact_manifest: dict[str, Any] | None
    registered: bool = False
    skipped_candidates: list[dict[str, Any]] = field(default_factory=list)

