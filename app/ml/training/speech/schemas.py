"""Typed containers for Speech baseline training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SpeechSplitManifest:
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    source_fingerprint: str
    preprocessing_artifact_hash: str
    manifest_hash: str
    payload: dict[str, Any]

    @property
    def all_ids(self) -> list[str]:
        return [*self.train_ids, *self.validation_ids, *self.test_ids]


@dataclass(frozen=True)
class SpeechTrainingBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    target: str
    split_manifest: SpeechSplitManifest
    source_fingerprint: str
    preprocessing_artifact_hash: str
    feature_schema: dict[str, Any]
    preprocessing_report: dict[str, Any]
    duplicate_manifest: dict[str, Any]
    corpus_summary: dict[str, Any]
    speaker_isolation_report: dict[str, Any]
    duplicate_isolation_report: dict[str, Any]
    feature_coverage: dict[str, Any]


@dataclass(frozen=True)
class SpeechPreprocessorResult:
    preprocessor: Any
    feature_names: list[str]
    removed_constant_features: list[str]
    missing_value_report: dict[str, Any]
    scale_numeric: bool


@dataclass(frozen=True)
class CandidateSpec:
    estimator_type: str
    name: str
    hyperparameters: dict[str, Any]


@dataclass
class CandidateResult:
    candidate_id: str
    spec: CandidateSpec
    estimator: Any
    preprocessor: Any
    feature_names: list[str]
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    overfitting_gap: dict[str, Any]
    selected_score: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SpeechRunArtifacts:
    run_id: str
    report_dir: Path
    run_dir: Path | None
    selected_candidate: CandidateResult | None
    metrics: dict[str, Any]
    artifact_manifest: dict[str, Any] | None
    registered: bool = False
    skipped_candidates: list[dict[str, Any]] = field(default_factory=list)

