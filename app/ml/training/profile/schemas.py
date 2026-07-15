"""Lightweight typed containers for the Profile baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProfileSplitManifest:
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
class ProfileTrainingBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    target: str
    split_manifest: ProfileSplitManifest
    source_fingerprint: str
    preprocessing_artifact_hash: str


@dataclass(frozen=True)
class ProfilePreprocessorResult:
    preprocessor: Any
    feature_names: list[str]
    feature_count: int
    numeric_features: list[str]
    categorical_features: list[str]
    constant_features: list[str]
    all_null_features: list[str]


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
    threshold_strategy: str
    threshold: float
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    threshold_selection: dict[str, Any]
    overfitting_gap: dict[str, Any]
    selected_score: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProfileRunArtifacts:
    run_id: str
    report_dir: Path
    run_dir: Path | None
    selected_candidate: CandidateResult | None
    metrics: dict[str, Any]
    artifact_manifest: dict[str, Any] | None
    registered: bool = False
