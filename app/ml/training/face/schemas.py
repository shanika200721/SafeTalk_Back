"""Typed containers for Phase 3I face training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaceTrainingBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    source_fingerprint: str
    split_manifest_hash: str
    deduplicated_manifest_hash: str
    split_manifest: dict[str, Any]
    contract: dict[str, Any]


@dataclass(frozen=True)
class FaceImageBundle:
    X: np.ndarray
    y: list[str]
    rows: pd.DataFrame
    feature_names: list[str]


@dataclass(frozen=True)
class FaceCandidateSpec:
    estimator_type: str
    name: str
    hyperparameters: dict[str, Any]
    feature_set: str = "flattened_pixels"
    scale_features: bool = True


@dataclass
class FaceCandidateResult:
    candidate_id: str
    spec: FaceCandidateSpec
    estimator: Any
    preprocessor: Any
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    selected_score: tuple[Any, ...]
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class FaceRunArtifacts:
    run_id: str
    report_dir: Path
    run_dir: Path
    selected_candidate: FaceCandidateResult | None
    metrics: dict[str, Any]
    artifact_manifest: dict[str, Any]
    registered: bool = False

