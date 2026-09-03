"""Typed containers for Speech domain-shift evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


class CorpusEvaluationStrategy(str, Enum):
    leave_one_corpus_out = "leave_one_corpus_out"
    train_on_single_test_on_other = "train_on_single_test_on_other"
    pooled_within_corpus = "pooled_within_corpus"
    corpus_transfer_matrix = "corpus_transfer_matrix"


@dataclass(frozen=True)
class SpeechDomainBundle:
    records: pd.DataFrame
    features: list[str]
    feature_schema: dict[str, Any]
    label_policy: dict[str, Any]
    feature_file_hash: str
    source_fingerprints: dict[str, str]


@dataclass(frozen=True)
class CorpusFoldConfig:
    fold_name: str
    training_corpora: list[str]
    validation_corpora: list[str]
    test_corpus: str
    included_labels: list[str]
    excluded_labels: list[str]
    random_seed: int
    speaker_grouping_required: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CorpusFoldData:
    config: CorpusFoldConfig
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class CorpusFoldResult:
    fold_name: str
    training_corpora: list[str]
    validation_corpora: list[str]
    test_corpus: str
    train_count: int
    validation_count: int
    test_count: int
    feature_count: int
    candidate_model: str | None
    selected_hyperparameters: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    per_class_metrics: dict[str, Any]
    confusion_matrix: dict[str, Any]
    unsupported_labels: list[str]
    missing_classes: list[str]
    warnings: list[str]
    runtime_seconds: float
    artifact_paths: list[str] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class CorpusTransferResult:
    train_corpus: str
    test_corpus: str
    shared_labels: list[str]
    train_count: int
    test_count: int
    metrics: dict[str, Any]
    per_class_metrics: dict[str, Any]
    warnings: list[str]


@dataclass
class SpeechDomainShiftReport:
    evaluation_version: str
    policy_version: str
    generated_at: datetime
    folds: list[CorpusFoldResult]
    transfer_matrix: list[CorpusTransferResult]
    pooled_baseline_reference: dict[str, Any]
    cross_corpus_summary: dict[str, Any]
    corpus_gap_summary: dict[str, Any]
    shortcut_risk_findings: dict[str, Any]
    blockers: list[str]
    recommendations: list[str]
    research_readiness: str
    report_dir: Path | None = None

