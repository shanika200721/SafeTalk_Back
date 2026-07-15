"""LOCO fold construction for Speech domain-shift evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import hashing, paths
from app.ml.evaluation.speech_domain.constants import (
    CORPORA,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SPEAKER_FRACTION,
    SPEECH_DOMAIN_EVALUATION_VERSION,
    SPEECH_LOCO_POLICY_VERSION,
)
from app.ml.evaluation.speech_domain.data import (
    select_shared_label_records,
    validate_no_duplicate_audio_overlap,
    validate_no_speaker_overlap,
)
from app.ml.evaluation.speech_domain.schemas import CorpusFoldConfig, CorpusFoldData, SpeechDomainBundle
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.speech.constants import (
    SPEECH_CORPUS_COLUMN,
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_SPEAKER_KEY_COLUMN,
    SPEECH_TARGET_COLUMN,
)


def _fold_policy(label_policy: dict[str, Any], held_out: str) -> tuple[list[str], list[str]]:
    fold = ((label_policy.get("fold_rules") or {}).get(f"hold_out_{held_out}") or {})
    included = [str(label) for label in fold.get("included_labels_shared") or label_policy.get("shared_labels_all_corpora") or []]
    excluded = [str(label) for label in fold.get("excluded_labels") or []]
    if not included:
        raise ValueError(f"label policy has no included labels for {held_out}")
    return included, excluded


def _validation_speakers(training: pd.DataFrame, *, fraction: float) -> set[str]:
    validation: set[str] = set()
    for corpus, group in training.groupby(SPEECH_CORPUS_COLUMN, sort=True):
        speakers = sorted(group[SPEECH_SPEAKER_KEY_COLUMN].astype(str).unique())
        if len(speakers) < 2:
            continue
        count = max(1, int(round(len(speakers) * fraction)))
        validation.update(speakers[:: max(1, len(speakers) // count)][:count])
    return validation


def create_loco_folds(
    bundle: SpeechDomainBundle,
    *,
    held_out_corpus: str | None = None,
    shared_labels_only: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    validation_speaker_fraction: float = DEFAULT_VALIDATION_SPEAKER_FRACTION,
) -> list[CorpusFoldData]:
    selected_corpora = [held_out_corpus] if held_out_corpus else CORPORA
    invalid = sorted(set(selected_corpora) - set(CORPORA))
    if invalid:
        raise ValueError(f"unknown held-out corpus: {invalid}")
    folds: list[CorpusFoldData] = []
    for held_out in selected_corpora:
        training_corpora = [corpus for corpus in CORPORA if corpus != held_out]
        included, excluded = _fold_policy(bundle.label_policy, held_out)
        records = select_shared_label_records(bundle.records, included) if shared_labels_only else bundle.records.copy()
        train_validation = records.loc[records[SPEECH_CORPUS_COLUMN].astype(str).isin(training_corpora)].copy()
        test = records.loc[records[SPEECH_CORPUS_COLUMN].astype(str) == held_out].copy()
        if train_validation.empty or test.empty:
            raise ValueError(f"LOCO fold {held_out} has empty train/validation or test records")
        validation_speakers = _validation_speakers(train_validation, fraction=validation_speaker_fraction)
        validation = train_validation.loc[train_validation[SPEECH_SPEAKER_KEY_COLUMN].astype(str).isin(validation_speakers)].copy()
        train = train_validation.loc[~train_validation[SPEECH_SPEAKER_KEY_COLUMN].astype(str).isin(validation_speakers)].copy()
        if train.empty or validation.empty:
            raise ValueError(f"LOCO fold {held_out} cannot create non-empty speaker-isolated train and validation sets")
        train = train.sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)
        validation = validation.sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)
        test = test.sort_values(SPEECH_RECORD_ID_COLUMN).reset_index(drop=True)
        validate_no_speaker_overlap(train, validation, test)
        validate_no_duplicate_audio_overlap(train, validation, test)
        missing_classes = sorted(set(included) - set(test[SPEECH_TARGET_COLUMN].astype(str)))
        config = CorpusFoldConfig(
            fold_name=f"hold_out_{held_out.lower()}",
            training_corpora=training_corpora,
            validation_corpora=training_corpora,
            test_corpus=held_out,
            included_labels=included,
            excluded_labels=excluded,
            random_seed=random_seed,
            speaker_grouping_required=True,
            notes=[
                "Validation speakers are drawn only from training corpora.",
                "Held-out corpus records are test-only.",
                *([f"Missing held-out classes: {', '.join(missing_classes)}"] if missing_classes else []),
            ],
        )
        folds.append(CorpusFoldData(config=config, train=train, validation=validation, test=test))
    return folds


def fold_manifest_payload(fold: CorpusFoldData, *, feature_file_hash: str, source_fingerprints: dict[str, str]) -> dict[str, Any]:
    payload = {
        "manifest_version": SPEECH_DOMAIN_EVALUATION_VERSION,
        "policy_version": SPEECH_LOCO_POLICY_VERSION,
        "fold_name": fold.config.fold_name,
        "training_corpora": fold.config.training_corpora,
        "validation_corpora": fold.config.validation_corpora,
        "test_corpus": fold.config.test_corpus,
        "included_labels": fold.config.included_labels,
        "excluded_labels": fold.config.excluded_labels,
        "random_seed": fold.config.random_seed,
        "speaker_grouping_required": fold.config.speaker_grouping_required,
        "train_ids": fold.train[SPEECH_RECORD_ID_COLUMN].astype(str).tolist(),
        "validation_ids": fold.validation[SPEECH_RECORD_ID_COLUMN].astype(str).tolist(),
        "test_ids": fold.test[SPEECH_RECORD_ID_COLUMN].astype(str).tolist(),
        "train_count": int(len(fold.train)),
        "validation_count": int(len(fold.validation)),
        "test_count": int(len(fold.test)),
        "feature_file_hash": feature_file_hash,
        "source_fingerprints": source_fingerprints,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": fold.config.notes,
        "privacy": {
            "raw_speaker_ids_included": False,
            "raw_filenames_included": False,
            "audio_content_included": False
        },
    }
    payload["manifest_hash"] = hashing.hash_json_data({key: value for key, value in payload.items() if key != "manifest_hash"})
    return payload


def save_loco_manifests(
    folds: list[CorpusFoldData],
    *,
    output_dir: str | Path,
    feature_file_hash: str,
    source_fingerprints: dict[str, str],
    overwrite: bool = False,
) -> dict[str, Path]:
    root = Path(output_dir)
    if not root.is_absolute():
        root = paths.get_repository_root() / root
    root = root.resolve(strict=False)
    prevent_overwrite(root / "speech_loco_manifest_summary.json", overwrite=overwrite)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "manifest_version": SPEECH_DOMAIN_EVALUATION_VERSION,
        "policy_version": SPEECH_LOCO_POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds": [],
    }
    for fold in folds:
        payload = fold_manifest_payload(fold, feature_file_hash=feature_file_hash, source_fingerprints=source_fingerprints)
        name = f"speech_loco_{fold.config.test_corpus.lower()}_manifest.json"
        path = root / name
        prevent_overwrite(path, overwrite=overwrite)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs[fold.config.fold_name] = path
        summary["folds"].append(
            {
                "fold_name": fold.config.fold_name,
                "path": path.relative_to(paths.get_repository_root()).as_posix(),
                "train_count": payload["train_count"],
                "validation_count": payload["validation_count"],
                "test_count": payload["test_count"],
                "manifest_hash": payload["manifest_hash"],
            }
        )
    summary_path = root / "speech_loco_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs

