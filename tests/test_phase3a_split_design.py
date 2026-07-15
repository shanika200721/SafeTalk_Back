from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.splitting.common import (
    calculate_split_targets,
    compute_split_artifact_hash,
    deterministic_shuffle,
    grouped_stratified_split,
    stratified_split,
    validate_duplicate_isolation,
    validate_group_isolation,
    validate_manifest_coverage,
    validate_no_overlap,
)
from app.ml.splitting.constants import SPLIT_DESIGN_VERSION, SPLIT_MANIFEST_VERSION
from app.ml.splitting.profile import create_profile_split
from app.ml.splitting.speech import create_speech_split
from app.ml.splitting.text import create_text_split
from app.ml.splitting.validation import validate_no_forbidden_modalities


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def minimal_schema(path: Path, modality: str, label: str) -> Path:
    return write_json(
        path,
        {
            "dataset_name": f"{modality}-fixture",
            "dataset_version": "v1",
            "preprocessing_version": "1.0.0",
            "feature_schema_version": "1.0.0",
            "target_columns": [label],
        },
    )


def minimal_report(path: Path) -> Path:
    return write_json(
        path,
        {
            "source_fingerprint": "0" * 64,
            "preprocessing_version": "1.0.0",
            "feature_schema_version": "1.0.0",
            "source_fingerprints": {"fixture": "0" * 64},
        },
    )


def split_config(path: Path, *, stratify: str, group: str | None = None) -> Path:
    return write_json(
        path,
        {
            "split_version": "1.0.0",
            "dataset_version": "v1",
            "preprocessing_version": "1.0.0",
            "feature_schema_version": "1.0.0",
            "random_seed": 7,
            "train_proportion": 0.6,
            "validation_proportion": 0.2,
            "test_proportion": 0.2,
            "stratify_column": stratify,
            "grouping_column": group,
            "duplicate_policy": "isolate",
            "conflict_policy": "exclude",
            "minimum_class_count_per_split": 1,
            "retry_limit": 20,
            "deterministic_tie_break_rule": "sha256",
            "notes": [],
        },
    )


def test_common_split_logic_targets_determinism_overlap_coverage_and_hash():
    assert calculate_split_targets(10, 0.6, 0.2, 0.2) == {"train": 6, "validation": 2, "test": 2}
    with pytest.raises(ValueError):
        calculate_split_targets(10, 0.8, 0.2, 0.2)

    values = [f"id-{index}" for index in range(20)]
    assert deterministic_shuffle(values, 42) == deterministic_shuffle(reversed(values), 42)
    assert deterministic_shuffle(values, 42) != deterministic_shuffle(values, 43)

    df = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(30)],
            "label": ["a"] * 15 + ["b"] * 15,
        }
    )
    first = stratified_split(
        df,
        record_id_column="record_id",
        label_column="label",
        train_proportion=0.6,
        validation_proportion=0.2,
        test_proportion=0.2,
        seed=1,
    )
    second = stratified_split(
        df,
        record_id_column="record_id",
        label_column="label",
        train_proportion=0.6,
        validation_proportion=0.2,
        test_proportion=0.2,
        seed=1,
    )
    changed = stratified_split(
        df,
        record_id_column="record_id",
        label_column="label",
        train_proportion=0.6,
        validation_proportion=0.2,
        test_proportion=0.2,
        seed=2,
    )
    assert [(item.record_id, item.split) for item in first] == [(item.record_id, item.split) for item in second]
    assert [(item.record_id, item.split) for item in first] != [(item.record_id, item.split) for item in changed]
    validate_no_overlap(first)
    validate_manifest_coverage(first, df["record_id"])
    assert compute_split_artifact_hash([item.dict() for item in first]) == compute_split_artifact_hash([item.dict() for item in second])
    with pytest.raises(ValueError):
        validate_manifest_coverage(first[:-1], df["record_id"])


def test_grouped_logic_keeps_groups_and_duplicates_together():
    rows = []
    for group in range(12):
        for label in ["a", "b"]:
            rows.append(
                {
                    "record_id": f"r-{group}-{label}",
                    "label": label,
                    "group": f"g-{group}",
                    "dup": "dup-1" if group in {0, 1} else None,
                }
            )
    assignments = grouped_stratified_split(
        pd.DataFrame(rows),
        record_id_column="record_id",
        label_column="label",
        group_column="group",
        duplicate_column="dup",
        train_proportion=0.6,
        validation_proportion=0.2,
        test_proportion=0.2,
        seed=3,
    )
    validate_group_isolation(assignments)
    validate_duplicate_isolation(assignments)


def test_profile_split_fixture_has_both_classes_and_no_sensitive_stratification(tmp_path):
    df = pd.DataFrame(
        {
            "record_id": [f"p{i}" for i in range(30)],
            "target_depression": ["yes"] * 12 + ["no"] * 18,
            "gender": ["x"] * 30,
        }
    )
    input_path = tmp_path / "profile.csv"
    df.to_csv(input_path, index=False)
    result = create_profile_split(
        input_path=input_path,
        config_path=split_config(tmp_path / "config.json", stratify="target_depression"),
        preprocessing_report_path=minimal_report(tmp_path / "report.json"),
        feature_schema_path=minimal_schema(tmp_path / "schema.json", "profile", "target_depression"),
        output_dir=tmp_path / "out",
        overwrite=True,
    )
    assert result["manifest"].manifest_version == SPLIT_MANIFEST_VERSION
    assert result["manifest"].split_design_version == SPLIT_DESIGN_VERSION
    for distribution in [
        result["manifest"].validation_summary.train_distribution,
        result["manifest"].validation_summary.validation_distribution,
        result["manifest"].validation_summary.test_distribution,
    ]:
        assert set(distribution) == {"no", "yes"}
    assert result["manifest"].stratify_column == "target_depression"
    assert "gender" not in json.dumps(result["report"].to_safe_dict()).lower()


def test_text_split_excludes_conflicts_isolates_hashes_and_omits_raw_text(tmp_path):
    rows = []
    labels = ["anxiety", "depression", "normal", "suicidal"]
    for index in range(40):
        rows.append(
                {
                    "record_id": f"t{index}",
                    "normalized_text": f"private text {index}",
                    "canonical_label": labels[index % 4],
                    "source_name": "source.csv",
                    "text_hash": "h-dup-anxiety" if index in {0, 4} else f"h{index}",
                }
            )
    input_path = tmp_path / "text.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    duplicate_manifest = write_json(
        tmp_path / "dups.json",
        {"exact_duplicate_groups": [{"duplicate_hash": "h-dup-anxiety", "record_ids": ["t0", "t4"], "conflict": False}]},
    )
    conflict_path = tmp_path / "conflict.csv"
    pd.DataFrame({"record_id": ["t2"], "reason": ["conflict"]}).to_csv(conflict_path, index=False)
    overlap_path = write_json(tmp_path / "overlap.json", {"exact_overlap_count": 2})
    result = create_text_split(
        input_path=input_path,
        config_path=split_config(tmp_path / "config.json", stratify="canonical_label", group="text_hash"),
        preprocessing_report_path=minimal_report(tmp_path / "report.json"),
        feature_schema_path=minimal_schema(tmp_path / "schema.json", "text", "canonical_label"),
        duplicate_manifest_path=duplicate_manifest,
        conflict_manifest_path=conflict_path,
        source_overlap_report_path=overlap_path,
        output_dir=tmp_path / "out",
        overwrite=True,
    )
    assert result["exclusions"] == {"t2": "conflicting_duplicate_quarantine"}
    validate_group_isolation(result["assignments"])
    validate_duplicate_isolation(result["assignments"])
    assert set(result["manifest"].validation_summary.train_distribution) == set(labels)
    report_text = (tmp_path / "out" / "text_split_report.json").read_text(encoding="utf-8")
    assert "private text" not in report_text
    assert result["reference_policy"]["record_level_overlap_ids_available"] is False


def test_speech_split_is_speaker_safe_reports_corpus_limits_and_omits_filenames(tmp_path):
    rows = []
    labels = ["angry", "happy", "sad"]
    for speaker in range(12):
        corpus = "TESS" if speaker < 2 else ("SAVEE" if speaker < 6 else "CREMA")
        for label in labels:
            rows.append(
                {
                    "record_id": f"s{speaker}-{label}",
                    "safe_speaker_key": f"safe-speaker-{speaker}",
                    "corpus_name": corpus,
                    "canonical_emotion_label": label,
                    "audio_relative_path": f"raw-speaker-{speaker}.wav",
                    "original_audio_hash": f"audio-{speaker}-{label}",
                }
            )
    input_path = tmp_path / "speech.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    duplicate_manifest = write_json(
        tmp_path / "dups.json",
        {
            "duplicate_audio_hash_group_count": 1,
            "cross_corpus_duplicate_audio_hash_group_count": 0,
            "duplicate_audio_hash_groups": {"dup-a": ["s0-angry", "s0-happy"]},
        },
    )
    result = create_speech_split(
        input_path=input_path,
        config_path=split_config(tmp_path / "config.json", stratify="canonical_emotion_label", group="safe_speaker_key"),
        preprocessing_report_path=minimal_report(tmp_path / "report.json"),
        feature_schema_path=minimal_schema(tmp_path / "schema.json", "speech", "canonical_emotion_label"),
        duplicate_manifest_path=duplicate_manifest,
        output_dir=tmp_path / "out",
        overwrite=True,
    )
    validate_group_isolation(result["assignments"])
    validate_duplicate_isolation(result["assignments"])
    assert "TESS has only 2 speakers" in "\n".join(result["report"].limitations)
    report_text = (tmp_path / "out" / "speech_split_report.json").read_text(encoding="utf-8")
    assert "raw-speaker" not in report_text
    assert result["corpus_distribution"]


def test_forbidden_modalities_are_rejected():
    for modality in ["mood", "face", "behavioral", "DASS21", "fusion"]:
        with pytest.raises(ValueError):
            validate_no_forbidden_modalities(modality)


def run_cli(*args: str, timeout: int = 120):
    script = paths.get_backend_root() / "scripts" / "create_phase3a_splits.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def test_cli_profile_success_replay_validate_only_overwrite_and_hash_failures(tmp_path):
    output = tmp_path / "profile-out"
    success = run_cli("--modality", "profile", "--output-dir", str(output), "--overwrite", "--replay")
    assert success.returncode == 0, success.stderr
    assert "replay=passed" in success.stdout

    overwrite = run_cli("--modality", "profile", "--output-dir", str(output))
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr.lower()

    validate_only_dir = tmp_path / "validate-only"
    validate_only = run_cli("--modality", "profile", "--output-dir", str(validate_only_dir), "--validate-only")
    assert validate_only.returncode == 0, validate_only.stderr
    assert not validate_only_dir.exists()

    bad_fingerprint = write_json(tmp_path / "bad-fingerprint.json", {"combined_sha256": "1" * 64})
    mismatch = run_cli("--modality", "profile", "--source-fingerprint", str(bad_fingerprint), "--strict")
    assert mismatch.returncode == 1
    assert "fingerprint" in mismatch.stderr.lower()

    config = json.loads((paths.get_repository_root() / "ml-research/configs/profile.split.v1.json").read_text(encoding="utf-8"))
    config["expected_preprocessing_artifact_hash"] = "2" * 64
    bad_config = write_json(tmp_path / "bad-config.json", config)
    bad_hash = run_cli("--modality", "profile", "--config", str(bad_config), "--validate-only")
    assert bad_hash.returncode == 2
    assert "preprocessing hash" in bad_hash.stderr.lower()


def test_cli_text_speech_and_aggregate_validation(tmp_path):
    text_out = tmp_path / "text-out"
    speech_out = tmp_path / "speech-out"
    text = run_cli("--modality", "text", "--output-dir", str(text_out), "--overwrite", "--replay")
    assert text.returncode == 0, text.stderr
    speech = run_cli("--modality", "speech", "--output-dir", str(speech_out), "--overwrite", "--replay")
    assert speech.returncode == 0, speech.stderr

    validator = paths.get_backend_root() / "scripts" / "validate_phase3a_splits.py"
    aggregate = subprocess.run(
        [sys.executable, str(validator), "--output-dir", str(tmp_path / "reports"), "--overwrite"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert aggregate.returncode in {0, 1}
    assert (tmp_path / "reports" / "phase3a_split_validation.json").exists()
