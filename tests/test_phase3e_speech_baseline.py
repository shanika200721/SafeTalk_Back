import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.common import hashing, paths
from app.ml.training.speech.constants import FEATURE_SETS, REQUIRED_MODEL_CARD_DISCLAIMER, SPEECH_LABELS
from app.ml.training.speech.data import build_speech_training_bundle
from app.ml.training.speech.estimators import logistic_regression_candidate_specs, random_forest_candidate_specs, svm_candidate_specs
from app.ml.training.speech.evaluation import corpus_metrics, evaluate_speech_split, feature_interpretation
from app.ml.training.speech.preprocessing import build_speech_preprocessor, transform_speech_features
from app.ml.training.speech.runner import dry_run_speech_baseline, run_speech_baseline


def _temp_root() -> Path:
    root = paths.get_generated_root() / "temporary" / f"phase3e-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _fixture(root: Path):
    feature_columns = FEATURE_SETS["full_acoustic"]
    rows = []
    split_ids = {"train": [], "validation": [], "test": []}
    idx = 1
    for label_index, label in enumerate(SPEECH_LABELS):
        for split, count in {"train": 3, "validation": 1, "test": 1}.items():
            for item in range(count):
                record_id = f"speech-fixture-{idx:03d}"
                speaker = f"speech-v1-spk-{split}-{label}-{item}"
                corpus = "CREMA" if label_index % 2 == 0 else "RAVDESS"
                base = float(label_index + 1)
                values = {feature: base + (feature_pos / 100.0) + (item / 1000.0) for feature_pos, feature in enumerate(feature_columns)}
                rows.append(
                    {
                        "record_id": record_id,
                        "safe_speaker_key": speaker,
                        "corpus_name": corpus,
                        "canonical_emotion_label": label,
                        "audio_relative_path": f"Final Dataset/Speech Emotion/Fake/{record_id}.wav",
                        "original_audio_hash": hashing.hash_text(record_id),
                        "duration_seconds": values["duration_seconds"],
                        "sample_rate": 16000,
                        "channel_count": 1,
                        "sample_width": 2,
                        "frame_count": 16000,
                        "file_format": "wav",
                        "file_size_bytes": 100,
                        "readable": True,
                        "validation_warnings": "",
                        **values,
                        "feature_extraction_warnings": "",
                    }
                )
                split_ids[split].append(record_id)
                idx += 1
    all_df = pd.DataFrame(rows)
    canonical_cols = [
        "audio_relative_path",
        "canonical_emotion_label",
        "channel_count",
        "corpus_name",
        "duration_seconds",
        "file_format",
        "file_size_bytes",
        "frame_count",
        "original_audio_hash",
        "readable",
        "record_id",
        "safe_speaker_key",
        "sample_rate",
        "sample_width",
        "validation_warnings",
    ]
    canonical = root / "speech_canonical_manifest.csv"
    all_df[canonical_cols].to_csv(canonical, index=False)
    features = root / "speech_features.csv"
    all_df[["record_id", "safe_speaker_key", "corpus_name", "canonical_emotion_label", *feature_columns, "feature_extraction_warnings"]].to_csv(features, index=False)
    canonical_hash = hashing.sha256_file(canonical)
    source_fingerprints = {"CREMA": "a" * 64, "RAVDESS": "b" * 64, "SAVEE": "c" * 64, "TESS": "d" * 64}
    split_payload = {
        "train_ids": split_ids["train"],
        "validation_ids": split_ids["validation"],
        "test_ids": split_ids["test"],
        "source_fingerprint": hashing.hash_json_data(source_fingerprints),
        "preprocessing_artifact_hash": canonical_hash,
        "validation_summary": {"train_count": len(split_ids["train"]), "validation_count": len(split_ids["validation"]), "test_count": len(split_ids["test"])},
    }
    split = root / "speech_split_manifest.json"
    split.write_text(json.dumps(split_payload), encoding="utf-8")
    schema = root / "speech_feature_schema.json"
    schema.write_text(json.dumps({"features": [{"name": feature, "dtype": "float"} for feature in feature_columns], "target_label": "canonical_emotion_label"}), encoding="utf-8")
    report = root / "speech_preprocessing_report.json"
    report.write_text(json.dumps({"source_fingerprints": source_fingerprints, "output_record_count": len(all_df)}), encoding="utf-8")
    fingerprint_dir = root / "fingerprints"
    fingerprint_dir.mkdir()
    for corpus, digest in source_fingerprints.items():
        (fingerprint_dir / f"{corpus.lower()}-v1.json").write_text(json.dumps({"combined_sha256": digest}), encoding="utf-8")
    duplicate = root / "speech_duplicate_manifest.json"
    duplicate.write_text(json.dumps({"duplicate_audio_hash_groups": {}, "duplicate_audio_hash_group_count": 0}), encoding="utf-8")
    corpus_summary = root / "speech_corpus_summary.json"
    corpus_summary.write_text(json.dumps({"CREMA": {"record_count": 20}, "RAVDESS": {"record_count": 20}}), encoding="utf-8")
    assignments = root / "speech_split_assignments.csv"
    pd.DataFrame([{"record_id": record_id, "split": split} for split, ids in split_ids.items() for record_id in ids]).to_csv(assignments, index=False)
    speaker_report = root / "speech_speaker_isolation_report.json"
    speaker_report.write_text(json.dumps({"speaker_overlap_count": 0}), encoding="utf-8")
    dup_report = root / "speech_duplicate_isolation_report.json"
    dup_report.write_text(json.dumps({"duplicate_overlap_count": 0}), encoding="utf-8")
    config = root / "training_config.json"
    config.write_text(
        json.dumps(
            {
                "locked_split_manifest_hash": hashing.sha256_file(split),
                "feature_set": "full_acoustic",
                "max_candidate_count": 4,
                "hyperparameter_search": {
                    "logistic_regression": {"C": [1.0], "class_weight": [None], "max_iter": 100, "random_state": 42},
                    "random_forest": {"n_estimators": [20], "max_depth": [4], "min_samples_leaf": [1], "class_weight": [None], "random_state": 42, "n_jobs": 1},
                    "svm": {"kernel": ["linear"], "C": [1.0], "class_weight": [None], "max_iter": 1000, "random_state": 42},
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "features": features,
        "canonical": canonical,
        "split": split,
        "schema": schema,
        "report": report,
        "fingerprint_dir": fingerprint_dir,
        "duplicate": duplicate,
        "corpus_summary": corpus_summary,
        "assignments": assignments,
        "speaker_report": speaker_report,
        "duplicate_report": dup_report,
        "config": config,
    }


def _bundle(paths_):
    return build_speech_training_bundle(
        features_path=paths_["features"],
        canonical_manifest_path=paths_["canonical"],
        split_manifest_path=paths_["split"],
        feature_schema_path=paths_["schema"],
        preprocessing_report_path=paths_["report"],
        fingerprint_dir=paths_["fingerprint_dir"],
        duplicate_manifest_path=paths_["duplicate"],
        corpus_summary_path=paths_["corpus_summary"],
        split_assignments_path=paths_["assignments"],
        speaker_isolation_report_path=paths_["speaker_report"],
        duplicate_isolation_report_path=paths_["duplicate_report"],
    )


def test_speech_data_contract_locked_splits_fingerprints_and_leakage_rejections():
    root = _temp_root()
    fixture = _fixture(root)
    bundle = _bundle(fixture)
    assert bundle.train["record_id"].tolist() == json.loads(fixture["split"].read_text())["train_ids"]
    assert bundle.feature_coverage["complete"] is True

    bad_features = pd.read_csv(fixture["features"])
    bad_features = bad_features.drop(columns=["mfcc_13_std"])
    bad_path = root / "bad_features.csv"
    bad_features.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        build_speech_training_bundle(
            features_path=bad_path,
            canonical_manifest_path=fixture["canonical"],
            split_manifest_path=fixture["split"],
            feature_schema_path=fixture["schema"],
            preprocessing_report_path=fixture["report"],
            fingerprint_dir=fixture["fingerprint_dir"],
            duplicate_manifest_path=fixture["duplicate"],
            corpus_summary_path=fixture["corpus_summary"],
            split_assignments_path=fixture["assignments"],
            speaker_isolation_report_path=fixture["speaker_report"],
            duplicate_isolation_report_path=fixture["duplicate_report"],
        )

    bad_report = root / "bad_speaker_report.json"
    bad_report.write_text(json.dumps({"speaker_overlap_count": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="speaker"):
        build_speech_training_bundle(
            features_path=fixture["features"],
            canonical_manifest_path=fixture["canonical"],
            split_manifest_path=fixture["split"],
            feature_schema_path=fixture["schema"],
            preprocessing_report_path=fixture["report"],
            fingerprint_dir=fixture["fingerprint_dir"],
            duplicate_manifest_path=fixture["duplicate"],
            corpus_summary_path=fixture["corpus_summary"],
            split_assignments_path=fixture["assignments"],
            speaker_isolation_report_path=bad_report,
            duplicate_isolation_report_path=fixture["duplicate_report"],
        )

    with pytest.raises(ValueError, match="prohibited"):
        build_speech_training_bundle(
            features_path=fixture["features"],
            canonical_manifest_path=fixture["canonical"],
            split_manifest_path=fixture["split"],
            feature_schema_path=fixture["schema"],
            preprocessing_report_path=fixture["report"],
            fingerprint_dir=fixture["fingerprint_dir"],
            duplicate_manifest_path=fixture["duplicate"],
            corpus_summary_path=fixture["corpus_summary"],
            split_assignments_path=fixture["assignments"],
            speaker_isolation_report_path=fixture["speaker_report"],
            duplicate_isolation_report_path=fixture["duplicate_report"],
            features=["corpus_name"],
        )


def test_speech_preprocessing_is_train_only_deterministic_and_finite():
    root = _temp_root()
    fixture = _fixture(root)
    bundle = _bundle(fixture)
    train = bundle.train.copy()
    train.loc[train.index[0], "pitch_mean"] = np.nan
    train["constant_feature"] = 1.0
    features = [*bundle.features, "constant_feature"]
    prep = build_speech_preprocessor(train, features, estimator_type="logistic_regression")
    assert "constant_feature" in prep.removed_constant_features
    X_val_1 = transform_speech_features(prep.preprocessor, bundle.validation.assign(constant_feature=999.0), features)
    X_val_2 = transform_speech_features(prep.preprocessor, bundle.validation.assign(constant_feature=-999.0), features)
    assert X_val_1.shape[1] == len(prep.feature_names)
    assert np.isfinite(X_val_1).all()
    assert np.allclose(X_val_1, X_val_2)


def test_speech_estimators_metrics_interpretation_and_corpus_slices():
    assert len(logistic_regression_candidate_specs({"C": [0.1, 1.0], "class_weight": [None, "balanced"]})) == 4
    assert len(random_forest_candidate_specs({"n_estimators": [10], "max_depth": [2], "min_samples_leaf": [1], "class_weight": [None, "balanced"]})) == 2
    assert len(svm_candidate_specs({"kernel": ["linear"], "C": [1.0], "class_weight": [None, "balanced"]})) == 2
    y_true = ["sad", "sad", "calm", "neutral", "happy", "fearful", "angry", "disgust", "surprised"]
    y_pred = ["happy", "sad", "neutral", "neutral", "happy", "sad", "angry", "disgust", "happy"]
    metrics = evaluate_speech_split(y_true, y_pred, split_name="validation")
    assert metrics["macro_f1"] is not None
    assert metrics["per_class"]["sad"]["false_negatives"] == 1
    assert metrics["class_focus"]["calm"]["recall"] == 0.0
    df = pd.DataFrame({"corpus_name": ["CREMA"] * len(y_true)})
    slices = corpus_metrics(df, y_true, y_pred, min_support=2)
    assert "CREMA" in slices["by_corpus"]
    class Dummy:
        coef_ = np.array([[1.0, -2.0]])
        classes_ = np.array(["sad"])
    rows = feature_interpretation(Dummy(), ["pitch_mean", "mfcc_01_mean"])
    assert "causal" in rows[0]["warning"]


def test_speech_runner_dry_run_artifacts_model_card_and_overwrite_protection():
    root = _temp_root()
    fixture = _fixture(root)
    dry = dry_run_speech_baseline(
        config_path=fixture["config"],
        features_path=fixture["features"],
        canonical_manifest_path=fixture["canonical"],
        split_manifest_path=fixture["split"],
        feature_schema_path=fixture["schema"],
        preprocessing_report_path=fixture["report"],
        fingerprint_dir=fixture["fingerprint_dir"],
    )
    assert dry["status"] == "dry_run_ok"
    result = run_speech_baseline(
        config_path=fixture["config"],
        features_path=fixture["features"],
        canonical_manifest_path=fixture["canonical"],
        split_manifest_path=fixture["split"],
        feature_schema_path=fixture["schema"],
        preprocessing_report_path=fixture["report"],
        fingerprint_dir=fixture["fingerprint_dir"],
        duplicate_manifest_path=fixture["duplicate"],
        corpus_summary_path=fixture["corpus_summary"],
        split_assignments_path=fixture["assignments"],
        speaker_isolation_report_path=fixture["speaker_report"],
        duplicate_isolation_report_path=fixture["duplicate_report"],
        report_dir=root / "reports",
        model_root=root / "models",
        candidate="logistic_regression",
    )
    assert result.selected_candidate is not None
    assert result.artifact_manifest["active"] is False
    assert (result.run_dir / "pipeline.joblib").exists()
    assert REQUIRED_MODEL_CARD_DISCLAIMER in (result.run_dir / "model_card.md").read_text(encoding="utf-8")
    error_text = (root / "reports" / "speech_error_analysis.csv").read_text(encoding="utf-8")
    assert "safe_speaker_key" not in error_text
    with pytest.raises(FileExistsError):
        run_speech_baseline(
            config_path=fixture["config"],
            features_path=fixture["features"],
            canonical_manifest_path=fixture["canonical"],
            split_manifest_path=fixture["split"],
            feature_schema_path=fixture["schema"],
            preprocessing_report_path=fixture["report"],
            fingerprint_dir=fixture["fingerprint_dir"],
            duplicate_manifest_path=fixture["duplicate"],
            corpus_summary_path=fixture["corpus_summary"],
            split_assignments_path=fixture["assignments"],
            speaker_isolation_report_path=fixture["speaker_report"],
            duplicate_isolation_report_path=fixture["duplicate_report"],
            report_dir=root / "reports",
            model_root=root / "models",
            candidate="logistic_regression",
        )


def test_speech_cli_dry_run_bounded_smoke_and_coverage_failure():
    root = _temp_root()
    fixture = _fixture(root)
    dry = subprocess.run(
        [
            sys.executable,
            "scripts/train_speech_baseline.py",
            "--dry-run",
            "--config",
            str(fixture["config"]),
            "--features",
            str(fixture["features"]),
            "--canonical-manifest",
            str(fixture["canonical"]),
            "--split-manifest",
            str(fixture["split"]),
            "--feature-schema",
            str(fixture["schema"]),
            "--preprocessing-report",
            str(fixture["report"]),
            "--fingerprint-dir",
            str(fixture["fingerprint_dir"]),
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert "dry_run_ok" in dry.stdout

    smoke = subprocess.run(
        [
            sys.executable,
            "scripts/train_speech_baseline.py",
            "--config",
            str(fixture["config"]),
            "--features",
            str(fixture["features"]),
            "--canonical-manifest",
            str(fixture["canonical"]),
            "--split-manifest",
            str(fixture["split"]),
            "--feature-schema",
            str(fixture["schema"]),
            "--preprocessing-report",
            str(fixture["report"]),
            "--fingerprint-dir",
            str(fixture["fingerprint_dir"]),
            "--duplicate-manifest",
            str(fixture["duplicate"]),
            "--corpus-summary",
            str(fixture["corpus_summary"]),
            "--split-assignments",
            str(fixture["assignments"]),
            "--speaker-isolation-report",
            str(fixture["speaker_report"]),
            "--duplicate-isolation-report",
            str(fixture["duplicate_report"]),
            "--output-dir",
            str(root / "cli_reports"),
            "--model-root",
            str(root / "cli_models"),
            "--candidate",
            "logistic_regression",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert "completed" in smoke.stdout

    short = pd.read_csv(fixture["features"]).head(5)
    short_path = root / "short_features.csv"
    short.to_csv(short_path, index=False)
    fail = subprocess.run(
        [
            sys.executable,
            "scripts/train_speech_baseline.py",
            "--config",
            str(fixture["config"]),
            "--features",
            str(short_path),
            "--canonical-manifest",
            str(fixture["canonical"]),
            "--split-manifest",
            str(fixture["split"]),
            "--feature-schema",
            str(fixture["schema"]),
            "--preprocessing-report",
            str(fixture["report"]),
            "--fingerprint-dir",
            str(fixture["fingerprint_dir"]),
            "--duplicate-manifest",
            str(fixture["duplicate"]),
            "--corpus-summary",
            str(fixture["corpus_summary"]),
            "--split-assignments",
            str(fixture["assignments"]),
            "--speaker-isolation-report",
            str(fixture["speaker_report"]),
            "--duplicate-isolation-report",
            str(fixture["duplicate_report"]),
            "--output-dir",
            str(root / "fail_reports"),
            "--model-root",
            str(root / "fail_models"),
            "--candidate",
            "logistic_regression",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert fail.returncode == 1
    assert "coverage" in fail.stderr.lower()
