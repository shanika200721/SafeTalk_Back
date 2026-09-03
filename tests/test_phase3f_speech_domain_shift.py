import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.common import hashing, paths
from app.ml.evaluation.speech_domain.constants import (
    SPEECH_DOMAIN_EVALUATION_VERSION,
    SPEECH_LOCO_POLICY_VERSION,
)
from app.ml.evaluation.speech_domain.data import build_domain_evaluation_bundle
from app.ml.evaluation.speech_domain.metrics import corpus_generalization_gap, evaluate_domain_predictions
from app.ml.evaluation.speech_domain.runner import (
    run_shortcut_diagnostics,
    run_speech_domain_shift_evaluation,
    run_transfer_matrix,
)
from app.ml.evaluation.speech_domain.splits import create_loco_folds
from app.ml.training.speech.constants import FEATURE_SETS


def _temp_root() -> Path:
    root = paths.get_generated_root() / "temporary" / f"phase3f-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _policy(root: Path) -> Path:
    payload = {
        "policy_version": "1.0.0",
        "canonical_labels": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "shared_labels_all_corpora": ["neutral", "happy", "sad", "angry", "fearful", "disgust"],
        "global_rules": ["The RAVDESS calm label is not merged into neutral."],
        "corpora": {
            "CREMA": {"canonical_labels": ["neutral", "happy", "sad", "angry", "fearful", "disgust"], "original_labels": []},
            "RAVDESS": {"canonical_labels": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"], "original_labels": []},
            "SAVEE": {"canonical_labels": ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"], "original_labels": []},
            "TESS": {"canonical_labels": ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"], "original_labels": []},
        },
        "fold_rules": {
            f"hold_out_{corpus}": {
                "included_labels_shared": ["neutral", "happy", "sad", "angry", "fearful", "disgust"],
                "excluded_labels": ["calm", "surprised"],
            }
            for corpus in ["CREMA", "RAVDESS", "SAVEE", "TESS"]
        },
    }
    path = root / "speech.domain_label_policy.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(root: Path):
    features = FEATURE_SETS["full_acoustic"]
    rows = []
    corpus_labels = {
        "CREMA": ["neutral", "happy", "sad", "angry", "fearful", "disgust"],
        "RAVDESS": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "SAVEE": ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "TESS": ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
    }
    corpus_offset = {"CREMA": 0.0, "RAVDESS": 10.0, "SAVEE": 20.0, "TESS": 30.0}
    idx = 0
    for corpus, labels in corpus_labels.items():
        speaker_count = 2 if corpus == "TESS" else 3
        for speaker_idx in range(speaker_count):
            for label_idx, label in enumerate(labels):
                idx += 1
                record_id = f"domain-rec-{idx:04d}"
                values = {
                    feature: corpus_offset[corpus] + label_idx + speaker_idx / 10.0 + feature_pos / 1000.0
                    for feature_pos, feature in enumerate(features)
                }
                rows.append(
                    {
                        "record_id": record_id,
                        "safe_speaker_key": f"{corpus}-speaker-{speaker_idx}",
                        "corpus_name": corpus,
                        "canonical_emotion_label": label,
                        "original_audio_hash": hashing.hash_text(record_id),
                        "sample_rate": 16000 + int(corpus_offset[corpus]),
                        "channel_count": 1,
                        **values,
                        "feature_extraction_warnings": "",
                    }
                )
    df = pd.DataFrame(rows)
    feature_path = root / "speech_features.csv"
    df[["record_id", "safe_speaker_key", "corpus_name", "canonical_emotion_label", *features, "feature_extraction_warnings"]].to_csv(feature_path, index=False)
    canonical_path = root / "speech_canonical_manifest.csv"
    df[["record_id", "safe_speaker_key", "corpus_name", "canonical_emotion_label", "original_audio_hash", "duration_seconds", "sample_rate", "channel_count"]].to_csv(canonical_path, index=False)
    schema_path = root / "speech_feature_schema.json"
    schema_path.write_text(json.dumps({"features": [{"name": feature, "dtype": "float"} for feature in features]}), encoding="utf-8")
    fingerprint_dir = root / "fingerprints"
    fingerprint_dir.mkdir()
    for corpus in ["CREMA", "RAVDESS", "SAVEE", "TESS"]:
        (fingerprint_dir / f"{corpus.lower()}-v1.json").write_text(json.dumps({"combined_sha256": hashing.hash_text(corpus)}), encoding="utf-8")
    corpus_mapping = root / "speech.corpus_mapping.v1.json"
    corpus_mapping.write_text(json.dumps({"mapping_version": "1.0.0", "corpora": []}), encoding="utf-8")
    return {
        "features": feature_path,
        "canonical": canonical_path,
        "schema": schema_path,
        "fingerprints": fingerprint_dir,
        "policy": _policy(root),
        "corpus_mapping": corpus_mapping,
    }


def _bundle(fixture):
    return build_domain_evaluation_bundle(
        features_path=fixture["features"],
        canonical_manifest_path=fixture["canonical"],
        feature_schema_path=fixture["schema"],
        label_policy_path=fixture["policy"],
        fingerprint_dir=fixture["fingerprints"],
    )


def test_label_policy_shared_labels_and_calm_not_merged():
    root = _temp_root()
    fixture = _fixture(root)
    policy = json.loads(fixture["policy"].read_text(encoding="utf-8"))
    assert "calm" not in policy["shared_labels_all_corpora"]
    assert any("not merged" in rule for rule in policy["global_rules"])
    bundle = _bundle(fixture)
    assert bundle.label_policy["policy_version"] == SPEECH_LOCO_POLICY_VERSION


def test_loco_folds_are_speaker_safe_deterministic_and_holdout_only():
    root = _temp_root()
    fixture = _fixture(root)
    folds_a = create_loco_folds(_bundle(fixture))
    folds_b = create_loco_folds(_bundle(fixture))
    assert [len(fold.train) for fold in folds_a] == [len(fold.train) for fold in folds_b]
    for fold in folds_a:
        assert fold.config.test_corpus not in set(fold.train["corpus_name"])
        assert fold.config.test_corpus not in set(fold.validation["corpus_name"])
        assert set(fold.test["corpus_name"]) == {fold.config.test_corpus}
        assert set(fold.train["safe_speaker_key"]) & set(fold.validation["safe_speaker_key"]) == set()
        assert "calm" in fold.config.excluded_labels


def test_metrics_absent_classes_and_generalization_gap():
    metrics = evaluate_domain_predictions(
        ["sad", "sad", "happy"],
        ["sad", "happy", "happy"],
        labels=["sad", "happy", "calm"],
        split_name="pytest",
    )
    assert "calm" in metrics["missing_classes"]
    assert metrics["macro_f1"] is not None
    gap = corpus_generalization_gap(0.8, [0.2, 0.4])
    assert gap["loco_mean_macro_f1"] == pytest.approx(0.3)
    assert gap["corpus_generalization_gap"] == pytest.approx(0.5)


def test_transfer_matrix_and_shortcut_diagnostics_are_research_only():
    root = _temp_root()
    bundle = _bundle(_fixture(root))
    transfer = run_transfer_matrix(bundle, features=bundle.features)
    assert any(item.train_corpus == "CREMA" and item.test_corpus == "RAVDESS" for item in transfer)
    assert all("calm" not in item.shared_labels for item in transfer)
    shortcut = run_shortcut_diagnostics(bundle, features=bundle.features)
    assert shortcut["diagnostic_only"] is True
    assert shortcut["registered"] is False
    assert shortcut["active"] is False
    assert shortcut["accuracy"] >= 0.0


def test_runner_writes_safe_reports_and_blocks_overwrite():
    root = _temp_root()
    fixture = _fixture(root)
    result = run_speech_domain_shift_evaluation(
        features_path=fixture["features"],
        canonical_manifest_path=fixture["canonical"],
        feature_schema_path=fixture["schema"],
        corpus_mapping_path=fixture["corpus_mapping"],
        label_policy_path=fixture["policy"],
        fingerprint_dir=fixture["fingerprints"],
        output_dir=root / "reports",
        model_root=root / "models",
        held_out_corpus="CREMA",
        candidate="logistic_regression",
        run_transfer=True,
        run_shortcut=True,
        save_artifacts=False,
        evaluation_manifest_dir=root / "manifests",
    )
    assert result["evaluation_version"] == SPEECH_DOMAIN_EVALUATION_VERSION
    assert result["activation_status"] == "inactive"
    report_text = (root / "reports" / "speech_domain_shift_summary.json").read_text(encoding="utf-8")
    assert "speaker-0" not in report_text
    assert str(root) not in report_text
    assert "pipeline.joblib" not in report_text
    with pytest.raises(FileExistsError):
        run_speech_domain_shift_evaluation(
            features_path=fixture["features"],
            canonical_manifest_path=fixture["canonical"],
            feature_schema_path=fixture["schema"],
            label_policy_path=fixture["policy"],
            fingerprint_dir=fixture["fingerprints"],
            output_dir=root / "reports",
            model_root=root / "models",
            held_out_corpus="CREMA",
            candidate="logistic_regression",
            save_artifacts=False,
            evaluation_manifest_dir=root / "manifests",
        )


def test_runner_dry_run_and_cli_modes():
    root = _temp_root()
    fixture = _fixture(root)
    dry = run_speech_domain_shift_evaluation(
        features_path=fixture["features"],
        canonical_manifest_path=fixture["canonical"],
        feature_schema_path=fixture["schema"],
        label_policy_path=fixture["policy"],
        fingerprint_dir=fixture["fingerprints"],
        dry_run=True,
    )
    assert dry["status"] == "dry_run_ok"
    cli = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_speech_domain_shift.py",
            "--dry-run",
            "--features",
            str(fixture["features"]),
            "--canonical-manifest",
            str(fixture["canonical"]),
            "--feature-schema",
            str(fixture["schema"]),
            "--label-policy",
            str(fixture["policy"]),
            "--fingerprint-dir",
            str(fixture["fingerprints"]),
            "--corpus-mapping",
            str(fixture["corpus_mapping"]),
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert "dry_run_ok" in cli.stdout
    invalid = subprocess.run(
        [sys.executable, "scripts/evaluate_speech_domain_shift.py", "--held-out-corpus", "BAD"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode != 0
