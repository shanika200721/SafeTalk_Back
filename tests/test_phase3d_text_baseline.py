import json
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from app.ml.common import hashing, paths
from app.ml.training.text.constants import TEXT_LABELS
from app.ml.training.text.data import build_text_training_bundle
from app.ml.training.text.estimators import logistic_regression_candidate_specs, linear_svm_candidate_specs
from app.ml.training.text.evaluation import evaluate_text_split
from app.ml.training.text.preprocessing import fit_text_vectorizer, hash_vocabulary, transform_text_features
from app.ml.training.text.runner import dry_run_text_baseline, run_text_baseline


def _temp_root() -> Path:
    root = paths.get_generated_root() / "temporary" / f"phase3d-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _fixture(root: Path):
    records = []
    phrases = {
        "anxiety": ["not calm anxiety worry", "panic worry anxiety"],
        "depression": ["not happy depression low", "sad low depression"],
        "normal": ["not worried normal steady", "steady okay normal"],
        "suicidal": ["suicidal crisis help", "not safe suicidal"],
    }
    index = 1
    for label in TEXT_LABELS:
        for text in phrases[label] * 3:
            unique_text = f"{text} unique{index}"
            records.append(
                {
                    "record_id": f"txt-{index:03d}",
                    "normalized_text": unique_text,
                    "canonical_label": label,
                    "source_name": "fixture.csv",
                    "text_hash": hashing.hash_text(unique_text),
                    "character_count": len(unique_text),
                    "word_count": len(unique_text.split()),
                    "line_count": 1,
                    "placeholder_count": int("<USER>" in text),
                }
            )
            index += 1
    df = pd.DataFrame(records)
    canonical = root / "canonical_text.csv"
    df.to_csv(canonical, index=False)
    canonical_hash = hashing.sha256_file(canonical)

    train_ids = df.groupby("canonical_label").head(3)["record_id"].tolist()
    validation_ids = df.groupby("canonical_label").nth([3])["record_id"].tolist()
    test_ids = df.groupby("canonical_label").nth([4, 5])["record_id"].tolist()
    split = {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "excluded_ids": {"txt-quarantine": "conflicting_duplicate_quarantine"},
        "source_fingerprint": canonical_hash,
        "preprocessing_artifact_hash": canonical_hash,
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    fingerprint = root / "fingerprint.json"
    fingerprint.write_text(json.dumps({"combined_sha256": canonical_hash}), encoding="utf-8")
    duplicate = root / "duplicate.json"
    duplicate.write_text(json.dumps({"exact_duplicate_groups": [], "near_duplicate_candidates": []}), encoding="utf-8")
    quarantine = root / "quarantine.csv"
    pd.DataFrame([{"record_id": "txt-quarantine", "text_hash": "q", "canonical_label": "normal"}]).to_csv(quarantine, index=False)
    overlap = root / "overlap.json"
    overlap.write_text(json.dumps({"exact_overlap_count": 0}), encoding="utf-8")
    feature_schema = root / "feature_schema.json"
    feature_schema.write_text(json.dumps({"features": [{"name": "normalized_text"}]}), encoding="utf-8")
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "locked_split_manifest_hash": hashing.sha256_file(split_path),
                "min_validation_suicidal_recall": 0.0,
                "max_candidate_count": 8,
                "vectorizers": [
                    {"name": "word_tfidf", "kind": "word", "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "max_features": 100, "sublinear_tf": True}
                ],
                "hyperparameter_search": {
                    "logistic_regression": {"C": [1.0], "class_weight": [None], "max_iter": 100, "solver": "liblinear", "random_state": 42},
                    "linear_svm": {"C": [1.0], "class_weight": [None], "max_iter": 1000, "random_state": 42},
                },
            }
        ),
        encoding="utf-8",
    )
    return canonical, split_path, fingerprint, duplicate, quarantine, overlap, feature_schema, config


def test_text_locked_split_enforces_quarantine_hashes_labels_and_order():
    root = _temp_root()
    canonical, split_path, fingerprint, duplicate, quarantine, overlap, _, config = _fixture(root)
    cfg = json.loads(config.read_text(encoding="utf-8"))
    bundle = build_text_training_bundle(
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        source_fingerprint_path=fingerprint,
        duplicate_manifest_path=duplicate,
        conflict_quarantine_path=quarantine,
        source_overlap_report_path=overlap,
        expected_split_manifest_hash=cfg["locked_split_manifest_hash"],
    )
    assert set(bundle.train["canonical_label"]) == set(TEXT_LABELS)
    assert bundle.train["record_id"].tolist() == json.loads(split_path.read_text())["train_ids"]

    bad_split = json.loads(split_path.read_text())
    bad_split["train_ids"][0] = "txt-quarantine"
    bad_path = root / "bad_split.json"
    bad_path.write_text(json.dumps(bad_split), encoding="utf-8")
    with pytest.raises(ValueError):
        build_text_training_bundle(
            canonical_data_path=canonical,
            split_manifest_path=bad_path,
            source_fingerprint_path=fingerprint,
            duplicate_manifest_path=duplicate,
            conflict_quarantine_path=quarantine,
            source_overlap_report_path=overlap,
        )


def test_text_vectorizers_are_train_only_deterministic_and_sparse():
    train = ["not unsafe <USER> token", "calm normal words"]
    validation = ["brandnew validationword"]
    word = fit_text_vectorizer(train, {"name": "word", "kind": "word", "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0})
    char = fit_text_vectorizer(train, {"name": "char", "kind": "char", "ngram_range": [3, 5], "min_df": 1, "max_df": 1.0})
    combined = fit_text_vectorizer(
        train,
        {
            "name": "combined",
            "kind": "combined",
            "word": {"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0},
            "char": {"ngram_range": [3, 3], "min_df": 1, "max_df": 1.0},
        },
    )
    assert transform_text_features(word.vectorizer, validation).shape[1] == word.feature_count
    assert "validationword" not in set(word.feature_names)
    assert any("not" in feature for feature in word.feature_names)
    assert any("<USER>" in feature for feature in word.feature_names)
    assert word.vocabulary_hash == hash_vocabulary(word.feature_names)
    assert char.feature_count > 0
    assert combined.feature_count > word.feature_count


def test_text_estimators_and_metrics_cover_suicidal_counts():
    assert len(logistic_regression_candidate_specs({"C": [0.5, 1.0], "class_weight": [None, "balanced"]})) == 4
    assert len(linear_svm_candidate_specs({"C": [1.0], "class_weight": [None, "balanced"]})) == 2
    y_true = ["suicidal", "suicidal", "normal", "depression"]
    y_pred = ["normal", "suicidal", "normal", "suicidal"]
    metrics = evaluate_text_split(y_true, y_pred, split_name="validation")
    assert metrics["macro_f1"] is not None
    assert metrics["suicidal_class"]["false_negatives"] == 1
    assert metrics["suicidal_class"]["suicidal_predicted_as_normal"] == 1
    assert metrics["suicidal_class"]["depression_predicted_as_suicidal"] == 1


def test_text_runner_dry_run_artifacts_no_raw_text_and_overwrite_protection():
    root = _temp_root()
    canonical, split_path, fingerprint, duplicate, quarantine, overlap, feature_schema, config = _fixture(root)
    dry = dry_run_text_baseline(
        config_path=config,
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        source_fingerprint_path=fingerprint,
        duplicate_manifest_path=duplicate,
        conflict_quarantine_path=quarantine,
        source_overlap_report_path=overlap,
    )
    assert dry["status"] == "dry_run_ok"
    result = run_text_baseline(
        config_path=config,
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        feature_schema_path=feature_schema,
        source_fingerprint_path=fingerprint,
        duplicate_manifest_path=duplicate,
        conflict_quarantine_path=quarantine,
        source_overlap_report_path=overlap,
        report_dir=root / "reports",
        model_root=root / "models",
        overwrite=False,
    )
    assert result.selected_candidate is not None
    assert result.artifact_manifest["active"] is False
    assert (result.run_dir / "model_card.md").read_text(encoding="utf-8").count("research prototype") >= 1
    error_report = (root / "reports" / "text_error_analysis.csv").read_text(encoding="utf-8")
    assert "suicidal crisis help" not in error_report
    with pytest.raises(FileExistsError):
        run_text_baseline(
            config_path=config,
            canonical_data_path=canonical,
            split_manifest_path=split_path,
            feature_schema_path=feature_schema,
            source_fingerprint_path=fingerprint,
            duplicate_manifest_path=duplicate,
            conflict_quarantine_path=quarantine,
            source_overlap_report_path=overlap,
            report_dir=root / "reports",
            model_root=root / "models",
        )


def test_text_cli_dry_run_and_bounded_smoke():
    dry = subprocess.run(
        [sys.executable, "scripts/train_text_baseline.py", "--dry-run"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry.returncode == 0
    assert "dry_run_ok" in dry.stdout

    root = _temp_root()
    smoke = subprocess.run(
        [
            sys.executable,
            "scripts/train_text_baseline.py",
            "--config",
            "ml-research/configs/training.text.logistic_regression.v1.json",
            "--candidate",
            "logistic_regression",
            "--max-train-records",
            "120",
            "--output-dir",
            str(root / "reports"),
            "--model-root",
            str(root / "models"),
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert "completed" in smoke.stdout
