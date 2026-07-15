import json
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from app.ml.common import hashing, paths
from app.ml.training.profile.constants import (
    EXTENDED_SELF_REPORT_FEATURE_SET,
    MINIMAL_CONTEXTUAL_FEATURE_SET,
    PROFILE_TARGET_COLUMN,
    SENSITIVE_CONTEXT_FEATURE_SET,
)
from app.ml.training.profile.data import (
    build_profile_training_bundle,
    validate_profile_feature_policy,
)
from app.ml.training.profile.estimators import (
    logistic_regression_candidate_specs,
    profile_candidate_specs,
    random_forest_candidate_specs,
)
from app.ml.training.profile.evaluation import (
    evaluate_profile_split,
    fairness_exploration,
    select_profile_threshold,
)
from app.ml.training.profile.preprocessing import (
    build_profile_preprocessor,
    save_profile_preprocessor,
    transform_profile_features,
)
from app.ml.training.profile.runner import dry_run_profile_baseline, run_profile_baseline


def _temp_root() -> Path:
    root = paths.get_generated_root() / "temporary" / f"phase3c-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _fixture(root: Path):
    df = pd.DataFrame(
        {
            "record_id": [f"rec-{index}" for index in range(10)],
            "year_of_study": ["year 1", "year 2", "year 3", "year 4", "year 1", "year 2", "year 3", "year 4", "year 1", "year 2"],
            "self_reported_anxiety": ["yes", "no"] * 5,
            "self_reported_panic_attack": ["no", "yes"] * 5,
            "gender": ["female", "male"] * 5,
            "target_depression": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )
    canonical = root / "canonical.csv"
    df.to_csv(canonical, index=False)
    canonical_hash = hashing.sha256_file(canonical)
    split = {
        "train_ids": [f"rec-{index}" for index in range(6)],
        "validation_ids": ["rec-6", "rec-7"],
        "test_ids": ["rec-8", "rec-9"],
        "source_fingerprint": canonical_hash,
        "preprocessing_artifact_hash": canonical_hash,
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    fingerprint = root / "fingerprint.json"
    fingerprint.write_text(json.dumps({"combined_sha256": canonical_hash}), encoding="utf-8")
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "locked_split_manifest_hash": hashing.sha256_file(split_path),
                "threshold_strategies": ["default", "max_f1"],
                "max_candidate_count": 32,
                "hyperparameter_search": {
                    "logistic_regression": {"C": [1.0], "class_weight": [None], "max_iter": 100, "random_state": 42},
                    "random_forest": {"n_estimators": [10], "max_depth": [2], "min_samples_leaf": [1], "class_weight": [None], "random_state": 42},
                },
            }
        ),
        encoding="utf-8",
    )
    feature_schema = root / "feature_schema.json"
    feature_schema.write_text(json.dumps({"features": ["year_of_study"]}), encoding="utf-8")
    return canonical, split_path, fingerprint, config, feature_schema


def test_profile_feature_policy_blocks_leakage_and_sensitive_defaults():
    validate_profile_feature_policy(["year_of_study"], feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET)
    validate_profile_feature_policy(
        ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"],
        feature_set=EXTENDED_SELF_REPORT_FEATURE_SET,
    )
    with pytest.raises(ValueError):
        validate_profile_feature_policy(["year_of_study", "gender"], feature_set=SENSITIVE_CONTEXT_FEATURE_SET)
    validate_profile_feature_policy(["year_of_study", "gender"], feature_set=SENSITIVE_CONTEXT_FEATURE_SET, allow_sensitive_context=True)
    for forbidden in (PROFILE_TARGET_COLUMN, "source_timestamp", "Timestamp", "sought_specialist_treatment"):
        with pytest.raises(ValueError):
            validate_profile_feature_policy(["year_of_study", forbidden], feature_set=EXTENDED_SELF_REPORT_FEATURE_SET)


def test_profile_data_loading_validates_hashes_missing_extra_and_order():
    root = _temp_root()
    canonical, split_path, fingerprint, config, _ = _fixture(root)
    cfg = json.loads(config.read_text(encoding="utf-8"))
    bundle = build_profile_training_bundle(
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        source_fingerprint_path=fingerprint,
        features=["year_of_study"],
        feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
        expected_split_manifest_hash=cfg["locked_split_manifest_hash"],
    )
    assert bundle.train["record_id"].tolist() == [f"rec-{index}" for index in range(6)]

    bad = pd.read_csv(canonical)
    bad = bad[bad["record_id"] != "rec-9"]
    bad_path = root / "missing.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        build_profile_training_bundle(
            canonical_data_path=bad_path,
            split_manifest_path=split_path,
            source_fingerprint_path=fingerprint,
            features=["year_of_study"],
            feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
        )
    with pytest.raises(ValueError):
        build_profile_training_bundle(
            canonical_data_path=canonical,
            split_manifest_path=split_path,
            source_fingerprint_path=fingerprint,
            features=["target_depression"],
            feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
        )
    with pytest.raises(ValueError):
        build_profile_training_bundle(
            canonical_data_path=canonical,
            split_manifest_path=split_path,
            source_fingerprint_path=fingerprint,
            features=["year_of_study"],
            feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
            expected_split_manifest_hash="0" * 64,
        )


def test_profile_preprocessing_is_train_fitted_deterministic_and_saves():
    root = _temp_root()
    train = pd.DataFrame({"year_of_study": ["year 1", "year 2", "year 1"], "age": [20, None, 22]})
    validation = pd.DataFrame({"year_of_study": ["year 99"], "age": [30]})
    result = build_profile_preprocessor(train, ["year_of_study", "age"], estimator_type="logistic_regression")
    transformed_train = transform_profile_features(result.preprocessor, train, ["year_of_study", "age"])
    transformed_validation = transform_profile_features(result.preprocessor, validation, ["year_of_study", "age"])
    assert transformed_train.shape[1] == transformed_validation.shape[1]
    assert result.feature_names == list(result.preprocessor.get_feature_names_out())
    assert "age" in result.numeric_features
    assert "year_of_study" in result.categorical_features
    saved = save_profile_preprocessor(result.preprocessor, root / "preprocessor.joblib")
    assert saved.exists()
    with pytest.raises(FileExistsError):
        save_profile_preprocessor(result.preprocessor, saved)


def test_profile_estimators_are_bounded_and_deterministic():
    assert len(logistic_regression_candidate_specs({"C": [0.1, 1.0], "class_weight": [None, "balanced"]})) == 4
    assert len(random_forest_candidate_specs({"n_estimators": [10], "max_depth": [2], "min_samples_leaf": [1], "class_weight": [None, "balanced"]})) == 2
    specs = profile_candidate_specs({"max_candidate_count": 32}, candidate="all")
    assert len(specs) <= 32
    assert all(spec.hyperparameters.get("random_state") == 42 for spec in specs)


def test_profile_evaluation_threshold_fairness_and_absent_metrics():
    y = ["yes", "yes", "no", "no"]
    p = [0.2, 0.8, 0.7, 0.1]
    selection = select_profile_threshold(y, p, strategy="max_f1")
    metrics = evaluate_profile_split(y, p, threshold=selection["threshold"], feature_count=3, split_name="validation")
    assert "false_negatives" in metrics
    assert "confusion_matrix" in metrics
    one_class = evaluate_profile_split(["yes", "yes"], [0.8, 0.9], threshold=0.5, feature_count=1, split_name="tiny")
    assert one_class["roc_auc"] is None
    fairness = fairness_exploration(
        pd.DataFrame({"gender": ["a", "b", "b", "b"]}),
        y,
        p,
        threshold=0.5,
        sensitive_columns=["gender"],
        min_support=5,
    )
    assert fairness["slices"]["gender"]["a"]["status"] == "insufficient sample"


def test_profile_runner_dry_run_real_run_artifacts_and_overwrite_protection():
    root = _temp_root()
    canonical, split_path, fingerprint, config, feature_schema = _fixture(root)
    dry = dry_run_profile_baseline(
        config_path=config,
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        source_fingerprint_path=fingerprint,
        feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
    )
    assert dry["status"] == "dry_run_ok"

    result = run_profile_baseline(
        config_path=config,
        canonical_data_path=canonical,
        split_manifest_path=split_path,
        feature_schema_path=feature_schema,
        source_fingerprint_path=fingerprint,
        report_dir=root / "reports",
        model_root=root / "models",
        feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
        overwrite=False,
    )
    assert result.selected_candidate is not None
    assert result.metrics["summary"]["model_became_active"] is False
    assert (result.run_dir / "artifact_manifest.json").exists()
    assert (result.run_dir / "model_card.md").read_text(encoding="utf-8").count("research prototype") >= 1
    assert "active" not in (result.run_dir / "metrics.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_profile_baseline(
            config_path=config,
            canonical_data_path=canonical,
            split_manifest_path=split_path,
            feature_schema_path=feature_schema,
            source_fingerprint_path=fingerprint,
            report_dir=root / "reports",
            model_root=root / "models",
            feature_set=MINIMAL_CONTEXTUAL_FEATURE_SET,
        )


def test_profile_cli_dry_run_and_sensitive_refusal():
    completed = subprocess.run(
        [sys.executable, "scripts/train_profile_baseline.py", "--dry-run"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "dry_run_ok" in completed.stdout

    refused = subprocess.run(
        [sys.executable, "scripts/train_profile_baseline.py", "--dry-run", "--feature-set", "sensitive_context_exploratory"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "sensitive-context" in refused.stderr
