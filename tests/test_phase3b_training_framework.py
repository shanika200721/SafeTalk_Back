import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic.v1 import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.ml.common import hashing, paths
from app.ml.splitting.schemas import ModalitySplitManifest, SplitValidationSummary
from app.ml.training import (
    ARTIFACT_MANIFEST_VERSION,
    EVALUATION_SCHEMA_VERSION,
    MODEL_CARD_VERSION,
    TRAINING_FRAMEWORK_VERSION,
)
from app.ml.training.artifacts import (
    create_candidate_bundle,
    create_run_directory,
    load_artifact_manifest,
    save_metrics_json,
    save_model_artifact,
    save_training_config,
    verify_artifact_hashes,
)
from app.ml.training.data import (
    build_training_dataset_bundle,
    load_split_manifest,
    validate_feature_columns,
    validate_no_cross_split_overlap,
    validate_preprocessing_artifact_hash,
    validate_source_fingerprint,
    validate_target_column,
)
from app.ml.training.metrics import evaluate_binary_metrics, evaluate_multiclass_metrics
from app.ml.training.model_card import build_model_card
from app.ml.training.registry import (
    build_model_registry_payload,
    confirm_model_not_active,
    register_candidate_model,
    retrieve_registered_candidate,
)
from app.ml.training.reproducibility import (
    capture_environment_versions,
    deterministic_run_id,
    hash_training_config,
    save_reproducibility_report,
)
from app.ml.training.runner import (
    create_synthetic_smoke_fixture,
    run_training_pipeline,
    synthetic_estimator_factory,
)
from app.ml.training.schemas import ModelCard, TrainingConfig, TrainingRunResult, TrainingTask
from app.ml.training.thresholds import select_binary_threshold


def _temp_output() -> Path:
    root = paths.get_generated_root() / "temporary" / f"pytest-phase3b-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _config(manifest_path: Path, **overrides) -> TrainingConfig:
    payload = {
        "experiment_name": "pytest-phase3b",
        "experiment_version": "1.0.0",
        "model_name": "pytest-model",
        "model_version": "1.0.0",
        "modality": "synthetic",
        "task": "binary_classification",
        "framework": "scikit-learn",
        "estimator_type": "logistic_regression",
        "dataset_name": "pytest-synthetic",
        "dataset_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "split_manifest_path": manifest_path.relative_to(paths.get_repository_root()).as_posix(),
        "random_seed": 42,
        "hyperparameters": {"max_iter": 100, "random_state": 42},
        "class_weight_policy": None,
        "threshold_strategy": "max_f1",
        "target_column": "label",
        "feature_columns": ["feature_0", "feature_1"],
        "excluded_columns": ["record_id"],
        "sensitive_columns": [],
        "primary_metric": "f1_weighted",
        "secondary_metrics": ["balanced_accuracy"],
        "artifact_subdirectory": "synthetic/pytest",
        "notes": "pytest synthetic only",
    }
    payload.update(overrides)
    return TrainingConfig(**payload)


def _dataset_and_manifest(root: Path, *, excluded=False):
    root.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(
        {
            "record_id": [f"rec-{index}" for index in range(8)],
            "feature_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "feature_1": [0, 1, 0, 1, 0, 1, 0, 1],
            "label": ["class_0", "class_1"] * 4,
        }
    )
    if excluded:
        records.loc[len(records)] = ["excluded-rec", 9, 1, "class_1"]
    data_path = root / "canonical.csv"
    records.to_csv(data_path, index=False)
    source_hash = hashing.sha256_file(data_path)
    preprocessing_hash = hashing.hash_json_data({"features": ["feature_0", "feature_1"]})
    manifest = ModalitySplitManifest(
        manifest_version="1.0.0",
        split_design_version="1.0.0",
        modality="synthetic",
        dataset_name="pytest-synthetic",
        dataset_version="1.0.0",
        preprocessing_version="1.0.0",
        feature_schema_version="1.0.0",
        source_fingerprint=source_hash,
        preprocessing_artifact_hash=preprocessing_hash,
        config_hash=hashing.hash_json_data({"pytest": True}),
        random_seed=42,
        split_strategy="random_stratified",
        train_ids=["rec-0", "rec-1", "rec-2", "rec-3"],
        validation_ids=["rec-4", "rec-5"],
        test_ids=["rec-6", "rec-7"],
        excluded_ids={"excluded-rec": "quarantine"} if excluded else {},
        stratify_column="label",
        duplicate_policy="unique",
        created_at=datetime.now(timezone.utc),
        validation_summary=SplitValidationSummary(
            train_count=4,
            validation_count=2,
            test_count=2,
            total_count=8,
            deterministic_replay_passed=True,
        ),
    )
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_safe_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return data_path, manifest_path, source_hash, preprocessing_hash


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_schema_versions_and_training_config_validation():
    root = _temp_output()
    _, manifest_path, _, _ = _dataset_and_manifest(root)
    config = _config(manifest_path)
    assert config.model_version == "1.0.0"
    assert TRAINING_FRAMEWORK_VERSION == "1.0.0"
    assert MODEL_CARD_VERSION == "1.0.0"
    assert EVALUATION_SCHEMA_VERSION == "1.0.0"
    assert ARTIFACT_MANIFEST_VERSION == "1.0.0"
    with pytest.raises(ValidationError):
        _config(manifest_path, model_version="")
    with pytest.raises(ValidationError):
        _config(manifest_path, feature_columns=["feature_0", "label"])
    with pytest.raises(ValidationError):
        _config(manifest_path, feature_columns=["feature_0", "record_id"])
    with pytest.raises(ValidationError):
        TrainingRunResult(
            run_id="x",
            status="completed",
            started_at=datetime.now(timezone.utc),
            config_hash="0" * 64,
            dataset_reference={
                "train_ids": ["a"],
                "validation_ids": [],
                "test_ids": ["b"],
                "manifest_hash": "",
                "source_fingerprint": "0" * 64,
                "preprocessing_artifact_hash": "0" * 64,
            },
        )


def test_model_card_requires_clinical_disclaimer_and_sections():
    root = _temp_output()
    _, manifest_path, _, _ = _dataset_and_manifest(root)
    config = _config(manifest_path)
    card = build_model_card(config=config, metrics={}, split_summary="locked", dataset_summary="synthetic", preprocessing_summary="none")
    assert "research prototype" in card.clinical_disclaimer
    assert "Clinical" in card.prohibited_use or "clinical" in card.prohibited_use
    assert card.fairness_considerations
    assert card.privacy_considerations
    assert card.limitations
    with pytest.raises(ValidationError):
        ModelCard(**dict(card.to_safe_dict(), clinical_disclaimer="missing"))


def test_split_safe_data_loading_validates_coverage_and_leakage():
    root = _temp_output()
    data_path, manifest_path, source_hash, preprocessing_hash = _dataset_and_manifest(root)
    config = _config(manifest_path)
    bundle = build_training_dataset_bundle(config=config, split_manifest_path=manifest_path, canonical_data_path=data_path)
    assert bundle.train["record_id"].tolist() == ["rec-0", "rec-1", "rec-2", "rec-3"]
    manifest = load_split_manifest(manifest_path)
    validate_no_cross_split_overlap(manifest)
    validate_source_fingerprint(manifest, source_hash)
    validate_preprocessing_artifact_hash(manifest, preprocessing_hash)
    with pytest.raises(ValueError):
        validate_source_fingerprint(manifest, "1" * 64)
    with pytest.raises(ValueError):
        validate_preprocessing_artifact_hash(manifest, "2" * 64)
    with pytest.raises(ValueError):
        validate_target_column(pd.read_csv(data_path), "missing", config.feature_columns)
    with pytest.raises(ValueError):
        validate_feature_columns(pd.read_csv(data_path), ["feature_0", "record_id"])


def test_data_loading_rejects_missing_and_excluded_records():
    root = _temp_output()
    data_path, manifest_path, _, _ = _dataset_and_manifest(root)
    config = _config(manifest_path)
    df = pd.read_csv(data_path)
    df = df[df["record_id"] != "rec-7"]
    missing_path = root / "missing.csv"
    df.to_csv(missing_path, index=False)
    with pytest.raises(ValueError):
        build_training_dataset_bundle(config=config, split_manifest_path=manifest_path, canonical_data_path=missing_path)

    excluded_data, excluded_manifest, _, _ = _dataset_and_manifest(root / "excluded", excluded=True)
    with pytest.raises(ValueError):
        build_training_dataset_bundle(config=_config(excluded_manifest), split_manifest_path=excluded_manifest, canonical_data_path=excluded_data)


def test_reproducibility_is_deterministic_and_portable():
    root = _temp_output()
    _, manifest_path, _, _ = _dataset_and_manifest(root)
    config = _config(manifest_path)
    assert deterministic_run_id(config) == deterministic_run_id(config)
    assert hash_training_config(config) == hash_training_config(config)
    changed = _config(manifest_path, hyperparameters={"max_iter": 101, "random_state": 42})
    assert hash_training_config(config) != hash_training_config(changed)
    report = capture_environment_versions()
    text = json.dumps(report)
    assert "PASSWORD" not in text.upper()
    assert ":\\Users\\" not in text
    report_path = save_reproducibility_report(report, root / "reproducibility.json")
    assert report_path.exists()


def test_binary_and_multiclass_metrics_are_deterministic_and_warn():
    binary = evaluate_binary_metrics(
        ["class_0", "class_1", "class_1", "class_0"],
        ["class_0", "class_0", "class_1", "class_1"],
        y_probability=[0.1, 0.4, 0.8, 0.7],
        positive_label="class_1",
    )
    assert binary.false_negative_count == 1
    assert binary.false_positive_count == 1
    assert binary.confusion_matrix == [[1, 1], [1, 1]]
    multi = evaluate_multiclass_metrics(
        ["a", "a", "b"],
        ["a", "b", "b"],
        y_probability=[[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.8, 0.1]],
        labels=["a", "b", "c"],
    )
    assert any("class absent" in warning for warning in multi.warnings)
    with pytest.raises(ValueError):
        evaluate_binary_metrics([0, 1], [0, 1], y_probability=[1.2, 0.1])


def test_threshold_strategies_use_validation_inputs():
    y = ["class_0", "class_1", "class_1", "class_0"]
    p = [0.1, 0.4, 0.8, 0.7]
    assert select_binary_threshold(y, p, strategy="default", positive_label="class_1").threshold == 0.5
    assert 0 <= select_binary_threshold(y, p, strategy="max_f1", positive_label="class_1").threshold <= 1
    assert 0 <= select_binary_threshold(y, p, strategy="recall_priority", positive_label="class_1", min_recall=0.5).threshold <= 1
    assert select_binary_threshold(y, p, strategy="fixed", fixed_threshold=0.3, positive_label="class_1").threshold == 0.3
    cost = select_binary_threshold(y, p, strategy="cost_sensitive", positive_label="class_1", false_negative_cost=10)
    assert "validation" in cost.to_dict()["objective"]


def test_artifact_management_hashes_and_rejects_untrusted_load():
    root = _temp_output()
    _, manifest_path, _, _ = _dataset_and_manifest(root)
    config = _config(manifest_path)
    run_dir = create_run_directory(modality="synthetic", model_name=f"pytest-{uuid.uuid4().hex}", model_version="1.0.0", run_id="run-1")
    save_training_config(config, run_dir)
    model_path = save_model_artifact({"model": "synthetic"}, run_dir)
    metrics_path = save_metrics_json({"accuracy": 1.0}, run_dir)
    card_path = run_dir / "model_card.md"
    card_path.write_text("research prototype card", encoding="utf-8")
    manifest = create_candidate_bundle(
        run_dir=run_dir,
        config=config,
        run_id="run-1",
        split_manifest_hash="0" * 64,
        metrics_path=metrics_path,
        model_path=model_path,
        model_card_path=card_path,
    )
    assert verify_artifact_hashes(manifest)
    assert load_artifact_manifest(run_dir / "artifact_manifest.json").run_id == "run-1"
    with pytest.raises(FileExistsError):
        save_metrics_json({"accuracy": 0.0}, run_dir)
    with pytest.raises(ValueError):
        load_artifact_manifest(root / "artifact_manifest.json")


def test_registry_candidate_registration_keeps_model_inactive(db_session):
    root = _temp_output()
    fixture = create_synthetic_smoke_fixture(root, overwrite=True)
    result = run_training_pipeline(config=fixture["config"], canonical_data_path=fixture["canonical_data"], estimator_factory=synthetic_estimator_factory, overwrite=True)
    manifest = load_artifact_manifest(paths.get_repository_root() / result.artifact_manifest_path)
    config = _config(fixture["split_manifest"], model_name="registry-candidate", model_version=f"1.0.{uuid.uuid4().int % 10000}")
    payload = build_model_registry_payload(config=config, artifact_manifest=manifest, metrics_json={"validation": {}}, thresholds_json={})
    registered = register_candidate_model(db_session, payload)
    db_session.commit()
    assert registered.is_active is False
    assert confirm_model_not_active(db_session, model_name=config.model_name, modality=config.modality, version=config.model_version)
    assert retrieve_registered_candidate(db_session, model_name=config.model_name, modality=config.modality, version=config.model_version)
    with pytest.raises(IntegrityError):
        register_candidate_model(db_session, payload)
        db_session.commit()


def test_runner_synthetic_binary_and_multiclass_create_bundle():
    root = _temp_output()
    fixture = create_synthetic_smoke_fixture(root, overwrite=True)
    result = run_training_pipeline(config=fixture["config"], canonical_data_path=fixture["canonical_data"], estimator_factory=synthetic_estimator_factory, overwrite=True)
    assert result.status.value == "completed"
    assert result.test_metrics is not None
    assert "validation" in result.selected_thresholds["objective"]
    assert Path(paths.get_repository_root() / result.model_artifact_path).exists()
    assert Path(paths.get_repository_root() / result.model_card_path).exists()
    second = run_training_pipeline(config=fixture["config"], canonical_data_path=fixture["canonical_data"], estimator_factory=synthetic_estimator_factory, overwrite=True)
    assert second.run_id == result.run_id

    multi_root = _temp_output()
    multi_fixture = create_synthetic_smoke_fixture(multi_root, task=TrainingTask.MULTICLASS_CLASSIFICATION, overwrite=True)
    multi = run_training_pipeline(config=multi_fixture["config"], canonical_data_path=multi_fixture["canonical_data"], estimator_factory=synthetic_estimator_factory, overwrite=True)
    assert multi.status.value == "completed"
    assert multi.selected_thresholds["strategy"] == "argmax"
