"""Generic Phase 3B training runner.

The runner is deliberately modality-agnostic. Real Profile/Text/Speech training
commands are not connected in Phase 3B; tests and CLI smoke checks use tiny
synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from app.ml.common import hashing, paths
from app.ml.splitting.schemas import ModalitySplitManifest, SplitValidationSummary
from app.ml.training import artifacts
from app.ml.training.config import load_training_config
from app.ml.training.data import build_training_dataset_bundle
from app.ml.training.evaluation import evaluate_train_validation_test
from app.ml.training.model_card import build_model_card, save_model_card
from app.ml.training.registry import build_model_registry_payload, register_candidate_model
from app.ml.training.reproducibility import (
    capture_environment_versions,
    deterministic_run_id,
    hash_training_config,
    save_reproducibility_report,
    set_global_seed,
    validate_deterministic_estimator_settings,
)
from app.ml.training.reporting import artifact_inventory, training_summary_json, training_summary_markdown
from app.ml.training.schemas import TrainingConfig, TrainingRunResult, TrainingStatus, TrainingTask, utc_now


EstimatorFactory = Callable[[TrainingConfig], Any]


def _feature_target(bundle, config: TrainingConfig):
    return (
        bundle.train[config.feature_columns],
        bundle.train[config.target_column],
        bundle.validation[config.feature_columns],
        bundle.validation[config.target_column],
        bundle.test[config.feature_columns],
        bundle.test[config.target_column],
    )


def run_training_pipeline(
    *,
    config: TrainingConfig | str | Path,
    canonical_data_path: str | Path,
    estimator_factory: EstimatorFactory,
    output_dir: str | Path | None = None,
    register_candidate: bool = False,
    db_session=None,
    overwrite: bool = False,
) -> TrainingRunResult:
    loaded_config = load_training_config(config) if isinstance(config, (str, Path)) else config
    started_at = utc_now()
    set_global_seed(loaded_config.random_seed)
    config_hash = hash_training_config(loaded_config)
    run_id = deterministic_run_id(loaded_config)
    try:
        bundle = build_training_dataset_bundle(
            config=loaded_config,
            split_manifest_path=loaded_config.split_manifest_path,
            canonical_data_path=canonical_data_path,
        )
        estimator = estimator_factory(loaded_config)
        warnings = validate_deterministic_estimator_settings(estimator)
        x_train, y_train, x_val, y_val, x_test, y_test = _feature_target(bundle, loaded_config)
        estimator.fit(x_train, y_train)
        evaluated = evaluate_train_validation_test(
            estimator,
            task=loaded_config.task,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_val,
            y_validation=y_val,
            x_test=x_test,
            y_test=y_test,
            threshold_strategy=loaded_config.threshold_strategy,
        )

        if output_dir is None:
            run_dir = artifacts.create_run_directory(
                modality=loaded_config.modality,
                model_name=loaded_config.model_name,
                model_version=loaded_config.model_version,
                run_id=run_id,
                overwrite=overwrite,
            )
        else:
            run_dir = Path(output_dir).resolve(strict=False)
            run_dir.mkdir(parents=True, exist_ok=True)
            if any(run_dir.iterdir()) and not overwrite:
                raise FileExistsError(f"Output directory already contains artifacts: {run_dir}")

        artifacts.save_training_config(loaded_config, run_dir, overwrite=overwrite)
        model_path = artifacts.save_model_artifact(estimator, run_dir, overwrite=overwrite)
        metrics_payload = {
            "train": evaluated["train_metrics"].to_safe_dict(),
            "validation": evaluated["validation_metrics"].to_safe_dict(),
            "test": evaluated["test_metrics"].to_safe_dict(),
            "selected_thresholds": evaluated["selected_thresholds"],
            "selection_policy": "validation metrics only; test metrics ignored for selection",
        }
        metrics_path = artifacts.save_metrics_json(metrics_payload, run_dir, overwrite=overwrite)
        repro_path = save_reproducibility_report(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "environment": capture_environment_versions(),
                "determinism_warnings": warnings,
            },
            Path(run_dir) / "reproducibility_report.json",
            overwrite=overwrite,
        )
        card = build_model_card(
            config=loaded_config,
            metrics=metrics_payload,
            split_summary=(
                f"Locked split manifest with train={len(bundle.dataset_reference.train_ids)}, "
                f"validation={len(bundle.dataset_reference.validation_ids)}, "
                f"test={len(bundle.dataset_reference.test_ids)}. No resplitting performed."
            ),
            dataset_summary=f"Synthetic or canonical dataset `{loaded_config.dataset_name}` version `{loaded_config.dataset_version}`.",
            preprocessing_summary=f"Preprocessing version `{loaded_config.preprocessing_version}` and feature schema `{loaded_config.feature_schema_version}`.",
        )
        _, model_card_md_path = save_model_card(card, run_dir, overwrite=overwrite)
        manifest = artifacts.create_candidate_bundle(
            run_dir=run_dir,
            config=loaded_config,
            run_id=run_id,
            split_manifest_hash=bundle.dataset_reference.manifest_hash,
            metrics_path=metrics_path,
            model_path=model_path,
            model_card_path=model_card_md_path,
            extra_files=[repro_path],
            overwrite=overwrite,
        )
        artifact_manifest_path = Path(run_dir) / "artifact_manifest.json"
        registered = None
        if register_candidate:
            if db_session is None:
                raise ValueError("candidate registration requires a temporary/test database session")
            payload = build_model_registry_payload(
                config=loaded_config,
                artifact_manifest=manifest,
                metrics_json=metrics_payload,
                thresholds_json=evaluated["selected_thresholds"],
            )
            registered = register_candidate_model(db_session, payload)

        result = TrainingRunResult(
            run_id=run_id,
            status=TrainingStatus.COMPLETED,
            started_at=started_at,
            completed_at=utc_now(),
            config_hash=config_hash,
            dataset_reference=bundle.dataset_reference,
            model_artifact_path=_repo_relative(model_path),
            metrics_path=_repo_relative(metrics_path),
            model_card_path=_repo_relative(model_card_md_path),
            artifact_manifest_path=_repo_relative(artifact_manifest_path),
            train_metrics=evaluated["train_metrics"],
            validation_metrics=evaluated["validation_metrics"],
            test_metrics=evaluated["test_metrics"],
            selected_thresholds=evaluated["selected_thresholds"],
            warnings=warnings + (["candidate registered inactive"] if registered is not None else []),
        )
        training_summary_json(result, Path(run_dir) / "training_summary.json", overwrite=overwrite)
        training_summary_markdown(result, Path(run_dir) / "training_summary.md", overwrite=overwrite)
        artifact_inventory(run_dir, Path(run_dir) / "artifact_inventory.json", overwrite=overwrite)
        return result
    except Exception as exc:
        return TrainingRunResult(
            run_id=run_id,
            status=TrainingStatus.FAILED,
            started_at=started_at,
            completed_at=utc_now(),
            config_hash=config_hash,
            dataset_reference={
                "train_ids": ["placeholder-train"],
                "validation_ids": [],
                "test_ids": ["placeholder-test"],
                "manifest_hash": "0" * 64,
                "source_fingerprint": "0" * 64,
                "preprocessing_artifact_hash": "0" * 64,
            },
            warnings=[],
            failure_reason=str(exc),
        )


def _repo_relative(path: str | Path) -> str:
    return Path(path).resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite synthetic fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def create_synthetic_smoke_fixture(
    output_root: str | Path,
    *,
    task: TrainingTask = TrainingTask.BINARY_CLASSIFICATION,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Path]:
    output = Path(output_root)
    if not output.is_absolute():
        cwd_candidate = (Path.cwd() / output).resolve(strict=False)
        repo_candidate = (paths.get_repository_root() / output).resolve(strict=False)
        output = cwd_candidate if paths.is_path_inside(paths.get_generated_root(), cwd_candidate) else repo_candidate
    output = output.resolve(strict=False)
    if not paths.is_path_inside(paths.get_generated_root(), output):
        raise ValueError("synthetic smoke fixtures must be written under generated/")
    output.mkdir(parents=True, exist_ok=True)

    n_classes = 2 if task == TrainingTask.BINARY_CLASSIFICATION else 3
    n_samples = 48 if n_classes == 2 else 60
    x, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    labels = [f"class_{value}" for value in y]
    records = pd.DataFrame(x, columns=[f"feature_{index}" for index in range(4)])
    records.insert(0, "record_id", [f"synthetic-{index:03d}" for index in range(n_samples)])
    records["label"] = labels
    canonical_path = output / ("synthetic_binary.csv" if n_classes == 2 else "synthetic_multiclass.csv")
    if canonical_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite synthetic fixture: {canonical_path}")
    records.to_csv(canonical_path, index=False)

    train_count = int(n_samples * 0.6)
    validation_count = int(n_samples * 0.2)
    train_ids = records["record_id"].iloc[:train_count].tolist()
    validation_ids = records["record_id"].iloc[train_count : train_count + validation_count].tolist()
    test_ids = records["record_id"].iloc[train_count + validation_count :].tolist()
    source_hash = hashing.sha256_file(canonical_path)
    feature_schema = {
        "features": [f"feature_{index}" for index in range(4)],
        "target": "label",
        "note": "Synthetic smoke fixture only; no clinical or suicide-risk labels.",
    }
    preprocessing_hash = hashing.hash_json_data(feature_schema)
    manifest = ModalitySplitManifest(
        manifest_version="1.0.0",
        split_design_version="1.0.0",
        modality="synthetic",
        dataset_name="training-framework-smoke",
        dataset_version="1.0.0",
        preprocessing_version="1.0.0",
        feature_schema_version="1.0.0",
        source_fingerprint=source_hash,
        preprocessing_artifact_hash=preprocessing_hash,
        config_hash=hashing.hash_json_data({"seed": seed, "task": task.value}),
        random_seed=seed,
        split_strategy="random_stratified",
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        excluded_ids={},
        stratify_column="label",
        duplicate_policy="not_applicable_synthetic_unique_ids",
        created_at=utc_now(),
        validation_summary=SplitValidationSummary(
            train_count=len(train_ids),
            validation_count=len(validation_ids),
            test_count=len(test_ids),
            total_count=n_samples,
            train_distribution=records.iloc[:train_count]["label"].value_counts().sort_index().to_dict(),
            validation_distribution=records.iloc[train_count : train_count + validation_count]["label"].value_counts().sort_index().to_dict(),
            test_distribution=records.iloc[train_count + validation_count :]["label"].value_counts().sort_index().to_dict(),
            deterministic_replay_passed=True,
        ),
        notes=["Synthetic framework smoke fixture only; exclude from production/model-selection reports."],
    )
    manifest_path = _write_json(output / ("synthetic_binary_split_manifest.json" if n_classes == 2 else "synthetic_multiclass_split_manifest.json"), manifest.to_safe_dict(), overwrite=overwrite)
    config = TrainingConfig(
        experiment_name="training-framework-smoke",
        experiment_version="1.0.0",
        model_name="synthetic-smoke-baseline",
        model_version="1.0.0" if n_classes == 2 else "1.0.0-multiclass",
        modality="synthetic",
        task=task,
        framework="scikit-learn",
        estimator_type="logistic_regression",
        dataset_name="training-framework-smoke",
        dataset_version="1.0.0",
        preprocessing_version="1.0.0",
        feature_schema_version="1.0.0",
        split_manifest_path=_repo_relative(manifest_path),
        random_seed=seed,
        hyperparameters={"max_iter": 200, "random_state": seed},
        class_weight_policy=None,
        threshold_strategy="max_f1" if n_classes == 2 else "default",
        target_column="label",
        feature_columns=[f"feature_{index}" for index in range(4)],
        excluded_columns=["record_id"],
        sensitive_columns=[],
        primary_metric="f1_weighted",
        secondary_metrics=["balanced_accuracy", "recall_weighted"],
        artifact_subdirectory="synthetic/smoke",
        notes="Synthetic framework smoke fixture only.",
    )
    config_path = _write_json(output / ("training.synthetic.binary.json" if n_classes == 2 else "training.synthetic.multiclass.json"), config.to_safe_dict(), overwrite=overwrite)
    feature_schema_path = _write_json(output / ("synthetic_feature_schema.json" if n_classes == 2 else "synthetic_multiclass_feature_schema.json"), feature_schema, overwrite=overwrite)
    return {
        "canonical_data": canonical_path,
        "split_manifest": manifest_path,
        "config": config_path,
        "feature_schema": feature_schema_path,
    }


def synthetic_estimator_factory(config: TrainingConfig):
    params = {"max_iter": 200, "random_state": config.random_seed}
    params.update(config.hyperparameters)
    return LogisticRegression(**params)
