"""Runner for the restricted Phase 3I Face research baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from app.ml.common import hashing, paths
from app.ml.training.face.constants import (
    DEFAULT_DEDUPLICATED_MANIFEST,
    DEFAULT_DUPLICATE_DECISIONS,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_MODEL_ROOT,
    DEFAULT_QUARANTINE,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SPLIT_ASSIGNMENTS,
    DEFAULT_SPLIT_MANIFEST,
    FACE_BASELINE_EXPERIMENT_VERSION,
    FACE_IMAGE_PIPELINE_VERSION,
    FACE_MODEL_FAMILY_VERSION,
    REQUIRED_MODEL_CARD_DISCLAIMER,
)
from app.ml.training.face.data import build_face_training_bundle, load_face_images_for_split, validate_face_training_contract
from app.ml.training.face.estimators import create_face_estimator, face_candidate_specs
from app.ml.training.face.evaluation import _prediction_scores, error_analysis_rows, evaluate_face_split, selection_score, train_validation_gap
from app.ml.training.face.preprocessing import fit_train_only_scaler, transform_with_scaler
from app.ml.training.face.reporting import (
    artifact_manifest,
    build_face_model_card,
    build_summary_markdown,
    file_inventory,
    limitations_markdown,
    save_joblib_artifact,
    write_csv,
    write_json,
    write_markdown,
)
from app.ml.training.face.schemas import FaceCandidateResult, FaceRunArtifacts
from app.ml.training.reproducibility import capture_environment_versions, set_global_seed


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def default_face_training_config() -> dict[str, Any]:
    return {
        "experiment_name": "face-emotion-restricted-baseline",
        "experiment_version": FACE_BASELINE_EXPERIMENT_VERSION,
        "model_version": FACE_MODEL_FAMILY_VERSION,
        "image_pipeline_version": FACE_IMAGE_PIPELINE_VERSION,
        "random_seed": 43107,
        "feature_set": "flattened_pixels",
        "max_candidate_count": 8,
        "hyperparameter_search": {
            "logistic_regression": {"C": [0.1], "class_weight": [None, "balanced"], "max_iter": 120, "random_state": 43107},
            "linear_svm": {"C": [0.1], "class_weight": [None, "balanced"], "max_iter": 1000, "random_state": 43107},
            "random_forest": {"n_estimators": [40], "max_depth": [10], "min_samples_leaf": 2, "class_weight": [None], "random_state": 43107, "n_jobs": 1},
        },
        "reviewer_independence_status": "reviewer_independence_unverified",
    }


def load_face_training_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_face_training_config()
    payload = _load_json(path)
    merged = default_face_training_config()
    for key, value in payload.items():
        if key == "hyperparameter_search":
            hyper = dict(merged["hyperparameter_search"])
            for model_key, model_value in value.items():
                hyper[model_key] = {**hyper.get(model_key, {}), **model_value}
            merged[key] = hyper
        else:
            merged[key] = value
    return merged


def dry_run_face_baseline(
    *,
    deduplicated_manifest_path: str | Path = DEFAULT_DEDUPLICATED_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS,
    quarantine_path: str | Path = DEFAULT_QUARANTINE,
    duplicate_decisions_path: str | Path = DEFAULT_DUPLICATE_DECISIONS,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    require_replay: bool = True,
) -> dict[str, Any]:
    contract = validate_face_training_contract(
        deduplicated_manifest_path=deduplicated_manifest_path,
        split_manifest_path=split_manifest_path,
        split_assignments_path=split_assignments_path,
        quarantine_path=quarantine_path,
        duplicate_decisions_path=duplicate_decisions_path,
        source_fingerprint_path=source_fingerprint_path,
        require_replay=require_replay,
    )
    return {"status": "dry_run_ok", **contract}


def _train_candidate(spec, train_bundle, validation_bundle) -> FaceCandidateResult:
    estimator = create_face_estimator(spec)
    scaler = fit_train_only_scaler(train_bundle.X.copy()) if spec.scale_features else None
    X_train = transform_with_scaler(scaler, train_bundle.X.copy())
    X_validation = transform_with_scaler(scaler, validation_bundle.X.copy())
    estimator.fit(X_train, train_bundle.y)
    train_metrics = evaluate_face_split(estimator, X_train, train_bundle.y, split_name="train")
    validation_metrics = evaluate_face_split(estimator, X_validation, validation_bundle.y, split_name="validation")
    gap = train_validation_gap(train_metrics, validation_metrics)
    train_metrics["train_validation_gap"] = gap
    return FaceCandidateResult(
        candidate_id=f"{spec.name}_{spec.feature_set}",
        spec=spec,
        estimator=estimator,
        preprocessor=scaler,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        selected_score=selection_score(validation_metrics, gap, spec.estimator_type),
    )


def _candidate_rows(candidates: list[FaceCandidateResult]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "estimator_type": item.spec.estimator_type,
                "feature_set": item.spec.feature_set,
                "hyperparameters": item.spec.hyperparameters,
                "validation_macro_f1": item.validation_metrics.get("macro_f1"),
                "validation_macro_recall": item.validation_metrics.get("macro_recall"),
                "validation_balanced_accuracy": item.validation_metrics.get("balanced_accuracy"),
                "validation_minimum_class_recall": item.validation_metrics.get("minimum_class_recall"),
                "validation_disgust_recall": item.validation_metrics.get("disgust_recall"),
                "train_validation_gap": item.train_metrics.get("train_validation_gap"),
                "selected_score": item.selected_score,
            }
        )
    return rows


def _confusion_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    labels = metrics.get("labels", [])
    matrix = metrics.get("confusion_matrix", [])
    return [{"true_label": label, **{f"pred_{pred}": int(matrix[i][j]) for j, pred in enumerate(labels)}} for i, label in enumerate(labels)]


def _per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"label": label, **values} for label, values in (metrics.get("per_class") or {}).items()]


def run_face_baseline(
    *,
    config_path: str | Path | None = None,
    ablation_config_path: str | Path | None = None,
    deduplicated_manifest_path: str | Path = DEFAULT_DEDUPLICATED_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS,
    quarantine_path: str | Path = DEFAULT_QUARANTINE,
    duplicate_decisions_path: str | Path = DEFAULT_DUPLICATE_DECISIONS,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    candidate: str = "all",
    feature_set: str | None = None,
    dry_run: bool = False,
    max_train_records: int | None = None,
    overwrite: bool = False,
    register_candidate: bool = False,
    test_database_url: str | None = None,
    require_replay: bool = True,
) -> FaceRunArtifacts | dict[str, Any]:
    if register_candidate:
        if not test_database_url or not test_database_url.startswith("sqlite"):
            raise ValueError("Face candidate registration requires isolated sqlite test database URL and remains inactive")
    config = load_face_training_config(config_path)
    if ablation_config_path:
        config["ablation_config_hash"] = hashing.sha256_file(ablation_config_path, allow_outside_project=True)
    selected_feature_set = feature_set or str(config.get("feature_set", "flattened_pixels"))
    if dry_run:
        return dry_run_face_baseline(
            deduplicated_manifest_path=deduplicated_manifest_path,
            split_manifest_path=split_manifest_path,
            split_assignments_path=split_assignments_path,
            quarantine_path=quarantine_path,
            duplicate_decisions_path=duplicate_decisions_path,
            source_fingerprint_path=source_fingerprint_path,
            require_replay=require_replay,
        )
    set_global_seed(int(config.get("random_seed", 43107)))
    bundle = build_face_training_bundle(
        deduplicated_manifest_path=deduplicated_manifest_path,
        split_manifest_path=split_manifest_path,
        split_assignments_path=split_assignments_path,
        quarantine_path=quarantine_path,
        duplicate_decisions_path=duplicate_decisions_path,
        source_fingerprint_path=source_fingerprint_path,
        max_train_records=max_train_records,
        require_replay=require_replay,
    )
    train_images = load_face_images_for_split(bundle.train, feature_set=selected_feature_set)
    validation_images = load_face_images_for_split(bundle.validation, feature_set=selected_feature_set)
    test_images = load_face_images_for_split(bundle.test, feature_set=selected_feature_set)

    candidates = [_train_candidate(spec, train_images, validation_images) for spec in face_candidate_specs(config, candidate=candidate, feature_set=selected_feature_set)]
    candidates.sort(key=lambda item: item.selected_score, reverse=True)
    selected = candidates[0] if candidates else None

    test_metrics: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    selected_pipeline = None
    if selected:
        X_test = transform_with_scaler(selected.preprocessor, test_images.X.copy())
        test_metrics = evaluate_face_split(selected.estimator, X_test, test_images.y, split_name="test")
        y_pred, confidence = _prediction_scores(selected.estimator, X_test)
        error_rows = error_analysis_rows(test_images.rows, test_images.y, y_pred, confidence)
        selected_pipeline = Pipeline([("preprocessor", selected.preprocessor), ("model", selected.estimator)])

    config_hash = hashing.hash_json_data({**config, "candidate": candidate, "feature_set": selected_feature_set, "max_train_records": max_train_records})
    run_id = f"face-{selected_feature_set}-{config_hash[:12]}"
    model_name = f"face-emotion-{selected.spec.estimator_type if selected else 'no-selected-candidate'}".replace("_", "-")
    model_version = str(config.get("model_version", FACE_MODEL_FAMILY_VERSION))
    run_dir = _resolve(model_root) / "face" / model_name / model_version / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = _resolve(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "face_model_family_version": FACE_MODEL_FAMILY_VERSION,
        "experiment_version": FACE_BASELINE_EXPERIMENT_VERSION,
        "image_pipeline_version": FACE_IMAGE_PIPELINE_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
        "feature_set": selected_feature_set,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "contract": bundle.contract,
        "candidate_count": len(candidates),
        "selected_candidate": {"candidate_id": selected.candidate_id, "estimator_type": selected.spec.estimator_type, "hyperparameters": selected.spec.hyperparameters}
        if selected
        else None,
        "selection_rationale": "Validation-only hierarchy: macro F1, macro recall, balanced accuracy, minimum-class recall, disgust recall, lower train-validation gap, simplicity, inference cost.",
        "train_metrics": selected.train_metrics if selected else {},
        "validation_metrics": selected.validation_metrics if selected else {},
        "test_metrics": test_metrics,
        "reviewer_independence_status": "reviewer_independence_unverified",
        "registration_status": "not_registered",
        "activation_status": "inactive",
        "research_readiness_decision": "restricted research baseline only; not deployable and not clinical",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    model_card = build_face_model_card(summary)
    if REQUIRED_MODEL_CARD_DISCLAIMER not in model_card:
        raise ValueError("Face model card missing required disclaimer")

    artifact_files: list[Path] = []
    if selected:
        artifact_files.append(save_joblib_artifact(selected.estimator, run_dir / "model.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(selected.preprocessor, run_dir / "preprocessor.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(selected_pipeline, run_dir / "pipeline.joblib", overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "training_config.json", {**config, "feature_set": selected_feature_set}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "metrics.json", {"train": summary["train_metrics"], "validation": summary["validation_metrics"], "test": test_metrics}, overwrite=overwrite))
    artifact_files.append(write_csv(run_dir / "candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "feature_schema.json", {"source": str(feature_schema_path), "feature_set": selected_feature_set, "feature_names": train_images.feature_names}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "split_manifest_reference.json", bundle.split_manifest, overwrite=overwrite))
    artifact_files.append(write_markdown(run_dir / "model_card.md", model_card, overwrite=overwrite))
    artifact_files.append(
        write_json(
            run_dir / "reproducibility_report.json",
            {
                "environment": capture_environment_versions(),
                "source_fingerprint": bundle.source_fingerprint,
                "split_manifest_hash": bundle.split_manifest_hash,
                "deduplicated_manifest_hash": bundle.deduplicated_manifest_hash,
                "config_hash": config_hash,
                "reviewer_independence_status": "reviewer_independence_unverified",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=overwrite,
        )
    )
    manifest = artifact_manifest(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        files=artifact_files,
        split_manifest_hash=bundle.split_manifest_hash,
        source_fingerprint=bundle.source_fingerprint,
        config_hash=config_hash,
    )
    manifest_path = write_json(run_dir / "artifact_manifest.json", manifest, overwrite=overwrite)
    artifact_files.append(manifest_path)

    report_outputs: list[Path] = []
    report_outputs.append(write_json(report_path / "face_baseline_summary.json", summary, overwrite=overwrite))
    report_outputs.append(write_markdown(report_path / "face_baseline_summary.md", build_summary_markdown(summary), overwrite=overwrite))
    report_outputs.append(write_csv(report_path / "face_candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite))
    report_outputs.append(write_json(report_path / "face_metrics_train.json", summary["train_metrics"], overwrite=overwrite))
    report_outputs.append(write_json(report_path / "face_metrics_validation.json", summary["validation_metrics"], overwrite=overwrite))
    report_outputs.append(write_json(report_path / "face_metrics_test.json", test_metrics, overwrite=overwrite))
    report_outputs.append(write_csv(report_path / "face_confusion_matrix_test.csv", _confusion_rows(test_metrics), overwrite=overwrite))
    report_outputs.append(write_csv(report_path / "face_per_class_metrics_test.csv", _per_class_rows(test_metrics), overwrite=overwrite))
    report_outputs.append(write_csv(report_path / "face_error_analysis.csv", error_rows, overwrite=overwrite))
    report_outputs.append(
        write_json(
            report_path / "face_class_imbalance_analysis.json",
            {
                "label_distributions": bundle.contract["label_distributions"],
                "disgust_low_support": True,
                "minimum_class_recall": test_metrics.get("minimum_class_recall"),
                "worst_performing_class": test_metrics.get("worst_performing_class"),
            },
            overwrite=overwrite,
        )
    )
    report_outputs.append(write_markdown(report_path / "face_review_governance_limitations.md", limitations_markdown(), overwrite=overwrite))
    report_outputs.append(write_markdown(report_path / "face_dataset_limitations.md", limitations_markdown(), overwrite=overwrite))
    inventory = file_inventory([*artifact_files, *report_outputs])
    report_outputs.append(write_json(report_path / "face_artifact_inventory.json", inventory, overwrite=overwrite))

    return FaceRunArtifacts(
        run_id=run_id,
        report_dir=report_path,
        run_dir=run_dir,
        selected_candidate=selected,
        metrics={"summary": summary, "reports": {path.name: str(path) for path in report_outputs}},
        artifact_manifest=manifest,
        registered=False,
    )
