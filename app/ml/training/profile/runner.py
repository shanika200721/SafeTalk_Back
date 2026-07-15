"""End-to-end runner for the Student Profile depression baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.common import hashing, paths
from app.ml.training.profile.constants import (
    DEFAULT_CANONICAL_DATA,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SPLIT_MANIFEST,
    FEATURE_SETS,
    PROFILE_BASELINE_EXPERIMENT_VERSION,
    PROFILE_MODEL_FAMILY_VERSION,
    PROFILE_TARGET_COLUMN,
    REQUIRED_MODEL_CARD_DISCLAIMER,
    SENSITIVE_CONTEXT_FEATURES,
    SENSITIVE_CONTEXT_FEATURE_SET,
)
from app.ml.training.profile.data import build_profile_training_bundle, resolve_profile_feature_set, validate_profile_feature_policy
from app.ml.training.profile.estimators import create_profile_estimator, profile_candidate_specs
from app.ml.training.profile.evaluation import (
    evaluate_profile_split,
    fairness_exploration,
    feature_interpretation,
    overfitting_gap,
    select_profile_threshold,
)
from app.ml.training.profile.preprocessing import build_profile_preprocessor, transform_profile_features
from app.ml.training.profile.reporting import (
    build_artifact_manifest,
    build_limitations_markdown,
    build_profile_model_card,
    build_summary_markdown,
    file_inventory,
    save_joblib_artifact,
    write_csv,
    write_json,
    write_markdown,
)
from app.ml.training.profile.schemas import CandidateResult, ProfileRunArtifacts
from app.ml.training.reproducibility import capture_environment_versions, set_global_seed


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_profile_training_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_profile_training_config()
    payload = _load_json(path)
    return {**default_profile_training_config(), **payload}


def default_profile_training_config() -> dict[str, Any]:
    return {
        "experiment_name": "profile-depression-baseline",
        "experiment_version": PROFILE_BASELINE_EXPERIMENT_VERSION,
        "model_version": PROFILE_MODEL_FAMILY_VERSION,
        "task": "binary_classification",
        "primary_metric": "validation_recall_then_f1",
        "primary_metric_rationale": "False negatives matter in a screening-style research baseline, but F1 guards against selecting a model that predicts every record as positive.",
        "secondary_metrics": ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"],
        "random_seed": 42,
        "feature_set": "minimal_contextual",
        "feature_sets": FEATURE_SETS,
        "allow_sensitive_context": False,
        "threshold_strategies": ["default", "max_f1", "recall_priority"],
        "min_validation_recall": 0.6,
        "max_candidate_count": 32,
        "hyperparameter_search": {
            "logistic_regression": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"], "max_iter": 500, "random_state": 42},
            "random_forest": {
                "n_estimators": [50, 100],
                "max_depth": [2, 4],
                "min_samples_leaf": [1, 3],
                "class_weight": [None, "balanced"],
                "random_state": 42,
            },
        },
    }


def dry_run_profile_baseline(
    *,
    config_path: str | Path | None = None,
    canonical_data_path: str | Path = DEFAULT_CANONICAL_DATA,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    feature_set: str | None = None,
    allow_sensitive_context: bool = False,
) -> dict[str, Any]:
    config = load_profile_training_config(config_path)
    selected_feature_set, features = resolve_profile_feature_set(config, feature_set=feature_set)
    allow_sensitive = bool(allow_sensitive_context or config.get("allow_sensitive_context"))
    bundle = build_profile_training_bundle(
        canonical_data_path=canonical_data_path,
        split_manifest_path=split_manifest_path,
        source_fingerprint_path=source_fingerprint_path,
        features=features,
        feature_set=selected_feature_set,
        allow_sensitive_context=allow_sensitive,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    validate_profile_feature_policy(features, feature_set=selected_feature_set, allow_sensitive_context=allow_sensitive)
    return {
        "status": "dry_run_ok",
        "feature_set": selected_feature_set,
        "features": features,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "source_fingerprint": bundle.source_fingerprint,
        "preprocessing_artifact_hash": bundle.preprocessing_artifact_hash,
        "split_manifest_hash": bundle.split_manifest.manifest_hash,
    }


def _positive_probabilities(estimator, X) -> list[float]:
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if "yes" not in classes:
        raise ValueError("trained estimator does not expose positive class probability")
    return [float(value) for value in proba[:, classes.index("yes")]]


def _candidate_score(validation_metrics: Mapping[str, Any], train_gap: Mapping[str, Any], *, min_recall: float) -> tuple[Any, ...]:
    recall = validation_metrics.get("recall") or 0.0
    f1 = validation_metrics.get("f1") or 0.0
    balanced = validation_metrics.get("balanced_accuracy") or 0.0
    brier = validation_metrics.get("brier_score")
    gap = abs(train_gap.get("f1_train_minus_validation") or 0.0)
    return (
        1 if recall >= min_recall else 0,
        float(recall),
        float(f1),
        float(balanced),
        -float(brier if brier is not None else 1.0),
        -float(gap),
    )


def _train_candidate(
    *,
    spec,
    bundle,
    threshold_strategy: str,
    min_recall: float,
) -> CandidateResult:
    prep = build_profile_preprocessor(bundle.train, bundle.features, estimator_type=spec.estimator_type)
    X_train = transform_profile_features(prep.preprocessor, bundle.train, bundle.features)
    X_validation = transform_profile_features(prep.preprocessor, bundle.validation, bundle.features)
    y_train = bundle.train[bundle.target].astype(str).tolist()
    y_validation = bundle.validation[bundle.target].astype(str).tolist()

    estimator = create_profile_estimator(spec)
    estimator.fit(X_train, y_train)
    train_prob = _positive_probabilities(estimator, X_train)
    validation_prob = _positive_probabilities(estimator, X_validation)
    threshold_selection = select_profile_threshold(
        y_validation,
        validation_prob,
        strategy=threshold_strategy,
        min_recall=min_recall,
    )
    threshold = float(threshold_selection["threshold"])
    train_metrics = evaluate_profile_split(y_train, train_prob, threshold=threshold, feature_count=prep.feature_count, split_name="train")
    validation_metrics = evaluate_profile_split(
        y_validation,
        validation_prob,
        threshold=threshold,
        feature_count=prep.feature_count,
        split_name="validation",
    )
    gap = overfitting_gap(train_metrics, validation_metrics)
    candidate_id = f"{spec.name}_threshold={threshold_strategy}"
    result = CandidateResult(
        candidate_id=candidate_id,
        spec=spec,
        estimator=estimator,
        preprocessor=prep.preprocessor,
        feature_names=prep.feature_names,
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        threshold_selection=threshold_selection,
        overfitting_gap=gap,
    )
    result.selected_score = _candidate_score(validation_metrics, gap, min_recall=min_recall)
    result.train_metrics["preprocessing"] = {
        "numeric_features": prep.numeric_features,
        "categorical_features": prep.categorical_features,
        "constant_features": prep.constant_features,
        "all_null_features": prep.all_null_features,
        "feature_count_after_encoding": prep.feature_count,
    }
    return result


def _candidate_rows(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "estimator_type": candidate.spec.estimator_type,
                "hyperparameters": candidate.spec.hyperparameters,
                "threshold_strategy": candidate.threshold_strategy,
                "threshold": candidate.threshold,
                "validation_recall": candidate.validation_metrics.get("recall"),
                "validation_f1": candidate.validation_metrics.get("f1"),
                "validation_balanced_accuracy": candidate.validation_metrics.get("balanced_accuracy"),
                "validation_brier_score": candidate.validation_metrics.get("brier_score"),
                "false_negatives_validation": candidate.validation_metrics.get("false_negatives"),
                "overfitting_gap": candidate.overfitting_gap,
            }
        )
    return rows


def _write_reports(
    *,
    report_dir: Path,
    summary: dict[str, Any],
    candidates: list[CandidateResult],
    selected: CandidateResult | None,
    test_metrics: dict[str, Any] | None,
    feature_names: list[str],
    interpretation_rows: list[dict[str, Any]],
    fairness: dict[str, Any],
    files_for_inventory: list[Path],
    overwrite: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["summary_json"] = write_json(report_dir / "profile_baseline_summary.json", summary, overwrite=overwrite)
    outputs["summary_md"] = write_markdown(report_dir / "profile_baseline_summary.md", build_summary_markdown(summary), overwrite=overwrite)
    outputs["candidate_comparison"] = write_csv(report_dir / "profile_candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite)
    outputs["metrics_train"] = write_json(report_dir / "profile_metrics_train.json", selected.train_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_validation"] = write_json(report_dir / "profile_metrics_validation.json", selected.validation_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_test"] = write_json(report_dir / "profile_metrics_test.json", test_metrics or {}, overwrite=overwrite)
    cm = (test_metrics or {}).get("confusion_matrix") or {}
    outputs["confusion_matrix_test"] = write_csv(report_dir / "profile_confusion_matrix_test.csv", [cm], overwrite=overwrite)
    outputs["threshold_selection"] = write_json(report_dir / "profile_threshold_selection.json", selected.threshold_selection if selected else {}, overwrite=overwrite)
    outputs["feature_names"] = write_json(report_dir / "profile_feature_names.json", {"features": feature_names, "count": len(feature_names)}, overwrite=overwrite)
    outputs["feature_interpretation"] = write_csv(report_dir / "profile_feature_interpretation.csv", interpretation_rows, overwrite=overwrite)
    outputs["fairness_exploration"] = write_json(report_dir / "profile_fairness_exploration.json", fairness, overwrite=overwrite)
    outputs["limitations"] = write_markdown(report_dir / "profile_limitations.md", build_limitations_markdown(), overwrite=overwrite)
    inventory = file_inventory([*files_for_inventory, *outputs.values()])
    outputs["artifact_inventory"] = write_json(report_dir / "profile_artifact_inventory.json", inventory, overwrite=overwrite)
    return outputs


def run_profile_baseline(
    *,
    config_path: str | Path | None = None,
    canonical_data_path: str | Path = DEFAULT_CANONICAL_DATA,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    feature_set: str | None = None,
    candidate: str = "all",
    threshold_strategy: str | None = None,
    dry_run: bool = False,
    allow_sensitive_context: bool = False,
    overwrite: bool = False,
    register_candidate: bool = False,
    test_database_url: str | None = None,
) -> ProfileRunArtifacts | dict[str, Any]:
    config = load_profile_training_config(config_path)
    selected_feature_set, features = resolve_profile_feature_set(config, feature_set=feature_set)
    allow_sensitive = bool(allow_sensitive_context or config.get("allow_sensitive_context"))
    if selected_feature_set == SENSITIVE_CONTEXT_FEATURE_SET and not allow_sensitive:
        raise ValueError("sensitive-context feature set requires explicit exploratory flag")
    if dry_run:
        return dry_run_profile_baseline(
            config_path=config_path,
            canonical_data_path=canonical_data_path,
            split_manifest_path=split_manifest_path,
            source_fingerprint_path=source_fingerprint_path,
            feature_set=selected_feature_set,
            allow_sensitive_context=allow_sensitive,
        )

    set_global_seed(int(config.get("random_seed", 42)))
    bundle = build_profile_training_bundle(
        canonical_data_path=canonical_data_path,
        split_manifest_path=split_manifest_path,
        source_fingerprint_path=source_fingerprint_path,
        features=features,
        feature_set=selected_feature_set,
        allow_sensitive_context=allow_sensitive,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    strategies = [threshold_strategy] if threshold_strategy else list(config.get("threshold_strategies") or ["default"])
    specs = profile_candidate_specs(config, candidate=candidate)
    candidates: list[CandidateResult] = []
    min_recall = float(config.get("min_validation_recall", 0.6))
    for spec in specs:
        for strategy in strategies:
            candidates.append(_train_candidate(spec=spec, bundle=bundle, threshold_strategy=strategy, min_recall=min_recall))
    candidates.sort(key=lambda item: item.selected_score, reverse=True)
    selected = candidates[0] if candidates else None

    test_metrics = None
    feature_names: list[str] = []
    interpretation_rows: list[dict[str, Any]] = []
    fairness: dict[str, Any] = {"enabled": False, "reason": "no selected candidate"}
    if selected is not None:
        X_test = transform_profile_features(selected.preprocessor, bundle.test, bundle.features)
        test_prob = _positive_probabilities(selected.estimator, X_test)
        test_metrics = evaluate_profile_split(
            bundle.test[bundle.target].astype(str).tolist(),
            test_prob,
            threshold=selected.threshold,
            feature_count=len(selected.feature_names),
            split_name="test",
        )
        feature_names = selected.feature_names
        interpretation_rows = feature_interpretation(selected.estimator, feature_names)
        sensitive_columns = sorted(SENSITIVE_CONTEXT_FEATURES & set(features)) if allow_sensitive else []
        fairness = fairness_exploration(
            bundle.test,
            bundle.test[bundle.target].astype(str).tolist(),
            test_prob,
            threshold=selected.threshold,
            sensitive_columns=sensitive_columns,
        )

    config_hash = hashing.hash_json_data({**config, "feature_set": selected_feature_set, "features": features, "candidate": candidate, "threshold_strategy": threshold_strategy})
    run_id = f"profile-{selected_feature_set}-{config_hash[:12]}"
    model_name = selected.spec.estimator_type.replace("_", "-") if selected else "profile-depression-no-selected-candidate"
    model_name = f"profile-depression-{model_name}"
    model_version = str(config.get("model_version", PROFILE_MODEL_FAMILY_VERSION))
    run_dir = _resolve(model_root) / "profile" / model_name / model_version / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_files: list[Path] = []
    model_card = build_profile_model_card(
        model_name=model_name,
        model_version=model_version,
        feature_set=selected_feature_set,
        selected_candidate=selected.candidate_id if selected else None,
        threshold_strategy=selected.threshold_strategy if selected else None,
        threshold=selected.threshold if selected else None,
        metrics={"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}},
    )
    if REQUIRED_MODEL_CARD_DISCLAIMER not in model_card:
        raise ValueError("Profile model card is missing required disclaimer")

    if selected is not None:
        artifact_files.append(save_joblib_artifact(selected.estimator, run_dir / "model.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(selected.preprocessor, run_dir / "preprocessor.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(Pipeline([("preprocessor", selected.preprocessor), ("model", selected.estimator)]), run_dir / "pipeline.joblib", overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "training_config.json", {**config, "selected_feature_set": selected_feature_set, "features": features}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "metrics.json", {"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}}, overwrite=overwrite))
    artifact_files.append(write_csv(run_dir / "candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "feature_schema.json", {"source": str(feature_schema_path), "selected_features": features, "transformed_feature_names": feature_names}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "split_manifest_reference.json", bundle.split_manifest.payload, overwrite=overwrite))
    artifact_files.append(write_markdown(run_dir / "model_card.md", model_card, overwrite=overwrite))
    artifact_files.append(
        write_json(
            run_dir / "reproducibility_report.json",
            {
                "environment": capture_environment_versions(),
                "source_fingerprint": bundle.source_fingerprint,
                "preprocessing_artifact_hash": bundle.preprocessing_artifact_hash,
                "split_manifest_hash": bundle.split_manifest.manifest_hash,
                "config_hash": config_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=overwrite,
        )
    )
    manifest = build_artifact_manifest(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        feature_set=selected_feature_set,
        files=artifact_files,
        split_manifest_hash=bundle.split_manifest.manifest_hash,
        source_fingerprint=bundle.source_fingerprint,
        preprocessing_artifact_hash=bundle.preprocessing_artifact_hash,
        config_hash=config_hash,
    )
    manifest_path = write_json(run_dir / "artifact_manifest.json", manifest, overwrite=overwrite)
    artifact_files.append(manifest_path)

    summary = {
        "profile_model_family_version": PROFILE_MODEL_FAMILY_VERSION,
        "experiment_version": PROFILE_BASELINE_EXPERIMENT_VERSION,
        "feature_set": selected_feature_set,
        "features": features,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "candidate_count": len(candidates),
        "selected_candidate": {
            "candidate_id": selected.candidate_id,
            "estimator_type": selected.spec.estimator_type,
            "hyperparameters": selected.spec.hyperparameters,
        }
        if selected
        else None,
        "selection_rationale": "Validation-only hierarchy: minimum recall, validation F1, balanced accuracy, calibration, lower overfitting gap, and interpretability.",
        "threshold_strategy": selected.threshold_strategy if selected else None,
        "selected_threshold": selected.threshold if selected else None,
        "train_metrics": selected.train_metrics if selected else {},
        "validation_metrics": selected.validation_metrics if selected else {},
        "test_metrics": test_metrics or {},
        "overfitting_gap": selected.overfitting_gap if selected else {},
        "fairness_summary": fairness,
        "artifact_path": str(run_dir),
        "model_card_path": str(run_dir / "model_card.md"),
        "reproducibility_report_path": str(run_dir / "reproducibility_report.json"),
        "candidate_registration_occurred": False,
        "model_became_active": False,
        "research_readiness_decision": "research baseline only; not clinically validated and not deployable",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    report_outputs = _write_reports(
        report_dir=_resolve(report_dir),
        summary=summary,
        candidates=candidates,
        selected=selected,
        test_metrics=test_metrics,
        feature_names=feature_names,
        interpretation_rows=interpretation_rows,
        fairness=fairness,
        files_for_inventory=artifact_files,
        overwrite=overwrite,
    )
    registered = False
    if register_candidate:
        if not test_database_url or not test_database_url.startswith("sqlite"):
            raise ValueError("candidate registration requires an isolated sqlite test database URL in this step")
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.db.base import Base
        from app.ml.training.registry import register_candidate_model

        engine = create_engine(test_database_url)
        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()
        try:
            payload = {
                "model_name": model_name,
                "modality": "profile",
                "version": model_version,
                "framework": "scikit-learn",
                "artifact_path": str((run_dir / "pipeline.joblib").relative_to(paths.get_repository_root())).replace("\\", "/"),
                "preprocessing_path": str((run_dir / "preprocessor.joblib").relative_to(paths.get_repository_root())).replace("\\", "/") if selected else None,
                "dataset_version": "v1",
                "feature_schema_version": "1.0.0",
                "metrics_json": {"validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}},
                "thresholds_json": selected.threshold_selection if selected else {},
                "is_active": False,
            }
            register_candidate_model(session, payload)
            session.commit()
            registered = True
            summary["candidate_registration_occurred"] = True
        finally:
            session.close()

    return ProfileRunArtifacts(
        run_id=run_id,
        report_dir=_resolve(report_dir),
        run_dir=run_dir,
        selected_candidate=selected,
        metrics={"summary": summary, "test": test_metrics or {}, "reports": {key: str(value) for key, value in report_outputs.items()}},
        artifact_manifest=manifest,
        registered=registered,
    )
