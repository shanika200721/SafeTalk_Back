"""End-to-end runner for the Phase 3D Text baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.common import hashing, paths
from app.ml.training.reproducibility import capture_environment_versions, set_global_seed
from app.ml.training.text.constants import (
    DEFAULT_CANONICAL_DATA,
    DEFAULT_CONFLICT_QUARANTINE,
    DEFAULT_DUPLICATE_MANIFEST,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SOURCE_OVERLAP_REPORT,
    DEFAULT_SPLIT_MANIFEST,
    REQUIRED_MODEL_CARD_DISCLAIMER,
    TEXT_BASELINE_EXPERIMENT_VERSION,
    TEXT_FEATURE_SET,
    TEXT_LABELS,
    TEXT_MODEL_FAMILY_VERSION,
    TEXT_RECORD_ID_COLUMN,
    TEXT_TARGET_COLUMN,
    TEXT_TEXT_COLUMN,
    TEXT_VECTORIZER_VERSION,
)
from app.ml.training.text.data import build_text_training_bundle
from app.ml.training.text.estimators import create_text_estimator, text_candidate_specs
from app.ml.training.text.evaluation import (
    aggregate_feature_interpretation,
    confusion_matrix_rows,
    evaluate_text_split,
    per_class_metric_rows,
    predict_with_scores,
    privacy_safe_error_analysis,
    train_validation_gap,
)
from app.ml.training.text.preprocessing import fit_text_vectorizer, transform_text_features, vectorizer_summary
from app.ml.training.text.reporting import (
    build_artifact_manifest,
    build_dataset_limitations_markdown,
    build_summary_markdown,
    build_text_model_card,
    file_inventory,
    save_joblib_artifact,
    write_csv,
    write_json,
    write_markdown,
)
from app.ml.training.text.schemas import CandidateResult, TextRunArtifacts


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_text_training_config() -> dict[str, Any]:
    return {
        "experiment_name": "text-classification-baseline",
        "experiment_version": TEXT_BASELINE_EXPERIMENT_VERSION,
        "model_name": "text-classification-baseline",
        "model_version": TEXT_MODEL_FAMILY_VERSION,
        "task": "multiclass_classification",
        "primary_metric": "validation_suicidal_recall_then_macro_f1",
        "primary_metric_rationale": "Prioritize suicidal-class recall, then macro F1 and balanced performance.",
        "random_seed": 42,
        "feature_set": TEXT_FEATURE_SET,
        "min_validation_suicidal_recall": 0.7,
        "max_candidate_count": 24,
        "vectorizers": [
            {
                "name": "word_tfidf",
                "kind": "word",
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.95,
                "max_features": 20000,
                "sublinear_tf": True,
            }
        ],
        "hyperparameter_search": {
            "logistic_regression": {"C": [0.5, 1.0, 2.0], "class_weight": [None, "balanced"], "max_iter": 300, "solver": "liblinear", "random_state": 42},
            "linear_svm": {"C": [0.5, 1.0, 2.0], "class_weight": [None, "balanced"], "max_iter": 3000, "random_state": 42},
        },
    }


def load_text_training_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_text_training_config()
    payload = _load_json(path)
    merged = default_text_training_config()
    merged.update(payload)
    return merged


def _apply_train_limit(bundle, max_train_records: int | None):
    if not max_train_records:
        return bundle
    if max_train_records <= 0:
        raise ValueError("max_train_records must be positive")
    limited = bundle.train.groupby(TEXT_TARGET_COLUMN, group_keys=False).apply(
        lambda group: group.sort_values(TEXT_RECORD_ID_COLUMN).head(max(1, int(max_train_records / len(TEXT_LABELS))))
    )
    if len(limited) > max_train_records:
        limited = limited.sort_values(TEXT_RECORD_ID_COLUMN).head(max_train_records)
    return type(bundle)(
        train=limited.reset_index(drop=True),
        validation=bundle.validation,
        test=bundle.test,
        text_column=bundle.text_column,
        target=bundle.target,
        split_manifest=bundle.split_manifest,
        source_fingerprint=bundle.source_fingerprint,
        preprocessing_artifact_hash=bundle.preprocessing_artifact_hash,
        duplicate_manifest=bundle.duplicate_manifest,
        source_overlap_report=bundle.source_overlap_report,
    )


def dry_run_text_baseline(
    *,
    config_path: str | Path | None = None,
    canonical_data_path: str | Path = DEFAULT_CANONICAL_DATA,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    duplicate_manifest_path: str | Path = DEFAULT_DUPLICATE_MANIFEST,
    conflict_quarantine_path: str | Path = DEFAULT_CONFLICT_QUARANTINE,
    source_overlap_report_path: str | Path = DEFAULT_SOURCE_OVERLAP_REPORT,
) -> dict[str, Any]:
    config = load_text_training_config(config_path)
    bundle = build_text_training_bundle(
        canonical_data_path=canonical_data_path,
        split_manifest_path=split_manifest_path,
        source_fingerprint_path=source_fingerprint_path,
        duplicate_manifest_path=duplicate_manifest_path,
        conflict_quarantine_path=conflict_quarantine_path,
        source_overlap_report_path=source_overlap_report_path,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    return {
        "status": "dry_run_ok",
        "feature_set": TEXT_FEATURE_SET,
        "text_column": TEXT_TEXT_COLUMN,
        "target_column": TEXT_TARGET_COLUMN,
        "labels": TEXT_LABELS,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "source_fingerprint": bundle.source_fingerprint,
        "preprocessing_artifact_hash": bundle.preprocessing_artifact_hash,
        "split_manifest_hash": bundle.split_manifest.manifest_hash,
        "source_overlap_count": bundle.source_overlap_report.get("exact_overlap_count"),
    }


def _candidate_score(validation_metrics: Mapping[str, Any], gap: Mapping[str, Any], *, min_suicidal_recall: float) -> tuple[Any, ...]:
    suicidal = validation_metrics.get("suicidal_class") or {}
    suicidal_recall = suicidal.get("recall") or 0.0
    macro_f1 = validation_metrics.get("macro_f1") or 0.0
    macro_recall = validation_metrics.get("macro_recall") or 0.0
    balanced = validation_metrics.get("balanced_accuracy") or 0.0
    suicidal_f1 = suicidal.get("f1") or 0.0
    gap_value = abs(gap.get("macro_f1_train_minus_validation") or 0.0)
    return (
        1 if suicidal_recall >= min_suicidal_recall else 0,
        float(suicidal_recall),
        float(macro_f1),
        float(macro_recall),
        float(balanced),
        float(suicidal_f1),
        -float(gap_value),
    )


def _train_candidate(*, spec, vectorizer_config: Mapping[str, Any], bundle) -> CandidateResult:
    vec_result = fit_text_vectorizer(bundle.train[bundle.text_column], vectorizer_config)
    X_train = transform_text_features(vec_result.vectorizer, bundle.train[bundle.text_column])
    X_validation = transform_text_features(vec_result.vectorizer, bundle.validation[bundle.text_column])
    y_train = bundle.train[bundle.target].astype(str).tolist()
    y_validation = bundle.validation[bundle.target].astype(str).tolist()
    estimator = create_text_estimator(spec)
    estimator.fit(X_train, y_train)
    train_pred, train_scores, train_score_kind = predict_with_scores(estimator, X_train)
    validation_pred, validation_scores, validation_score_kind = predict_with_scores(estimator, X_validation)
    train_metrics = evaluate_text_split(y_train, train_pred, scores=train_scores, score_kind=train_score_kind, split_name="train")
    validation_metrics = evaluate_text_split(y_validation, validation_pred, scores=validation_scores, score_kind=validation_score_kind, split_name="validation")
    gap = train_validation_gap(train_metrics, validation_metrics)
    vectorizer_name = str(vectorizer_config.get("name") or vectorizer_config.get("kind") or "tfidf")
    result = CandidateResult(
        candidate_id=f"{vectorizer_name}_{spec.name}",
        spec=spec,
        vectorizer_name=vectorizer_name,
        vectorizer_config=dict(vectorizer_config),
        estimator=estimator,
        vectorizer=vec_result.vectorizer,
        feature_names=vec_result.feature_names,
        vocabulary_hash=vec_result.vocabulary_hash,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        overfitting_gap=gap,
    )
    return result


def _candidate_rows(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "estimator_type": candidate.spec.estimator_type,
            "hyperparameters": candidate.spec.hyperparameters,
            "vectorizer_name": candidate.vectorizer_name,
            "vectorizer_config": candidate.vectorizer_config,
            "feature_count": len(candidate.feature_names),
            "vocabulary_hash": candidate.vocabulary_hash,
            "validation_macro_f1": candidate.validation_metrics.get("macro_f1"),
            "validation_macro_recall": candidate.validation_metrics.get("macro_recall"),
            "validation_balanced_accuracy": candidate.validation_metrics.get("balanced_accuracy"),
            "validation_suicidal_recall": (candidate.validation_metrics.get("suicidal_class") or {}).get("recall"),
            "validation_suicidal_f1": (candidate.validation_metrics.get("suicidal_class") or {}).get("f1"),
            "validation_suicidal_false_negatives": (candidate.validation_metrics.get("suicidal_class") or {}).get("false_negatives"),
            "overfitting_gap": candidate.overfitting_gap,
        }
        for candidate in candidates
    ]


def _write_reports(
    *,
    report_dir: Path,
    summary: dict[str, Any],
    candidates: list[CandidateResult],
    selected: CandidateResult | None,
    test_metrics: dict[str, Any] | None,
    confusion_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    suicidal_analysis: dict[str, Any],
    vectorizer_payload: dict[str, Any],
    interpretation_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    files_for_inventory: list[Path],
    overwrite: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["summary_json"] = write_json(report_dir / "text_baseline_summary.json", summary, overwrite=overwrite)
    outputs["summary_md"] = write_markdown(report_dir / "text_baseline_summary.md", build_summary_markdown(summary), overwrite=overwrite)
    outputs["candidate_comparison"] = write_csv(report_dir / "text_candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite)
    outputs["metrics_train"] = write_json(report_dir / "text_metrics_train.json", selected.train_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_validation"] = write_json(report_dir / "text_metrics_validation.json", selected.validation_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_test"] = write_json(report_dir / "text_metrics_test.json", test_metrics or {}, overwrite=overwrite)
    outputs["confusion_matrix_test"] = write_csv(report_dir / "text_confusion_matrix_test.csv", confusion_rows, overwrite=overwrite)
    outputs["per_class_metrics_test"] = write_csv(report_dir / "text_per_class_metrics_test.csv", per_class_rows, overwrite=overwrite)
    outputs["suicidal_analysis"] = write_json(report_dir / "text_suicidal_class_analysis.json", suicidal_analysis, overwrite=overwrite)
    outputs["vectorizer_summary"] = write_json(report_dir / "text_vectorizer_summary.json", vectorizer_payload, overwrite=overwrite)
    outputs["vocabulary_hash"] = write_json(report_dir / "text_vocabulary_hash.json", {"vocabulary_hash": vectorizer_payload.get("vocabulary_hash")}, overwrite=overwrite)
    outputs["feature_interpretation"] = write_csv(report_dir / "text_feature_interpretation.csv", interpretation_rows, overwrite=overwrite)
    outputs["error_analysis"] = write_csv(report_dir / "text_error_analysis.csv", error_rows, overwrite=overwrite)
    outputs["limitations"] = write_markdown(report_dir / "text_dataset_limitations.md", build_dataset_limitations_markdown(), overwrite=overwrite)
    outputs["artifact_inventory"] = write_json(report_dir / "text_artifact_inventory.json", file_inventory([*files_for_inventory, *outputs.values()]), overwrite=overwrite)
    return outputs


def run_text_baseline(
    *,
    config_path: str | Path | None = None,
    canonical_data_path: str | Path = DEFAULT_CANONICAL_DATA,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    source_fingerprint_path: str | Path = DEFAULT_SOURCE_FINGERPRINT,
    duplicate_manifest_path: str | Path = DEFAULT_DUPLICATE_MANIFEST,
    conflict_quarantine_path: str | Path = DEFAULT_CONFLICT_QUARANTINE,
    source_overlap_report_path: str | Path = DEFAULT_SOURCE_OVERLAP_REPORT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    feature_set: str | None = None,
    candidate: str = "all",
    dry_run: bool = False,
    max_train_records: int | None = None,
    overwrite: bool = False,
    register_candidate: bool = False,
    test_database_url: str | None = None,
) -> TextRunArtifacts | dict[str, Any]:
    if feature_set and feature_set != TEXT_FEATURE_SET:
        raise ValueError("Text baseline supports only normalized_text_tfidf")
    config = load_text_training_config(config_path)
    if dry_run:
        return dry_run_text_baseline(
            config_path=config_path,
            canonical_data_path=canonical_data_path,
            split_manifest_path=split_manifest_path,
            source_fingerprint_path=source_fingerprint_path,
            duplicate_manifest_path=duplicate_manifest_path,
            conflict_quarantine_path=conflict_quarantine_path,
            source_overlap_report_path=source_overlap_report_path,
        )

    set_global_seed(int(config.get("random_seed", 42)))
    bundle = build_text_training_bundle(
        canonical_data_path=canonical_data_path,
        split_manifest_path=split_manifest_path,
        source_fingerprint_path=source_fingerprint_path,
        duplicate_manifest_path=duplicate_manifest_path,
        conflict_quarantine_path=conflict_quarantine_path,
        source_overlap_report_path=source_overlap_report_path,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    bundle = _apply_train_limit(bundle, max_train_records)
    specs = text_candidate_specs(config, candidate=candidate)
    vectorizers = [dict(value) for value in config.get("vectorizers", [])]
    if not vectorizers:
        raise ValueError("Text training config must define at least one vectorizer")
    total = len(specs) * len(vectorizers)
    if total > int(config.get("max_candidate_count", 24)):
        raise ValueError("bounded Text candidate/vectorizer grid exceeded max_candidate_count")

    candidates: list[CandidateResult] = []
    skipped: list[dict[str, Any]] = []
    min_suicidal_recall = float(config.get("min_validation_suicidal_recall", 0.7))
    for vectorizer_config in vectorizers:
        if vectorizer_config.get("kind") == "combined" and vectorizer_config.get("enabled_by_default") is False:
            skipped.append({"vectorizer": vectorizer_config.get("name", "combined"), "reason": "combined vectorizer disabled by config"})
            continue
        for spec in specs:
            result = _train_candidate(spec=spec, vectorizer_config=vectorizer_config, bundle=bundle)
            result.selected_score = _candidate_score(result.validation_metrics, result.overfitting_gap, min_suicidal_recall=min_suicidal_recall)
            candidates.append(result)
    candidates.sort(key=lambda item: item.selected_score, reverse=True)
    selected = candidates[0] if candidates and candidates[0].selected_score[0] == 1 else None

    test_metrics = None
    confusion_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    suicidal_analysis: dict[str, Any] = {"selected": False, "reason": "no candidate met minimum validation suicidal recall"}
    interpretation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    vectorizer_payload: dict[str, Any] = {}
    if selected is not None:
        X_test = transform_text_features(selected.vectorizer, bundle.test[bundle.text_column])
        y_test = bundle.test[bundle.target].astype(str).tolist()
        test_pred, test_scores, test_score_kind = predict_with_scores(selected.estimator, X_test)
        test_metrics = evaluate_text_split(y_test, test_pred, scores=test_scores, score_kind=test_score_kind, split_name="test")
        confusion_rows = confusion_matrix_rows(y_test, test_pred)
        per_class_rows = per_class_metric_rows(test_metrics)
        suicidal_analysis = {"selected": True, **(test_metrics.get("suicidal_class") or {})}
        interpretation_rows = aggregate_feature_interpretation(selected.estimator, selected.feature_names)
        error_rows = privacy_safe_error_analysis(bundle.test, y_test, test_pred, test_scores, score_kind=test_score_kind)
        vectorizer_payload = {
            "vectorizer_name": selected.vectorizer_name,
            "vectorizer_version": TEXT_VECTORIZER_VERSION,
            "feature_count": len(selected.feature_names),
            "vocabulary_hash": selected.vocabulary_hash,
            "vectorizer_config": selected.vectorizer_config,
            "complete_vocabulary_exposed": False,
        }

    config_hash = hashing.hash_json_data({**config, "candidate": candidate, "max_train_records": max_train_records})
    run_id = f"text-{config_hash[:12]}"
    model_name = f"text-classification-{selected.spec.estimator_type.replace('_', '-')}" if selected else "text-classification-no-selected-candidate"
    model_version = str(config.get("model_version", TEXT_MODEL_FAMILY_VERSION))
    run_dir = _resolve(model_root) / "text" / model_name / model_version / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_files: list[Path] = []
    if selected is not None:
        artifact_files.append(save_joblib_artifact(selected.estimator, run_dir / "model.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(selected.vectorizer, run_dir / "vectorizer.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(Pipeline([("vectorizer", selected.vectorizer), ("model", selected.estimator)]), run_dir / "pipeline.joblib", overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "training_config.json", {**config, "feature_set": TEXT_FEATURE_SET}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "metrics.json", {"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}}, overwrite=overwrite))
    artifact_files.append(write_csv(run_dir / "candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "feature_schema.json", {"source": str(feature_schema_path), "text_column": TEXT_TEXT_COLUMN, "target_column": TEXT_TARGET_COLUMN}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "split_manifest_reference.json", bundle.split_manifest.payload, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "vocabulary_metadata.json", vectorizer_payload, overwrite=overwrite))
    model_card = build_text_model_card(
        model_name=model_name,
        model_version=model_version,
        selected_candidate=selected.candidate_id if selected else None,
        vectorizer_name=selected.vectorizer_name if selected else None,
        metrics={"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}},
    )
    if REQUIRED_MODEL_CARD_DISCLAIMER not in model_card:
        raise ValueError("Text model card is missing required disclaimer")
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
                "calibration_policy": "No SVM probability calibration in Phase 3D; Logistic Regression probabilities are native estimator outputs.",
            },
            overwrite=overwrite,
        )
    )
    manifest = build_artifact_manifest(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        files=artifact_files,
        split_manifest_hash=bundle.split_manifest.manifest_hash,
        source_fingerprint=bundle.source_fingerprint,
        preprocessing_artifact_hash=bundle.preprocessing_artifact_hash,
        config_hash=config_hash,
    )
    manifest_path = write_json(run_dir / "artifact_manifest.json", manifest, overwrite=overwrite)
    artifact_files.append(manifest_path)

    summary = {
        "text_model_family_version": TEXT_MODEL_FAMILY_VERSION,
        "experiment_version": TEXT_BASELINE_EXPERIMENT_VERSION,
        "vectorizer_version": TEXT_VECTORIZER_VERSION,
        "feature_set": TEXT_FEATURE_SET,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "candidate_count": len(candidates),
        "skipped_candidates": skipped,
        "selected_candidate": {
            "candidate_id": selected.candidate_id,
            "estimator_type": selected.spec.estimator_type,
            "hyperparameters": selected.spec.hyperparameters,
            "vectorizer_name": selected.vectorizer_name,
            "vectorizer_config": selected.vectorizer_config,
        }
        if selected
        else None,
        "selection_rationale": "Validation-only hierarchy: minimum suicidal recall, macro F1, macro recall, balanced accuracy, suicidal F1, lower train-validation gap, and simplicity.",
        "train_metrics": selected.train_metrics if selected else {},
        "validation_metrics": selected.validation_metrics if selected else {},
        "test_metrics": test_metrics or {},
        "overfitting_gap": selected.overfitting_gap if selected else {},
        "feature_count": len(selected.feature_names) if selected else 0,
        "vocabulary_hash": selected.vocabulary_hash if selected else None,
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
        confusion_rows=confusion_rows,
        per_class_rows=per_class_rows,
        suicidal_analysis=suicidal_analysis,
        vectorizer_payload=vectorizer_payload,
        interpretation_rows=interpretation_rows,
        error_rows=error_rows,
        files_for_inventory=artifact_files,
        overwrite=overwrite,
    )

    registered = False
    if register_candidate:
        if not test_database_url or not test_database_url.startswith("sqlite"):
            raise ValueError("candidate registration requires an isolated sqlite test database URL in this step")
        if selected is None:
            raise ValueError("cannot register because no Text candidate met the selection policy")
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
                "modality": "text",
                "version": model_version,
                "framework": "scikit-learn",
                "artifact_path": str((run_dir / "pipeline.joblib").relative_to(paths.get_repository_root())).replace("\\", "/"),
                "preprocessing_path": str((run_dir / "vectorizer.joblib").relative_to(paths.get_repository_root())).replace("\\", "/"),
                "dataset_version": "v1",
                "feature_schema_version": "1.0.0",
                "metrics_json": {"validation": selected.validation_metrics, "test": test_metrics or {}},
                "thresholds_json": {},
                "is_active": False,
            }
            register_candidate_model(session, payload)
            session.commit()
            registered = True
            summary["candidate_registration_occurred"] = True
        finally:
            session.close()

    return TextRunArtifacts(
        run_id=run_id,
        report_dir=_resolve(report_dir),
        run_dir=run_dir,
        selected_candidate=selected,
        metrics={"summary": summary, "test": test_metrics or {}, "reports": {key: str(value) for key, value in report_outputs.items()}},
        artifact_manifest=manifest,
        registered=registered,
        skipped_candidates=skipped,
    )

