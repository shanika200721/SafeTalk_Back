"""End-to-end runner for the Phase 3E Speech acoustic baseline."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.common import hashing, paths
from app.ml.preprocessing.speech.features import extract_acoustic_features
from app.ml.training.reproducibility import capture_environment_versions, set_global_seed
from app.ml.training.speech.constants import (
    DEFAULT_CANONICAL_MANIFEST,
    DEFAULT_CORPUS_DISTRIBUTION,
    DEFAULT_CORPUS_SUMMARY,
    DEFAULT_DUPLICATE_ISOLATION_REPORT,
    DEFAULT_DUPLICATE_MANIFEST,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_FEATURES,
    DEFAULT_FINGERPRINT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PREPROCESSING_REPORT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SPEAKER_ISOLATION_REPORT,
    DEFAULT_SPLIT_ASSIGNMENTS,
    DEFAULT_SPLIT_MANIFEST,
    FEATURE_SETS,
    REQUIRED_MODEL_CARD_DISCLAIMER,
    SPEECH_BASELINE_EXPERIMENT_VERSION,
    SPEECH_CORPUS_COLUMN,
    SPEECH_FEATURE_PIPELINE_VERSION,
    SPEECH_MODEL_FAMILY_VERSION,
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_TARGET_COLUMN,
)
from app.ml.training.speech.data import (
    build_speech_training_bundle,
    inspect_speech_feature_coverage,
    load_speech_canonical_manifest,
    load_speech_split_manifest,
    resolve_speech_feature_set,
    verify_speech_integrity,
)
from app.ml.training.speech.estimators import create_speech_estimator, speech_candidate_specs
from app.ml.training.speech.evaluation import (
    confusion_matrix_rows,
    corpus_distribution,
    corpus_metrics,
    evaluate_speech_split,
    feature_interpretation,
    per_class_metric_rows,
    predict_with_scores,
    privacy_safe_error_analysis,
    train_validation_gap,
)
from app.ml.training.speech.preprocessing import build_speech_preprocessor, transform_speech_features
from app.ml.training.speech.reporting import (
    build_artifact_manifest,
    build_dataset_limitations_markdown,
    build_speech_model_card,
    build_summary_markdown,
    file_inventory,
    save_joblib_artifact,
    write_csv,
    write_json,
    write_markdown,
)
from app.ml.training.speech.schemas import CandidateResult, SpeechRunArtifacts


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def default_speech_training_config() -> dict[str, Any]:
    return {
        "experiment_name": "speech-emotion-acoustic-baseline",
        "experiment_version": SPEECH_BASELINE_EXPERIMENT_VERSION,
        "model_version": SPEECH_MODEL_FAMILY_VERSION,
        "task": "multiclass_classification",
        "primary_metric": "validation_macro_f1",
        "random_seed": 42,
        "feature_set": "full_acoustic",
        "feature_sets": FEATURE_SETS,
        "max_candidate_count": 64,
        "min_validation_macro_f1": 0.0,
        "hyperparameter_search": {
            "logistic_regression": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"], "solver": "lbfgs", "multi_class": "multinomial", "max_iter": 300, "random_state": 42},
            "random_forest": {"n_estimators": [100], "max_depth": [8, 12], "min_samples_leaf": [1, 3], "class_weight": [None, "balanced"], "random_state": 42, "n_jobs": 1},
            "svm": {"kernel": ["linear"], "C": [0.1, 1.0], "class_weight": [None, "balanced"], "max_iter": 3000, "random_state": 42},
        },
    }


def load_speech_training_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_speech_training_config()
    payload = _load_json(path)
    merged = default_speech_training_config()
    merged.update(payload)
    if "hyperparameter_search" in payload:
        hyper = default_speech_training_config()["hyperparameter_search"]
        hyper.update(payload["hyperparameter_search"])
        merged["hyperparameter_search"] = hyper
    return merged


def extract_missing_speech_features(
    *,
    canonical_manifest_path: str | Path = DEFAULT_CANONICAL_MANIFEST,
    features_path: str | Path = DEFAULT_FEATURES,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    overwrite: bool = False,
) -> dict[str, Any]:
    canonical = load_speech_canonical_manifest(canonical_manifest_path)
    schema = _load_json(feature_schema_path)
    feature_columns = [str(item["name"]) for item in schema.get("features", [])]
    if not feature_columns:
        raise ValueError("Speech feature schema has no feature columns")
    output = _resolve(features_path)
    if output.exists() and not overwrite:
        try:
            existing = pd.read_csv(output)
            if len(existing) >= len(canonical):
                return {
                    "status": "features_already_complete",
                    "feature_rows": int(len(existing)),
                    "missing_feature_count": 0,
                    "feature_file_hash": hashing.sha256_file(output),
                    "runtime_seconds": 0.0,
                    "failures": [],
                }
        except Exception:
            pass
    if not paths.is_path_inside(paths.get_generated_preprocessing_root() / "speech" / "v1", output):
        raise ValueError("Speech feature extraction may write only to generated/preprocessing/speech/v1")
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, record in canonical.sort_values([SPEECH_CORPUS_COLUMN, SPEECH_RECORD_ID_COLUMN]).iterrows():
        source = _resolve(str(record["audio_relative_path"]))
        values, warnings = extract_acoustic_features(source)
        if any(str(warning).startswith("feature extraction failed") for warning in warnings):
            failures.append({"record_id": record[SPEECH_RECORD_ID_COLUMN], "corpus": record[SPEECH_CORPUS_COLUMN], "warning": "|".join(warnings)})
        row = {
            SPEECH_RECORD_ID_COLUMN: record[SPEECH_RECORD_ID_COLUMN],
            "safe_speaker_key": record["safe_speaker_key"],
            SPEECH_CORPUS_COLUMN: record[SPEECH_CORPUS_COLUMN],
            SPEECH_TARGET_COLUMN: record[SPEECH_TARGET_COLUMN],
            **{feature: values.get(feature, 0.0) for feature in feature_columns},
            "feature_extraction_warnings": "|".join(warnings),
        }
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Speech feature file: {output}")
    tmp = output.with_name(f".{output.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[SPEECH_RECORD_ID_COLUMN, "safe_speaker_key", SPEECH_CORPUS_COLUMN, SPEECH_TARGET_COLUMN, *feature_columns, "feature_extraction_warnings"])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)
    runtime = time.perf_counter() - start
    return {
        "status": "features_extracted",
        "feature_rows": len(rows),
        "missing_feature_count": 0,
        "feature_count": len(feature_columns),
        "feature_file_hash": hashing.sha256_file(output),
        "runtime_seconds": round(runtime, 3),
        "failures": failures,
    }


def dry_run_speech_baseline(
    *,
    config_path: str | Path | None = None,
    features_path: str | Path = DEFAULT_FEATURES,
    canonical_manifest_path: str | Path = DEFAULT_CANONICAL_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    preprocessing_report_path: str | Path = DEFAULT_PREPROCESSING_REPORT,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    feature_set: str | None = None,
) -> dict[str, Any]:
    config = load_speech_training_config(config_path)
    selected_feature_set, features = resolve_speech_feature_set(config, feature_set=feature_set)
    split_manifest = load_speech_split_manifest(split_manifest_path)
    verify_speech_integrity(
        canonical_manifest_path=canonical_manifest_path,
        split_manifest=split_manifest,
        preprocessing_report_path=preprocessing_report_path,
        fingerprint_dir=fingerprint_dir,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    coverage = inspect_speech_feature_coverage(
        features_path=features_path,
        split_manifest_path=split_manifest_path,
        feature_schema_path=feature_schema_path,
        feature_set=selected_feature_set,
    )
    coverage["feature_columns"] = features
    return {
        "status": "dry_run_ok" if coverage["complete"] else "dry_run_feature_coverage_incomplete",
        "feature_set": selected_feature_set,
        "target_column": SPEECH_TARGET_COLUMN,
        "record_id_column": SPEECH_RECORD_ID_COLUMN,
        "safe_speaker_key": "safe_speaker_key",
        "corpus_metadata": SPEECH_CORPUS_COLUMN,
        "train_count": len(split_manifest.train_ids),
        "validation_count": len(split_manifest.validation_ids),
        "test_count": len(split_manifest.test_ids),
        "source_fingerprint": split_manifest.source_fingerprint,
        "preprocessing_artifact_hash": split_manifest.preprocessing_artifact_hash,
        "split_manifest_hash": split_manifest.manifest_hash,
        "feature_coverage": coverage,
    }


def _apply_train_limit(bundle, max_train_records: int | None):
    if not max_train_records:
        return bundle
    if max_train_records <= 0:
        raise ValueError("max_train_records must be positive")
    limited = bundle.train.groupby(bundle.target, group_keys=False).apply(lambda group: group.sort_values(SPEECH_RECORD_ID_COLUMN).head(max(1, int(max_train_records / 8))))
    if len(limited) > max_train_records:
        limited = limited.sort_values(SPEECH_RECORD_ID_COLUMN).head(max_train_records)
    return type(bundle)(
        train=limited.reset_index(drop=True),
        validation=bundle.validation,
        test=bundle.test,
        features=bundle.features,
        target=bundle.target,
        split_manifest=bundle.split_manifest,
        source_fingerprint=bundle.source_fingerprint,
        preprocessing_artifact_hash=bundle.preprocessing_artifact_hash,
        feature_schema=bundle.feature_schema,
        preprocessing_report=bundle.preprocessing_report,
        duplicate_manifest=bundle.duplicate_manifest,
        corpus_summary=bundle.corpus_summary,
        speaker_isolation_report=bundle.speaker_isolation_report,
        duplicate_isolation_report=bundle.duplicate_isolation_report,
        feature_coverage=bundle.feature_coverage,
    )


def _candidate_score(validation_metrics: Mapping[str, Any], gap: Mapping[str, Any]) -> tuple[Any, ...]:
    recalls = [(values or {}).get("recall") or 0.0 for values in (validation_metrics.get("per_class") or {}).values()]
    min_recall = min(recalls) if recalls else 0.0
    return (
        float(validation_metrics.get("macro_f1") or 0.0),
        float(validation_metrics.get("macro_recall") or 0.0),
        float(validation_metrics.get("balanced_accuracy") or 0.0),
        float(min_recall),
        -abs(float(gap.get("macro_f1_train_minus_validation") or 0.0)),
    )


def _train_candidate(*, spec, bundle) -> CandidateResult:
    prep = build_speech_preprocessor(bundle.train, bundle.features, estimator_type=spec.estimator_type)
    X_train = transform_speech_features(prep.preprocessor, bundle.train, bundle.features)
    X_validation = transform_speech_features(prep.preprocessor, bundle.validation, bundle.features)
    y_train = bundle.train[bundle.target].astype(str).tolist()
    y_validation = bundle.validation[bundle.target].astype(str).tolist()
    estimator = create_speech_estimator(spec)
    estimator.fit(X_train, y_train)
    train_pred, train_scores, train_score_kind = predict_with_scores(estimator, X_train)
    validation_pred, validation_scores, validation_score_kind = predict_with_scores(estimator, X_validation)
    train_metrics = evaluate_speech_split(y_train, train_pred, scores=train_scores, score_kind=train_score_kind, split_name="train")
    validation_metrics = evaluate_speech_split(y_validation, validation_pred, scores=validation_scores, score_kind=validation_score_kind, split_name="validation")
    train_metrics["preprocessing"] = {
        "feature_pipeline_version": SPEECH_FEATURE_PIPELINE_VERSION,
        "removed_constant_features": prep.removed_constant_features,
        "missing_value_handling": prep.missing_value_report,
        "scaled": prep.scale_numeric,
        "final_feature_count": len(prep.feature_names),
    }
    gap = train_validation_gap(train_metrics, validation_metrics)
    result = CandidateResult(
        candidate_id=spec.name,
        spec=spec,
        estimator=estimator,
        preprocessor=prep.preprocessor,
        feature_names=prep.feature_names,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        overfitting_gap=gap,
    )
    result.selected_score = _candidate_score(validation_metrics, gap)
    return result


def _candidate_rows(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "estimator_type": candidate.spec.estimator_type,
            "hyperparameters": candidate.spec.hyperparameters,
            "feature_count": len(candidate.feature_names),
            "validation_macro_f1": candidate.validation_metrics.get("macro_f1"),
            "validation_macro_recall": candidate.validation_metrics.get("macro_recall"),
            "validation_balanced_accuracy": candidate.validation_metrics.get("balanced_accuracy"),
            "validation_min_class_recall": min((value.get("recall") or 0.0) for value in (candidate.validation_metrics.get("per_class") or {}).values()),
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
    corpus_metrics_payload: dict[str, Any],
    corpus_distribution_rows: list[dict[str, Any]],
    interpretation_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    feature_coverage: dict[str, Any],
    files_for_inventory: list[Path],
    overwrite: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["summary_json"] = write_json(report_dir / "speech_baseline_summary.json", summary, overwrite=overwrite)
    outputs["summary_md"] = write_markdown(report_dir / "speech_baseline_summary.md", build_summary_markdown(summary), overwrite=overwrite)
    outputs["candidate_comparison"] = write_csv(report_dir / "speech_candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite)
    outputs["metrics_train"] = write_json(report_dir / "speech_metrics_train.json", selected.train_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_validation"] = write_json(report_dir / "speech_metrics_validation.json", selected.validation_metrics if selected else {}, overwrite=overwrite)
    outputs["metrics_test"] = write_json(report_dir / "speech_metrics_test.json", test_metrics or {}, overwrite=overwrite)
    outputs["confusion_matrix_test"] = write_csv(report_dir / "speech_confusion_matrix_test.csv", confusion_rows, overwrite=overwrite)
    outputs["per_class_metrics_test"] = write_csv(report_dir / "speech_per_class_metrics_test.csv", per_class_rows, overwrite=overwrite)
    outputs["corpus_metrics"] = write_json(report_dir / "speech_corpus_metrics.json", corpus_metrics_payload, overwrite=overwrite)
    outputs["corpus_distribution"] = write_csv(report_dir / "speech_corpus_distribution.csv", corpus_distribution_rows, overwrite=overwrite)
    outputs["feature_interpretation"] = write_csv(report_dir / "speech_feature_interpretation.csv", interpretation_rows, overwrite=overwrite)
    outputs["error_analysis"] = write_csv(report_dir / "speech_error_analysis.csv", error_rows, overwrite=overwrite)
    outputs["feature_coverage"] = write_json(report_dir / "speech_feature_coverage.json", feature_coverage, overwrite=overwrite)
    outputs["limitations"] = write_markdown(report_dir / "speech_dataset_limitations.md", build_dataset_limitations_markdown(), overwrite=overwrite)
    outputs["artifact_inventory"] = write_json(report_dir / "speech_artifact_inventory.json", file_inventory([*files_for_inventory, *outputs.values()]), overwrite=overwrite)
    return outputs


def run_speech_baseline(
    *,
    config_path: str | Path | None = None,
    ablation_config_path: str | Path | None = None,
    features_path: str | Path = DEFAULT_FEATURES,
    canonical_manifest_path: str | Path = DEFAULT_CANONICAL_MANIFEST,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    preprocessing_report_path: str | Path = DEFAULT_PREPROCESSING_REPORT,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    duplicate_manifest_path: str | Path = DEFAULT_DUPLICATE_MANIFEST,
    corpus_summary_path: str | Path = DEFAULT_CORPUS_SUMMARY,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS,
    speaker_isolation_report_path: str | Path = DEFAULT_SPEAKER_ISOLATION_REPORT,
    duplicate_isolation_report_path: str | Path = DEFAULT_DUPLICATE_ISOLATION_REPORT,
    corpus_distribution_path: str | Path = DEFAULT_CORPUS_DISTRIBUTION,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    feature_set: str | None = None,
    candidate: str = "all",
    dry_run: bool = False,
    max_train_records: int | None = None,
    extract_missing_features: bool = False,
    overwrite: bool = False,
    register_candidate: bool = False,
    test_database_url: str | None = None,
) -> SpeechRunArtifacts | dict[str, Any]:
    config = load_speech_training_config(config_path)
    selected_feature_set, selected_features = resolve_speech_feature_set(config, feature_set=feature_set)
    if dry_run:
        return dry_run_speech_baseline(
            config_path=config_path,
            features_path=features_path,
            canonical_manifest_path=canonical_manifest_path,
            split_manifest_path=split_manifest_path,
            feature_schema_path=feature_schema_path,
            preprocessing_report_path=preprocessing_report_path,
            fingerprint_dir=fingerprint_dir,
            feature_set=selected_feature_set,
        )
    extraction_result = None
    coverage = inspect_speech_feature_coverage(
        features_path=features_path,
        split_manifest_path=split_manifest_path,
        feature_schema_path=feature_schema_path,
        feature_set=selected_feature_set,
    )
    if not coverage["complete"]:
        if not extract_missing_features:
            raise ValueError(f"full Speech feature coverage is insufficient: missing {coverage['missing_feature_count']} records")
        extraction_result = extract_missing_speech_features(
            canonical_manifest_path=canonical_manifest_path,
            features_path=features_path,
            feature_schema_path=feature_schema_path,
            overwrite=True,
        )
    set_global_seed(int(config.get("random_seed", 42)))
    bundle = build_speech_training_bundle(
        features_path=features_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        feature_schema_path=feature_schema_path,
        preprocessing_report_path=preprocessing_report_path,
        fingerprint_dir=fingerprint_dir,
        duplicate_manifest_path=duplicate_manifest_path,
        corpus_summary_path=corpus_summary_path,
        split_assignments_path=split_assignments_path,
        speaker_isolation_report_path=speaker_isolation_report_path,
        duplicate_isolation_report_path=duplicate_isolation_report_path,
        feature_set=selected_feature_set,
        features=selected_features,
        expected_split_manifest_hash=config.get("locked_split_manifest_hash"),
    )
    bundle = _apply_train_limit(bundle, max_train_records)
    specs = speech_candidate_specs(config, candidate=candidate)
    candidates: list[CandidateResult] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        if spec.estimator_type == "rbf_svm" and (len(bundle.train) > int(config.get("max_rbf_svm_train_records", 2500))):
            skipped.append({"candidate_id": spec.name, "reason": "RBF SVM skipped because runtime bound would be unsafe"})
            continue
        candidates.append(_train_candidate(spec=spec, bundle=bundle))
    if not candidates:
        raise ValueError("no Speech candidates were trained")
    candidates.sort(key=lambda item: item.selected_score, reverse=True)
    selected = candidates[0] if candidates[0].validation_metrics.get("macro_f1", 0.0) >= float(config.get("min_validation_macro_f1", 0.0)) else None

    test_metrics = None
    confusion_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    corpus_metrics_payload: dict[str, Any] = {}
    interpretation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    if selected is not None:
        X_test = transform_speech_features(selected.preprocessor, bundle.test, bundle.features)
        y_test = bundle.test[bundle.target].astype(str).tolist()
        test_pred, test_scores, test_score_kind = predict_with_scores(selected.estimator, X_test)
        test_metrics = evaluate_speech_split(y_test, test_pred, scores=test_scores, score_kind=test_score_kind, split_name="test")
        confusion_rows = confusion_matrix_rows(y_test, test_pred)
        per_class_rows = per_class_metric_rows(test_metrics)
        corpus_metrics_payload = {
            "test": corpus_metrics(bundle.test, y_test, test_pred),
            "train_validation_test_distribution_source": str(corpus_distribution_path),
            "warning": "Corpus analysis is stratified evaluation only; corpus was not used as a predictive feature.",
        }
        interpretation_rows = feature_interpretation(selected.estimator, selected.feature_names)
        error_rows = privacy_safe_error_analysis(bundle.test, y_test, test_pred, test_scores, score_kind=test_score_kind)

    config_hash = hashing.hash_json_data({**config, "feature_set": selected_feature_set, "candidate": candidate, "max_train_records": max_train_records})
    run_id = f"speech-{config_hash[:12]}"
    model_name = f"speech-emotion-{selected.spec.estimator_type.replace('_', '-')}" if selected else "speech-emotion-no-selected-candidate"
    model_version = str(config.get("model_version", SPEECH_MODEL_FAMILY_VERSION))
    run_dir = _resolve(model_root) / "speech" / model_name / model_version / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_file_hash = hashing.sha256_file(features_path)
    source_fingerprints = dict(bundle.preprocessing_report.get("source_fingerprints") or {})
    artifact_files: list[Path] = []
    model_card = build_speech_model_card(
        model_name=model_name,
        model_version=model_version,
        selected_candidate=selected.candidate_id if selected else None,
        feature_set=selected_feature_set,
        metrics={"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}},
    )
    if REQUIRED_MODEL_CARD_DISCLAIMER not in model_card:
        raise ValueError("Speech model card is missing required disclaimer")
    if selected is not None:
        artifact_files.append(save_joblib_artifact(selected.estimator, run_dir / "model.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(selected.preprocessor, run_dir / "preprocessor.joblib", overwrite=overwrite))
        artifact_files.append(save_joblib_artifact(Pipeline([("preprocessor", selected.preprocessor), ("model", selected.estimator)]), run_dir / "pipeline.joblib", overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "training_config.json", {**config, "selected_feature_set": selected_feature_set, "features": selected_features}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "metrics.json", {"train": selected.train_metrics if selected else {}, "validation": selected.validation_metrics if selected else {}, "test": test_metrics or {}}, overwrite=overwrite))
    artifact_files.append(write_csv(run_dir / "candidate_comparison.csv", _candidate_rows(candidates), overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "feature_schema.json", {"source": str(feature_schema_path), "selected_features": selected_features, "transformed_feature_names": selected.feature_names if selected else []}, overwrite=overwrite))
    artifact_files.append(write_json(run_dir / "split_manifest_reference.json", bundle.split_manifest.payload, overwrite=overwrite))
    artifact_files.append(write_markdown(run_dir / "model_card.md", model_card, overwrite=overwrite))
    artifact_files.append(
        write_json(
            run_dir / "reproducibility_report.json",
            {
                "environment": capture_environment_versions(),
                "source_fingerprint": bundle.source_fingerprint,
                "corpus_fingerprints": source_fingerprints,
                "preprocessing_artifact_hash": bundle.preprocessing_artifact_hash,
                "feature_file_hash": feature_file_hash,
                "split_manifest_hash": bundle.split_manifest.manifest_hash,
                "config_hash": config_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "feature_extraction_result": extraction_result,
                "no_transcription": True,
                "no_deep_learning": True,
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
        feature_file_hash=feature_file_hash,
        corpus_fingerprints=source_fingerprints,
        config_hash=config_hash,
    )
    manifest_path = write_json(run_dir / "artifact_manifest.json", manifest, overwrite=overwrite)
    artifact_files.append(manifest_path)

    distribution_rows = corpus_distribution({"train": bundle.train, "validation": bundle.validation, "test": bundle.test})
    summary = {
        "speech_model_family_version": SPEECH_MODEL_FAMILY_VERSION,
        "experiment_version": SPEECH_BASELINE_EXPERIMENT_VERSION,
        "feature_pipeline_version": SPEECH_FEATURE_PIPELINE_VERSION,
        "feature_set": selected_feature_set,
        "features": selected_features,
        "train_count": len(bundle.train),
        "validation_count": len(bundle.validation),
        "test_count": len(bundle.test),
        "candidate_count": len(candidates),
        "skipped_candidates": skipped,
        "selected_candidate": {"candidate_id": selected.candidate_id, "estimator_type": selected.spec.estimator_type, "hyperparameters": selected.spec.hyperparameters} if selected else None,
        "selection_rationale": "Validation-only hierarchy: macro F1, macro recall, balanced accuracy, minimum class recall, corpus consistency, lower train-validation gap, simplicity, and inference cost.",
        "train_metrics": selected.train_metrics if selected else {},
        "validation_metrics": selected.validation_metrics if selected else {},
        "test_metrics": test_metrics or {},
        "overfitting_gap": selected.overfitting_gap if selected else {},
        "feature_coverage": bundle.feature_coverage,
        "feature_extraction_result": extraction_result,
        "feature_count": len(selected.feature_names) if selected else 0,
        "artifact_path": str(run_dir),
        "model_card_path": str(run_dir / "model_card.md"),
        "reproducibility_report_path": str(run_dir / "reproducibility_report.json"),
        "candidate_registration_occurred": False,
        "model_became_active": False,
        "research_readiness_decision": "research baseline only; emotion classification is not clinical validation and not deployable",
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
        corpus_metrics_payload=corpus_metrics_payload,
        corpus_distribution_rows=distribution_rows,
        interpretation_rows=interpretation_rows,
        error_rows=error_rows,
        feature_coverage=bundle.feature_coverage,
        files_for_inventory=artifact_files,
        overwrite=overwrite,
    )
    registered = False
    if register_candidate:
        if not test_database_url or not test_database_url.startswith("sqlite"):
            raise ValueError("candidate registration requires an isolated sqlite test database URL")
        if selected is None:
            raise ValueError("cannot register because no Speech candidate was selected")
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.db.base import Base
        from app.ml.training.registry import register_candidate_model

        engine = create_engine(test_database_url)
        Base.metadata.create_all(bind=engine)
        session = sessionmaker(bind=engine)()
        try:
            register_candidate_model(
                session,
                {
                    "model_name": model_name,
                    "modality": "speech",
                    "version": model_version,
                    "framework": "scikit-learn",
                    "artifact_path": str((run_dir / "pipeline.joblib").relative_to(paths.get_repository_root())).replace("\\", "/"),
                    "preprocessing_path": str((run_dir / "preprocessor.joblib").relative_to(paths.get_repository_root())).replace("\\", "/"),
                    "dataset_version": "v1",
                    "feature_schema_version": SPEECH_FEATURE_PIPELINE_VERSION,
                    "metrics_json": {"validation": selected.validation_metrics, "test": test_metrics or {}},
                    "thresholds_json": {},
                    "is_active": False,
                },
            )
            session.commit()
            registered = True
            summary["candidate_registration_occurred"] = True
        finally:
            session.close()

    return SpeechRunArtifacts(
        run_id=run_id,
        report_dir=_resolve(report_dir),
        run_dir=run_dir,
        selected_candidate=selected,
        metrics={"summary": summary, "test": test_metrics or {}, "reports": {key: str(value) for key, value in report_outputs.items()}},
        artifact_manifest=manifest,
        registered=registered,
        skipped_candidates=skipped,
    )

