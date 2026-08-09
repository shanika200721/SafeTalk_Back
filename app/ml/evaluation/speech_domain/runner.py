"""Runner for Phase 3F Speech leave-one-corpus-out domain-shift evaluation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from app.ml.common import hashing, paths
from app.ml.evaluation.speech_domain.constants import (
    CORPORA,
    DEFAULT_CANONICAL_MANIFEST,
    DEFAULT_EVALUATION_MANIFEST_DIR,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_FEATURES,
    DEFAULT_FINGERPRINT_DIR,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ROOT,
    DEFAULT_POOLED_BASELINE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REPORT_DIR,
    DIAGNOSTIC_ONLY_WARNING,
    SPEECH_DOMAIN_EVALUATION_VERSION,
    SPEECH_LOCO_POLICY_VERSION,
)
from app.ml.evaluation.speech_domain.data import build_domain_evaluation_bundle, load_json, resolve_project_path
from app.ml.evaluation.speech_domain.metrics import (
    corpus_generalization_gap,
    evaluate_domain_predictions,
    feature_distribution_rows,
    per_class_rows,
)
from app.ml.evaluation.speech_domain.reporting import artifact_inventory, repo_relative, save_domain_shift_reports, write_json
from app.ml.evaluation.speech_domain.schemas import CorpusFoldData, CorpusFoldResult, CorpusTransferResult
from app.ml.evaluation.speech_domain.splits import create_loco_folds, save_loco_manifests
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.reproducibility import set_global_seed
from app.ml.training.speech.estimators import create_speech_estimator
from app.ml.training.speech.preprocessing import build_speech_preprocessor, transform_speech_features
from app.ml.training.speech.schemas import CandidateSpec
from app.ml.training.speech.constants import (
    SPEECH_CORPUS_COLUMN,
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_TARGET_COLUMN,
)


def _candidate_specs(candidate: str, *, random_seed: int = DEFAULT_RANDOM_SEED) -> list[CandidateSpec]:
    selected = candidate or "all"
    specs: list[CandidateSpec] = []
    if selected in ("all", "logistic_regression"):
        specs.append(
            CandidateSpec(
                "logistic_regression",
                "logistic_regression_C=1.0_class_weight=balanced",
                {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 250, "random_state": random_seed},
            )
        )
    if selected in ("all", "random_forest"):
        specs.append(
            CandidateSpec(
                "random_forest",
                "random_forest_n=100_depth=12_leaf=3_class_weight=balanced",
                {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 3, "class_weight": "balanced", "random_state": random_seed, "n_jobs": 1},
            )
        )
    if selected in ("all", "linear_svm", "svm"):
        specs.append(
            CandidateSpec(
                "linear_svm",
                "linear_svm_C=1.0_class_weight=balanced",
                {"C": 1.0, "class_weight": "balanced", "max_iter": 2000, "random_state": random_seed},
            )
        )
    if not specs:
        raise ValueError(f"unsupported Speech domain candidate: {candidate}")
    return specs


def _fit_candidate(spec: CandidateSpec, fold: CorpusFoldData, features: list[str]) -> dict[str, Any]:
    prep = build_speech_preprocessor(fold.train, features, estimator_type=spec.estimator_type)
    X_train = transform_speech_features(prep.preprocessor, fold.train, features)
    X_validation = transform_speech_features(prep.preprocessor, fold.validation, features)
    y_train = fold.train[SPEECH_TARGET_COLUMN].astype(str).tolist()
    y_validation = fold.validation[SPEECH_TARGET_COLUMN].astype(str).tolist()
    estimator = create_speech_estimator(spec)
    estimator.fit(X_train, y_train)
    validation_pred = estimator.predict(X_validation)
    validation_metrics = evaluate_domain_predictions(
        y_validation,
        validation_pred,
        labels=fold.config.included_labels,
        split_name=f"{fold.config.fold_name}:validation",
    )
    score = (
        validation_metrics.get("macro_f1") or 0.0,
        validation_metrics.get("macro_recall") or 0.0,
        validation_metrics.get("balanced_accuracy") or 0.0,
        validation_metrics.get("worst_class_recall") or 0.0,
    )
    return {
        "spec": spec,
        "preprocessor": prep.preprocessor,
        "feature_names": prep.feature_names,
        "estimator": estimator,
        "validation_metrics": validation_metrics,
        "score": score,
    }


def _save_fold_artifacts(
    *,
    fold: CorpusFoldData,
    selected: dict[str, Any],
    metrics: dict[str, Any],
    model_root: str | Path,
    overwrite: bool,
) -> tuple[list[str], dict[str, str]]:
    root = Path(model_root)
    if not root.is_absolute():
        root = paths.get_repository_root() / root
    config_hash = hashing.hash_json_data(
        {
            "fold": fold.config.fold_name,
            "candidate": selected["spec"].name,
            "labels": fold.config.included_labels,
            "version": SPEECH_DOMAIN_EVALUATION_VERSION,
        }
    )
    run_id = f"domain-{config_hash[:12]}"
    run_dir = (root / fold.config.fold_name / run_id).resolve(strict=False)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Speech domain evaluation artifact directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    pipeline = Pipeline([("preprocessor", selected["preprocessor"]), ("model", selected["estimator"])])
    for name, obj in {"pipeline.joblib": pipeline, "model.joblib": selected["estimator"], "preprocessor.joblib": selected["preprocessor"]}.items():
        path = run_dir / name
        prevent_overwrite(path, overwrite=overwrite)
        joblib.dump(obj, path)
        outputs.append(path)
    config_path = write_json(
        run_dir / "evaluation_config.json",
        {
            "evaluation_version": SPEECH_DOMAIN_EVALUATION_VERSION,
            "policy_version": SPEECH_LOCO_POLICY_VERSION,
            "fold": fold.config.__dict__,
            "candidate": selected["spec"].name,
            "hyperparameters": selected["spec"].hyperparameters,
            "research_artifact_only": True,
            "registered": False,
            "active": False,
        },
        overwrite=overwrite,
    )
    metrics_path = write_json(run_dir / "metrics.json", metrics, overwrite=overwrite)
    outputs.extend([config_path, metrics_path])
    manifest = artifact_inventory(outputs)
    manifest.update({"run_id": run_id, "fold_name": fold.config.fold_name, "registered": False, "active": False})
    manifest_path = write_json(run_dir / "artifact_manifest.json", manifest, overwrite=overwrite)
    outputs.append(manifest_path)
    return [repo_relative(path) for path in outputs], {repo_relative(path): hashing.sha256_file(path) for path in outputs}


def _run_fold(
    fold: CorpusFoldData,
    *,
    features: list[str],
    candidate: str,
    model_root: str | Path,
    save_artifacts: bool,
    overwrite: bool,
) -> CorpusFoldResult:
    start = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for spec in _candidate_specs(candidate):
        try:
            candidates.append(_fit_candidate(spec, fold, features))
        except Exception as exc:
            warnings.append(f"candidate skipped: {spec.name}: {exc}")
    if not candidates:
        raise ValueError(f"no valid candidates for {fold.config.fold_name}")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    X_test = transform_speech_features(selected["preprocessor"], fold.test, features)
    y_test = fold.test[SPEECH_TARGET_COLUMN].astype(str).tolist()
    test_pred = selected["estimator"].predict(X_test)
    test_metrics = evaluate_domain_predictions(
        y_test,
        test_pred,
        labels=fold.config.included_labels,
        split_name=f"{fold.config.fold_name}:test",
    )
    warnings.extend(test_metrics.get("warnings") or [])
    missing_classes = sorted(set(fold.config.included_labels) - set(y_test))
    artifact_paths: list[str] = []
    artifact_hashes: dict[str, str] = {}
    if save_artifacts:
        artifact_paths, artifact_hashes = _save_fold_artifacts(
            fold=fold,
            selected=selected,
            metrics={"validation": selected["validation_metrics"], "test": test_metrics},
            model_root=model_root,
            overwrite=overwrite,
        )
    return CorpusFoldResult(
        fold_name=fold.config.fold_name,
        training_corpora=fold.config.training_corpora,
        validation_corpora=fold.config.validation_corpora,
        test_corpus=fold.config.test_corpus,
        train_count=int(len(fold.train)),
        validation_count=int(len(fold.validation)),
        test_count=int(len(fold.test)),
        feature_count=len(selected["feature_names"]),
        candidate_model=selected["spec"].name,
        selected_hyperparameters=selected["spec"].hyperparameters,
        validation_metrics=selected["validation_metrics"],
        test_metrics=test_metrics,
        per_class_metrics=test_metrics.get("per_class") or {},
        confusion_matrix=test_metrics.get("confusion_matrix") or {},
        unsupported_labels=fold.config.excluded_labels,
        missing_classes=missing_classes,
        warnings=warnings,
        runtime_seconds=round(time.perf_counter() - start, 3),
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
    )


def _fixed_transfer_estimator(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[Any, list[str]]:
    spec = CandidateSpec(
        "logistic_regression",
        "logistic_regression_C=1.0_class_weight=balanced",
        {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 250, "random_state": DEFAULT_RANDOM_SEED},
    )
    prep = build_speech_preprocessor(train, features, estimator_type=spec.estimator_type)
    X_train = transform_speech_features(prep.preprocessor, train, features)
    y_train = train[SPEECH_TARGET_COLUMN].astype(str).tolist()
    estimator = create_speech_estimator(spec)
    estimator.fit(X_train, y_train)
    X_test = transform_speech_features(prep.preprocessor, test, features)
    return (estimator.predict(X_test), y_train)


def run_transfer_matrix(bundle, *, features: list[str]) -> list[CorpusTransferResult]:
    labels = [str(label) for label in bundle.label_policy.get("shared_labels_all_corpora", [])]
    output: list[CorpusTransferResult] = []
    for train_corpus in CORPORA:
        for test_corpus in CORPORA:
            if train_corpus == test_corpus:
                continue
            train = bundle.records.loc[
                (bundle.records[SPEECH_CORPUS_COLUMN].astype(str) == train_corpus)
                & (bundle.records[SPEECH_TARGET_COLUMN].astype(str).isin(labels))
            ].copy()
            test = bundle.records.loc[
                (bundle.records[SPEECH_CORPUS_COLUMN].astype(str) == test_corpus)
                & (bundle.records[SPEECH_TARGET_COLUMN].astype(str).isin(labels))
            ].copy()
            warnings: list[str] = []
            if train[SPEECH_TARGET_COLUMN].nunique() < 2 or test[SPEECH_TARGET_COLUMN].nunique() < 2:
                output.append(CorpusTransferResult(train_corpus, test_corpus, labels, int(len(train)), int(len(test)), {}, {}, ["pair skipped because fewer than two labels are present"]))
                continue
            try:
                pred, _ = _fixed_transfer_estimator(train, test, features)
                y_test = test[SPEECH_TARGET_COLUMN].astype(str).tolist()
                metrics = evaluate_domain_predictions(y_test, pred, labels=labels, split_name=f"{train_corpus}_to_{test_corpus}")
                warnings.extend(metrics.get("warnings") or [])
                output.append(CorpusTransferResult(train_corpus, test_corpus, labels, int(len(train)), int(len(test)), metrics, metrics.get("per_class") or {}, warnings))
            except Exception as exc:
                output.append(CorpusTransferResult(train_corpus, test_corpus, labels, int(len(train)), int(len(test)), {}, {}, [f"pair skipped: {exc}"]))
    return output


def run_shortcut_diagnostics(bundle, *, features: list[str]) -> dict[str, Any]:
    records = bundle.records.copy()
    y = records[SPEECH_CORPUS_COLUMN].astype(str)
    if y.nunique() < 2:
        return {"status": "skipped", "warning": "fewer than two corpora", "diagnostic_only": True}
    train_df, test_df = train_test_split(records, test_size=0.25, random_state=DEFAULT_RANDOM_SEED, stratify=y)
    prep = build_speech_preprocessor(train_df, features, estimator_type="random_forest", scale_numeric=False)
    X_train = transform_speech_features(prep.preprocessor, train_df, features)
    X_test = transform_speech_features(prep.preprocessor, test_df, features)
    clf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=3, random_state=DEFAULT_RANDOM_SEED, n_jobs=1)
    clf.fit(X_train, train_df[SPEECH_CORPUS_COLUMN].astype(str))
    pred = clf.predict(X_test)
    accuracy = float(np.mean(pred == test_df[SPEECH_CORPUS_COLUMN].astype(str).to_numpy()))
    importances = getattr(clf, "feature_importances_", np.zeros(len(prep.feature_names)))
    top = sorted(
        [{"feature": feature, "importance": float(value)} for feature, value in zip(prep.feature_names, importances)],
        key=lambda row: row["importance"],
        reverse=True,
    )[:10]
    return {
        "status": "completed",
        "diagnostic_only": True,
        "warning": DIAGNOSTIC_ONLY_WARNING,
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "accuracy": accuracy,
        "strong_domain_shortcut_risk": accuracy >= 0.8,
        "strongest_corpus_separating_features": top,
        "registered": False,
        "active": False,
    }


def _pooled_baseline(path: str | Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except FileNotFoundError:
        return {"available": False, "path": str(path), "test_macro_f1": None}
    test = payload.get("test_metrics") or {}
    return {
        "available": True,
        "path": repo_relative(resolve_project_path(path)),
        "selected_candidate": payload.get("selected_candidate"),
        "test_macro_f1": test.get("macro_f1"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "warning": "Pooled locked-split performance does not represent unseen-corpus generalization.",
    }


def dry_run_domain_shift(*, held_out_corpus: str | None = None, shared_labels_only: bool = True, **kwargs) -> dict[str, Any]:
    bundle = build_domain_evaluation_bundle(**kwargs)
    folds = create_loco_folds(bundle, held_out_corpus=held_out_corpus, shared_labels_only=shared_labels_only)
    return {
        "status": "dry_run_ok",
        "evaluation_version": SPEECH_DOMAIN_EVALUATION_VERSION,
        "policy_version": SPEECH_LOCO_POLICY_VERSION,
        "folds": [
            {
                "fold_name": fold.config.fold_name,
                "training_corpora": fold.config.training_corpora,
                "test_corpus": fold.config.test_corpus,
                "train_count": int(len(fold.train)),
                "validation_count": int(len(fold.validation)),
                "test_count": int(len(fold.test)),
                "included_labels": fold.config.included_labels,
                "excluded_labels": fold.config.excluded_labels,
            }
            for fold in folds
        ],
        "feature_file_hash": bundle.feature_file_hash,
        "source_fingerprints_verified": True,
    }


def run_speech_domain_shift_evaluation(
    *,
    features_path: str | Path = DEFAULT_FEATURES,
    canonical_manifest_path: str | Path = DEFAULT_CANONICAL_MANIFEST,
    feature_schema_path: str | Path = DEFAULT_FEATURE_SCHEMA,
    corpus_mapping_path: str | Path | None = None,
    label_policy_path: str | Path = DEFAULT_LABEL_POLICY,
    fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    strategy: str = "leave_one_corpus_out",
    held_out_corpus: str | None = None,
    candidate: str = "all",
    shared_labels_only: bool = True,
    run_transfer: bool = False,
    run_shortcut: bool = False,
    dry_run: bool = False,
    max_records_per_corpus: int | None = None,
    overwrite: bool = False,
    report_path: str | Path | None = None,
    save_artifacts: bool = True,
    evaluation_manifest_dir: str | Path = DEFAULT_EVALUATION_MANIFEST_DIR,
) -> dict[str, Any]:
    if strategy not in {"leave_one_corpus_out", "corpus_transfer_matrix"}:
        raise ValueError(f"unsupported Speech domain evaluation strategy: {strategy}")
    if corpus_mapping_path:
        load_json(corpus_mapping_path)
    set_global_seed(DEFAULT_RANDOM_SEED)
    bundle_kwargs = {
        "features_path": features_path,
        "canonical_manifest_path": canonical_manifest_path,
        "feature_schema_path": feature_schema_path,
        "label_policy_path": label_policy_path,
        "fingerprint_dir": fingerprint_dir,
        "max_records_per_corpus": max_records_per_corpus,
    }
    if dry_run:
        result = dry_run_domain_shift(held_out_corpus=held_out_corpus, shared_labels_only=shared_labels_only, **bundle_kwargs)
        if report_path:
            write_json(report_path, result, overwrite=overwrite)
        return result
    bundle = build_domain_evaluation_bundle(**bundle_kwargs)
    folds = create_loco_folds(bundle, held_out_corpus=held_out_corpus, shared_labels_only=shared_labels_only)
    manifest_outputs = save_loco_manifests(
        folds,
        output_dir=evaluation_manifest_dir,
        feature_file_hash=bundle.feature_file_hash,
        source_fingerprints=bundle.source_fingerprints,
        overwrite=overwrite,
    )
    fold_results = [
        _run_fold(fold, features=bundle.features, candidate=candidate, model_root=model_root, save_artifacts=save_artifacts, overwrite=overwrite)
        for fold in folds
    ]
    transfer_results = run_transfer_matrix(bundle, features=bundle.features) if run_transfer else []
    shortcut = run_shortcut_diagnostics(bundle, features=bundle.features) if run_shortcut else {"status": "skipped", "diagnostic_only": True}
    pooled = _pooled_baseline(DEFAULT_POOLED_BASELINE)
    gap = corpus_generalization_gap(pooled.get("test_macro_f1"), [(fold.test_metrics or {}).get("macro_f1") for fold in fold_results])
    blockers = []
    if shortcut.get("strong_domain_shortcut_risk"):
        blockers.append("Strong corpus identity predictability indicates domain shortcut risk.")
    if gap.get("corpus_generalization_gap") is not None and (gap.get("loco_mean_macro_f1") or 0.0) < (pooled.get("test_macro_f1") or 0.0):
        blockers.append("LOCO performance is below pooled locked-split performance; pooled results should not be used as unseen-corpus evidence.")
    recommendations = [
        "Treat Speech baseline as research-only and inactive.",
        "Use LOCO and transfer results to decide whether any Speech claim survives corpus/domain shift.",
        "Collect natural, consented Sri Lankan speech data before production or clinical claims.",
        "Do not use corpus identity, speaker keys, filenames, or raw audio metadata as predictive features.",
    ]
    fold_dicts = [fold.__dict__ for fold in fold_results]
    summary = {
        "evaluation_version": SPEECH_DOMAIN_EVALUATION_VERSION,
        "policy_version": SPEECH_LOCO_POLICY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "feature_file_hash": bundle.feature_file_hash,
        "source_fingerprints": bundle.source_fingerprints,
        "folds": fold_dicts,
        "transfer_matrix": [item.__dict__ for item in transfer_results],
        "pooled_baseline_reference": pooled,
        "cross_corpus_summary": {
            "fold_count": len(fold_results),
            "held_out_corpora": [fold.test_corpus for fold in fold_results],
            "shared_labels_only": shared_labels_only,
            "fusion_training_valid": False,
            "interpretation": "LOCO estimates unseen-corpus degradation; it is not clinical validation.",
        },
        "corpus_gap_summary": gap,
        "shortcut_risk_findings": shortcut,
        "blockers": blockers,
        "recommendations": recommendations,
        "research_readiness": "research_only_domain_shift_evaluated_not_deployable",
        "registration_status": "not_registered",
        "activation_status": "inactive",
        "production_outputs_created": False,
    }
    loco_rows = [
        {
            "fold_name": fold.fold_name,
            "test_corpus": fold.test_corpus,
            "candidate_model": fold.candidate_model,
            "train_count": fold.train_count,
            "validation_count": fold.validation_count,
            "test_count": fold.test_count,
            "validation_macro_f1": fold.validation_metrics.get("macro_f1"),
            "test_macro_f1": fold.test_metrics.get("macro_f1"),
            "test_balanced_accuracy": fold.test_metrics.get("balanced_accuracy"),
            "worst_class_recall": fold.test_metrics.get("worst_class_recall"),
            "unsupported_labels": fold.unsupported_labels,
            "warnings": fold.warnings,
        }
        for fold in fold_results
    ]
    class_rows = [row for fold in fold_results for row in per_class_rows(fold.fold_name, "test", fold.test_metrics)]
    confusion = {fold.fold_name: fold.confusion_matrix for fold in fold_results}
    transfer_macro_rows = [
        {"train_corpus": item.train_corpus, "test_corpus": item.test_corpus, "macro_f1": item.metrics.get("macro_f1"), "warnings": item.warnings}
        for item in transfer_results
    ]
    transfer_balanced_rows = [
        {"train_corpus": item.train_corpus, "test_corpus": item.test_corpus, "balanced_accuracy": item.metrics.get("balanced_accuracy"), "warnings": item.warnings}
        for item in transfer_results
    ]
    files_for_inventory = [*manifest_outputs.values(), *[path for fold in fold_results for path in fold.artifact_paths]]
    report_outputs = save_domain_shift_reports(
        output_dir=output_dir,
        summary=summary,
        loco_rows=loco_rows,
        per_class_rows=class_rows,
        confusion_matrices=confusion,
        transfer_macro_rows=transfer_macro_rows,
        transfer_balanced_rows=transfer_balanced_rows,
        shortcut_diagnostics=shortcut,
        feature_distribution=feature_distribution_rows(bundle.records, bundle.features),
        comparison=gap,
        files_for_inventory=files_for_inventory,
        overwrite=overwrite,
    )
    summary["report_paths"] = {key: repo_relative(value) for key, value in report_outputs.items()}
    if report_path:
        write_json(report_path, summary, overwrite=overwrite)
    return summary
