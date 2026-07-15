"""Modality-specific Phase 2 validation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from app.ml.common import paths
from app.ml.common.serialization import load_dataset_fingerprint
from app.ml.validation.checks import _directory_fingerprint_manifest_matches
from app.ml.validation import checks as common_checks
from app.ml.validation.readiness import (
    classify_modality_readiness,
    model_training_status_for_readiness,
    split_status_for_readiness,
)
from app.ml.validation.schemas import ModalityValidationResult, ValidationCheckResult, ValidationStatus


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _config(config_dir: str | Path | None, name: str) -> Path:
    root = _repo_path(config_dir or paths.get_ml_research_root() / "configs")
    return root / name


def _generated(generated_root: str | Path | None, relative: str) -> Path:
    root = _repo_path(generated_root or paths.get_generated_root())
    return root / relative


def _load_json(path: str | Path) -> dict[str, Any]:
    with _repo_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _safe_load_json(path: str | Path) -> dict[str, Any]:
    try:
        return _load_json(path)
    except Exception:
        return {}


def _features(schema: dict[str, Any]) -> list[str]:
    return [str(item.get("name")) for item in schema.get("features", []) if isinstance(item, dict) and item.get("name")]


def _versions(report: dict[str, Any], schema: dict[str, Any] | None = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    schema = schema or {}
    return (
        str(report.get("dataset_version") or schema.get("dataset_version") or "v1"),
        str(report.get("preprocessing_version") or schema.get("preprocessing_version") or "") or None,
        str(report.get("feature_schema_version") or schema.get("feature_schema_version") or "") or None,
    )


def _audit_status(audit: dict[str, Any]) -> ValidationStatus:
    status = str(audit.get("summary_status") or audit.get("status") or "").lower()
    if status == "passed":
        return ValidationStatus.PASSED
    if status == "warning":
        return ValidationStatus.PASSED_WITH_WARNINGS
    if status in {"error", "failed"}:
        return ValidationStatus.FAILED
    return ValidationStatus.NOT_APPLICABLE


def _result(
    modality: str,
    dataset_name: str,
    report: dict[str, Any],
    schema: dict[str, Any],
    audit: dict[str, Any],
    checks: list[ValidationCheckResult],
    signals: dict[str, Any],
    *,
    source_fingerprint_verified: bool = False,
    grouping_columns: Optional[list[str]] = None,
    leakage_findings: Optional[list[str]] = None,
    duplicate_findings: Optional[list[str]] = None,
    missing_value_findings: Optional[list[str]] = None,
    identifier_findings: Optional[list[str]] = None,
    privacy_findings: Optional[list[str]] = None,
) -> ModalityValidationResult:
    readiness, warnings, blockers = classify_modality_readiness(modality, signals, checks)
    dataset_version, preprocessing_version, feature_schema_version = _versions(report, schema)
    features = _features(schema)
    return ModalityValidationResult(
        modality=modality,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        preprocessing_version=preprocessing_version,
        feature_schema_version=feature_schema_version,
        source_fingerprint=report.get("source_fingerprint")
        or report.get("source_fingerprint_hash")
        or report.get("fingerprint")
        or next(iter((report.get("source_fingerprints") or {}).values()), None),
        source_fingerprint_verified=source_fingerprint_verified,
        audit_status=_audit_status(audit),
        preprocessing_status=ValidationStatus.PASSED_WITH_WARNINGS if warnings else ValidationStatus.PASSED,
        row_or_file_count=int(report.get("source_row_count") or report.get("source_file_count") or report.get("source_records") or audit.get("row_count") or 0),
        output_record_count=int(report.get("output_record_count") or report.get("output_row_count") or report.get("output_records") or 0),
        feature_count=len(features),
        target_columns=[str(item) for item in schema.get("target_columns", [])],
        grouping_columns=grouping_columns or [],
        sensitive_columns=[str(item) for item in schema.get("sensitive_columns", [])],
        excluded_columns=[str(item) for item in schema.get("excluded_columns", [])],
        leakage_findings=leakage_findings or [],
        duplicate_findings=duplicate_findings or [],
        missing_value_findings=missing_value_findings or [],
        identifier_findings=identifier_findings or [],
        privacy_findings=privacy_findings or [],
        split_readiness=split_status_for_readiness(readiness),
        model_training_readiness=model_training_status_for_readiness(readiness),
        readiness_classification=readiness,
        checks=checks,
        warnings=warnings,
        blockers=blockers,
    )


def _basic_output_checks(modality: str, output_root: Path, report_path: Path, schema_path: Path | None = None) -> list[ValidationCheckResult]:
    checks = [
        common_checks.check_output_outside_raw_dataset(output_root, modality=modality),
        common_checks.check_split_manifest_absent(output_root, modality=modality),
        common_checks.check_model_artifacts_absent(output_root, modality=modality),
        common_checks.check_training_outputs_absent(output_root, modality=modality),
        common_checks.check_report_privacy(report_path, modality=modality),
    ]
    if schema_path is not None:
        checks.extend(
            [
                common_checks.check_json_schema_loads(schema_path, modality=modality, required_keys=("feature_schema_version", "features")),
                common_checks.check_feature_schema_version_present(_safe_load_json(schema_path), modality=modality),
                common_checks.check_deterministic_versions(_safe_load_json(schema_path), modality=modality),
            ]
        )
    return checks


def _fingerprint_checks(
    modality: str,
    fingerprint_path: Path,
    dataset_config_path: Path,
    report_hash: Optional[str],
    *,
    skip_source_reverification: bool,
) -> tuple[list[ValidationCheckResult], bool]:
    checks = [
        common_checks.check_fingerprint_present(fingerprint_path, modality=modality),
        common_checks.check_source_hash_matches_manifest(report_hash, fingerprint_path, modality=modality),
    ]
    verify = common_checks.check_fingerprint_matches_source(
        fingerprint_path,
        dataset_config_path,
        modality=modality,
        skip_reverification=skip_source_reverification,
    )
    checks.append(verify)
    return checks, str(verify.status) == ValidationStatus.PASSED.value


def validate_dass21_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    report_path = _generated(generated_root, "reports/dass21/scoring_validation_v1.json")
    mapping_path = _config(config_dir, "dass21.item_mapping.v1.json")
    fp_path = _generated(generated_root, "manifests/fingerprints/dass21-v1.json")
    dataset_path = _config(config_dir, "dass21.dataset.example.json")
    report = _safe_load_json(report_path)
    mapping = _safe_load_json(mapping_path)
    checks = [
        common_checks.check_required_file_exists(report_path, modality="dass21"),
        common_checks.check_json_schema_loads(report_path, modality="dass21", required_keys=("scoring_version", "mapping_version")),
        common_checks.check_required_file_exists(_repo_path("backend/app/ml/preprocessing/dass21/scoring.py"), modality="dass21", check_name="authoritative_scoring_module_exists"),
        common_checks.check_deterministic_versions(report, modality="dass21"),
    ]
    counts = Counter(item.get("subscale") for item in mapping.get("selected_dass21_items", []))
    if counts == {"depression": 7, "anxiety": 7, "stress": 7}:
        checks.append(common_checks.passed("dass21_subscale_item_count", "dass21", "DASS21 mapping has exactly 7 items per subscale."))
    else:
        checks.append(
            common_checks.blocked(
                "dass21_subscale_item_count",
                "dass21",
                "DASS21 mapping does not have exactly 7 items per subscale.",
                "Fix the authoritative DASS21 item mapping before using scoring output.",
                details={"subscale_count": dict(counts)},
            )
        )
    checks.append(common_checks.check_fingerprint_present(fp_path, modality="dass21"))
    verify = common_checks.check_fingerprint_matches_source(
        fp_path,
        dataset_path,
        modality="dass21",
        skip_reverification=skip_source_reverification,
    )
    checks.append(verify)
    verified = str(verify.status) == ValidationStatus.PASSED.value
    signals = {"warnings": ["Rule-based scoring only; no ML dataset output is required."]}
    return _result(
        "dass21",
        "dass21",
        report,
        {},
        {},
        checks,
        signals,
        source_fingerprint_verified=verified,
    )


def validate_profile_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/profile/v1")
    report_path = output_root / "profile_preprocessing_report.json"
    schema_path = output_root / "profile_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    audit = _safe_load_json(_generated(generated_root, "audits/student-profile/v1/audit.json"))
    checks = _basic_output_checks("profile", output_root, report_path, schema_path)
    checks.extend(
        [
            common_checks.check_required_file_exists(output_root / "canonical_profile.csv", modality="profile"),
            common_checks.check_required_file_exists(output_root / "profile_record_manifest.json", modality="profile"),
            common_checks.check_target_not_in_features(_features(schema), schema.get("target_columns", []), modality="profile"),
            common_checks.check_metadata_not_in_features(_features(schema), ["source_timestamp", "sought_specialist_treatment", "Timestamp"], modality="profile"),
            common_checks.check_no_nan_or_infinity(output_root / "canonical_profile.csv", modality="profile"),
        ]
    )
    fp_checks, verified = _fingerprint_checks(
        "profile",
        _generated(generated_root, "manifests/fingerprints/student-profile-v1.json"),
        _config(config_dir, "profile.dataset.example.json"),
        report.get("source_fingerprint"),
        skip_source_reverification=skip_source_reverification,
    )
    checks.extend(fp_checks)
    signals = {
        "warnings": report.get("warnings", []),
        "small_sample_size": int(report.get("source_row_count") or 0) < 500,
        "weak_or_self_reported_labels": True,
        "fairness_concerns": True,
    }
    return _result(
        "profile",
        "student-profile",
        report,
        schema,
        audit,
        checks,
        signals,
        source_fingerprint_verified=verified,
        leakage_findings=["Treatment-seeking and timestamp are excluded by default."],
        missing_value_findings=[f"{key}: {value}" for key, value in (report.get("missing_value_summary") or {}).items()],
    )


def validate_mood_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/mood/v1")
    report_path = output_root / "mood_readiness_report.json"
    schema_path = output_root / "mood_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    audit = _safe_load_json(_generated(generated_root, "audits/mood/v1/audit.json"))
    checks = _basic_output_checks("mood", output_root, report_path, schema_path)
    checks.append(common_checks.check_required_file_exists(output_root / "generated-synthetic/mood_features.csv", modality="mood"))
    fp_checks, verified = _fingerprint_checks(
        "mood",
        _generated(generated_root, "manifests/fingerprints/daily-mood-v1.json"),
        _config(config_dir, "mood.dataset.example.json"),
        report.get("fingerprint"),
        skip_source_reverification=skip_source_reverification,
    )
    checks.extend(fp_checks)
    pending_real = not bool(report.get("real_offline_production_export_exists"))
    if pending_real:
        checks.append(
            common_checks.blocked(
                "real_production_mood_export_absent",
                "mood",
                "No reviewed real offline production-like mood export is available.",
                "Accept a real-data policy and validate an approved longitudinal export before mood split design.",
            )
        )
    signals = {
        "warnings": report.get("generated_or_synthetic_limitations", []),
        "pending_real_data": pending_real,
        "non_clinical_label": True,
    }
    return _result(
        "mood",
        "daily-mood",
        report,
        schema,
        audit,
        checks,
        signals,
        source_fingerprint_verified=verified,
        grouping_columns=["ParticipantID"],
        leakage_findings=["Temporal feature schema states current/prior-only windows; no clinical risk label is present."],
    )


def validate_text_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/text/v1")
    report_path = output_root / "text_preprocessing_report.json"
    schema_path = output_root / "text_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    audit = _safe_load_json(_generated(generated_root, "audits/text/v1/audit.json"))
    checks = _basic_output_checks("text", output_root, report_path, schema_path)
    checks.extend(
        [
            common_checks.check_required_file_exists(output_root / "canonical_text.csv", modality="text"),
            common_checks.check_duplicate_manifest_present(output_root / "text_duplicate_manifest.json", modality="text"),
            common_checks.check_conflict_quarantine_present(output_root / "text_conflict_quarantine.csv", modality="text"),
            common_checks.check_required_file_exists(output_root / "text_source_overlap_report.json", modality="text", check_name="raw_test_overlap_report_exists"),
            common_checks.check_target_not_in_features(_features(schema), schema.get("target_columns", []), modality="text"),
            common_checks.check_identifiers_not_in_features(_features(schema), schema.get("excluded_columns", []), modality="text"),
        ]
    )
    fp_checks, verified = _fingerprint_checks(
        "text",
        _generated(generated_root, "manifests/fingerprints/mental-health-text-v1.json"),
        _config(config_dir, "text.dataset.example.json"),
        report.get("source_fingerprint"),
        skip_source_reverification=skip_source_reverification,
    )
    checks.extend(fp_checks)
    signals = {
        "warnings": report.get("warnings", []),
        "duplicate_restrictions": int(report.get("conflicting_duplicate_group_count") or 0) > 0,
        "incomplete_group_ids": True,
        "non_clinical_label": True,
    }
    return _result(
        "text",
        "mental-health-text",
        report,
        schema,
        audit,
        checks,
        signals,
        source_fingerprint_verified=verified,
        duplicate_findings=[
            f"Exact duplicate groups: {report.get('exact_duplicate_group_count', 0)}",
            f"Conflicting duplicate groups quarantined: {report.get('conflicting_duplicate_group_count', 0)}",
            f"Reference overlap count: {(report.get('leakage_checks') or {}).get('reference_test_overlap_count', 0)}",
        ],
        privacy_findings=["Privacy replacement is conservative; no raw text is included in validation reports."],
        identifier_findings=["No complete user/group identifier is available for group-aware splitting."],
    )


def _verify_speech_fingerprint(fingerprint_path: Path, *, skip_source_reverification: bool) -> ValidationCheckResult:
    modality = "speech"
    if skip_source_reverification:
        return common_checks.warning(
            "fingerprint_matches_source",
            modality,
            "Source fingerprint reverification was explicitly skipped.",
            "Run validation without --skip-source-reverification before split design.",
            artifact_paths=[fingerprint_path],
        )
    try:
        fingerprint = load_dataset_fingerprint(fingerprint_path)
        source = _repo_path(fingerprint.source_relative_path)
    except Exception as exc:
        return common_checks.failed(
            "fingerprint_matches_source",
            modality,
            "Speech fingerprint could not be loaded.",
            f"Restore the fingerprint manifest: {exc}",
            artifact_paths=[fingerprint_path],
        )
    if fingerprint.source_type == "directory":
        verified = _directory_fingerprint_manifest_matches(fingerprint)
    else:
        from app.ml.common.fingerprinting import verify_fingerprint_against_path

        verified = verify_fingerprint_against_path(fingerprint, source)
    if verified:
        return common_checks.passed("fingerprint_matches_source", modality, "Speech corpus fingerprint matches current source.", artifact_paths=[fingerprint_path])
    return common_checks.blocked(
        "fingerprint_matches_source",
        modality,
        "Speech corpus fingerprint mismatch detected.",
        "Freeze split design until the source/fingerprint mismatch is reconciled.",
        artifact_paths=[fingerprint_path],
    )


def validate_speech_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/speech/v1")
    report_path = output_root / "speech_preprocessing_report.json"
    schema_path = output_root / "speech_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    checks = _basic_output_checks("speech", output_root, report_path, schema_path)
    checks.extend(
        [
            common_checks.check_required_file_exists(output_root / "speech_canonical_manifest.csv", modality="speech"),
            common_checks.check_required_file_exists(output_root / "speech_features.csv", modality="speech"),
            common_checks.check_duplicate_manifest_present(output_root / "speech_duplicate_manifest.json", modality="speech"),
            common_checks.check_target_not_in_features(_features(schema), schema.get("target_columns", []), modality="speech"),
            common_checks.check_metadata_not_in_features(_features(schema), ["corpus_name", "source_file", "speaker_id", "safe_speaker_key"], modality="speech"),
        ]
    )
    verified_results = []
    for name in ("crema", "ravdess", "savee", "tess"):
        fp_path = _generated(generated_root, f"manifests/fingerprints/speech/{name}-v1.json")
        checks.append(common_checks.check_fingerprint_present(fp_path, modality="speech"))
        verify = _verify_speech_fingerprint(fp_path, skip_source_reverification=skip_source_reverification)
        verified_results.append(str(verify.status) == ValidationStatus.PASSED.value)
        checks.append(verify)
    signals = {
        "warnings": report.get("warnings", []),
        "corpus_variation": True,
        "missing_full_feature_extraction": bool(report.get("feature_missing_summary")),
        "non_clinical_label": True,
    }
    return _result(
        "speech",
        "speech-emotion",
        report,
        schema,
        {},
        checks,
        signals,
        source_fingerprint_verified=all(verified_results),
        grouping_columns=["safe_speaker_key"],
        duplicate_findings=[f"Duplicate audio hash groups: {(report.get('duplicate_summary') or {}).get('duplicate_audio_hash_group_count', 0)}"],
        missing_value_findings=[f"Feature extraction status documented for {len(report.get('feature_missing_summary') or {})} fields."],
        identifier_findings=["safe_speaker_key is available for grouping and excluded from predictive features."],
    )


def validate_face_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/face/v1")
    report_path = output_root / "face_preprocessing_report.json"
    schema_path = output_root / "face_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    audit = _safe_load_json(_generated(generated_root, "audits/face/v1/audit.json"))
    conflicts = _safe_load_json(output_root / "face_cross_label_conflicts.json")
    checks = _basic_output_checks("face", output_root, report_path, schema_path)
    checks.extend(
        [
            common_checks.check_required_file_exists(output_root / "face_canonical_manifest.csv", modality="face"),
            common_checks.check_duplicate_manifest_present(output_root / "face_duplicate_manifest.json", modality="face"),
            common_checks.check_required_file_exists(output_root / "face_cross_split_overlap.json", modality="face", check_name="cross_split_duplicate_report_exists"),
            common_checks.check_required_file_exists(output_root / "face_cross_label_conflicts.json", modality="face", check_name="cross_label_conflict_report_exists"),
            common_checks.check_target_not_in_features(_features(schema), schema.get("target_columns", []), modality="face"),
        ]
    )
    fp_checks, verified = _fingerprint_checks(
        "face",
        _generated(generated_root, "manifests/fingerprints/face/facial-emotion-v1.json"),
        _config(config_dir, "face.dataset.example.json"),
        report.get("source_fingerprint"),
        skip_source_reverification=skip_source_reverification,
    )
    checks.extend(fp_checks)
    critical_conflicts = str(conflicts.get("severity")) == "critical" and int(report.get("cross_label_duplicate_count") or 0) > 0
    if critical_conflicts:
        checks.append(
            common_checks.blocked(
                "critical_cross_label_conflicts_unresolved",
                "face",
                "Critical cross-label duplicate conflicts remain unresolved.",
                "Resolve or exclude conflicting duplicate image groups before split design.",
                details={"cross_label_duplicate_count": int(report.get("cross_label_duplicate_count") or 0)},
                artifact_paths=[output_root / "face_cross_label_conflicts.json"],
            )
        )
    signals = {
        "warnings": report.get("warnings", []),
        "critical_duplicate_conflicts": critical_conflicts,
        "unresolved_leakage_conflicts": int(report.get("cross_split_duplicate_count") or 0) > 0,
        "leakage_related": True,
    }
    return _result(
        "face",
        "facial-emotion",
        report,
        schema,
        audit,
        checks,
        signals,
        source_fingerprint_verified=verified,
        duplicate_findings=[
            f"Cross-split duplicate groups: {report.get('cross_split_duplicate_count', 0)}",
            f"Cross-label duplicate groups: {report.get('cross_label_duplicate_count', 0)}",
        ],
        leakage_findings=["Subject identifiers are not available; predefined train/test folders are metadata only."],
    )


def validate_behavioral_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> ModalityValidationResult:
    output_root = _generated(generated_root, "preprocessing/behavioral/v1")
    report_path = output_root / "behavioral_readiness_report.json"
    schema_path = output_root / "behavioral_feature_schema.json"
    report = _safe_load_json(report_path)
    schema = _safe_load_json(schema_path)
    checks = _basic_output_checks("behavioral", output_root, report_path, schema_path)
    checks.extend(
        [
            common_checks.check_required_file_exists(output_root / "generated-synthetic/behavioral_features.csv", modality="behavioral"),
            common_checks.check_required_file_exists(output_root / "generated-synthetic/behavioral_record_manifest.json", modality="behavioral"),
        ]
    )
    no_real_fp = not _generated(generated_root, "manifests/fingerprints/behavioral-v1.json").exists()
    if no_real_fp:
        checks.append(
            common_checks.passed(
                "no_real_behavioral_fingerprint",
                "behavioral",
                "No real behavioral source fingerprint exists, matching the current source status.",
            )
        )
        checks.append(
            common_checks.blocked(
                "real_behavioral_source_absent",
                "behavioral",
                "No real consented behavioral source dataset is available.",
                "Collect or approve a real offline behavioral dataset before model-training readiness can be evaluated.",
            )
        )
    signals = {
        "warnings": [report.get("reason", "Behavioral data are schema/synthetic only.")],
        "engineering_only": True,
        "synthetic_only": True,
        "real_source_exists": False,
    }
    return _result(
        "behavioral",
        "behavioral-telemetry",
        report,
        schema,
        {},
        checks,
        signals,
        source_fingerprint_verified=False,
        grouping_columns=["participant_key"],
        leakage_findings=["Synthetic engineering schema is prior/current-only; no clinical labels are present."],
        privacy_findings=["No real telemetry payloads are included in Phase 2 outputs."],
    )


VALIDATORS = {
    "dass21": validate_dass21_modality,
    "profile": validate_profile_modality,
    "mood": validate_mood_modality,
    "text": validate_text_modality,
    "speech": validate_speech_modality,
    "face": validate_face_modality,
    "behavioral": validate_behavioral_modality,
}


def validate_modalities(
    modalities: Iterable[str],
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    skip_source_reverification: bool = False,
) -> list[ModalityValidationResult]:
    results = []
    for modality in modalities:
        validator = VALIDATORS[modality]
        results.append(
            validator(
                config_dir=config_dir,
                generated_root=generated_root,
                skip_source_reverification=skip_source_reverification,
            )
        )
    return results
