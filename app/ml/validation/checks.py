"""Reusable read-only validation checks for Phase 2 artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from pydantic.v1 import ValidationError

from app.ml.common import paths
from app.ml.common.fingerprinting import verify_dataset_fingerprint
from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.common.serialization import load_dataset_config, load_dataset_fingerprint
from app.ml.validation.constants import MODEL_ARTIFACT_EXTENSIONS, MODEL_ARTIFACT_NAMES, TEXT_PRIVACY_PATTERNS
from app.ml.validation.schemas import (
    ValidationCheckResult,
    ValidationSeverity,
    ValidationStatus,
    repo_relative,
)


MAX_TEXT_READ_BYTES = 256 * 1024
MAX_CSV_ROWS = 5000


def _path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _artifact(path_like: str | Path) -> list[str]:
    try:
        return [repo_relative(_path(path_like))]
    except Exception:
        return []


def _result(
    check_name: str,
    modality: str,
    status: ValidationStatus | str,
    severity: ValidationSeverity | str,
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    recommendation: Optional[str] = None,
    artifact_paths: Optional[Iterable[str | Path]] = None,
) -> ValidationCheckResult:
    artifacts = []
    for path_like in artifact_paths or []:
        try:
            artifacts.append(repo_relative(_path(path_like)))
        except Exception:
            # External temp fixtures are valid for unit checks, but validation
            # reports must never serialize machine-specific absolute paths.
            continue
    return ValidationCheckResult(
        check_name=check_name,
        modality=modality,
        status=status,
        severity=severity,
        message=message,
        details=dict(details or {}),
        recommendation=recommendation,
        artifact_paths=artifacts,
    )


def passed(check_name: str, modality: str, message: str, **kwargs: Any) -> ValidationCheckResult:
    return _result(check_name, modality, ValidationStatus.PASSED, ValidationSeverity.INFO, message, **kwargs)


def warning(check_name: str, modality: str, message: str, recommendation: str, **kwargs: Any) -> ValidationCheckResult:
    return _result(
        check_name,
        modality,
        ValidationStatus.PASSED_WITH_WARNINGS,
        ValidationSeverity.WARNING,
        message,
        recommendation=recommendation,
        **kwargs,
    )


def failed(check_name: str, modality: str, message: str, recommendation: str, **kwargs: Any) -> ValidationCheckResult:
    return _result(
        check_name,
        modality,
        ValidationStatus.FAILED,
        ValidationSeverity.ERROR,
        message,
        recommendation=recommendation,
        **kwargs,
    )


def blocked(check_name: str, modality: str, message: str, recommendation: str, **kwargs: Any) -> ValidationCheckResult:
    return _result(
        check_name,
        modality,
        ValidationStatus.BLOCKED,
        ValidationSeverity.CRITICAL,
        message,
        recommendation=recommendation,
        **kwargs,
    )


def check_required_file_exists(path: str | Path, *, modality: str, check_name: str = "required_file_exists") -> ValidationCheckResult:
    resolved = _path(path)
    if resolved.exists() and resolved.is_file():
        return passed(check_name, modality, "Required artifact exists.", artifact_paths=[resolved])
    return failed(
        check_name,
        modality,
        "Required artifact is missing.",
        "Generate or restore the expected Phase 2 artifact before proceeding.",
        artifact_paths=[resolved],
    )


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        with _path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def check_json_schema_loads(
    path: str | Path,
    *,
    modality: str,
    required_keys: Iterable[str] = (),
    check_name: str = "json_schema_loads",
) -> ValidationCheckResult:
    try:
        payload = _load_json_object(path)
    except Exception as exc:
        return failed(check_name, modality, "JSON artifact is malformed.", f"Fix malformed JSON: {exc}", artifact_paths=[path])
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return failed(
            check_name,
            modality,
            "JSON artifact is missing required fields.",
            "Regenerate the artifact with the expected Phase 2 schema.",
            details={"missing_key_count": len(missing), "missing_keys": missing},
            artifact_paths=[path],
        )
    return passed(check_name, modality, "JSON artifact loads with required fields.", artifact_paths=[path])


def check_fingerprint_present(path: str | Path, *, modality: str, check_name: str = "fingerprint_present") -> ValidationCheckResult:
    try:
        payload = _load_json_object(path)
    except Exception as exc:
        return failed(check_name, modality, "Fingerprint manifest cannot be loaded.", f"Restore a valid fingerprint manifest: {exc}", artifact_paths=[path])
    if payload.get("combined_sha256") and payload.get("dataset_version"):
        return passed(
            check_name,
            modality,
            "Source fingerprint manifest is present.",
            details={"file_count": int(payload.get("file_count") or 0)},
            artifact_paths=[path],
        )
    return failed(
        check_name,
        modality,
        "Fingerprint manifest is missing required identity fields.",
        "Regenerate the source fingerprint from the reviewed dataset config.",
        artifact_paths=[path],
    )


def check_fingerprint_matches_source(
    fingerprint_path: str | Path,
    dataset_config_path: str | Path,
    *,
    modality: str,
    skip_reverification: bool = False,
    check_name: str = "fingerprint_matches_source",
) -> ValidationCheckResult:
    if skip_reverification:
        return warning(
            check_name,
            modality,
            "Source fingerprint reverification was explicitly skipped.",
            "Run validation without --skip-source-reverification before split design.",
            artifact_paths=[fingerprint_path, dataset_config_path],
        )
    try:
        fingerprint = load_dataset_fingerprint(_path(fingerprint_path))
        dataset_config = load_dataset_config(_path(dataset_config_path))
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        return failed(
            check_name,
            modality,
            "Fingerprint or dataset config could not be loaded for reverification.",
            f"Fix the malformed artifact before proceeding: {exc}",
            artifact_paths=[fingerprint_path, dataset_config_path],
        )
    if fingerprint.source_type == "directory":
        verified = _directory_fingerprint_manifest_matches(fingerprint)
    else:
        verified = verify_dataset_fingerprint(fingerprint, dataset_config)
    if verified:
        return passed(check_name, modality, "Source fingerprint matches the current source.", artifact_paths=[fingerprint_path, dataset_config_path])
    return blocked(
        check_name,
        modality,
        "Source fingerprint mismatch detected.",
        "Freeze preprocessing work and regenerate fingerprints only after confirming the reviewed source change.",
        artifact_paths=[fingerprint_path, dataset_config_path],
    )


def _directory_fingerprint_manifest_matches(fingerprint) -> bool:
    """Verify large directory manifests without rereading all media bytes."""
    root = paths.get_repository_root()
    files = getattr(fingerprint, "files", []) or []
    if fingerprint.file_count != len(files):
        return False
    total = 0
    for file_entry in files:
        path = root / file_entry.relative_path
        if not path.exists() or not path.is_file():
            return False
        size = path.stat().st_size
        if size != file_entry.size_bytes:
            return False
        total += size
    return total == fingerprint.total_bytes


def check_dataset_version_present(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    return _check_version_present(payload_or_path, "dataset_version", modality=modality)


def check_preprocessing_version_present(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    return _check_version_present(payload_or_path, "preprocessing_version", modality=modality)


def check_feature_schema_version_present(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    return _check_version_present(payload_or_path, "feature_schema_version", modality=modality)


def check_config_hash_present(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    return _check_version_present(payload_or_path, "config_hash", modality=modality, check_name="config_hash_present")


def check_source_hash_present(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    payload = _payload(payload_or_path)
    value = payload.get("source_fingerprint") or payload.get("source_fingerprint_hash") or payload.get("combined_sha256")
    if value:
        return passed("source_hash_present", modality, "Source hash is present.")
    return failed("source_hash_present", modality, "Source hash is missing.", "Record the source fingerprint/hash in the artifact.")


def check_source_hash_matches_manifest(
    source_hash: Optional[str],
    manifest_path: str | Path,
    *,
    modality: str,
    check_name: str = "source_hash_matches_manifest",
) -> ValidationCheckResult:
    try:
        manifest = _load_json_object(manifest_path)
    except Exception as exc:
        return failed(check_name, modality, "Fingerprint manifest cannot be loaded.", f"Restore the manifest: {exc}", artifact_paths=[manifest_path])
    expected = manifest.get("combined_sha256")
    if source_hash and expected and str(source_hash).lower() == str(expected).lower():
        return passed(check_name, modality, "Reported source hash matches the fingerprint manifest.", artifact_paths=[manifest_path])
    return blocked(
        check_name,
        modality,
        "Reported source hash does not match the fingerprint manifest.",
        "Reconcile the report and fingerprint before proceeding.",
        details={"has_report_hash": bool(source_hash), "has_manifest_hash": bool(expected)},
        artifact_paths=[manifest_path],
    )


def _payload(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, Mapping):
        return dict(payload_or_path)
    return _load_json_object(payload_or_path)


def _check_version_present(
    payload_or_path: Mapping[str, Any] | str | Path,
    key: str,
    *,
    modality: str,
    check_name: Optional[str] = None,
) -> ValidationCheckResult:
    try:
        payload = _payload(payload_or_path)
    except Exception as exc:
        return failed(check_name or f"{key}_present", modality, "Artifact cannot be loaded.", f"Fix artifact loading: {exc}")
    if str(payload.get(key) or "").strip():
        return passed(check_name or f"{key}_present", modality, f"{key} is present.")
    return failed(check_name or f"{key}_present", modality, f"{key} is missing.", f"Add {key} to the Phase 2 artifact.")


def check_output_outside_raw_dataset(output_path: str | Path, *, modality: str) -> ValidationCheckResult:
    resolved = _path(output_path)
    if not paths.is_path_inside(paths.get_raw_dataset_root(), resolved):
        return passed("output_outside_raw_dataset", modality, "Output path is outside the raw dataset root.", artifact_paths=[resolved])
    return blocked(
        "output_outside_raw_dataset",
        modality,
        "Generated output path points inside the raw dataset root.",
        "Move generated outputs under generated/ and leave raw datasets immutable.",
        artifact_paths=[resolved],
    )


def check_no_absolute_paths(value: Any, *, modality: str, artifact_path: str | Path | None = None) -> ValidationCheckResult:
    text = _bounded_text(value)
    matches = []
    for name in ("windows_absolute_path", "posix_absolute_path"):
        matches.extend(re.findall(TEXT_PRIVACY_PATTERNS[name], text))
    if not matches:
        return passed("no_absolute_paths", modality, "No machine-specific absolute paths were detected.", artifact_paths=_artifact(artifact_path) if artifact_path else [])
    return failed(
        "no_absolute_paths",
        modality,
        "Machine-specific absolute paths were detected.",
        "Rewrite reports with repository-relative paths only.",
        details={"absolute_path_finding_count": len(matches)},
        artifact_paths=_artifact(artifact_path) if artifact_path else [],
    )


def check_no_path_traversal(value: Any, *, modality: str, artifact_path: str | Path | None = None) -> ValidationCheckResult:
    text = _bounded_text(value)
    if "../" not in text.replace("\\", "/"):
        return passed("no_path_traversal", modality, "No path traversal references were detected.", artifact_paths=_artifact(artifact_path) if artifact_path else [])
    return failed(
        "no_path_traversal",
        modality,
        "Path traversal references were detected.",
        "Rewrite artifact paths as repository-relative paths without '..'.",
        artifact_paths=_artifact(artifact_path) if artifact_path else [],
    )


def _bounded_text(value: Any) -> str:
    if isinstance(value, (str, Path)) and Path(str(value)).exists() and Path(str(value)).is_file():
        with _path(value).open("rb") as handle:
            return handle.read(MAX_TEXT_READ_BYTES).decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)[:MAX_TEXT_READ_BYTES]
    return str(value)[:MAX_TEXT_READ_BYTES]


def check_no_nan_or_infinity(path: str | Path, *, modality: str, max_rows: int = MAX_CSV_ROWS) -> ValidationCheckResult:
    resolved = _path(path)
    try:
        if resolved.suffix.lower() == ".json":
            payload = _load_json_object(resolved)
            count = _count_non_finite(payload)
        else:
            count = _count_non_finite_csv(resolved, max_rows=max_rows)
    except Exception as exc:
        return failed("no_nan_or_infinity", modality, "Artifact could not be scanned for non-finite values.", f"Fix malformed data artifact: {exc}", artifact_paths=[resolved])
    if count == 0:
        return passed("no_nan_or_infinity", modality, "No NaN or infinity values were detected in the bounded scan.", artifact_paths=[resolved])
    return blocked(
        "no_nan_or_infinity",
        modality,
        "NaN or infinity values were detected.",
        "Repair preprocessing so feature outputs contain only finite values or explicit documented missing values.",
        details={"non_finite_count": count, "bounded_rows": max_rows},
        artifact_paths=[resolved],
    )


def _count_non_finite(value: Any) -> int:
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, dict):
        return sum(_count_non_finite(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_non_finite(item) for item in value)
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return 1
    return 0


def _count_non_finite_csv(path: Path, *, max_rows: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            for value in row.values():
                cleaned = str(value).strip().lower()
                if cleaned in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                    count += 1
    return count


def check_required_columns(path: str | Path, required_columns: Iterable[str], *, modality: str) -> ValidationCheckResult:
    resolved = _path(path)
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
    except Exception as exc:
        return failed("required_columns", modality, "CSV header could not be loaded.", f"Fix malformed CSV: {exc}", artifact_paths=[resolved])
    missing = sorted(set(required_columns) - set(fieldnames))
    if not missing:
        return passed("required_columns", modality, "Required columns are present.", artifact_paths=[resolved])
    return failed(
        "required_columns",
        modality,
        "Required columns are missing.",
        "Regenerate the canonical artifact with the required schema.",
        details={"missing_columns": missing},
        artifact_paths=[resolved],
    )


def check_target_not_in_features(feature_names: Iterable[str], target_columns: Iterable[str], *, modality: str) -> ValidationCheckResult:
    overlap = sorted(set(feature_names) & set(target_columns))
    if not overlap:
        return passed("target_not_in_features", modality, "Target columns are excluded from features.")
    return blocked(
        "target_not_in_features",
        modality,
        "Target leakage detected in feature columns.",
        "Remove target columns from feature schemas and regenerate preprocessing artifacts.",
        details={"leakage_column_count": len(overlap), "columns": overlap},
    )


def check_identifiers_not_in_features(feature_names: Iterable[str], identifier_columns: Iterable[str], *, modality: str) -> ValidationCheckResult:
    overlap = sorted(set(feature_names) & set(identifier_columns))
    if not overlap:
        return passed("identifiers_not_in_features", modality, "Identifier columns are excluded from features.")
    return blocked(
        "identifiers_not_in_features",
        modality,
        "Identifier leakage detected in feature columns.",
        "Remove identifiers from predictive features and retain them only for grouping/manifests.",
        details={"identifier_feature_count": len(overlap), "columns": overlap},
    )


def check_metadata_not_in_features(feature_names: Iterable[str], metadata_columns: Iterable[str], *, modality: str) -> ValidationCheckResult:
    overlap = sorted(set(feature_names) & set(metadata_columns))
    if not overlap:
        return passed("metadata_not_in_features", modality, "Metadata columns are excluded from features.")
    return failed(
        "metadata_not_in_features",
        modality,
        "Metadata leakage detected in feature columns.",
        "Exclude metadata such as source file, timestamp, split, and corpus identity from predictive features.",
        details={"metadata_feature_count": len(overlap), "columns": overlap},
    )


def check_sensitive_columns_documented(sensitive_columns: Iterable[str], documented_columns: Iterable[str], *, modality: str) -> ValidationCheckResult:
    missing = sorted(set(sensitive_columns) - set(documented_columns))
    if not missing:
        return passed("sensitive_columns_documented", modality, "Sensitive columns are documented or absent.")
    return warning(
        "sensitive_columns_documented",
        modality,
        "Sensitive columns need explicit documentation.",
        "Document sensitive-feature handling and fairness restrictions before split design.",
        details={"undocumented_sensitive_column_count": len(missing), "columns": missing},
    )


def check_duplicate_manifest_present(path: str | Path, *, modality: str) -> ValidationCheckResult:
    return check_required_file_exists(path, modality=modality, check_name="duplicate_manifest_present")


def check_conflict_quarantine_present(path: str | Path, *, modality: str) -> ValidationCheckResult:
    return check_required_file_exists(path, modality=modality, check_name="conflict_quarantine_present")


def check_split_manifest_absent(root: str | Path, *, modality: str) -> ValidationCheckResult:
    def is_split_manifest(path: Path) -> bool:
        name = path.name.lower()
        stem = path.stem.lower()
        return path.suffix.lower() == ".json" and (
            stem in {"split_manifest", "train_validation_test_split", "phase3_split_manifest"}
            or name.endswith("_split_manifest.json")
            or name.startswith("split_manifest")
        )

    findings = _find_files(root, is_split_manifest)
    if not findings:
        return passed("split_manifest_absent", modality, "No split manifest was found.", artifact_paths=[root])
    return blocked(
        "split_manifest_absent",
        modality,
        "Split manifests were found during Phase 2 validation.",
        "Remove split artifacts from Phase 2 outputs and create splits only in the approved next phase.",
        details={"split_manifest_count": len(findings)},
        artifact_paths=findings[:20],
    )


def check_model_artifacts_absent(root: str | Path, *, modality: str) -> ValidationCheckResult:
    findings = _find_files(
        root,
        lambda path: path.suffix.lower() in MODEL_ARTIFACT_EXTENSIONS or path.stem.lower() in MODEL_ARTIFACT_NAMES,
    )
    if not findings:
        return passed("model_artifacts_absent", modality, "No model, scaler, encoder, or tokenizer artifacts were found.", artifact_paths=[root])
    return blocked(
        "model_artifacts_absent",
        modality,
        "Unexpected model-related artifacts were found.",
        "Delete or relocate model artifacts; Phase 2 validation must remain preprocessing-only.",
        details={"model_artifact_count": len(findings)},
        artifact_paths=findings[:20],
    )


def check_training_outputs_absent(root: str | Path, *, modality: str) -> ValidationCheckResult:
    findings = _find_files(root, lambda path: any(token in path.name.lower() for token in ("training", "evaluation", "metrics", "predictions")))
    if not findings:
        return passed("training_outputs_absent", modality, "No training/evaluation/prediction outputs were found.", artifact_paths=[root])
    return blocked(
        "training_outputs_absent",
        modality,
        "Unexpected training, evaluation, or prediction outputs were found.",
        "Keep Phase 2 outputs limited to preprocessing validation and readiness reports.",
        details={"training_output_count": len(findings)},
        artifact_paths=findings[:20],
    )


def _find_files(root: str | Path, predicate) -> list[Path]:
    resolved = _path(root)
    if not resolved.exists():
        return []
    if resolved.is_file():
        return [resolved] if predicate(resolved) else []
    findings = []
    for child in resolved.rglob("*"):
        if child.is_file() and predicate(child):
            findings.append(child)
    return findings


def check_report_privacy(path: str | Path, *, modality: str) -> ValidationCheckResult:
    text = _bounded_text(path)
    findings: dict[str, int] = {}
    for name, pattern in TEXT_PRIVACY_PATTERNS.items():
        count = len(re.findall(pattern, text))
        if count:
            findings[name] = count
    if not findings:
        return passed("report_privacy", modality, "No obvious raw identifiers or absolute paths were found in the bounded report scan.", artifact_paths=[path])
    return failed(
        "report_privacy",
        modality,
        "Potential privacy leakage was detected in a report.",
        "Remove raw identifiers and machine-specific paths from generated reports.",
        details={"finding_types": sorted(findings), "finding_count": sum(findings.values())},
        artifact_paths=[path],
    )


def check_deterministic_versions(payload_or_path: Mapping[str, Any] | str | Path, *, modality: str) -> ValidationCheckResult:
    payload = _payload(payload_or_path)
    version_keys = [
        key
        for key in payload
        if (key.endswith("_version") or key == "version") and key not in {"dataset_version", "source_questionnaire_version", "target_questionnaire_version"}
    ]
    malformed = [key for key in version_keys if not re.match(r"^\d+\.\d+\.\d+$", str(payload.get(key) or ""))]
    if not malformed:
        return passed("deterministic_versions", modality, "Version fields are deterministic semantic versions.", details={"version_field_count": len(version_keys)})
    return failed(
        "deterministic_versions",
        modality,
        "Some version fields are missing or not semantic versions.",
        "Use deterministic x.y.z version strings in Phase 2 configs and reports.",
        details={"malformed_version_fields": malformed},
    )


def check_generated_output_hashes(manifest_path: str | Path, *, modality: str) -> ValidationCheckResult:
    try:
        payload = _load_json_object(manifest_path)
    except Exception as exc:
        return failed("generated_output_hashes", modality, "Record manifest could not be loaded.", f"Restore the manifest: {exc}", artifact_paths=[manifest_path])
    manifest_hash = hash_json_data(payload)
    if manifest_hash:
        return passed("generated_output_hashes", modality, "Generated manifest has a deterministic hash.", details={"manifest_sha256": manifest_hash}, artifact_paths=[manifest_path])
    return failed("generated_output_hashes", modality, "Generated manifest hash could not be computed.", "Repair the manifest.")
