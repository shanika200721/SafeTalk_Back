"""Deterministic Phase 2 readiness classification rules."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.ml.validation.schemas import ModalityReadiness, ValidationCheckResult, ValidationSeverity, ValidationStatus


def _has_status(checks: Iterable[ValidationCheckResult], statuses: set[str]) -> bool:
    return any(str(check.status) in statuses for check in checks)


def _has_critical(checks: Iterable[ValidationCheckResult]) -> bool:
    return any(str(check.severity) == ValidationSeverity.CRITICAL.value or str(check.status) == ValidationStatus.BLOCKED.value for check in checks)


def _has_check(checks: Iterable[ValidationCheckResult], check_name: str, status: str | None = None) -> bool:
    for check in checks:
        if check.check_name == check_name and (status is None or str(check.status) == status):
            return True
    return False


def classify_modality_readiness(
    modality: str,
    signals: Mapping[str, Any],
    checks: Iterable[ValidationCheckResult],
) -> tuple[ModalityReadiness, list[str], list[str]]:
    """Return deterministic readiness, warnings, and blockers.

    The rules intentionally prefer explicit blocking when leakage, source, or
    schema integrity signals are unresolved. Warnings can restrict split design
    without blocking preprocessing readiness.
    """
    checks = list(checks)
    warnings: list[str] = list(signals.get("warnings") or [])
    blockers: list[str] = []

    if modality == "dass21":
        return ModalityReadiness.SCORING_ONLY_NOT_ML, warnings, blockers

    if signals.get("engineering_only") or signals.get("synthetic_only"):
        if not signals.get("real_source_exists", False):
            blockers.append("Real source data are absent; only schema or synthetic engineering fixtures are available.")
        return ModalityReadiness.ENGINEERING_TESTS_ONLY, warnings, blockers

    if signals.get("source_missing") or signals.get("pending_real_data"):
        blockers.append("Reviewed real source data are missing or not approved for model-training readiness.")
        return ModalityReadiness.BLOCKED_PENDING_DATA, warnings, blockers

    if signals.get("fingerprint_mismatch") or _has_check(checks, "fingerprint_matches_source", ValidationStatus.BLOCKED.value):
        blockers.append("Source fingerprint mismatch must be resolved before split design.")
        return ModalityReadiness.BLOCKED_PENDING_DATA, warnings, blockers

    if signals.get("target_in_features") or _has_check(checks, "target_not_in_features", ValidationStatus.BLOCKED.value):
        blockers.append("Target columns are present in predictive features.")
        return ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION, warnings, blockers

    if signals.get("identifiers_in_features") or _has_check(checks, "identifiers_not_in_features", ValidationStatus.BLOCKED.value):
        blockers.append("Identifier columns are present in predictive features.")
        return ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION, warnings, blockers

    if signals.get("critical_duplicate_conflicts") or signals.get("unresolved_leakage_conflicts"):
        blockers.append("Unresolved critical duplicate, conflict, or cross-split leakage findings remain.")
        return ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION, warnings, blockers

    if signals.get("nan_or_infinity") or _has_check(checks, "no_nan_or_infinity", ValidationStatus.BLOCKED.value):
        blockers.append("Feature outputs contain NaN or infinity values.")
        return ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION, warnings, blockers

    if signals.get("audit_critical_errors") or _has_critical(checks):
        blockers.append("Critical validation checks remain unresolved.")
        if signals.get("leakage_related", False):
            return ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION, warnings, blockers
        return ModalityReadiness.BLOCKED_PENDING_DATA, warnings, blockers

    if signals.get("feature_schema_missing"):
        blockers.append("Feature schema is missing.")
        return ModalityReadiness.BLOCKED_PENDING_DATA, warnings, blockers

    restriction_flags = (
        "small_sample_size",
        "incomplete_group_ids",
        "missing_grouping_key",
        "corpus_variation",
        "missing_full_feature_extraction",
        "weak_or_self_reported_labels",
        "fairness_concerns",
        "non_clinical_label",
        "duplicate_restrictions",
    )
    if any(signals.get(flag) for flag in restriction_flags) or warnings or _has_status(checks, {ValidationStatus.PASSED_WITH_WARNINGS.value}):
        return ModalityReadiness.READY_WITH_RESTRICTIONS, warnings, blockers

    return ModalityReadiness.READY_FOR_SPLIT_DESIGN, warnings, blockers


def split_status_for_readiness(readiness: ModalityReadiness) -> ValidationStatus:
    if readiness in {ModalityReadiness.READY_FOR_SPLIT_DESIGN, ModalityReadiness.READY_WITH_RESTRICTIONS}:
        return ValidationStatus.PASSED_WITH_WARNINGS if readiness == ModalityReadiness.READY_WITH_RESTRICTIONS else ValidationStatus.PASSED
    if readiness == ModalityReadiness.SCORING_ONLY_NOT_ML:
        return ValidationStatus.NOT_APPLICABLE
    return ValidationStatus.BLOCKED


def model_training_status_for_readiness(readiness: ModalityReadiness) -> ValidationStatus:
    if readiness == ModalityReadiness.READY_FOR_SPLIT_DESIGN:
        return ValidationStatus.PASSED_WITH_WARNINGS
    if readiness == ModalityReadiness.READY_WITH_RESTRICTIONS:
        return ValidationStatus.PASSED_WITH_WARNINGS
    if readiness == ModalityReadiness.SCORING_ONLY_NOT_ML:
        return ValidationStatus.NOT_APPLICABLE
    if readiness == ModalityReadiness.ENGINEERING_TESTS_ONLY:
        return ValidationStatus.NOT_APPLICABLE
    return ValidationStatus.BLOCKED
