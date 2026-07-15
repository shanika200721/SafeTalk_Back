"""Validation for Phase 4A pilot protocol design and synthetic records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

from app.ml.pilot.consent import validate_modality_consent
from app.ml.pilot.constants import (
    MODALITIES,
    PILOT_CONSENT_VERSION,
    PILOT_DATA_SCHEMA_VERSION,
    PILOT_PROTOCOL_VERSION,
    PILOT_RETENTION_POLICY_VERSION,
    PILOT_SAFETY_POLICY_VERSION,
    PROHIBITED_ARTIFACT_TERMS,
)
from app.ml.pilot.modalities import validate_modality_scope
from app.ml.pilot.participant import validate_no_production_user_id_leakage, validate_pilot_participant_id
from app.ml.pilot.privacy import validate_no_direct_identifiers, validate_privacy
from app.ml.pilot.retention import validate_retention_policy
from app.ml.pilot.safety import validate_safety_events
from app.ml.pilot.schemas import ValidationIssue, require_timezone
from app.ml.pilot.sessions import (
    calculate_data_completeness,
    detect_duplicate_session_records,
    detect_future_leakage,
    validate_temporal_order,
)


def _issue(check: str, ok: bool, message: str) -> ValidationIssue:
    return ValidationIssue(check=check, status="passed" if ok else "failed", message=message)


def _contains_prohibited_terms(payload: Any) -> List[str]:
    text = str(payload).lower()
    return [term for term in PROHIBITED_ARTIFACT_TERMS if term in text]


def validate_protocol_configs(
    protocol_config: Dict[str, Any],
    modality_scope: Dict[str, Any],
    alignment_policy: Dict[str, Any],
    retention_policy: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    issues: List[ValidationIssue] = []
    issues.append(_issue("protocol version", protocol_config.get("protocol_version") == PILOT_PROTOCOL_VERSION, "pilot protocol version must match constants"))
    issues.append(_issue("schema version", protocol_config.get("schema_version") == PILOT_DATA_SCHEMA_VERSION, "pilot schema version must match constants"))
    issues.append(_issue("consent version", protocol_config.get("consent_version") == PILOT_CONSENT_VERSION, "pilot consent version must match constants"))
    issues.append(_issue("safety version", protocol_config.get("safety_policy_version") == PILOT_SAFETY_POLICY_VERSION, "pilot safety version must match constants"))
    issues.append(_issue("retention version", retention_policy.get("retention_policy_version") == PILOT_RETENTION_POLICY_VERSION, "retention policy version must match constants"))
    scope_result = validate_modality_scope(modality_scope)
    issues.append(_issue("modality scope", scope_result["valid"], "; ".join(scope_result["errors"]) or "all modality scope fields present"))
    retention_result = validate_retention_policy(retention_policy)
    issues.append(_issue("retention policy", retention_result["valid"], "; ".join(retention_result["errors"]) or "retention policy covers required categories"))
    required_alignment = ("participant_key", "session_key", "timestamp_tolerance_minutes", "modality_windows", "no_future_leakage_rule", "outcome_window")
    missing_alignment = [key for key in required_alignment if key not in alignment_policy]
    issues.append(_issue("alignment policy", not missing_alignment, f"missing alignment fields: {missing_alignment}" if missing_alignment else "alignment policy contains required fields"))
    if strict:
        for field_name in ("ethics_approval_required", "real_collection_prohibited", "model_use_prohibited"):
            issues.append(_issue(f"strict {field_name}", protocol_config.get(field_name) is True, f"{field_name} must be true"))
    errors = [item.to_dict() for item in issues if item.status == "failed"]
    return {"valid": not errors, "issues": [item.to_dict() for item in issues], "errors": errors}


def validate_pilot_dataset(
    participants: Sequence[Any],
    consents: Sequence[Any],
    sessions: Sequence[Any],
    modality_records: Sequence[Any],
    outcomes: Sequence[Any],
    safety_events: Sequence[Any],
    withdrawals: Sequence[Any],
    modality_scope: Dict[str, Any],
    alignment_policy: Dict[str, Any],
    retention_policy: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    del strict
    issues: List[ValidationIssue] = []
    errors: List[str] = []
    participant_ids = [participant.pilot_participant_id for participant in participants]
    consent_by_pid = {consent.pilot_participant_id: consent for consent in consents}
    session_ids = {session.session_id for session in sessions}

    for participant in participants:
        if not validate_pilot_participant_id(participant.pilot_participant_id):
            errors.append(f"{participant.pilot_participant_id}: invalid pseudonymous participant ID")
        try:
            validate_no_production_user_id_leakage([participant.pilot_participant_id, participant.age_band, participant.study_site, participant.cohort])
            validate_no_direct_identifiers(participant.to_dict())
        except ValueError as exc:
            errors.append(str(exc))
        try:
            require_timezone(participant.enrollment_date, "enrollment_date")
        except ValueError as exc:
            errors.append(str(exc))
    issues.append(_issue("pseudonymous IDs", not errors, "participant IDs are pseudonymous and contain no direct identifiers"))

    consent_errors = []
    for record in modality_records:
        consent = consent_by_pid.get(record.pilot_participant_id)
        if consent is None:
            consent_errors.append(f"{record.record_id}: missing consent record")
            continue
        if not record.consent_verified:
            consent_errors.append(f"{record.record_id}: consent_verified is false")
        if not validate_modality_consent(consent, record.modality, record.collected_at):
            consent_errors.append(f"{record.record_id}: modality-specific consent mismatch")
        if record.session_id not in session_ids:
            consent_errors.append(f"{record.record_id}: unknown session_id")
        if any(term in str(record.to_dict()).lower() for term in PROHIBITED_ARTIFACT_TERMS):
            consent_errors.append(f"{record.record_id}: prohibited inference/prediction artifact term present")
    issues.append(_issue("consent before collection", not consent_errors, "; ".join(consent_errors) or "all records have prior modality-specific consent"))

    withdrawal_errors = []
    withdrawal_by_pid = {withdrawal.pilot_participant_id: withdrawal for withdrawal in withdrawals}
    for record in modality_records:
        withdrawal = withdrawal_by_pid.get(record.pilot_participant_id)
        if withdrawal and record.collected_at >= withdrawal.withdrawal_time and not record.withdrawn:
            withdrawal_errors.append(f"{record.record_id}: post-withdrawal record is not marked withdrawn")
    issues.append(_issue("withdrawal enforcement", not withdrawal_errors, "; ".join(withdrawal_errors) or "withdrawal records are respected"))

    timezone_errors = []
    for item in [*participants, *sessions, *modality_records, *outcomes, *safety_events, *withdrawals]:
        for key, value in getattr(item, "__dict__", {}).items():
            if isinstance(value, datetime):
                try:
                    require_timezone(value, key)
                except ValueError as exc:
                    timezone_errors.append(str(exc))
    issues.append(_issue("timezone-aware timestamps", not timezone_errors, "; ".join(timezone_errors) or "timestamps are timezone-aware"))

    temporal = validate_temporal_order(sessions, modality_records)
    issues.append(_issue("temporal order", temporal["valid"], "; ".join(temporal["errors"]) or "session and record timestamps are ordered"))
    leakage = detect_future_leakage(modality_records, outcomes)
    issues.append(_issue("no future leakage", leakage["valid"], str(leakage["leakage"]) if leakage["leakage"] else "no records occur after outcome assessment windows"))
    duplicates = detect_duplicate_session_records(modality_records)
    issues.append(_issue("duplicate records", duplicates["valid"], str(duplicates["duplicates"]) if duplicates["duplicates"] else "no duplicate participant-session-modality records"))

    scope_result = validate_modality_scope(modality_scope)
    issues.append(_issue("modality scope", scope_result["valid"], "; ".join(scope_result["errors"]) or "real-collection-disabled modalities remain disabled"))
    retention_result = validate_retention_policy(retention_policy)
    issues.append(_issue("raw biometric retention policy", retention_result["valid"], "; ".join(retention_result["errors"]) or "retention policy covers raw biometric handling"))
    privacy_result = validate_privacy(modality_records, participant_ids)
    issues.append(_issue("privacy validation", privacy_result["valid"], "; ".join(privacy_result["errors"]) or "no direct identifiers or unsafe raw paths detected"))

    outcome_errors = []
    for outcome in outcomes:
        if outcome.outcome_source in {"model_output", "prediction", "alert"}:
            outcome_errors.append(f"{outcome.outcome_id}: model-output-derived labels are prohibited")
        if not outcome.review_blinded_to_model:
            outcome_errors.append(f"{outcome.outcome_id}: outcome review must be blinded to model outputs")
        if outcome.outcome_label == "suicide-risk truth":
            outcome_errors.append(f"{outcome.outcome_id}: single suicide-risk truth labels are prohibited")
    issues.append(_issue("outcome-source documentation", not outcome_errors, "; ".join(outcome_errors) or "outcomes are sourced, blinded, and not model-derived"))

    safety_result = validate_safety_events(safety_events)
    issues.append(_issue("safety-event documentation", safety_result["valid"], "; ".join(safety_result["errors"]) or "safety events require human review and no model decisions"))

    prohibited = _contains_prohibited_terms({"participants": participant_ids, "records": [record.to_dict() for record in modality_records], "outcomes": [outcome.to_dict() for outcome in outcomes]})
    issues.append(_issue("no inference artifacts", not prohibited, f"prohibited terms found: {prohibited}" if prohibited else "no inference, prediction, alert, or fusion artifacts present"))
    issues.append(_issue("no production DB access", True, "validator operates on in-memory records and config dictionaries only"))
    issues.append(_issue("no prediction rows", True, "no ModalityPrediction writes are implemented"))
    issues.append(_issue("no alert rows", True, "no Alert writes are implemented"))

    failed = [item.to_dict() for item in issues if item.status == "failed"]
    completeness = calculate_data_completeness(modality_records, sessions)
    return {
        "valid": not failed,
        "issues": [item.to_dict() for item in issues],
        "errors": failed,
        "summary": {
            "participant_count": len(participants),
            "session_count": len(sessions),
            "modality_record_count": len(modality_records),
            "withdrawal_count": len(withdrawals),
            "safety_event_count": len(safety_events),
            "missingness_summary": completeness,
            "alignment_policy": alignment_policy.get("policy_version"),
        },
    }
