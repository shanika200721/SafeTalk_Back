"""Typed in-memory schemas for the Phase 4A pilot protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.ml.pilot.constants import (
    MODALITIES,
    PILOT_CONSENT_VERSION,
    PILOT_DATA_SCHEMA_VERSION,
    PILOT_PROTOCOL_VERSION,
)


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def isoformat(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    return value


@dataclass(frozen=True)
class PilotParticipant:
    pilot_participant_id: str
    enrollment_status: str
    consent_status: str
    consent_version: str
    enrollment_date: datetime
    withdrawal_date: Optional[datetime] = None
    age_band: Optional[str] = None
    study_site: Optional[str] = None
    cohort: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        require_timezone(self.enrollment_date, "enrollment_date")
        require_timezone(self.created_at, "created_at")
        if self.withdrawal_date:
            require_timezone(self.withdrawal_date, "withdrawal_date")
        if self.consent_version != PILOT_CONSENT_VERSION:
            raise ValueError("participant consent_version does not match pilot consent version")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotConsentRecord:
    pilot_participant_id: str
    consent_version: str
    consented_modalities: List[str]
    declined_modalities: List[str]
    consent_timestamp: datetime
    data_retention_choice: str
    future_research_permission: bool
    contact_permission: bool
    reviewer_alias: str
    withdrawal_timestamp: Optional[datetime] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        require_timezone(self.consent_timestamp, "consent_timestamp")
        if self.withdrawal_timestamp:
            require_timezone(self.withdrawal_timestamp, "withdrawal_timestamp")
        if self.consent_version != PILOT_CONSENT_VERSION:
            raise ValueError("consent_version does not match pilot consent version")
        unknown = (set(self.consented_modalities) | set(self.declined_modalities)) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown consent modality: {sorted(unknown)}")
        overlap = set(self.consented_modalities).intersection(self.declined_modalities)
        if overlap:
            raise ValueError(f"modality cannot be both consented and declined: {sorted(overlap)}")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotSession:
    session_id: str
    pilot_participant_id: str
    session_type: str
    scheduled_at: datetime
    modality_status: Dict[str, str]
    completion_status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    interruption_reason: Optional[str] = None
    safety_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        require_timezone(self.scheduled_at, "scheduled_at")
        if self.started_at:
            require_timezone(self.started_at, "started_at")
        if self.completed_at:
            require_timezone(self.completed_at, "completed_at")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotModalityRecord:
    record_id: str
    pilot_participant_id: str
    session_id: str
    modality: str
    source_type: str
    collected_at: datetime
    local_timezone: str
    consent_verified: bool
    withdrawn: bool
    completeness: float
    quality_flags: List[str]
    raw_artifact_reference: Optional[str] = None
    derived_artifact_reference: Optional[str] = None
    preprocessing_version: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        require_timezone(self.collected_at, "collected_at")
        require_timezone(self.created_at, "created_at")
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality: {self.modality}")
        if not 0 <= self.completeness <= 1:
            raise ValueError("completeness must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotOutcomeRecord:
    outcome_id: str
    pilot_participant_id: str
    assessment_time: datetime
    outcome_source: str
    counselor_or_clinician_reviewed: bool
    review_blinded_to_model: bool
    outcome_label: str
    outcome_confidence: str
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        require_timezone(self.assessment_time, "assessment_time")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotSafetyEvent:
    safety_event_id: str
    pilot_participant_id: str
    event_source: str
    event_category: str
    detected_by: str
    immediate_action: str
    referred_to_human: bool
    resolved_status: str
    created_at: datetime
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        require_timezone(self.created_at, "created_at")
        if self.detected_by not in {"participant", "researcher", "counselor", "questionnaire_response"}:
            raise ValueError("detected_by must name a human/protocol source")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotWithdrawalRecord:
    pilot_participant_id: str
    withdrawal_time: datetime
    withdrawal_scope: str
    retain_existing_data: bool
    destroy_raw_data: bool
    destroy_derived_data: bool
    reason_optional: Optional[str]
    processed_at: datetime

    def __post_init__(self) -> None:
        require_timezone(self.withdrawal_time, "withdrawal_time")
        require_timezone(self.processed_at, "processed_at")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class PilotDatasetManifest:
    schema_version: str
    protocol_version: str
    generated_at: datetime
    participant_count: int
    session_count: int
    modality_record_count: int
    modality_distribution: Dict[str, int]
    consent_version_distribution: Dict[str, int]
    date_range: Dict[str, Optional[str]]
    missingness_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    safety_event_count: int
    withdrawal_count: int
    alignment_summary: Dict[str, Any]
    source_hashes: Dict[str, str]
    warnings: List[str]

    def __post_init__(self) -> None:
        require_timezone(self.generated_at, "generated_at")
        if self.schema_version != PILOT_DATA_SCHEMA_VERSION:
            raise ValueError("manifest schema_version mismatch")
        if self.protocol_version != PILOT_PROTOCOL_VERSION:
            raise ValueError("manifest protocol_version mismatch")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    status: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"check": self.check, "status": self.status, "message": self.message}
