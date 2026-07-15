"""Consent validation for the Phase 4A pilot protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from app.ml.pilot.constants import MODALITIES, PILOT_CONSENT_VERSION, SEPARATE_CONSENT_MODALITIES
from app.ml.pilot.schemas import PilotConsentRecord, require_timezone


@dataclass(frozen=True)
class ConsentDecision:
    modality: str
    consented: bool
    separate_consent_required: bool
    required_for_participation: bool = False


def build_consent_record(
    pilot_participant_id: str,
    decisions: Iterable[ConsentDecision],
    consent_timestamp: datetime,
    data_retention_choice: str,
    future_research_permission: bool,
    contact_permission: bool,
    reviewer_alias: str,
    consent_version: str = PILOT_CONSENT_VERSION,
    notes: Optional[str] = None,
) -> PilotConsentRecord:
    require_timezone(consent_timestamp, "consent_timestamp")
    consented: List[str] = []
    declined: List[str] = []
    seen = set()
    for decision in decisions:
        if decision.modality not in MODALITIES:
            raise ValueError(f"unknown modality: {decision.modality}")
        if decision.modality in seen:
            raise ValueError(f"duplicate consent decision for {decision.modality}")
        seen.add(decision.modality)
        if decision.modality in SEPARATE_CONSENT_MODALITIES and not decision.separate_consent_required:
            raise ValueError(f"{decision.modality} consent must be captured separately")
        if decision.consented:
            consented.append(decision.modality)
        else:
            declined.append(decision.modality)
    missing = set(MODALITIES) - seen
    if missing:
        raise ValueError(f"missing consent decisions: {sorted(missing)}")
    return PilotConsentRecord(
        pilot_participant_id=pilot_participant_id,
        consent_version=consent_version,
        consented_modalities=consented,
        declined_modalities=declined,
        consent_timestamp=consent_timestamp,
        data_retention_choice=data_retention_choice,
        future_research_permission=future_research_permission,
        contact_permission=contact_permission,
        reviewer_alias=reviewer_alias,
        notes=notes,
    )


def validate_general_study_consent(record: PilotConsentRecord) -> Dict[str, object]:
    valid = record.consent_version == PILOT_CONSENT_VERSION and bool(record.consented_modalities)
    return {"valid": valid, "consent_version": record.consent_version, "participation_allowed": valid}


def validate_modality_consent(record: PilotConsentRecord, modality: str, collection_time: datetime) -> bool:
    require_timezone(collection_time, "collection_time")
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality: {modality}")
    if modality not in record.consented_modalities:
        return False
    if record.withdrawal_timestamp and collection_time >= record.withdrawal_timestamp:
        return False
    return record.consent_timestamp <= collection_time


def validate_withdrawal_choices(retain_existing_data: bool, destroy_raw_data: bool, destroy_derived_data: bool) -> Dict[str, object]:
    if not retain_existing_data and not (destroy_raw_data or destroy_derived_data):
        return {"valid": False, "reason": "withdrawal requires retention or destruction instructions"}
    return {
        "valid": True,
        "retain_existing_data": retain_existing_data,
        "destroy_raw_data": destroy_raw_data,
        "destroy_derived_data": destroy_derived_data,
    }


def consent_matrix(record: PilotConsentRecord) -> List[Dict[str, object]]:
    return [
        {
            "pilot_participant_id": record.pilot_participant_id,
            "modality": modality,
            "consented": modality in record.consented_modalities,
            "declined": modality in record.declined_modalities,
            "separate_consent_required": modality in SEPARATE_CONSENT_MODALITIES,
            "consent_version": record.consent_version,
        }
        for modality in MODALITIES
    ]
