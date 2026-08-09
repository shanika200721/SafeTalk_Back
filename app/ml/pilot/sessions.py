"""Temporal alignment and completeness helpers for pilot records."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from app.ml.pilot.constants import MODALITIES, PILOT_DATA_SCHEMA_VERSION, PILOT_PROTOCOL_VERSION
from app.ml.pilot.schemas import (
    PilotDatasetManifest,
    PilotModalityRecord,
    PilotSession,
    require_timezone,
)


def create_session_windows(
    sessions: Sequence[PilotSession],
    baseline_hours: int = 72,
    daily_hours: int = 24,
    weekly_hours: int = 168,
) -> Dict[str, Dict[str, datetime]]:
    windows: Dict[str, Dict[str, datetime]] = {}
    for session in sessions:
        center = session.started_at or session.scheduled_at
        require_timezone(center, "session window center")
        hours = baseline_hours if session.session_type == "baseline" else weekly_hours if session.session_type == "weekly" else daily_hours
        windows[session.session_id] = {"start": center - timedelta(hours=hours / 2), "end": center + timedelta(hours=hours / 2)}
    return windows


def align_modality_records(
    records: Sequence[PilotModalityRecord],
    sessions: Sequence[PilotSession],
    tolerance_minutes: int = 120,
) -> Dict[str, Any]:
    session_by_id = {session.session_id: session for session in sessions}
    aligned = []
    unaligned = []
    tolerance = timedelta(minutes=tolerance_minutes)
    for record in records:
        session = session_by_id.get(record.session_id)
        if session is None:
            unaligned.append(record.record_id)
            continue
        anchor = session.started_at or session.scheduled_at
        delta = abs(record.collected_at - anchor)
        row = {"record_id": record.record_id, "session_id": record.session_id, "delta_minutes": round(delta.total_seconds() / 60, 2)}
        if delta <= tolerance:
            aligned.append(row)
        else:
            unaligned.append(record.record_id)
    return {"valid": not unaligned, "aligned_records": aligned, "unaligned_record_ids": unaligned}


def validate_temporal_order(sessions: Sequence[PilotSession], records: Sequence[PilotModalityRecord]) -> Dict[str, Any]:
    errors: List[str] = []
    for session in sessions:
        if session.started_at and session.started_at < session.scheduled_at - timedelta(days=1):
            errors.append(f"{session.session_id}: started before allowable schedule window")
        if session.started_at and session.completed_at and session.completed_at < session.started_at:
            errors.append(f"{session.session_id}: completed before start")
    for record in records:
        if record.created_at < record.collected_at:
            errors.append(f"{record.record_id}: created before collected timestamp")
    return {"valid": not errors, "errors": errors}


def detect_future_leakage(records: Sequence[PilotModalityRecord], outcomes: Sequence[Any]) -> Dict[str, Any]:
    leakage: List[Dict[str, str]] = []
    outcomes_by_participant: Dict[str, List[Any]] = defaultdict(list)
    for outcome in outcomes:
        outcomes_by_participant[outcome.pilot_participant_id].append(outcome)
    for record in records:
        for outcome in outcomes_by_participant.get(record.pilot_participant_id, []):
            if record.collected_at > outcome.assessment_time:
                leakage.append({"record_id": record.record_id, "outcome_id": outcome.outcome_id})
    return {"valid": not leakage, "leakage": leakage}


def detect_duplicate_session_records(records: Sequence[PilotModalityRecord]) -> Dict[str, Any]:
    keys = Counter((record.pilot_participant_id, record.session_id, record.modality) for record in records)
    duplicates = [
        {"pilot_participant_id": key[0], "session_id": key[1], "modality": key[2], "count": count}
        for key, count in keys.items()
        if count > 1
    ]
    return {"valid": not duplicates, "duplicates": duplicates}


def calculate_data_completeness(records: Sequence[PilotModalityRecord], sessions: Sequence[PilotSession]) -> Dict[str, Any]:
    expected: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    for session in sessions:
        for modality, status in session.modality_status.items():
            if status in {"scheduled", "completed", "missed"}:
                expected[modality] += 1
    for record in records:
        observed[record.modality] += record.completeness
    summary: Dict[str, Dict[str, float]] = {}
    for modality in MODALITIES:
        exp = expected[modality]
        obs = observed[modality]
        summary[modality] = {
            "expected": float(exp),
            "observed": float(obs),
            "completion_rate": round(obs / exp, 4) if exp else 1.0,
            "missingness": round(1 - (obs / exp), 4) if exp else 0.0,
        }
    return summary


def build_aligned_pilot_manifest(
    participants: Sequence[Any],
    sessions: Sequence[PilotSession],
    records: Sequence[PilotModalityRecord],
    consents: Sequence[Any],
    safety_events: Sequence[Any],
    withdrawals: Sequence[Any],
    source_hashes: Dict[str, str] | None = None,
) -> PilotDatasetManifest:
    modality_distribution = Counter(record.modality for record in records)
    consent_versions = Counter(consent.consent_version for consent in consents)
    timestamps = [record.collected_at for record in records]
    missingness = calculate_data_completeness(records, sessions)
    alignment = {
        "duplicate_records": detect_duplicate_session_records(records),
        "temporal_order": validate_temporal_order(sessions, records),
    }
    return PilotDatasetManifest(
        schema_version=PILOT_DATA_SCHEMA_VERSION,
        protocol_version=PILOT_PROTOCOL_VERSION,
        generated_at=datetime.now(tz=timestamps[0].tzinfo) if timestamps else datetime.now().astimezone(),
        participant_count=len(participants),
        session_count=len(sessions),
        modality_record_count=len(records),
        modality_distribution=dict(modality_distribution),
        consent_version_distribution=dict(consent_versions),
        date_range={
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
        },
        missingness_summary=missingness,
        quality_summary={"quality_flag_counts": dict(Counter(flag for record in records for flag in record.quality_flags))},
        safety_event_count=len(safety_events),
        withdrawal_count=len(withdrawals),
        alignment_summary=alignment,
        source_hashes=source_hashes or {},
        warnings=["Synthetic protocol simulation only; not a training-ready fusion dataset."],
    )
