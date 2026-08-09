"""Deterministic synthetic protocol simulation for Phase 4A smoke testing."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from app.ml.pilot.consent import ConsentDecision, build_consent_record
from app.ml.pilot.constants import MODALITIES, PILOT_CONSENT_VERSION, SYNTHETIC_MARKER
from app.ml.pilot.participant import (
    generate_modality_record_id,
    generate_pilot_participant_id,
    generate_session_id,
)
from app.ml.pilot.schemas import (
    PilotModalityRecord,
    PilotOutcomeRecord,
    PilotParticipant,
    PilotSafetyEvent,
    PilotSession,
    PilotWithdrawalRecord,
)
from app.ml.pilot.sessions import build_aligned_pilot_manifest


@dataclass(frozen=True)
class SyntheticPilotDataset:
    participants: List[PilotParticipant]
    consents: List[Any]
    sessions: List[PilotSession]
    modality_records: List[PilotModalityRecord]
    outcomes: List[PilotOutcomeRecord]
    safety_events: List[PilotSafetyEvent]
    withdrawals: List[PilotWithdrawalRecord]
    manifest: Any
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participants": [item.to_dict() for item in self.participants],
            "consents": [item.to_dict() for item in self.consents],
            "sessions": [item.to_dict() for item in self.sessions],
            "modality_records": [item.to_dict() for item in self.modality_records],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "safety_events": [item.to_dict() for item in self.safety_events],
            "withdrawals": [item.to_dict() for item in self.withdrawals],
            "manifest": self.manifest.to_dict(),
            "metadata": self.metadata,
        }


def _consent_decisions(index: int) -> List[ConsentDecision]:
    speech = index % 3 != 0
    face = False
    behavioral = False
    return [
        ConsentDecision("dass21", True, False),
        ConsentDecision("profile", True, False),
        ConsentDecision("mood", index % 4 != 0, False),
        ConsentDecision("text", index % 5 != 0, False),
        ConsentDecision("speech", speech, True),
        ConsentDecision("face", face, True),
        ConsentDecision("behavioral", behavioral, True),
    ]


def generate_synthetic_pilot_dataset(participants: int = 12, weeks: int = 4, seed: int = 404) -> SyntheticPilotDataset:
    rng = random.Random(seed)
    start = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
    pilot_participants: List[PilotParticipant] = []
    consents = []
    sessions: List[PilotSession] = []
    records: List[PilotModalityRecord] = []
    outcomes: List[PilotOutcomeRecord] = []
    withdrawals: List[PilotWithdrawalRecord] = []
    safety_events: List[PilotSafetyEvent] = []

    for index in range(participants):
        pid = generate_pilot_participant_id(seed=seed, ordinal=index + 1)
        enrollment = start + timedelta(hours=index)
        withdrawal_time = start + timedelta(days=16, hours=3) if index == 7 else None
        pilot_participants.append(
            PilotParticipant(
                pilot_participant_id=pid,
                enrollment_status="withdrawn" if withdrawal_time else "enrolled",
                consent_status="withdrawn" if withdrawal_time else "active",
                consent_version=PILOT_CONSENT_VERSION,
                enrollment_date=enrollment,
                withdrawal_date=withdrawal_time,
                age_band="18-24",
                study_site="synthetic_site_a",
                cohort="synthetic_2026_pilot",
                created_at=enrollment,
            )
        )
        consents.append(
            build_consent_record(
                pid,
                _consent_decisions(index),
                enrollment - timedelta(minutes=15),
                data_retention_choice="retain_deidentified_derived_destroy_raw_on_withdrawal",
                future_research_permission=index % 2 == 0,
                contact_permission=index % 3 == 0,
                reviewer_alias=f"synthetic_reviewer_{index % 3}",
                notes=SYNTHETIC_MARKER,
            )
        )
        if withdrawal_time:
            withdrawals.append(
                PilotWithdrawalRecord(
                    pilot_participant_id=pid,
                    withdrawal_time=withdrawal_time,
                    withdrawal_scope="all_future_collection",
                    retain_existing_data=True,
                    destroy_raw_data=True,
                    destroy_derived_data=False,
                    reason_optional="synthetic withdrawal scenario",
                    processed_at=withdrawal_time + timedelta(hours=2),
                )
            )

        session_index = 0
        baseline_id = generate_session_id(pid, session_index, seed=seed)
        baseline_at = enrollment + timedelta(hours=1)
        sessions.append(
            PilotSession(
                session_id=baseline_id,
                pilot_participant_id=pid,
                session_type="baseline",
                scheduled_at=baseline_at,
                started_at=baseline_at,
                completed_at=baseline_at + timedelta(minutes=30),
                modality_status={"profile": "completed", "dass21": "completed"},
                completion_status="completed",
            )
        )
        for modality in ("profile", "dass21"):
            records.append(_record(pid, baseline_id, modality, baseline_at + timedelta(minutes=5), 1.0, []))

        for week in range(weeks):
            week_start = enrollment + timedelta(days=week * 7 + 1)
            if withdrawal_time and week_start >= withdrawal_time:
                continue
            weekly_id = generate_session_id(pid, 100 + week, seed=seed)
            text_status = "completed" if "text" in consents[-1].consented_modalities and not (index == 2 and week == 2) else "missed"
            speech_status = "completed" if "speech" in consents[-1].consented_modalities and week in {1, 3} and index % 2 == 0 else "not_consented"
            sessions.append(
                PilotSession(
                    session_id=weekly_id,
                    pilot_participant_id=pid,
                    session_type="weekly",
                    scheduled_at=week_start,
                    started_at=week_start + timedelta(minutes=10),
                    completed_at=week_start + timedelta(minutes=35),
                    modality_status={"text": text_status, "speech": speech_status, "face": "disabled", "behavioral": "disabled"},
                    completion_status="partial" if text_status == "missed" else "completed",
                )
            )
            if text_status == "completed":
                records.append(_record(pid, weekly_id, "text", week_start + timedelta(minutes=20), 1.0, []))
            if speech_status == "completed":
                records.append(_record(pid, weekly_id, "speech", week_start + timedelta(minutes=25), 1.0, ["synthetic_audio_placeholder"]))

        for day in range(weeks * 7):
            daily_id = generate_session_id(pid, 1000 + day, seed=seed)
            scheduled = enrollment + timedelta(days=day, hours=20)
            if withdrawal_time and scheduled >= withdrawal_time:
                continue
            mood_consented = "mood" in consents[-1].consented_modalities
            missed = rng.random() < 0.18 or (index == 5 and day in {6, 7, 8})
            mood_status = "completed" if mood_consented and not missed else "missed" if mood_consented else "not_consented"
            sessions.append(
                PilotSession(
                    session_id=daily_id,
                    pilot_participant_id=pid,
                    session_type="daily",
                    scheduled_at=scheduled,
                    started_at=scheduled if mood_status == "completed" else None,
                    completed_at=scheduled + timedelta(minutes=5) if mood_status == "completed" else None,
                    modality_status={"mood": mood_status},
                    completion_status="completed" if mood_status == "completed" else "missed",
                )
            )
            if mood_status == "completed":
                records.append(_record(pid, daily_id, "mood", scheduled + timedelta(minutes=2), 1.0, []))

        outcomes.append(
            PilotOutcomeRecord(
                outcome_id=f"PILOT-O-{hashlib.sha256(('outcome' + pid).encode()).hexdigest()[:16].upper()}",
                pilot_participant_id=pid,
                assessment_time=enrollment + timedelta(days=weeks * 7 + 2),
                outcome_source="counselor_reviewed_follow_up",
                counselor_or_clinician_reviewed=True,
                review_blinded_to_model=True,
                outcome_label="counselor_concern_level_only",
                outcome_confidence="synthetic_low",
                notes="Synthetic outcome placeholder; no clinical truth label.",
            )
        )

    safety_pid = pilot_participants[3].pilot_participant_id
    safety_session = next(session for session in sessions if session.pilot_participant_id == safety_pid and session.session_type == "daily")
    safety_events.append(
        PilotSafetyEvent(
            safety_event_id=f"PILOT-E-{hashlib.sha256(('safety' + safety_pid).encode()).hexdigest()[:16].upper()}",
            pilot_participant_id=safety_pid,
            session_id=safety_session.session_id,
            event_source="questionnaire_response_protocol_simulation",
            event_category="participant_self_disclosure",
            detected_by="questionnaire_response",
            immediate_action="researcher pauses session and contacts qualified counselor per draft protocol",
            referred_to_human=True,
            resolved_status="synthetic_resolved_by_human_review",
            created_at=safety_session.scheduled_at + timedelta(minutes=3),
        )
    )

    manifest = build_aligned_pilot_manifest(
        pilot_participants,
        sessions,
        records,
        consents,
        safety_events,
        withdrawals,
        source_hashes={"synthetic_seed": hashlib.sha256(str(seed).encode()).hexdigest()},
    )
    metadata = {
        "synthetic": True,
        "synthetic_marker": SYNTHETIC_MARKER,
        "seed": seed,
        "weeks": weeks,
        "real_participant_data_collected": False,
        "model_execution": False,
        "prediction_generation": False,
        "alert_generation": False,
        "clinical_truth_labels": False,
    }
    return SyntheticPilotDataset(pilot_participants, consents, sessions, records, outcomes, safety_events, withdrawals, manifest, metadata)


def _record(
    pid: str,
    session_id: str,
    modality: str,
    collected_at: datetime,
    completeness: float,
    quality_flags: List[str],
) -> PilotModalityRecord:
    reference = None
    if modality == "speech":
        reference = f"generated/pilot-protocol-smoke/v1/synthetic/{pid}/{session_id}/speech_placeholder.json"
    return PilotModalityRecord(
        record_id=generate_modality_record_id(pid, session_id, modality),
        pilot_participant_id=pid,
        session_id=session_id,
        modality=modality,
        source_type="synthetic_protocol_simulation",
        collected_at=collected_at,
        local_timezone="UTC",
        raw_artifact_reference=reference,
        derived_artifact_reference=f"generated/pilot-protocol-smoke/v1/derived/{pid}/{session_id}/{modality}.json",
        preprocessing_version=None,
        consent_verified=True,
        withdrawn=False,
        completeness=completeness,
        quality_flags=quality_flags,
        created_at=collected_at + timedelta(minutes=1),
    )
