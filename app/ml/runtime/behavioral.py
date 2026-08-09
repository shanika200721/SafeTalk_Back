from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models.database_models import ChatMessage, DailyCheckIn, JournalEntry, User


BEHAVIORAL_MODEL_NAME = "behavioral-personal-baseline-anomaly"
BEHAVIORAL_MODEL_VERSION = "1.0.0"
BEHAVIORAL_PREPROCESSING_VERSION = "behavioral-runtime-v1"
BEHAVIORAL_FEATURE_SCHEMA_VERSION = "behavioral-feature-v1"
BEHAVIORAL_RISK_MAPPING_STATUS = "no_validated_risk_mapping"
BEHAVIORAL_RUNTIME_LIMITATION = (
    "Behavioral anomaly evidence is contextual only. The project has no validated mapping from these activity "
    "features to suicide risk, so it is excluded from fusion."
)


@dataclass(frozen=True)
class BehavioralSignal:
    features: dict[str, Any]
    anomaly_score: float
    confidence: float
    label: str
    data_sufficiency: str
    provenance: dict[str, Any]


def _count_between(query, column, start: datetime, end: datetime) -> int:
    return int(query.filter(column >= start, column < end).count())


def aggregate_behavioral_features(db: Session, user: User, *, now: datetime | None = None) -> dict[str, Any]:
    """Aggregate only persisted, privacy-preserving behavioral fields.

    This intentionally avoids typing cadence, pointer movement, and response latency because this
    application does not persist those telemetry fields.
    """

    now = now or datetime.utcnow()
    recent_start = now - timedelta(days=7)
    baseline_start = now - timedelta(days=28)

    checkins = db.query(DailyCheckIn).filter(DailyCheckIn.user_id == user.id)
    journals = db.query(JournalEntry).filter(JournalEntry.student_id == user.id, JournalEntry.deleted_at.is_(None))
    sent_messages = db.query(ChatMessage).filter(ChatMessage.sender_id == user.id)
    received_messages = db.query(ChatMessage).filter(ChatMessage.receiver_id == user.id)

    recent_checkins = _count_between(checkins, DailyCheckIn.created_at, recent_start, now)
    baseline_checkins = _count_between(checkins, DailyCheckIn.created_at, baseline_start, recent_start)
    recent_journals = _count_between(journals, JournalEntry.created_at, recent_start, now)
    baseline_journals = _count_between(journals, JournalEntry.created_at, baseline_start, recent_start)
    recent_sent_messages = _count_between(sent_messages, ChatMessage.created_at, recent_start, now)
    baseline_sent_messages = _count_between(sent_messages, ChatMessage.created_at, baseline_start, recent_start)
    recent_received_messages = _count_between(received_messages, ChatMessage.created_at, recent_start, now)
    baseline_received_messages = _count_between(received_messages, ChatMessage.created_at, baseline_start, recent_start)

    last_activity_candidates = [value for value in [user.last_login_at] if value is not None]
    latest_checkin = checkins.order_by(DailyCheckIn.created_at.desc()).first()
    latest_journal = journals.order_by(JournalEntry.created_at.desc()).first()
    latest_sent = sent_messages.order_by(ChatMessage.created_at.desc()).first()
    if latest_checkin:
        last_activity_candidates.append(latest_checkin.created_at)
    if latest_journal:
        last_activity_candidates.append(latest_journal.created_at)
    if latest_sent:
        last_activity_candidates.append(latest_sent.created_at)

    last_activity_at = max(last_activity_candidates) if last_activity_candidates else None
    days_since_last_activity = (now - last_activity_at).total_seconds() / 86400 if last_activity_at else None

    return {
        "window_days": 7,
        "baseline_days": 21,
        "recent_checkins": recent_checkins,
        "baseline_checkins": baseline_checkins,
        "recent_journal_entries": recent_journals,
        "baseline_journal_entries": baseline_journals,
        "recent_sent_chat_messages": recent_sent_messages,
        "baseline_sent_chat_messages": baseline_sent_messages,
        "recent_received_chat_messages": recent_received_messages,
        "baseline_received_chat_messages": baseline_received_messages,
        "has_login_timestamp": user.last_login_at is not None,
        "days_since_last_activity": round(days_since_last_activity, 4) if days_since_last_activity is not None else None,
        "last_activity_at": last_activity_at.isoformat() if last_activity_at else None,
    }


def _drop_score(recent_count: int, baseline_count: int) -> float:
    baseline_weekly = baseline_count / 3.0
    if baseline_weekly < 1:
        return 0.0
    drop_fraction = max(0.0, (baseline_weekly - recent_count) / baseline_weekly)
    return min(1.0, drop_fraction)


def score_behavioral_anomaly(features: dict[str, Any]) -> BehavioralSignal:
    checkin_drop = _drop_score(features["recent_checkins"], features["baseline_checkins"])
    journal_drop = _drop_score(features["recent_journal_entries"], features["baseline_journal_entries"])
    sent_drop = _drop_score(features["recent_sent_chat_messages"], features["baseline_sent_chat_messages"])
    received_drop = _drop_score(features["recent_received_chat_messages"], features["baseline_received_chat_messages"])
    inactivity_days = features.get("days_since_last_activity")
    inactivity_component = 0.0 if inactivity_days is None else min(1.0, max(0.0, (float(inactivity_days) - 3.0) / 11.0))

    historical_points = (
        features["baseline_checkins"]
        + features["baseline_journal_entries"]
        + features["baseline_sent_chat_messages"]
        + features["baseline_received_chat_messages"]
    )
    recent_points = (
        features["recent_checkins"]
        + features["recent_journal_entries"]
        + features["recent_sent_chat_messages"]
        + features["recent_received_chat_messages"]
        + int(bool(features.get("has_login_timestamp")))
    )
    if historical_points >= 6:
        data_sufficiency = "sufficient_personal_baseline"
        confidence = 0.65
    elif historical_points >= 2 or recent_points >= 3:
        data_sufficiency = "limited_personal_baseline"
        confidence = 0.35
    else:
        data_sufficiency = "insufficient_personal_baseline"
        confidence = 0.15

    anomaly_0_1 = (
        0.35 * checkin_drop
        + 0.20 * journal_drop
        + 0.20 * sent_drop
        + 0.10 * received_drop
        + 0.15 * inactivity_component
    )
    anomaly_score = round(max(0.0, min(100.0, anomaly_0_1 * 100.0)), 2)
    if anomaly_score >= 60:
        label = "high_activity_anomaly"
    elif anomaly_score >= 30:
        label = "moderate_activity_anomaly"
    else:
        label = "low_activity_anomaly"

    provenance = {
        "collected_fields_used": [
            "users.last_login_at",
            "daily_checkins.created_at",
            "journal_entries.created_at",
            "chat_messages.sender_id/receiver_id/created_at",
        ],
        "fields_not_available": [
            "typing_speed_cpm",
            "mouse_distance_px",
            "response_latency_ms",
            "session_duration_seconds",
        ],
        "algorithm": "recent_7_day_activity_drop_against_previous_21_day_personal_baseline_plus_inactivity",
        "risk_mapping_status": BEHAVIORAL_RISK_MAPPING_STATUS,
        "fusion_eligible": False,
    }
    return BehavioralSignal(
        features={
            **features,
            "checkin_drop_component": round(checkin_drop, 6),
            "journal_drop_component": round(journal_drop, 6),
            "sent_chat_drop_component": round(sent_drop, 6),
            "received_chat_drop_component": round(received_drop, 6),
            "inactivity_component": round(inactivity_component, 6),
        },
        anomaly_score=anomaly_score,
        confidence=confidence,
        label=label,
        data_sufficiency=data_sufficiency,
        provenance=provenance,
    )


def predict_behavioral_signal(db: Session, user: User, *, now: datetime | None = None) -> BehavioralSignal:
    return score_behavioral_anomaly(aggregate_behavioral_features(db, user, now=now))
