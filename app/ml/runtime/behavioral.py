from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models.database_models import BehavioralTelemetryEvent, ChatMessage, DailyCheckIn, JournalEntry, User


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


def _sum_between(query, column, timestamp_column, start: datetime, end: datetime) -> float:
    rows = query.filter(timestamp_column >= start, timestamp_column < end).all()
    return float(sum(float(getattr(item, column.key) or 0.0) for item in rows))


def _mean_between(query, column, timestamp_column, start: datetime, end: datetime) -> float | None:
    values = [
        float(getattr(item, column.key))
        for item in query.filter(timestamp_column >= start, timestamp_column < end).all()
        if getattr(item, column.key) is not None
    ]
    return float(sum(values) / len(values)) if values else None


def aggregate_behavioral_features(db: Session, user: User, *, now: datetime | None = None) -> dict[str, Any]:
    """Aggregate only persisted, privacy-preserving behavioral fields.

    The optional telemetry channel stores only aggregate timing/count summaries.
    It never stores raw keystroke contents or pointer paths.
    """

    now = now or datetime.utcnow()
    recent_start = now - timedelta(days=7)
    baseline_start = now - timedelta(days=28)

    checkins = db.query(DailyCheckIn).filter(DailyCheckIn.user_id == user.id)
    journals = db.query(JournalEntry).filter(JournalEntry.student_id == user.id, JournalEntry.deleted_at.is_(None))
    sent_messages = db.query(ChatMessage).filter(ChatMessage.sender_id == user.id)
    received_messages = db.query(ChatMessage).filter(ChatMessage.receiver_id == user.id)
    telemetry = db.query(BehavioralTelemetryEvent).filter(BehavioralTelemetryEvent.student_id == user.id)

    recent_checkins = _count_between(checkins, DailyCheckIn.created_at, recent_start, now)
    baseline_checkins = _count_between(checkins, DailyCheckIn.created_at, baseline_start, recent_start)
    recent_journals = _count_between(journals, JournalEntry.created_at, recent_start, now)
    baseline_journals = _count_between(journals, JournalEntry.created_at, baseline_start, recent_start)
    recent_sent_messages = _count_between(sent_messages, ChatMessage.created_at, recent_start, now)
    baseline_sent_messages = _count_between(sent_messages, ChatMessage.created_at, baseline_start, recent_start)
    recent_received_messages = _count_between(received_messages, ChatMessage.created_at, recent_start, now)
    baseline_received_messages = _count_between(received_messages, ChatMessage.created_at, baseline_start, recent_start)
    recent_telemetry_events = _count_between(telemetry, BehavioralTelemetryEvent.created_at, recent_start, now)
    baseline_telemetry_events = _count_between(telemetry, BehavioralTelemetryEvent.created_at, baseline_start, recent_start)
    recent_session_duration_seconds = _sum_between(
        telemetry, BehavioralTelemetryEvent.session_duration_seconds, BehavioralTelemetryEvent.created_at, recent_start, now
    )
    baseline_session_duration_seconds = _sum_between(
        telemetry, BehavioralTelemetryEvent.session_duration_seconds, BehavioralTelemetryEvent.created_at, baseline_start, recent_start
    )
    recent_interaction_count = int(
        _sum_between(telemetry, BehavioralTelemetryEvent.interaction_count, BehavioralTelemetryEvent.created_at, recent_start, now)
    )
    baseline_interaction_count = int(
        _sum_between(telemetry, BehavioralTelemetryEvent.interaction_count, BehavioralTelemetryEvent.created_at, baseline_start, recent_start)
    )
    recent_typing_active_ms = _sum_between(
        telemetry, BehavioralTelemetryEvent.typing_active_ms, BehavioralTelemetryEvent.created_at, recent_start, now
    )
    baseline_typing_active_ms = _sum_between(
        telemetry, BehavioralTelemetryEvent.typing_active_ms, BehavioralTelemetryEvent.created_at, baseline_start, recent_start
    )
    recent_response_latency_ms_mean = _mean_between(
        telemetry, BehavioralTelemetryEvent.response_latency_ms, BehavioralTelemetryEvent.created_at, recent_start, now
    )
    baseline_response_latency_ms_mean = _mean_between(
        telemetry, BehavioralTelemetryEvent.response_latency_ms, BehavioralTelemetryEvent.created_at, baseline_start, recent_start
    )

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
        "recent_telemetry_events": recent_telemetry_events,
        "baseline_telemetry_events": baseline_telemetry_events,
        "recent_session_duration_seconds": round(recent_session_duration_seconds, 4),
        "baseline_session_duration_seconds": round(baseline_session_duration_seconds, 4),
        "recent_interaction_count": recent_interaction_count,
        "baseline_interaction_count": baseline_interaction_count,
        "recent_typing_active_ms": round(recent_typing_active_ms, 4),
        "baseline_typing_active_ms": round(baseline_typing_active_ms, 4),
        "recent_response_latency_ms_mean": round(recent_response_latency_ms_mean, 4) if recent_response_latency_ms_mean is not None else None,
        "baseline_response_latency_ms_mean": round(baseline_response_latency_ms_mean, 4) if baseline_response_latency_ms_mean is not None else None,
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
    telemetry_drop = _drop_score(features["recent_telemetry_events"], features["baseline_telemetry_events"])
    session_duration_drop = _drop_score(
        int(features["recent_session_duration_seconds"] // 60),
        int(features["baseline_session_duration_seconds"] // 60),
    )
    interaction_drop = _drop_score(features["recent_interaction_count"], features["baseline_interaction_count"])
    typing_drop = _drop_score(
        int(features["recent_typing_active_ms"] // 1000),
        int(features["baseline_typing_active_ms"] // 1000),
    )
    latency_component = 0.0
    recent_latency = features.get("recent_response_latency_ms_mean")
    baseline_latency = features.get("baseline_response_latency_ms_mean")
    if recent_latency is not None and baseline_latency is not None and baseline_latency >= 100:
        latency_component = min(1.0, max(0.0, (float(recent_latency) - float(baseline_latency)) / max(float(baseline_latency), 1.0)))
    inactivity_days = features.get("days_since_last_activity")
    inactivity_component = 0.0 if inactivity_days is None else min(1.0, max(0.0, (float(inactivity_days) - 3.0) / 11.0))

    historical_points = (
        features["baseline_checkins"]
        + features["baseline_journal_entries"]
        + features["baseline_sent_chat_messages"]
        + features["baseline_received_chat_messages"]
        + features["baseline_telemetry_events"]
        + int(features["baseline_session_duration_seconds"] > 0)
        + int(features["baseline_interaction_count"] > 0)
    )
    recent_points = (
        features["recent_checkins"]
        + features["recent_journal_entries"]
        + features["recent_sent_chat_messages"]
        + features["recent_received_chat_messages"]
        + features["recent_telemetry_events"]
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
        0.25 * checkin_drop
        + 0.15 * journal_drop
        + 0.15 * sent_drop
        + 0.08 * received_drop
        + 0.12 * telemetry_drop
        + 0.08 * session_duration_drop
        + 0.07 * interaction_drop
        + 0.03 * typing_drop
        + 0.02 * latency_component
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
            "behavioral_telemetry_events.session_duration_seconds",
            "behavioral_telemetry_events.interaction_count",
            "behavioral_telemetry_events.response_latency_ms",
            "behavioral_telemetry_events.typing_active_ms",
            "behavioral_telemetry_events.typing_pause_count",
            "behavioral_telemetry_events.typed_character_count",
        ],
        "fields_not_available": [
            "typing_speed_cpm",
            "raw_keystroke_content",
            "raw_mouse_path",
            "mouse_distance_px",
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
            "telemetry_event_drop_component": round(telemetry_drop, 6),
            "session_duration_drop_component": round(session_duration_drop, 6),
            "interaction_drop_component": round(interaction_drop, 6),
            "typing_activity_drop_component": round(typing_drop, 6),
            "response_latency_increase_component": round(latency_component, 6),
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
