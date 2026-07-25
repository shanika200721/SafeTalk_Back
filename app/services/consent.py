from datetime import datetime
from typing import Dict, Iterable, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.database_models import ConsentRecord, User


CURRENT_POLICY_VERSION = "1.0"

CONSENT_TYPES = {
    "profile_processing": "Profile information processing",
    "dass21_processing": "DASS-21 questionnaire processing",
    "mood_processing": "Daily mood and check-in processing",
    "text_processing": "Automated text analysis",
    "voice_processing": "Voice-message upload, storage, and processing",
    "face_processing": "Facial data processing",
    "behavioral_processing": "Behavioral signal processing",
    "counselor_escalation": "Counselor escalation and review",
    "research_data_use": "Research data use",
}


def validate_consent_type(consent_type: str) -> str:
    if consent_type not in CONSENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "UNKNOWN_CONSENT_TYPE",
                "message": "Unknown consent type.",
            },
        )
    return consent_type


def get_latest_consent(
    db: Session,
    user_id: int,
    consent_type: str,
) -> Optional[ConsentRecord]:
    validate_consent_type(consent_type)
    return (
        db.query(ConsentRecord)
        .filter(
            ConsentRecord.user_id == user_id,
            ConsentRecord.consent_type == consent_type,
        )
        .order_by(ConsentRecord.created_at.desc(), ConsentRecord.id.desc())
        .first()
    )


def has_active_consent(db: Session, user_id: int, consent_type: str) -> bool:
    latest = get_latest_consent(db, user_id, consent_type)
    return bool(latest and latest.is_granted and latest.withdrawn_at is None)


def require_active_consent(
    db: Session,
    user: User,
    consent_type: str,
    feature: str,
) -> None:
    if has_active_consent(db, user.id, consent_type):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": "CONSENT_REQUIRED",
            "consent_type": consent_type,
            "message": f"Active {CONSENT_TYPES[consent_type]} consent is required before {feature}.",
        },
    )


def latest_consent_state(db: Session, user_id: int) -> Dict[str, dict]:
    state = {}
    for consent_type, description in CONSENT_TYPES.items():
        latest = get_latest_consent(db, user_id, consent_type)
        state[consent_type] = {
            "consent_type": consent_type,
            "description": description,
            "is_granted": bool(latest and latest.is_granted and latest.withdrawn_at is None),
            "policy_version": latest.policy_version if latest else None,
            "granted_at": latest.granted_at if latest else None,
            "withdrawn_at": latest.withdrawn_at if latest else None,
            "updated_at": latest.updated_at if latest else None,
        }
    return state


def create_consent_record(
    db: Session,
    user_id: int,
    consent_type: str,
    is_granted: bool,
    policy_version: str,
    source: str = "api",
    metadata_json: Optional[dict] = None,
) -> ConsentRecord:
    validate_consent_type(consent_type)
    now = datetime.utcnow()
    record = ConsentRecord(
        user_id=user_id,
        consent_type=consent_type,
        is_granted=is_granted,
        policy_version=policy_version,
        granted_at=now if is_granted else None,
        withdrawn_at=None if is_granted else now,
        source=source,
        metadata_json=metadata_json,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def require_consents_for_modalities(
    db: Session,
    user: User,
    consent_types: Iterable[str],
    feature: str,
) -> None:
    for consent_type in consent_types:
        require_active_consent(db, user, consent_type, feature)
