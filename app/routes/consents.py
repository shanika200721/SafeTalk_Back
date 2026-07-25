from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import ConsentRecord, User
from app.routes.auth import get_current_user
from app.schemas import ConsentRecordSchema, ConsentUpdate
from app.services.consent import (
    CONSENT_TYPES,
    CURRENT_POLICY_VERSION,
    create_consent_record,
    latest_consent_state,
    validate_consent_type,
)

router = APIRouter(prefix="/api/consents", tags=["Consents"])


@router.get("/policy")
def get_consent_policy():
    """Return the current consent policy version and supported consent types."""
    return {
        "policy_version": CURRENT_POLICY_VERSION,
        "consent_types": [
            {"consent_type": consent_type, "description": description}
            for consent_type, description in CONSENT_TYPES.items()
        ],
        "notes": (
            "Consent controls future collection and processing. Withdrawal does not "
            "delete historical records during Phase 4B."
        ),
    }


@router.get("")
def get_current_consents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's latest consent state by type."""
    return {
        "user_id": current_user.id,
        "policy_version": CURRENT_POLICY_VERSION,
        "consents": latest_consent_state(db, current_user.id),
    }


@router.get("/history", response_model=List[ConsentRecordSchema])
def get_consent_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's append-only consent history."""
    return (
        db.query(ConsentRecord)
        .filter(ConsentRecord.user_id == current_user.id)
        .order_by(ConsentRecord.created_at.desc(), ConsentRecord.id.desc())
        .all()
    )


@router.put("/{consent_type}", response_model=ConsentRecordSchema)
def update_consent(
    consent_type: str,
    request: ConsentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Grant or withdraw a specific consent for the current user.

    Consent decisions are append-only and control future processing. Admin users do
    not use this endpoint to silently grant consent for another user.
    """
    validate_consent_type(consent_type)
    return create_consent_record(
        db=db,
        user_id=current_user.id,
        consent_type=consent_type,
        is_granted=request.is_granted,
        policy_version=request.policy_version,
        source="settings",
    )
