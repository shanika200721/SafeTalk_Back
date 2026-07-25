from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import User
from app.routes.auth import get_current_user
from app.services.support_contacts import (
    record_support_action,
    require_student,
    safe_university,
    select_support_contact,
)


router = APIRouter(prefix="/api/support", tags=["Support Contacts"])


class SupportActionCreate(BaseModel):
    action_type: str
    contact_type: Optional[str] = None


@router.get("/contact")
def get_support_contact(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_student(current_user)
    payload, _ = select_support_contact(db, current_user)
    return payload


@router.get("/university")
def get_support_university(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_student(current_user)
    university = current_user.university
    return {
        "university": safe_university(university),
        "has_university_scope": university is not None and university.active,
    }


@router.post("/actions")
def create_support_action(
    payload: SupportActionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return record_support_action(
        db,
        user=current_user,
        action_type=payload.action_type,
        contact_type=payload.contact_type,
    )
