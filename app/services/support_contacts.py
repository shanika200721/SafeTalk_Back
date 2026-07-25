from __future__ import annotations

import re
import uuid
from datetime import datetime
from urllib.parse import quote

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.database_models import (
    CounselorAssignment,
    CounselorProfile,
    CounselorProfileAudit,
    CounselorReview,
    CounselorUniversityAssignment,
    SupportContact,
    SupportContactAction,
    University,
    User,
    UserRole,
)


SUPPORT_ACTION_TYPES = {
    "support_panel_opened",
    "telephone_action_selected",
    "whatsapp_action_selected",
    "number_copied",
    "support_details_viewed",
}
AVAILABILITY_STATUSES = {"available", "busy", "offline", "on_leave"}
SUPPORT_LIMITATIONS = [
    "Opening a call or WhatsApp link does not guarantee an immediate response.",
    "SafeTalk does not automatically contact the counselor.",
    "This support contact is not a substitute for local emergency services during immediate danger.",
]
DEFAULT_WHATSAPP_MESSAGE = "Hello, I am a student using SafeTalk and I would like to speak with a counselor."


def public_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def normalize_e164(value: str | None, *, required: bool = False) -> str | None:
    if value is None or str(value).strip() == "":
        if required:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid international phone number is required")
        return None

    raw = str(value).strip()
    if re.search(r"[A-Za-z:/\\<>\"'`;]", raw):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone numbers must use international E.164 format")
    if raw.count("+") > 1 or ("+" in raw and not raw.startswith("+")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone numbers must use international E.164 format")

    normalized = re.sub(r"[\s().-]", "", raw)
    if not re.fullmatch(r"\+[1-9]\d{7,14}", normalized):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone numbers must use international E.164 format")
    return normalized


def validate_time_range(start_value: str | None, end_value: str | None) -> None:
    if not start_value and not end_value:
        return
    if not start_value or not end_value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Both availability start and end are required")
    if not re.fullmatch(r"\d{2}:\d{2}", start_value) or not re.fullmatch(r"\d{2}:\d{2}", end_value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Availability times must use HH:MM format")
    if start_value >= end_value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Availability start must be before availability end")


def telephone_uri(number: str | None) -> str | None:
    if not number:
        return None
    return f"tel:{number}"


def whatsapp_uri(number: str | None, message: str = DEFAULT_WHATSAPP_MESSAGE) -> str | None:
    if not number:
        return None
    digits_only = number.lstrip("+")
    return f"https://wa.me/{digits_only}?text={quote(message)}"


def available_hours(days: str | None, start_value: str | None, end_value: str | None) -> str | None:
    if days and start_value and end_value:
        return f"{days}, {start_value}-{end_value}"
    if days:
        return days
    if start_value and end_value:
        return f"{start_value}-{end_value}"
    return None


def require_admin(user: User) -> None:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required")


def require_student(user: User) -> None:
    if user.role != UserRole.STUDENT:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student access is required")


def require_counselor_or_admin(user: User) -> None:
    if user.role not in {UserRole.COUNSELOR, UserRole.PSYCHIATRIST, UserRole.ADMIN}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Counselor access is required")


def safe_university(university: University | None) -> dict | None:
    if not university:
        return None
    return {
        "university_name": university.university_name,
        "university_code": university.university_code,
        "campus_name": university.campus_name,
        "district": university.district,
        "province": university.province,
        "counseling_unit_phone": university.counseling_unit_phone,
        "general_phone": university.general_phone,
        "email": university.email,
        "website": university.website,
        "active": university.active,
    }


def contact_payload_from_profile(profile: CounselorProfile, *, contact_type: str = "assigned_counselor") -> dict:
    telephone_available = bool(profile.accepts_voice_calls and profile.telephone_number)
    whatsapp_available = bool((profile.accepts_whatsapp_messages or profile.accepts_whatsapp_calls) and profile.whatsapp_number)
    university = profile.university
    return {
        "contact_type": contact_type,
        "display_name": profile.full_name or "University Counselor",
        "professional_title": profile.professional_title,
        "university_name": university.university_name if university else None,
        "counseling_unit": university.campus_name if university else None,
        "office": profile.office_location or profile.office_name,
        "availability": profile.availability_status,
        "available_hours": available_hours(profile.available_days, profile.available_from, profile.available_until),
        "languages": profile.languages_json or [],
        "telephone_available": telephone_available,
        "whatsapp_available": whatsapp_available,
        "telephone_uri": telephone_uri(profile.telephone_number) if telephone_available else None,
        "whatsapp_uri": whatsapp_uri(profile.whatsapp_number) if whatsapp_available else None,
        "display_number": profile.telephone_number or profile.whatsapp_number,
        "emergency_service": False,
        "limitations": SUPPORT_LIMITATIONS,
    }


def contact_payload_from_support(contact: SupportContact) -> dict:
    telephone_available = bool(contact.telephone_enabled and contact.telephone_number)
    whatsapp_available = bool(contact.whatsapp_enabled and contact.whatsapp_number)
    university = contact.university
    return {
        "contact_type": contact.contact_type,
        "display_name": contact.display_name,
        "university_name": university.university_name if university else None,
        "counseling_unit": university.campus_name if university else None,
        "availability": "available" if contact.active else "offline",
        "available_hours": available_hours(contact.available_days, contact.available_from, contact.available_until),
        "telephone_available": telephone_available,
        "whatsapp_available": whatsapp_available,
        "telephone_uri": telephone_uri(contact.telephone_number) if telephone_available else None,
        "whatsapp_uri": whatsapp_uri(contact.whatsapp_number) if whatsapp_available else None,
        "display_number": contact.telephone_number or contact.whatsapp_number,
        "emergency_service": contact.emergency_service,
        "limitations": SUPPORT_LIMITATIONS,
    }


def _profile_is_student_visible(profile: CounselorProfile, student: User) -> bool:
    if not profile or not profile.active or not profile.approved or not profile.student_visible:
        return False
    if profile.availability_status not in {"available", "busy"}:
        return False
    if student.university_id and profile.university_id != student.university_id:
        return False
    if not student.university_id:
        return False
    return bool((profile.accepts_voice_calls and profile.telephone_number) or (profile.accepts_whatsapp_messages and profile.whatsapp_number))


def select_support_contact(db: Session, student: User) -> tuple[dict, SupportContact | None]:
    require_student(student)

    assignments = (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.student_id == student.id, CounselorAssignment.active.is_(True))
        .order_by(CounselorAssignment.assigned_date.desc())
        .all()
    )
    for assignment in assignments:
        profile = (
            db.query(CounselorProfile)
            .filter(CounselorProfile.user_id == assignment.counselor_id)
            .first()
        )
        if _profile_is_student_visible(profile, student):
            return contact_payload_from_profile(profile, contact_type="assigned_counselor"), None

    university_contact = (
        db.query(SupportContact)
        .filter(
            SupportContact.university_id == student.university_id,
            SupportContact.contact_type == "university_unit",
            SupportContact.active.is_(True),
            SupportContact.student_visible.is_(True),
            SupportContact.verified.is_(True),
        )
        .order_by(SupportContact.priority.asc(), SupportContact.created_at.asc())
        .first()
        if student.university_id
        else None
    )
    if university_contact:
        return contact_payload_from_support(university_contact), university_contact

    fallback_profile = (
        db.query(CounselorProfile)
        .filter(
            CounselorProfile.university_id == student.university_id,
            CounselorProfile.active.is_(True),
            CounselorProfile.approved.is_(True),
            CounselorProfile.student_visible.is_(True),
            CounselorProfile.availability_status.in_(["available", "busy"]),
        )
        .order_by(CounselorProfile.updated_at.desc())
        .first()
        if student.university_id
        else None
    )
    if fallback_profile and _profile_is_student_visible(fallback_profile, student):
        return contact_payload_from_profile(fallback_profile, contact_type="university_fallback"), None

    system_contact = (
        db.query(SupportContact)
        .filter(
            SupportContact.contact_type == "system_fallback",
            SupportContact.active.is_(True),
            SupportContact.student_visible.is_(True),
            SupportContact.verified.is_(True),
        )
        .order_by(SupportContact.priority.asc(), SupportContact.created_at.asc())
        .first()
    )
    if not system_contact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No approved support contact is configured")
    return contact_payload_from_support(system_contact), system_contact


def serialize_profile(profile: CounselorProfile, *, include_private: bool = False) -> dict:
    data = {
        "id": profile.id,
        "counselor_profile_id": profile.counselor_profile_id,
        "user_id": profile.user_id,
        "university_id": profile.university_id,
        "full_name": profile.full_name,
        "professional_title": profile.professional_title,
        "qualification": profile.qualification,
        "specialization": profile.specialization,
        "office_name": profile.office_name,
        "office_location": profile.office_location,
        "telephone_number": profile.telephone_number,
        "whatsapp_number": profile.whatsapp_number,
        "email": profile.email,
        "available_days": profile.available_days,
        "available_from": profile.available_from,
        "available_until": profile.available_until,
        "accepts_voice_calls": profile.accepts_voice_calls,
        "accepts_whatsapp_calls": profile.accepts_whatsapp_calls,
        "accepts_whatsapp_messages": profile.accepts_whatsapp_messages,
        "emergency_contact_enabled": profile.emergency_contact_enabled,
        "languages_json": profile.languages_json or [],
        "availability_status": profile.availability_status,
        "approved": profile.approved,
        "student_visible": profile.student_visible,
        "active": profile.active,
        "university": safe_university(profile.university),
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }
    if include_private:
        data["registration_number"] = profile.registration_number
        data["admin_notes"] = profile.admin_notes
    return data


def audit_profile_change(
    db: Session,
    *,
    profile: CounselorProfile | None,
    changed_by: User | None,
    change_type: str,
    before: dict | None = None,
    after: dict | None = None,
    notes: str | None = None,
) -> None:
    changed_fields = sorted(set((before or {}).keys()) | set((after or {}).keys()))
    audit = CounselorProfileAudit(
        audit_id=public_id("audit"),
        counselor_profile_id=profile.id if profile else None,
        changed_by_user_id=changed_by.id if changed_by else None,
        change_type=change_type,
        changed_fields=changed_fields,
        before_json=before,
        after_json=after,
        notes=notes,
    )
    db.add(audit)


def record_support_action(db: Session, *, user: User, action_type: str, contact_type: str | None = None) -> dict:
    require_student(user)
    if action_type not in SUPPORT_ACTION_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported support action")
    payload, contact = select_support_contact(db, user)
    action = SupportContactAction(
        action_id=public_id("support-action"),
        user_id=user.id,
        support_contact_id=contact.id if contact else None,
        contact_type=contact_type or payload.get("contact_type"),
        action_type=action_type,
        metadata_json={
            "privacy_minimized": True,
            "call_answered_recorded": False,
            "message_content_recorded": False,
        },
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return {
        "action_id": action.action_id,
        "action_type": action.action_type,
        "recorded": True,
        "counselor_contacted": False,
        "emergency_services_contacted": False,
    }


def serialize_support_contact(contact: SupportContact, *, include_id: bool = True) -> dict:
    data = {
        "support_contact_id": contact.support_contact_id,
        "university_id": contact.university_id,
        "counselor_profile_id": contact.counselor_profile_id,
        "contact_type": contact.contact_type,
        "display_name": contact.display_name,
        "telephone_number": contact.telephone_number,
        "whatsapp_number": contact.whatsapp_number,
        "email": contact.email,
        "available_days": contact.available_days,
        "available_from": contact.available_from,
        "available_until": contact.available_until,
        "telephone_enabled": contact.telephone_enabled,
        "whatsapp_enabled": contact.whatsapp_enabled,
        "student_visible": contact.student_visible,
        "emergency_service": contact.emergency_service,
        "verified": contact.verified,
        "active": contact.active,
        "priority": contact.priority,
        "telephone_uri": telephone_uri(contact.telephone_number) if contact.telephone_enabled else None,
        "whatsapp_uri": whatsapp_uri(contact.whatsapp_number) if contact.whatsapp_enabled else None,
    }
    if include_id:
        data["id"] = contact.id
    return data


def has_unresolved_reviews(db: Session, counselor_user_id: int) -> bool:
    return (
        db.query(CounselorReview)
        .filter(CounselorReview.counselor_id == counselor_user_id, CounselorReview.status != "CLOSED")
        .first()
        is not None
    )


def transfer_active_student_assignments(db: Session, *, from_counselor_id: int, to_counselor_id: int, admin_user: User) -> int:
    now = datetime.utcnow()
    active_assignments = (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.counselor_id == from_counselor_id, CounselorAssignment.active.is_(True))
        .all()
    )
    for assignment in active_assignments:
        assignment.active = False
        assignment.end_date = assignment.end_date or now
        db.add(
            CounselorAssignment(
                assignment_id=public_id("asg"),
                student_id=assignment.student_id,
                counselor_id=to_counselor_id,
                assigned_by=admin_user.id,
                assignment_reason="Transfer during counselor deactivation or university reassignment",
                active=True,
            )
        )
    return len(active_assignments)


def preserve_university_assignment(
    db: Session,
    *,
    profile: CounselorProfile,
    university_id: int,
    admin_user: User,
    reason: str | None = None,
) -> None:
    now = datetime.utcnow()
    active_rows = (
        db.query(CounselorUniversityAssignment)
        .filter(CounselorUniversityAssignment.counselor_profile_id == profile.id, CounselorUniversityAssignment.active.is_(True))
        .all()
    )
    for row in active_rows:
        row.active = False
        row.ended_at = row.ended_at or now
    db.add(
        CounselorUniversityAssignment(
            assignment_id=public_id("cuni"),
            counselor_profile_id=profile.id,
            university_id=university_id,
            assigned_by=admin_user.id,
            assignment_reason=reason,
            active=True,
        )
    )
