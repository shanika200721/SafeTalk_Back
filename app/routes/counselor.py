from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import (
    Alert,
    Assessment,
    CounselorAssignment,
    CounselorProfile,
    CounselorNote,
    CounselorReview,
    CounselorSession,
    ModalityPrediction,
    RiskAssessment,
    User,
    UserRole,
)
from app.routes.auth import get_current_user
from app.services.counselor_workflow import (
    REVIEW_STATUSES,
    assigned_student_ids,
    authorize_counselor_student_access,
    dashboard_payload,
    is_admin,
    is_counselor_role,
    parse_date,
    public_id,
    report_csv,
    report_payload,
    report_pdf_bytes,
    scoped_students_query,
    serialize_assignment,
    serialize_note,
    serialize_review,
    serialize_student_row,
    student_detail,
    timeline,
)
from app.services.support_contacts import (
    AVAILABILITY_STATUSES,
    audit_profile_change,
    normalize_e164,
    require_counselor_or_admin,
    serialize_profile,
    validate_time_range,
)


router = APIRouter(prefix="/api/counselor", tags=["Counselor"])


class CounselorSessionCreate(BaseModel):
    user_id: int
    session_type: str = "manual"
    risk_level_at_escalation: str = "UNKNOWN"
    counselor_notes: Optional[str] = None


class CounselorSessionUpdate(BaseModel):
    status: Optional[str] = None
    counselor_notes: Optional[str] = None
    intervention_type: Optional[str] = None
    outcome: Optional[str] = None
    follow_up_needed: Optional[bool] = None
    follow_up_date: Optional[datetime] = None


class AssignmentCreate(BaseModel):
    student_id: int
    counselor_id: int
    assignment_reason: Optional[str] = None
    notes: Optional[str] = None
    active: bool = True


class AssignmentUpdate(BaseModel):
    active: Optional[bool] = None
    end_date: Optional[datetime] = None
    notes: Optional[str] = None
    assignment_reason: Optional[str] = None


class ReviewCreate(BaseModel):
    assessment_id: Optional[int] = None
    student_id: int
    status: str = "NEW"
    review_notes: Optional[str] = None
    decision: Optional[str] = None
    risk_judgement: Optional[str] = None


class ReviewUpdate(BaseModel):
    status: Optional[str] = None
    review_notes: Optional[str] = None
    decision: Optional[str] = None
    risk_judgement: Optional[str] = None


class NoteCreate(BaseModel):
    student_id: int
    note_text: str
    note_type: str = "clinical"


class NoteUpdate(BaseModel):
    note_text: Optional[str] = None
    note_type: Optional[str] = None
    active: Optional[bool] = None


class CounselorProfileUpdate(BaseModel):
    telephone_number: Optional[str] = None
    whatsapp_number: Optional[str] = None
    office_location: Optional[str] = None
    office_name: Optional[str] = None
    available_days: Optional[str] = None
    available_from: Optional[str] = None
    available_until: Optional[str] = None
    accepts_voice_calls: Optional[bool] = None
    accepts_whatsapp_calls: Optional[bool] = None
    accepts_whatsapp_messages: Optional[bool] = None
    availability_status: Optional[str] = None
    qualification: Optional[str] = None
    specialization: Optional[str] = None
    languages_json: Optional[list[str]] = None


def verify_counselor(current_user: User = Depends(get_current_user)) -> User:
    if not (is_admin(current_user) or is_counselor_role(current_user)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only assigned counselors and admins can access this endpoint",
        )
    return current_user


def _require_admin(current_user: User) -> None:
    if not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required")


def _validate_review_status(status_value: str) -> None:
    if status_value not in REVIEW_STATUSES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid review status")


def _get_own_profile(db: Session, current_user: User) -> CounselorProfile:
    require_counselor_or_admin(current_user)
    profile = db.query(CounselorProfile).filter(CounselorProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Counselor profile not found")
    return profile


@router.get("/profile")
def get_counselor_profile(current_user: User = Depends(verify_counselor), db: Session = Depends(get_db)):
    profile = _get_own_profile(db, current_user)
    return serialize_profile(profile, include_private=False)


@router.patch("/profile")
def update_counselor_profile(
    payload: CounselorProfileUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    profile = _get_own_profile(db, current_user)
    updates = payload.dict(exclude_unset=True)
    if "availability_status" in updates and updates["availability_status"] not in AVAILABILITY_STATUSES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid availability status")
    validate_time_range(updates.get("available_from", profile.available_from), updates.get("available_until", profile.available_until))
    before = serialize_profile(profile, include_private=False)
    for field in ("telephone_number", "whatsapp_number"):
        if field in updates:
            updates[field] = normalize_e164(updates[field])
    for field, value in updates.items():
        setattr(profile, field, value)
    profile.updated_at = datetime.utcnow()
    audit_profile_change(
        db,
        profile=profile,
        changed_by=current_user,
        change_type="counselor_profile_update",
        before=before,
        after=serialize_profile(profile, include_private=False),
    )
    db.commit()
    db.refresh(profile)
    return serialize_profile(profile, include_private=False)


@router.get("/availability")
def get_counselor_availability(current_user: User = Depends(verify_counselor), db: Session = Depends(get_db)):
    profile = _get_own_profile(db, current_user)
    return {
        "availability_status": profile.availability_status,
        "available_days": profile.available_days,
        "available_from": profile.available_from,
        "available_until": profile.available_until,
        "accepts_voice_calls": profile.accepts_voice_calls,
        "accepts_whatsapp_calls": profile.accepts_whatsapp_calls,
        "accepts_whatsapp_messages": profile.accepts_whatsapp_messages,
    }


@router.patch("/availability")
def update_counselor_availability(
    payload: CounselorProfileUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    allowed = {
        "available_days",
        "available_from",
        "available_until",
        "accepts_voice_calls",
        "accepts_whatsapp_calls",
        "accepts_whatsapp_messages",
        "availability_status",
    }
    filtered = {key: value for key, value in payload.dict(exclude_unset=True).items() if key in allowed}
    return update_counselor_profile(CounselorProfileUpdate(**filtered), current_user=current_user, db=db)


def _scoped_alert_query(db: Session, current_user: User):
    ids = assigned_student_ids(db, current_user)
    query = db.query(Alert)
    if not is_admin(current_user):
        query = query.filter(Alert.user_id.in_(ids) if ids else Alert.user_id == -1)
    return query


@router.post("/assignments")
def create_assignment(
    payload: AssignmentCreate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    student = db.query(User).filter(User.id == payload.student_id, User.role == UserRole.STUDENT).first()
    counselor = db.query(User).filter(User.id == payload.counselor_id).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    if not counselor or not is_counselor_role(counselor):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Counselor not found")

    if payload.active:
        existing = (
            db.query(CounselorAssignment)
            .filter(CounselorAssignment.student_id == student.id, CounselorAssignment.active.is_(True))
            .all()
        )
        now = datetime.utcnow()
        for assignment in existing:
            assignment.active = False
            assignment.end_date = assignment.end_date or now

    assignment = CounselorAssignment(
        assignment_id=public_id("asg"),
        student_id=student.id,
        counselor_id=counselor.id,
        assigned_by=current_user.id,
        assignment_reason=payload.assignment_reason,
        notes=payload.notes,
        active=payload.active,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return serialize_assignment(assignment)


@router.patch("/assignments/{assignment_id}")
def update_assignment(
    assignment_id: int,
    payload: AssignmentUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    assignment = db.query(CounselorAssignment).filter(CounselorAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    if payload.active is not None:
        assignment.active = payload.active
        if payload.active is False:
            assignment.end_date = payload.end_date or datetime.utcnow()
    if payload.end_date is not None:
        assignment.end_date = payload.end_date
    if payload.notes is not None:
        assignment.notes = payload.notes
    if payload.assignment_reason is not None:
        assignment.assignment_reason = payload.assignment_reason
    assignment.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(assignment)
    return serialize_assignment(assignment)


@router.get("/dashboard")
def get_dashboard(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    return dashboard_payload(db, current_user, parse_date(start_date), parse_date(end_date))


@router.get("/students")
def get_students(
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    query = scoped_students_query(db, current_user)
    if search:
        pattern = f"%{search.lower()}%"
        query = query.filter((User.full_name.ilike(pattern)) | (User.email.ilike(pattern)) | (User.username.ilike(pattern)))

    students = query.order_by(User.full_name.asc()).all()
    rows = [serialize_student_row(db, student, current_user) for student in students]
    if risk_level and risk_level != "ALL":
        rows = [row for row in rows if row.get("risk_level") == risk_level]

    safe_limit = max(1, min(limit, 200))
    safe_page = max(1, page)
    start = (safe_page - 1) * safe_limit
    paginated = rows[start : start + safe_limit]
    return {
        "counselor_id": current_user.id,
        "scope": "all_students" if is_admin(current_user) else "assigned_students",
        "total": len(rows),
        "total_students": len(rows),
        "page": safe_page,
        "limit": safe_limit,
        "students": paginated,
    }


@router.get("/student/{student_id}")
def get_student(
    student_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    return student_detail(db, student_id, current_user)


@router.get("/student/{student_id}/dashboard")
def get_student_dashboard(
    student_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    return student_detail(db, student_id, current_user)


@router.get("/student/{student_id}/timeline")
def get_student_timeline(
    student_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    return timeline(db, student_id, current_user)


@router.get("/student/{student_id}/evidence")
def get_student_evidence(
    student_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    authorize_counselor_student_access(current_user, student_id, db)
    latest_assessment = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.student_id == student_id)
        .order_by(RiskAssessment.created_at.desc())
        .first()
    )
    inputs_by_prediction = {}
    if latest_assessment:
        inputs_by_prediction = {
            item.modality_prediction_id: item
            for item in latest_assessment.inputs
            if item.modality_prediction_id
        }
    evidence = []
    for modality in ["profile", "dass21", "mood", "text", "speech", "face", "behavioral"]:
        prediction = (
            db.query(ModalityPrediction)
            .filter(ModalityPrediction.student_id == student_id, ModalityPrediction.modality == modality)
            .order_by(ModalityPrediction.created_at.desc())
            .first()
        )
        fusion_input = inputs_by_prediction.get(prediction.id) if prediction else None
        evidence.append(
            {
                "modality": modality,
                "status": prediction.status if prediction else "missing",
                "captured_at": prediction.source_timestamp if prediction else None,
                "model_version": prediction.model_version if prediction else None,
                "preprocessing_version": prediction.preprocessing_version if prediction else None,
                "confidence_band": _confidence_band(prediction.confidence if prediction else None),
                "normalized_contribution": fusion_input.mapped_score if fusion_input else None,
                "effective_fusion_weight": fusion_input.effective_weight if fusion_input else None,
                "included": bool(fusion_input and fusion_input.included),
                "inclusion_or_exclusion_reason": None if fusion_input and fusion_input.included else (fusion_input.exclusion_reason if fusion_input else "not_in_latest_fusion"),
                "source": _safe_evidence_source(prediction) if prediction else None,
                "limitation": _evidence_limitation(modality),
            }
        )
    return {
        "student_id": student_id,
        "assessment_id": latest_assessment.id if latest_assessment else None,
        "modalities": evidence,
        "privacy": {
            "raw_audio_autoplay": False,
            "raw_journal_body_included": False,
            "journal_prediction_does_not_grant_body_access": True,
        },
    }


def _confidence_band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "unavailable"
    if confidence >= 0.75:
        return "higher"
    if confidence >= 0.45:
        return "moderate"
    return "low"


def _safe_evidence_source(prediction: ModalityPrediction) -> dict:
    source = {"type": prediction.source_type, "reference": prediction.source_record_id}
    if prediction.modality == "speech":
        source["display"] = "counselor-chat student voice message"
        source["raw_audio_access"] = "authorized_media_route_only"
    if prediction.modality == "face":
        source["display"] = "explicit student capture"
        source["raw_image_display_default"] = False
    if prediction.source_type == "journal_entry":
        source["display"] = "consented journal text prediction"
        source["raw_body_included"] = False
    return source


def _evidence_limitation(modality: str) -> str:
    if modality == "speech":
        return "Voice-emotion signal is supporting evidence only and not a diagnosis."
    if modality == "face":
        return "Facial-emotion signal is experimental or unavailable unless explicitly approved."
    if modality == "behavioral":
        return "Behavioral modality is unavailable without a verified behavioral model."
    if modality == "text":
        return "Text signal is screening support only; journal entries are not continuously monitored."
    return "Screening-support signal only; counselor judgement remains separate."


@router.get("/student/{student_id}/reports")
def get_student_report(
    student_id: int,
    format: str = "json",
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    payload = report_payload(db, student_id, current_user)
    if format == "json":
        return payload
    if format == "csv":
        return Response(
            content=report_csv(payload),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=student_{student_id}_report.csv"},
        )
    if format == "pdf":
        return Response(
            content=report_pdf_bytes(payload),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=student_{student_id}_report.pdf"},
        )
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="format must be json, csv, or pdf")


@router.post("/reviews")
def create_review(
    payload: ReviewCreate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    _validate_review_status(payload.status)
    authorize_counselor_student_access(current_user, payload.student_id, db)
    if payload.assessment_id is not None:
        assessment = db.query(RiskAssessment).filter(RiskAssessment.id == payload.assessment_id).first()
        if not assessment or assessment.student_id != payload.student_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Risk assessment not found for student")

    review = CounselorReview(
        review_id=public_id("rev"),
        assessment_id=payload.assessment_id,
        student_id=payload.student_id,
        counselor_id=current_user.id,
        status=payload.status,
        review_notes=payload.review_notes,
        decision=payload.decision,
        risk_judgement=payload.risk_judgement,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return serialize_review(review)


@router.patch("/reviews/{review_id}")
def update_review(
    review_id: int,
    payload: ReviewUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    review = db.query(CounselorReview).filter(CounselorReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    authorize_counselor_student_access(current_user, review.student_id, db)
    if payload.status is not None:
        _validate_review_status(payload.status)
        review.status = payload.status
    if payload.review_notes is not None:
        review.review_notes = payload.review_notes
    if payload.decision is not None:
        review.decision = payload.decision
    if payload.risk_judgement is not None:
        review.risk_judgement = payload.risk_judgement
    review.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(review)
    return serialize_review(review)


@router.post("/notes")
def create_note(
    payload: NoteCreate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    authorize_counselor_student_access(current_user, payload.student_id, db)
    if not payload.note_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Note text is required")
    note = CounselorNote(
        note_id=public_id("note"),
        student_id=payload.student_id,
        counselor_id=current_user.id,
        note_text=payload.note_text,
        note_type=payload.note_type,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return serialize_note(note)


@router.patch("/notes/{note_id}")
def update_note(
    note_id: int,
    payload: NoteUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    note = db.query(CounselorNote).filter(CounselorNote.id == note_id).first()
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    authorize_counselor_student_access(current_user, note.student_id, db)
    if payload.note_text is not None:
        if not payload.note_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Note text is required")
        note.note_text = payload.note_text
    if payload.note_type is not None:
        note.note_type = payload.note_type
    if payload.active is not None:
        note.active = payload.active
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return serialize_note(note)


@router.get("/alerts")
def get_alerts(
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    query = _scoped_alert_query(db, current_user)
    if unread_only:
        query = query.filter(Alert.is_read.is_(False))
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    return {
        "counselor_id": current_user.id,
        "total_alerts": len(alerts),
        "alerts": [
            {
                "id": alert.id,
                "user_id": alert.user_id,
                "student_name": db.query(User.full_name).filter(User.id == alert.user_id).scalar() or "Unknown Student",
                "alert_type": alert.alert_type,
                "risk_level": alert.risk_level,
                "message": alert.message,
                "is_read": alert.is_read,
                "created_at": alert.created_at,
            }
            for alert in alerts
        ],
    }


@router.put("/alerts/{alert_id}/read")
def mark_alert_read(
    alert_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    authorize_counselor_student_access(current_user, alert.user_id, db)
    alert.is_read = True
    db.commit()
    return {"message": "Alert marked as read"}


@router.get("/high-risk-users")
def get_high_risk_users(
    risk_level: str = "HIGH",
    limit: int = 50,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    ids = assigned_student_ids(db, current_user)
    if not ids:
        return {"counselor_id": current_user.id, "total_users": 0, "high_risk_users": []}
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    assessments = (
        db.query(RiskAssessment)
        .filter(
            RiskAssessment.student_id.in_(ids),
            RiskAssessment.created_at >= seven_days_ago,
            RiskAssessment.risk_level.in_([risk_level, "SEVERE"]),
        )
        .order_by(RiskAssessment.created_at.desc())
        .limit(limit)
        .all()
    )
    rows = []
    seen = set()
    for assessment in assessments:
        if assessment.student_id in seen:
            continue
        student = db.query(User).filter(User.id == assessment.student_id).first()
        if student:
            rows.append(
                {
                    "user_id": student.id,
                    "name": student.full_name,
                    "email": student.email,
                    "risk_level": assessment.risk_level,
                    "composite_score": assessment.final_score,
                    "last_assessment": assessment.created_at,
                }
            )
            seen.add(student.id)
    return {"counselor_id": current_user.id, "total_users": len(rows), "high_risk_users": rows}


@router.post("/sessions")
def create_session(
    session_data: CounselorSessionCreate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    authorize_counselor_student_access(current_user, session_data.user_id, db)
    session = CounselorSession(
        user_id=session_data.user_id,
        counselor_id=current_user.id,
        session_type=session_data.session_type,
        risk_level_at_escalation=session_data.risk_level_at_escalation,
        counselor_notes=session_data.counselor_notes,
        status="pending",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"message": "Session created successfully", "session_id": session.id, "session": _serialize_session(session)}


@router.get("/sessions/{session_id}")
def get_session(
    session_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    session = db.query(CounselorSession).filter(CounselorSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    authorize_counselor_student_access(current_user, session.user_id, db)
    return _serialize_session(session)


@router.put("/sessions/{session_id}")
def update_session(
    session_id: int,
    session_update: CounselorSessionUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    session = db.query(CounselorSession).filter(CounselorSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    authorize_counselor_student_access(current_user, session.user_id, db)
    for field, value in session_update.dict(exclude_unset=True).items():
        setattr(session, field, value)
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return {"message": "Session updated successfully", "session": _serialize_session(session)}


@router.get("/sessions/user/{student_id}")
def get_student_sessions(
    student_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    authorize_counselor_student_access(current_user, student_id, db)
    sessions = (
        db.query(CounselorSession)
        .filter(CounselorSession.user_id == student_id)
        .order_by(CounselorSession.created_at.desc())
        .all()
    )
    return {"user_id": student_id, "total_sessions": len(sessions), "sessions": [_serialize_session(s) for s in sessions]}


@router.get("/analytics/summary")
def get_analytics_summary(
    days: int = 30,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db),
):
    start = datetime.utcnow() - timedelta(days=max(1, days))
    return dashboard_payload(db, current_user, start, None)


def _serialize_session(session: CounselorSession) -> dict:
    return {
        "id": session.id,
        "user_id": session.user_id,
        "counselor_id": session.counselor_id,
        "session_type": session.session_type,
        "status": session.status,
        "risk_level_at_escalation": session.risk_level_at_escalation,
        "counselor_notes": session.counselor_notes,
        "intervention_type": session.intervention_type,
        "outcome": session.outcome,
        "follow_up_needed": session.follow_up_needed,
        "follow_up_date": session.follow_up_date,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
