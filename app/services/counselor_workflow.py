from __future__ import annotations

import csv
import io
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database_models import (
    Alert,
    Assessment,
    CounselorAssignment,
    CounselorNote,
    CounselorReview,
    DASS21Assessment,
    DailyCheckIn,
    ModalityPrediction,
    ProfileAssessment,
    RiskAssessment,
    User,
    UserRole,
)
from app.services.fusion import controlled_fusion_config


REVIEW_STATUSES = ("NEW", "UNDER_REVIEW", "FOLLOW_UP_REQUIRED", "REFERRED", "CLOSED")
MODEL_DISCLAIMER = (
    "AI outputs are screening-support signals only. Counselor review notes, decisions, "
    "and risk judgements are recorded separately and do not overwrite model output."
)
DASS21_TOTAL_MAX_SCORE = 126.0
DASS21_SUBSCALE_MAX_SCORE = 42.0
CANONICAL_MODALITIES = ("profile", "dass21", "mood", "text", "speech", "face", "behavioral")
MODALITY_LABELS = {
    "profile": "Profile",
    "dass21": "DASS-21",
    "mood": "Mood",
    "text": "Text",
    "speech": "Speech",
    "face": "Face",
    "behavioral": "Behavioral",
}


def public_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def iso(value):
    return value.isoformat() if value else None


def _bounded_percentage(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(100.0, numeric)), 2)


def _score_to_percentage(score_0_100: float | int | None = None, probability: float | int | None = None) -> float | None:
    if score_0_100 is not None:
        return _bounded_percentage(score_0_100)
    if probability is not None:
        return _bounded_percentage(float(probability) * 100)
    return None


def _dass_percentage(score: float | int | None, max_score: float = DASS21_TOTAL_MAX_SCORE) -> float | None:
    if score is None:
        return None
    try:
        return _bounded_percentage((float(score) / max_score) * 100)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _risk_assessment_percentage(assessment: RiskAssessment) -> float | None:
    if assessment.final_score is not None:
        return _bounded_percentage(assessment.final_score)
    if assessment.final_probability is not None:
        return _bounded_percentage(float(assessment.final_probability) * 100)
    if assessment.model_score is not None:
        model_score = float(assessment.model_score)
        return _bounded_percentage(model_score * 100 if model_score <= 1 else model_score)
    return None


def is_admin(user: User) -> bool:
    return user.role == UserRole.ADMIN or getattr(user.role, "value", None) == "admin"


def is_counselor_role(user: User) -> bool:
    return user.role in {UserRole.COUNSELOR, UserRole.PSYCHIATRIST} or getattr(user.role, "value", None) in {
        "counselor",
        "psychiatrist",
    }


def get_student_or_404(db: Session, student_id: int) -> User:
    student = db.query(User).filter(User.id == student_id, User.role == UserRole.STUDENT).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return student


def has_active_assignment(db: Session, counselor_id: int, student_id: int) -> bool:
    return (
        db.query(CounselorAssignment)
        .filter(
            CounselorAssignment.student_id == student_id,
            CounselorAssignment.counselor_id == counselor_id,
            CounselorAssignment.active.is_(True),
        )
        .first()
        is not None
    )


def authorize_counselor_student_access(current_user: User, student_id: int, db: Session) -> User:
    student = get_student_or_404(db, student_id)
    if is_admin(current_user):
        return student
    if is_counselor_role(current_user) and has_active_assignment(db, current_user.id, student_id):
        return student
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Counselor assignment is required for this student")


def assigned_student_ids(db: Session, current_user: User) -> list[int]:
    if is_admin(current_user):
        return [row[0] for row in db.query(User.id).filter(User.role == UserRole.STUDENT).all()]
    return [
        row[0]
        for row in (
            db.query(CounselorAssignment.student_id)
            .filter(
                CounselorAssignment.counselor_id == current_user.id,
                CounselorAssignment.active.is_(True),
            )
            .distinct()
            .all()
        )
    ]


def scoped_students_query(db: Session, current_user: User):
    ids = assigned_student_ids(db, current_user)
    query = db.query(User).filter(User.role == UserRole.STUDENT)
    if not ids:
        return query.filter(User.id == -1)
    return query.filter(User.id.in_(ids))


def latest_dass(db: Session, student_id: int) -> DASS21Assessment | None:
    return (
        db.query(DASS21Assessment)
        .filter(DASS21Assessment.user_id == student_id)
        .order_by(DASS21Assessment.created_at.desc())
        .first()
    )


def latest_checkin(db: Session, student_id: int) -> DailyCheckIn | None:
    return (
        db.query(DailyCheckIn)
        .filter(DailyCheckIn.user_id == student_id)
        .order_by(DailyCheckIn.created_at.desc())
        .first()
    )


def latest_risk(db: Session, student_id: int) -> RiskAssessment | None:
    return (
        db.query(RiskAssessment)
        .filter(RiskAssessment.student_id == student_id)
        .order_by(RiskAssessment.created_at.desc())
        .first()
    )


def latest_profile(db: Session, student_id: int) -> ProfileAssessment | None:
    return (
        db.query(ProfileAssessment)
        .filter(ProfileAssessment.user_id == student_id)
        .order_by(ProfileAssessment.updated_at.desc())
        .first()
    )


def serialize_assignment(assignment: CounselorAssignment | None) -> dict | None:
    if not assignment:
        return None
    return {
        "id": assignment.id,
        "assignment_id": assignment.assignment_id,
        "student_id": assignment.student_id,
        "counselor_id": assignment.counselor_id,
        "assigned_by": assignment.assigned_by,
        "assignment_reason": assignment.assignment_reason,
        "assigned_date": iso(assignment.assigned_date),
        "active": assignment.active,
        "end_date": iso(assignment.end_date),
        "notes": assignment.notes,
    }


def serialize_student_row(db: Session, student: User, current_user: User) -> dict:
    dass = latest_dass(db, student.id)
    checkin = latest_checkin(db, student.id)
    risk = latest_risk(db, student.id)
    assignment = (
        db.query(CounselorAssignment)
        .filter(
            CounselorAssignment.student_id == student.id,
            CounselorAssignment.counselor_id == current_user.id,
            CounselorAssignment.active.is_(True),
        )
        .order_by(CounselorAssignment.assigned_date.desc())
        .first()
        if not is_admin(current_user)
        else db.query(CounselorAssignment)
        .filter(CounselorAssignment.student_id == student.id, CounselorAssignment.active.is_(True))
        .order_by(CounselorAssignment.assigned_date.desc())
        .first()
    )
    open_reviews = (
        db.query(CounselorReview)
        .filter(CounselorReview.student_id == student.id, CounselorReview.status != "CLOSED")
        .count()
    )
    return {
        "id": student.id,
        "user_id": student.id,
        "full_name": student.full_name,
        "name": student.full_name,
        "email": student.email,
        "department": student.department,
        "year_of_study": student.year_of_study,
        "assigned_date": iso(assignment.assigned_date) if assignment else None,
        "assignment": serialize_assignment(assignment),
        "risk_level": (risk.risk_level or risk.model_risk_level) if risk else None,
        "fusion_score": risk.final_score if risk else None,
        "model_score": risk.model_score if risk else None,
        "evidence_coverage": risk.evidence_coverage if risk else None,
        "coverage_category": risk.coverage_category if risk else None,
        "last_assessment": iso(risk.created_at) if risk else None,
        "avg_dass21_score": dass.total_dass21_score if dass else None,
        "last_mood": checkin.mood if checkin else None,
        "last_stress": checkin.stress_level if checkin else None,
        "last_checkin": iso(checkin.created_at) if checkin else None,
        "open_reviews": open_reviews,
    }


def serialize_risk_assessment(assessment: RiskAssessment) -> dict:
    risk_percentage = _risk_assessment_percentage(assessment)
    return {
        "id": assessment.id,
        "assessment_id": assessment.id,
        "student_id": assessment.student_id,
        "final_probability": assessment.final_probability,
        "final_score": assessment.final_score,
        "risk_percentage": risk_percentage,
        "risk_level": assessment.risk_level,
        "confidence": assessment.confidence,
        "status": assessment.status,
        "assessment_type": assessment.assessment_type,
        "model_score": assessment.model_score,
        "model_risk_level": assessment.model_risk_level,
        "evidence_coverage": assessment.evidence_coverage,
        "coverage_category": assessment.coverage_category,
        "available_modalities": assessment.available_modalities,
        "used_modalities": assessment.used_modalities,
        "missing_modalities": assessment.missing_modalities,
        "screening_only": assessment.screening_only,
        "model_output_only": assessment.model_output_only,
        "created_at": iso(assessment.created_at),
    }


def serialize_dass(assessment: DASS21Assessment) -> dict:
    component_percentages = {
        "depression": _dass_percentage(assessment.depression_score, DASS21_SUBSCALE_MAX_SCORE),
        "anxiety": _dass_percentage(assessment.anxiety_score, DASS21_SUBSCALE_MAX_SCORE),
        "stress": _dass_percentage(assessment.stress_score, DASS21_SUBSCALE_MAX_SCORE),
    }
    return {
        "id": assessment.id,
        "depression_score": assessment.depression_score,
        "anxiety_score": assessment.anxiety_score,
        "stress_score": assessment.stress_score,
        "total_dass21_score": assessment.total_dass21_score,
        "risk_percentage": _dass_percentage(assessment.total_dass21_score),
        "overall_percentage": _dass_percentage(assessment.total_dass21_score),
        "component_percentages": component_percentages,
        "depression_severity": assessment.depression_severity,
        "anxiety_severity": assessment.anxiety_severity,
        "stress_severity": assessment.stress_severity,
        "is_complete": assessment.is_complete,
        "created_at": iso(assessment.created_at),
    }


def serialize_checkin(checkin: DailyCheckIn) -> dict:
    return {
        "id": checkin.id,
        "mood": checkin.mood,
        "stress_level": checkin.stress_level,
        "anxiety_level": checkin.anxiety_level,
        "sleep_hours": checkin.sleep_hours,
        "self_harm_thoughts": checkin.self_harm_thoughts,
        "negative_thoughts": checkin.negative_thoughts,
        "created_at": iso(checkin.created_at),
    }


def serialize_prediction(prediction: ModalityPrediction) -> dict:
    return {
        "id": prediction.id,
        "modality": prediction.modality,
        "modality_label": MODALITY_LABELS.get(prediction.modality, prediction.modality),
        "predicted_class": prediction.predicted_class,
        "probability": prediction.probability,
        "score_0_100": prediction.score_0_100,
        "risk_percentage": _score_to_percentage(prediction.score_0_100, prediction.probability),
        "confidence": prediction.confidence,
        "label": prediction.label or prediction.predicted_class,
        "raw_output": prediction.raw_output_json or {},
        "metadata": prediction.metadata_json or {},
        "data_quality_status": prediction.data_quality_status,
        "data_quality_flags": prediction.data_quality_flags or [],
        "status": prediction.status,
        "is_available": prediction.is_available,
        "output_type": prediction.output_type,
        "evidence_available": prediction.evidence_available,
        "clinical_use_boundary": prediction.clinical_use_boundary,
        "source_timestamp": iso(prediction.source_timestamp),
        "generated_at": iso(prediction.generated_at),
    }


def serialize_model_component_summary(
    latest: RiskAssessment | None,
    predictions: list[ModalityPrediction],
    dass_items: list[DASS21Assessment],
    profile: ProfileAssessment | None,
) -> list[dict]:
    latest_predictions = {}
    for prediction in predictions:
        latest_predictions.setdefault(prediction.modality, prediction)

    inputs_by_modality = {}
    if latest:
        for item in latest.inputs:
            if item.modality:
                inputs_by_modality[item.modality] = item

    latest_dass_item = dass_items[0] if dass_items else None
    base_weights = controlled_fusion_config()["base_weights"]
    rows = []
    for modality in CANONICAL_MODALITIES:
        fusion_input = inputs_by_modality.get(modality)
        prediction = fusion_input.modality_prediction if fusion_input and fusion_input.modality_prediction else latest_predictions.get(modality)
        risk_percentage = _score_to_percentage(
            prediction.score_0_100 if prediction else None,
            prediction.probability if prediction else None,
        )
        if risk_percentage is None and fusion_input and fusion_input.mapped_score is not None:
            risk_percentage = _bounded_percentage(fusion_input.mapped_score * 100)
        if risk_percentage is None and modality == "dass21" and latest_dass_item:
            risk_percentage = _dass_percentage(latest_dass_item.total_dass21_score)
        if risk_percentage is None and modality == "profile" and profile:
            risk_percentage = _bounded_percentage(profile.profile_score)

        contribution = None
        if fusion_input and fusion_input.included and fusion_input.mapped_score is not None and fusion_input.effective_weight is not None:
            contribution = _bounded_percentage(fusion_input.mapped_score * fusion_input.effective_weight * 100)

        rows.append(
            {
                "modality": modality,
                "label": MODALITY_LABELS.get(modality, modality),
                "status": prediction.status if prediction else "missing",
                "risk_percentage": risk_percentage,
                "component_percentage": risk_percentage,
                "contribution_percentage": contribution,
                "base_weight_percentage": _bounded_percentage(
                    (fusion_input.base_weight if fusion_input and fusion_input.base_weight is not None else base_weights.get(modality, 0.0)) * 100
                ),
                "effective_weight_percentage": _bounded_percentage(fusion_input.effective_weight * 100)
                if fusion_input and fusion_input.effective_weight is not None
                else None,
                "included": bool(fusion_input and fusion_input.included),
                "source_timestamp": iso(fusion_input.source_timestamp if fusion_input else (prediction.source_timestamp if prediction else None)),
                "generated_at": iso(prediction.generated_at) if prediction else None,
                "evidence_id": prediction.id if prediction else None,
                "reason": None
                if fusion_input and fusion_input.included
                else (fusion_input.exclusion_reason if fusion_input else "not_in_latest_fusion"),
            }
        )
    return rows


def serialize_review(review: CounselorReview) -> dict:
    return {
        "id": review.id,
        "review_id": review.review_id,
        "assessment_id": review.assessment_id,
        "student_id": review.student_id,
        "counselor_id": review.counselor_id,
        "status": review.status,
        "review_notes": review.review_notes,
        "decision": review.decision,
        "risk_judgement": review.risk_judgement,
        "created_at": iso(review.created_at),
        "updated_at": iso(review.updated_at),
    }


def serialize_note(note: CounselorNote) -> dict:
    return {
        "id": note.id,
        "note_id": note.note_id,
        "student_id": note.student_id,
        "counselor_id": note.counselor_id,
        "note_text": note.note_text,
        "note_type": note.note_type,
        "active": note.active,
        "created_at": iso(note.created_at),
        "updated_at": iso(note.updated_at),
    }


def student_detail(db: Session, student_id: int, current_user: User) -> dict:
    student = authorize_counselor_student_access(current_user, student_id, db)
    assessments = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.student_id == student_id)
        .order_by(RiskAssessment.created_at.desc())
        .limit(25)
        .all()
    )
    dass_items = (
        db.query(DASS21Assessment)
        .filter(DASS21Assessment.user_id == student_id)
        .order_by(DASS21Assessment.created_at.desc())
        .limit(25)
        .all()
    )
    checkins = (
        db.query(DailyCheckIn)
        .filter(DailyCheckIn.user_id == student_id)
        .order_by(DailyCheckIn.created_at.desc())
        .limit(30)
        .all()
    )
    predictions = (
        db.query(ModalityPrediction)
        .filter(ModalityPrediction.student_id == student_id)
        .order_by(ModalityPrediction.generated_at.desc())
        .limit(50)
        .all()
    )
    reviews = (
        db.query(CounselorReview)
        .filter(CounselorReview.student_id == student_id)
        .order_by(CounselorReview.updated_at.desc())
        .all()
    )
    notes = (
        db.query(CounselorNote)
        .filter(CounselorNote.student_id == student_id)
        .order_by(CounselorNote.created_at.desc())
        .all()
    )
    profile = latest_profile(db, student_id)
    active_assignment = (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.student_id == student_id, CounselorAssignment.active.is_(True))
        .order_by(CounselorAssignment.assigned_date.desc())
        .first()
    )
    latest = assessments[0] if assessments else None
    return {
        "student": {
            "id": student.id,
            "user_id": student.id,
            "full_name": student.full_name,
            "name": student.full_name,
            "email": student.email,
            "department": student.department,
            "year_of_study": student.year_of_study,
        },
        "user": {
            "id": student.id,
            "name": student.full_name,
            "email": student.email,
            "department": student.department,
            "year_of_study": student.year_of_study,
        },
        "assignment": serialize_assignment(active_assignment),
        "summary": serialize_student_row(db, student, current_user),
        "profile_assessment": {
            "profile_score": profile.profile_score,
            "updated_at": iso(profile.updated_at),
        }
        if profile
        else None,
        "latest_assessment": serialize_risk_assessment(latest) if latest else None,
        "latest_risk_assessment": serialize_risk_assessment(latest) if latest else None,
        "model_component_summary": serialize_model_component_summary(latest, predictions, dass_items, profile),
        "assessments": [serialize_risk_assessment(a) for a in assessments],
        "dass21_assessments": [serialize_dass(a) for a in dass_items],
        "dass21_scores": serialize_dass(dass_items[0]) if dass_items else None,
        "today_checkin": serialize_checkin(checkins[0]) if checkins else None,
        "recent_checkins": [serialize_checkin(c) for c in reversed(checkins)],
        "modality_evidence": [serialize_prediction(p) for p in predictions],
        "reviews": [serialize_review(r) for r in reviews],
        "notes": [serialize_note(n) for n in notes],
        "model_disclaimer": MODEL_DISCLAIMER,
    }


def timeline(db: Session, student_id: int, current_user: User) -> dict:
    authorize_counselor_student_access(current_user, student_id, db)
    events = []
    for item in db.query(DASS21Assessment).filter(DASS21Assessment.user_id == student_id).all():
        events.append({"type": "dass21", "timestamp": iso(item.created_at), "label": "DASS-21", "data": serialize_dass(item)})
    for item in db.query(DailyCheckIn).filter(DailyCheckIn.user_id == student_id).all():
        events.append({"type": "mood", "timestamp": iso(item.created_at), "label": "Daily check-in", "data": serialize_checkin(item)})
    for item in db.query(RiskAssessment).filter(RiskAssessment.student_id == student_id).all():
        events.append({"type": "fusion", "timestamp": iso(item.created_at), "label": "Controlled fusion", "data": serialize_risk_assessment(item)})
    for item in db.query(CounselorReview).filter(CounselorReview.student_id == student_id).all():
        events.append({"type": "review", "timestamp": iso(item.updated_at), "label": f"Review {item.status}", "data": serialize_review(item)})
    for item in db.query(CounselorNote).filter(CounselorNote.student_id == student_id).all():
        events.append({"type": "note", "timestamp": iso(item.created_at), "label": "Counselor note", "data": serialize_note(item)})
    events.sort(key=lambda event: event["timestamp"] or "", reverse=True)
    return {"student_id": student_id, "events": events, "model_disclaimer": MODEL_DISCLAIMER}


def dashboard_payload(db: Session, current_user: User, start_date: datetime | None = None, end_date: datetime | None = None) -> dict:
    students = scoped_students_query(db, current_user).all()
    student_ids = [student.id for student in students]
    if not student_ids:
        return empty_dashboard(current_user)

    risk_query = db.query(RiskAssessment).filter(RiskAssessment.student_id.in_(student_ids))
    checkin_query = db.query(DailyCheckIn).filter(DailyCheckIn.user_id.in_(student_ids))
    dass_query = db.query(DASS21Assessment).filter(DASS21Assessment.user_id.in_(student_ids))
    review_query = db.query(CounselorReview).filter(CounselorReview.student_id.in_(student_ids))
    if start_date:
        risk_query = risk_query.filter(RiskAssessment.created_at >= start_date)
        checkin_query = checkin_query.filter(DailyCheckIn.created_at >= start_date)
        dass_query = dass_query.filter(DASS21Assessment.created_at >= start_date)
        review_query = review_query.filter(CounselorReview.created_at >= start_date)
    if end_date:
        risk_query = risk_query.filter(RiskAssessment.created_at <= end_date)
        checkin_query = checkin_query.filter(DailyCheckIn.created_at <= end_date)
        dass_query = dass_query.filter(DASS21Assessment.created_at <= end_date)
        review_query = review_query.filter(CounselorReview.created_at <= end_date)

    risks = risk_query.order_by(RiskAssessment.created_at.asc()).all()
    checkins = checkin_query.order_by(DailyCheckIn.created_at.asc()).all()
    dass_items = dass_query.order_by(DASS21Assessment.created_at.asc()).all()
    reviews = review_query.all()
    latest_by_student = {student.id: latest_risk(db, student.id) for student in students}
    distribution = Counter((risk.risk_level or risk.model_risk_level or "UNKNOWN") for risk in latest_by_student.values() if risk)
    completed_reviews = sum(1 for review in reviews if review.status == "CLOSED")
    follow_up_reviews = sum(1 for review in reviews if review.status == "FOLLOW_UP_REQUIRED")
    return {
        "counselor_id": current_user.id,
        "scope": "all_students" if is_admin(current_user) else "assigned_students",
        "total_students": len(students),
        "assigned_students": len(students),
        "average_dass21": round(_avg(item.total_dass21_score for item in dass_items), 2),
        "average_mood": round(_avg(item.mood for item in checkins), 2),
        "average_stress": round(_avg(item.stress_level for item in checkins), 2),
        "fusion_distribution": dict(distribution),
        "risk_distribution": [{"name": key, "value": value} for key, value in distribution.items()],
        "review_completion_rate": round(completed_reviews / len(reviews), 3) if reviews else 0,
        "follow_up_rate": round(follow_up_reviews / len(reviews), 3) if reviews else 0,
        "assessment_completion": {
            "dass21": len(dass_items),
            "mood_checkins": len(checkins),
            "fusion": len(risks),
        },
        "students": [serialize_student_row(db, student, current_user) for student in students],
        "charts": {
            "trend": _trend_chart(checkins, risks),
            "assessment_completion": [
                {"name": "DASS-21", "value": len(dass_items)},
                {"name": "Mood", "value": len(checkins)},
                {"name": "Fusion", "value": len(risks)},
                {"name": "Reviews", "value": len(reviews)},
            ],
            "risk_distribution": [{"name": key, "value": value} for key, value in distribution.items()],
        },
        "date_filter": {"start_date": iso(start_date), "end_date": iso(end_date)},
    }


def empty_dashboard(current_user: User) -> dict:
    return {
        "counselor_id": current_user.id,
        "scope": "all_students" if is_admin(current_user) else "assigned_students",
        "total_students": 0,
        "assigned_students": 0,
        "average_dass21": 0,
        "average_mood": 0,
        "average_stress": 0,
        "fusion_distribution": {},
        "risk_distribution": [],
        "review_completion_rate": 0,
        "follow_up_rate": 0,
        "assessment_completion": {"dass21": 0, "mood_checkins": 0, "fusion": 0},
        "students": [],
        "charts": {"trend": [], "assessment_completion": [], "risk_distribution": []},
        "date_filter": {"start_date": None, "end_date": None},
    }


def _avg(values: Iterable[float | int | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0


def _trend_chart(checkins: list[DailyCheckIn], risks: list[RiskAssessment]) -> list[dict]:
    by_date: dict[str, dict] = {}
    for checkin in checkins:
        key = checkin.created_at.date().isoformat()
        by_date.setdefault(key, {"date": key})
        by_date[key]["mood"] = checkin.mood
        by_date[key]["stress"] = checkin.stress_level
    for risk in risks:
        key = risk.created_at.date().isoformat()
        by_date.setdefault(key, {"date": key})
        by_date[key]["fusion"] = risk.final_score or risk.model_score
    return [by_date[key] for key in sorted(by_date)]


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dates must be ISO format")


def report_payload(db: Session, student_id: int, current_user: User) -> dict:
    detail = student_detail(db, student_id, current_user)
    detail["timeline"] = timeline(db, student_id, current_user)["events"]
    return detail


def report_csv(payload: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["section", "timestamp", "label", "value"])
    student = payload["student"]
    writer.writerow(["student", "", "name", student["full_name"]])
    writer.writerow(["student", "", "email", student["email"]])
    writer.writerow(["disclaimer", "", "model_disclaimer", payload["model_disclaimer"]])
    latest = payload.get("latest_assessment") or {}
    writer.writerow(["summary", latest.get("created_at"), "risk_status", latest.get("risk_level") or latest.get("model_risk_level") or "N/A"])
    writer.writerow(["summary", latest.get("created_at"), "final_risk_percentage", latest.get("risk_percentage")])
    for component in payload.get("model_component_summary", []):
        writer.writerow(
            [
                "model_component",
                component.get("source_timestamp") or component.get("generated_at"),
                component.get("label") or component.get("modality"),
                {
                    "risk_percentage": component.get("risk_percentage"),
                    "contribution_percentage": component.get("contribution_percentage"),
                    "effective_weight_percentage": component.get("effective_weight_percentage"),
                    "status": component.get("status"),
                    "included": component.get("included"),
                },
            ]
        )
    for assessment in payload.get("dass21_assessments", []):
        writer.writerow(
            [
                "dass21",
                assessment.get("created_at"),
                "overall_risk_percentage",
                {
                    "risk_percentage": assessment.get("risk_percentage"),
                    "total_dass21_score": assessment.get("total_dass21_score"),
                    "component_percentages": assessment.get("component_percentages"),
                },
            ]
        )
    for item in payload["timeline"]:
        writer.writerow([item["type"], item["timestamp"], item["label"], item["data"]])
    return output.getvalue()


def report_pdf_bytes(payload: dict) -> bytes:
    student = payload["student"]
    lines = [
        "Individual Student Report",
        f"Student: {student['full_name']} ({student['email']})",
        f"Department: {student.get('department') or 'N/A'}",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        MODEL_DISCLAIMER,
        "",
        "Latest Model Summary",
    ]
    latest = payload.get("latest_assessment") or {}
    lines.extend(
        [
            f"Risk level: {latest.get('risk_level') or latest.get('model_risk_level') or 'N/A'}",
            f"Final risk percentage: {latest.get('risk_percentage') if latest.get('risk_percentage') is not None else 'N/A'}%",
            f"Fusion score: {latest.get('final_score') or latest.get('model_score') or 'N/A'}",
            f"Evidence coverage: {latest.get('evidence_coverage') or 'N/A'}",
            "",
            "Seven-Model Component Summary",
        ]
    )
    for component in payload.get("model_component_summary", [])[:7]:
        risk_percentage = component.get("risk_percentage")
        contribution = component.get("contribution_percentage")
        lines.append(
            f"{component.get('label') or component.get('modality')}: "
            f"{risk_percentage if risk_percentage is not None else 'N/A'}% "
            f"({component.get('status')}, contribution {contribution if contribution is not None else 'N/A'}%)"
        )
    lines.extend(
        [
            "",
            "DASS-21 Risk History",
        ]
    )
    for assessment in payload.get("dass21_assessments", [])[:10]:
        lines.append(
            f"{assessment.get('created_at')} - risk {assessment.get('risk_percentage') if assessment.get('risk_percentage') is not None else 'N/A'}% "
            f"- total {assessment.get('total_dass21_score')}"
        )
    lines.extend(
        [
            "",
            "Review History",
        ]
    )
    for review in payload.get("reviews", [])[:10]:
        lines.append(f"{review['updated_at']} - {review['status']} - {review.get('decision') or ''}")
    lines.extend(["", "Counselor Notes"])
    for note in payload.get("notes", [])[:10]:
        lines.append(f"{note['created_at']} - {note['note_type']} - {note['note_text'][:120]}")
    text = "\n".join(lines)
    return _minimal_pdf(text)


def _minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_lines = []
    y = 760
    for line in escaped.splitlines()[:45]:
        content_lines.append(f"BT /F1 10 Tf 50 {y} Td ({line}) Tj ET")
        y -= 16
    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        b"5 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj + b"\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(pdf)
