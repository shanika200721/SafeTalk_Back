from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import User, UserRole
from app.routes.auth import get_current_user
from app.services.consent import has_active_consent, require_active_consent
from app.services.modalities import availability_contract
from app.services.profile_assessment import (
    QUESTIONNAIRE_VERSION,
    admin_statistics,
    assessment_summary,
    counselor_can_access_student,
    create_or_update_draft,
    latest_phase4l_assessment,
    preprocess_profile_responses,
    profile_status_payload,
    questionnaire_contract,
    submit_profile_assessment,
    validate_responses,
)


router = APIRouter(tags=["Profile Assessment"])


class ProfileAssessmentPayload(BaseModel):
    questionnaire_version: str = Field(default=QUESTIONNAIRE_VERSION)
    responses: dict[str, Any] = Field(default_factory=dict)


class ProfilePatchPayload(BaseModel):
    status: str | None = Field(default=None, pattern="^(draft|submitted|needs_update)$")
    responses: dict[str, Any] | None = None


def _require_student(user: User) -> None:
    if user.role != UserRole.STUDENT:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only students can use this profile assessment route")


def _validate_version(version: str) -> None:
    if version != QUESTIONNAIRE_VERSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "UNSUPPORTED_QUESTIONNAIRE_VERSION", "questionnaire_version": version},
        )


def _require_profile_storage_consent(db: Session, user: User, feature: str) -> None:
    if has_active_consent(db, user.id, "profile_data_storage") or has_active_consent(db, user.id, "profile_processing"):
        return
    require_active_consent(db, user, "profile_data_storage", feature)


@router.get("/api/student/profile-assessment/questions")
def get_questions(current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    return questionnaire_contract()


@router.get("/api/student/profile-assessment/status")
def get_profile_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    return profile_status_payload(db, current_user)


@router.get("/api/student/profile-assessment/current")
def get_current_profile_assessment(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    assessment = latest_phase4l_assessment(db, current_user.id)
    if not assessment:
        return {"status": "not_started", "questionnaire_version": QUESTIONNAIRE_VERSION, "responses": {}}
    return {
        "assessment_id": assessment.id,
        "profile_assessment_id": assessment.profile_assessment_id,
        "questionnaire_version": assessment.questionnaire_version,
        "status": assessment.status,
        "responses": assessment.responses_json or {},
        "submitted_at": assessment.submitted_at,
        "completed_at": assessment.completed_at,
        "stale_at": assessment.stale_at,
    }


@router.post("/api/student/profile-assessment/draft")
def save_profile_draft(
    payload: ProfileAssessmentPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    _validate_version(payload.questionnaire_version)
    _require_profile_storage_consent(db, current_user, "saving profile assessment drafts")
    assessment = create_or_update_draft(db, current_user, payload.responses)
    db.commit()
    db.refresh(assessment)
    return {
        "assessment_id": assessment.id,
        "profile_assessment_id": assessment.profile_assessment_id,
        "status": assessment.status,
        "questionnaire_version": assessment.questionnaire_version,
        "updated_at": assessment.updated_at,
    }


@router.post("/api/student/profile-assessment/submit")
def submit_profile(
    payload: ProfileAssessmentPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    _validate_version(payload.questionnaire_version)
    assessment = submit_profile_assessment(db, current_user, payload.responses)
    db.commit()
    db.refresh(assessment)
    return {
        "assessment_id": assessment.id,
        "profile_assessment_id": assessment.profile_assessment_id,
        "status": assessment.status,
        "questionnaire_version": assessment.questionnaire_version,
        "submitted_at": assessment.submitted_at,
        "completed_at": assessment.completed_at,
        "stale_at": assessment.stale_at,
        "prediction_id": assessment.prediction_id,
        "prediction_status": assessment.prediction.status if assessment.prediction else "unavailable",
        "message": "Your profile assessment has been saved and will be used as one source of background screening evidence. It is not a diagnosis.",
    }


@router.patch("/api/student/profile-assessment/{assessment_id}")
def patch_profile_assessment(
    assessment_id: int,
    payload: ProfilePatchPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    if payload.responses is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="responses are required for Phase 4L patch")
    _require_profile_storage_consent(db, current_user, "updating profile assessment drafts")
    assessment = create_or_update_draft(db, current_user, payload.responses, assessment_id=assessment_id)
    db.commit()
    db.refresh(assessment)
    return {"assessment_id": assessment.id, "status": assessment.status, "responses": assessment.responses_json or {}}


@router.get("/api/student/profile-assessment/{assessment_id}/summary")
def get_profile_summary(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    assessment = latest_phase4l_assessment(db, current_user.id)
    if not assessment or assessment.id != assessment_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile assessment not found")
    return assessment_summary(assessment, counselor_view=False)


@router.post("/api/student/profile-assessment/{assessment_id}/reprocess")
def reprocess_profile_assessment(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    require_active_consent(db, current_user, "profile_model_processing", "reprocessing profile assessment model evidence")
    assessment = latest_phase4l_assessment(db, current_user.id)
    if not assessment or assessment.id != assessment_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile assessment not found")
    normalized = preprocess_profile_responses(validate_responses(assessment.responses_json or {}, require_required=True))
    assessment.normalized_features_json = normalized
    assessment.preprocessing_version = normalized["preprocessing_version"]
    from app.services.profile_assessment import process_profile_prediction

    prediction = process_profile_prediction(db, current_user, assessment, normalized)
    assessment.prediction_id = prediction.id
    assessment.status = "completed" if prediction.status != "failed" else "failed"
    db.commit()
    db.refresh(assessment)
    return {"assessment_id": assessment.id, "prediction_id": prediction.id, "prediction_status": prediction.status}


@router.get("/api/counselor/student/{student_id}/profile-summary")
def counselor_profile_summary(
    student_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not counselor_can_access_student(db, current_user, student_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Counselor is not assigned to this student")
    assessment = latest_phase4l_assessment(db, student_id)
    return assessment_summary(assessment, counselor_view=True)


@router.get("/api/admin/profile-assessments/statistics")
def admin_profile_statistics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return admin_statistics(db)


@router.get("/api/admin/profile-questionnaires")
def admin_profile_questionnaires(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    contract = questionnaire_contract()
    return {
        "questionnaires": [
            {
                "questionnaire_version": contract["questionnaire_version"],
                "active": True,
                "question_count": len(contract["questions"]),
                "raw_responses_visible": False,
            }
        ]
    }


@router.post("/api/admin/profile-questionnaires")
def admin_create_profile_questionnaire(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Phase 4L questionnaire is version-controlled in backend configuration")


@router.patch("/api/admin/profile-questionnaires/{version}")
def admin_update_profile_questionnaire(version: str, current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    if version != QUESTIONNAIRE_VERSION:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Questionnaire version not found")
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Runtime edits are disabled for the versioned Phase 4L questionnaire")


@router.get("/api/student/facial-analysis/status")
def facial_analysis_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    face = next(item for item in availability_contract() if item["modality"] == "face")
    capture_consent = has_active_consent(db, current_user.id, "facial_capture") or has_active_consent(db, current_user.id, "face_processing")
    processing_consent = has_active_consent(db, current_user.id, "facial_model_processing") or has_active_consent(db, current_user.id, "face_processing")
    runtime_state = "verified_approved" if face["runtime_model_active"] else ("experimental" if face["implemented"] else "inactive")
    return {
        "modality": "face",
        "runtime_state": runtime_state,
        "runtime_model_active": face["runtime_model_active"],
        "implemented": face["implemented"],
        "consent": {
            "facial_capture": capture_consent,
            "facial_model_processing": processing_consent,
        },
        "last_capture_at": None,
        "message": (
            "Optional facial-emotion analysis is available."
            if runtime_state == "verified_approved"
            else "Facial analysis is currently unavailable. You may review the feature and privacy information, but no facial prediction will be generated."
        ),
        "limitations": face["limitations"],
    }
