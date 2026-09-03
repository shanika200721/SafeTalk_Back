from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import RiskAssessment, User, UserRole
from app.routes.auth import get_current_user
from app.schemas import (
    ControlledFusionAssessmentResponse,
    ControlledFusionAssessRequest,
    ControlledFusionConfigResponse,
)
from app.services.fusion import controlled_fusion_config, run_controlled_fusion, serialize_assessment


router = APIRouter(prefix="/api/fusion", tags=["Controlled Fusion"])


def _target_user_id(requested_user_id: int | None, current_user: User, *, admin_only: bool = False) -> int:
    if current_user.role == UserRole.STUDENT:
        if requested_user_id is not None and requested_user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Students may assess only their own evidence")
        if admin_only:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Preview is restricted to admins")
        return current_user.id
    if current_user.role == UserRole.ADMIN:
        return requested_user_id or current_user.id
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Generic counselor fusion access is not enabled in Phase 4E")


def _authorize_assessment_read(assessment: RiskAssessment, current_user: User) -> None:
    if current_user.role == UserRole.ADMIN:
        return
    if current_user.role == UserRole.STUDENT and assessment.student_id == current_user.id:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to access this fusion assessment")


@router.post("/assess", response_model=ControlledFusionAssessmentResponse)
def assess(
    request: ControlledFusionAssessRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = _target_user_id(request.user_id if request else None, current_user)
    return run_controlled_fusion(db, user_id=target_user_id, persist=True)


@router.post("/preview", response_model=ControlledFusionAssessmentResponse)
def preview(
    request: ControlledFusionAssessRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = _target_user_id(request.user_id if request else None, current_user, admin_only=True)
    return run_controlled_fusion(db, user_id=target_user_id, persist=False)


@router.get("/config", response_model=ControlledFusionConfigResponse)
def config(current_user: User = Depends(get_current_user)):
    return controlled_fusion_config()


@router.get("/assessments", response_model=list[ControlledFusionAssessmentResponse])
def list_assessments(
    user_id: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = _target_user_id(user_id, current_user)
    rows = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.student_id == target_user_id, RiskAssessment.assessment_type == "screening_support")
        .order_by(RiskAssessment.created_at.desc(), RiskAssessment.id.desc())
        .limit(limit)
        .all()
    )
    return [serialize_assessment(row) for row in rows]


@router.get("/assessments/latest", response_model=ControlledFusionAssessmentResponse)
def latest_assessment(
    user_id: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = _target_user_id(user_id, current_user)
    assessment = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.student_id == target_user_id, RiskAssessment.assessment_type == "screening_support")
        .order_by(RiskAssessment.created_at.desc(), RiskAssessment.id.desc())
        .first()
    )
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No controlled fusion assessment found")
    return serialize_assessment(assessment)


@router.get("/assessments/{assessment_id}", response_model=ControlledFusionAssessmentResponse)
def get_assessment(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assessment = db.query(RiskAssessment).filter(RiskAssessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Controlled fusion assessment not found")
    _authorize_assessment_read(assessment, current_user)
    return serialize_assessment(assessment)
