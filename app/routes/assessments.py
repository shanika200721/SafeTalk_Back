from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from app.database import get_db
from app.models.database_models import (
    User, ProfileAssessment, DASS21Assessment, Assessment, 
    AssessmentHistory, Alert
)
from app.routes.auth import get_current_user
from app.schemas import (
    ProfileAssessmentCreate, ProfileAssessment as ProfileAssessmentSchema,
    AssessmentResponse, ModalityScores, RiskAssessmentRequest, RiskAssessmentResponse,
    DASS21Request, DASS21Response
)
from app.models.risk_assessment import risk_assessor
from app.utils.assessment_calculator import (
    ProfileRiskCalculator, 
    AssessmentAggregator,
    DailyCheckInCalculator
)
from app.utils.dass21_calculator import DASS21Calculator

router = APIRouter(prefix="/api/assessments", tags=["Assessments"])

# ==================== Profile Assessment ====================

@router.post("/profile", response_model=ProfileAssessmentSchema)
def create_profile_assessment(
    assessment_data: ProfileAssessmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update profile assessment for user"""
    
    # Check if user owns this assessment
    if assessment_data.user_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create assessment for another user"
        )
    
    # Check if profile assessment already exists
    existing = db.query(ProfileAssessment).filter(
        ProfileAssessment.user_id == assessment_data.user_id
    ).first()
    
    # Calculate profile risk score
    profile_dict = assessment_data.dict()
    profile_score = ProfileRiskCalculator.calculate(profile_dict)
    profile_dict['profile_score'] = profile_score
    
    if existing:
        # Update existing
        for key, value in profile_dict.items():
            if key != 'user_id':
                setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
    else:
        # Create new
        db_assessment = ProfileAssessment(**profile_dict)
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)
        return db_assessment

@router.get("/profile/{user_id}", response_model=ProfileAssessmentSchema)
def get_profile_assessment(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get profile assessment for a user"""
    
    if user_id != current_user.id and current_user.role.value not in ["admin", "counselor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view profile assessment of another user"
        )
    
    assessment = db.query(ProfileAssessment).filter(
        ProfileAssessment.user_id == user_id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile assessment not found"
        )
    
    return assessment

# ==================== DASS21 Assessment ====================

@router.post("/dass21", response_model=DASS21Response)
def create_dass21_assessment(
    request: DASS21Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create DASS21 assessment from 21 responses
    
    Expected format:
    {
        "responses": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]
    }
    """
    
    try:
        resp_list = request.responses
        
        # Validate response values
        if not all(0 <= r <= 3 for r in resp_list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each response must be between 0 and 3"
            )
        
        # Calculate scores
        dass21_result = DASS21Calculator.calculate(resp_list)
        dass21_risk_score = DASS21Calculator.calculate_dass21_risk_score(dass21_result)
        
        # Create assessment record
        db_assessment = DASS21Assessment(
            user_id=current_user.id,
            responses=resp_list,
            depression_score=dass21_result["depression_score"],
            anxiety_score=dass21_result["anxiety_score"],
            stress_score=dass21_result["stress_score"],
            total_dass21_score=dass21_result["total_dass21_score"],
            depression_severity=dass21_result["depression_severity"],
            anxiety_severity=dass21_result["anxiety_severity"],
            stress_severity=dass21_result["stress_severity"]
        )
        
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)
        
        return db_assessment
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process DASS21 assessment: {str(e)}"
        )

@router.get("/dass21/latest")
def get_latest_dass21(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get latest DASS21 assessment"""
    
    assessment = db.query(DASS21Assessment).filter(
        DASS21Assessment.user_id == current_user.id
    ).order_by(DASS21Assessment.created_at.desc()).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No DASS21 assessment found"
        )
    
    return {
        "id": assessment.id,
        "depression_score": assessment.depression_score,
        "anxiety_score": assessment.anxiety_score,
        "stress_score": assessment.stress_score,
        "total_dass21_score": assessment.total_dass21_score,
        "depression_severity": assessment.depression_severity,
        "anxiety_severity": assessment.anxiety_severity,
        "stress_severity": assessment.stress_severity,
        "created_at": assessment.created_at
    }

@router.get("/dass21/history")
def get_dass21_history(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get DASS21 assessment history"""
    
    assessments = db.query(DASS21Assessment).filter(
        DASS21Assessment.user_id == current_user.id
    ).order_by(DASS21Assessment.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": current_user.id,
        "assessments": [
            {
                "id": a.id,
                "total_dass21_score": a.total_dass21_score,
                "depression_score": a.depression_score,
                "anxiety_score": a.anxiety_score,
                "stress_score": a.stress_score,
                "created_at": a.created_at
            }
            for a in assessments
        ]
    }

@router.get("/dass21/today")
def get_today_dass21(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get today's DASS21 assessment if it exists"""
    from datetime import date, datetime as dt
    
    today = date.today()
    
    assessment = db.query(DASS21Assessment).filter(
        DASS21Assessment.user_id == current_user.id,
        db.func.date(DASS21Assessment.created_at) == today
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No DASS21 assessment for today"
        )
    
    return {
        "id": assessment.id,
        "responses": assessment.responses,
        "depression_score": assessment.depression_score,
        "anxiety_score": assessment.anxiety_score,
        "stress_score": assessment.stress_score,
        "total_dass21_score": assessment.total_dass21_score,
        "depression_severity": assessment.depression_severity,
        "anxiety_severity": assessment.anxiety_severity,
        "stress_severity": assessment.stress_severity,
        "created_at": assessment.created_at
    }

@router.put("/dass21/{assessment_id}", response_model=DASS21Response)
def update_dass21_assessment(
    assessment_id: int,
    request: DASS21Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing DASS21 assessment (only own assessments)"""
    
    try:
        # Get the assessment
        assessment = db.query(DASS21Assessment).filter(
            DASS21Assessment.id == assessment_id,
            DASS21Assessment.user_id == current_user.id
        ).first()
        
        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        resp_list = request.responses
        
        # Validate response values
        if not all(0 <= r <= 3 for r in resp_list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each response must be between 0 and 3"
            )
        
        # Calculate scores
        dass21_result = DASS21Calculator.calculate(resp_list)
        
        # Update assessment
        assessment.responses = resp_list
        assessment.depression_score = dass21_result["depression_score"]
        assessment.anxiety_score = dass21_result["anxiety_score"]
        assessment.stress_score = dass21_result["stress_score"]
        assessment.total_dass21_score = dass21_result["total_dass21_score"]
        assessment.depression_severity = dass21_result["depression_severity"]
        assessment.anxiety_severity = dass21_result["anxiety_severity"]
        assessment.stress_severity = dass21_result["stress_severity"]
        
        db.commit()
        db.refresh(assessment)
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update DASS21 assessment: {str(e)}"
        )

# ==================== Multimodal Risk Assessment ====================

@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
async def perform_risk_assessment(
    request: RiskAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform comprehensive multimodal risk assessment"""
    
    try:
        # Verify user ID matches
        if request.user_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot assess risk for another user"
            )
        
        # Convert scores to dict
        scores_dict = request.scores.dict()
        
        # Use existing risk assessor
        assessment = risk_assessor.assess(
            modality_scores=scores_dict,
            profile_data=request.profile_data
        )
        
        # Add user_id
        assessment['user_id'] = request.user_id
        
        # Save to database
        db_assessment = Assessment(
            user_id=request.user_id,
            assessment_type="multimodal",
            profile_score=scores_dict.get("profile_score", 0),
            mood_score=scores_dict.get("mood_score", 0),
            dass21_score=scores_dict.get("dass21_score", 0),
            text_score=scores_dict.get("text_score", 0),
            voice_score=scores_dict.get("voice_score", 0),
            face_score=scores_dict.get("face_score", 0),
            behavioral_score=scores_dict.get("behavioral_score", 0),
            composite_score=assessment['composite_score'],
            risk_level=assessment['risk_level'],
            needs_escalation=assessment['needs_escalation'],
            recommendations=assessment['recommendations']
        )
        
        db.add(db_assessment)
        
        # Create alert if high risk
        if assessment['needs_escalation']:
            alert = Alert(
                user_id=request.user_id,
                alert_type="escalation",
                risk_level=assessment['risk_level'],
                message=f"User has {assessment['risk_level']} risk level and needs immediate attention"
            )
            db.add(alert)
        
        db.commit()
        db.refresh(db_assessment)
        
        return {
            "user_id": request.user_id,
            "composite_score": assessment['composite_score'],
            "risk_level": assessment['risk_level'],
            "needs_escalation": assessment['needs_escalation'],
            "recommendations": assessment['recommendations'],
            "timestamp": assessment['timestamp'],
            "modality_breakdown": assessment['modality_breakdown']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{user_id}")
def get_assessment_history(
    user_id: int,
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get assessment history for a user"""
    
    if user_id != current_user.id and current_user.role.value not in ["admin", "counselor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view assessment history of another user"
        )
    
    # Get assessments from last N days
    since_date = datetime.utcnow() - timedelta(days=days)
    
    assessments = db.query(Assessment).filter(
        (Assessment.user_id == user_id) &
        (Assessment.created_at >= since_date)
    ).order_by(Assessment.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": user_id,
        "date_range": {
            "start": since_date,
            "end": datetime.utcnow()
        },
        "total_assessments": len(assessments),
        "assessments": [
            {
                "id": a.id,
                "assessment_type": a.assessment_type,
                "composite_score": a.composite_score,
                "risk_level": a.risk_level,
                "created_at": a.created_at
            }
            for a in assessments
        ]
    }

@router.get("/latest")
def get_latest_assessment(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get latest assessment for current user"""
    
    assessment = db.query(Assessment).filter(
        Assessment.user_id == current_user.id
    ).order_by(Assessment.created_at.desc()).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No assessment found"
        )
    
    return {
        "id": assessment.id,
        "composite_score": assessment.composite_score,
        "risk_level": assessment.risk_level,
        "needs_escalation": assessment.needs_escalation,
        "recommendations": assessment.recommendations,
        "created_at": assessment.created_at
    }
