from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from app.database import get_db
from app.models.database_models import (
    User, UserRole, CounselorSession, Alert, Assessment,
    DailyCheckIn, ProfileAssessment, DASS21Assessment
)
from app.routes.auth import get_current_user
from pydantic import BaseModel

router = APIRouter(prefix="/api/counselor", tags=["Counselor"])

class CounselorSessionCreate(BaseModel):
    user_id: int
    session_type: str
    risk_level_at_escalation: str
    counselor_notes: Optional[str] = None

class CounselorSessionUpdate(BaseModel):
    status: Optional[str] = None
    counselor_notes: Optional[str] = None
    intervention_type: Optional[str] = None
    outcome: Optional[str] = None
    follow_up_needed: Optional[bool] = None
    follow_up_date: Optional[datetime] = None

def verify_counselor(current_user: User = Depends(get_current_user)):
    """Verify user is a counselor or admin"""
    if current_user.role not in [UserRole.COUNSELOR, UserRole.ADMIN, UserRole.PSYCHIATRIST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only counselors and admins can access this endpoint"
        )
    return current_user

# ==================== Alerts ====================

@router.get("/alerts")
def get_alerts(
    current_user: User = Depends(verify_counselor),
    unread_only: bool = False,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get high-risk alerts for counselor"""
    
    query = db.query(Alert).order_by(Alert.created_at.desc())
    
    if unread_only:
        query = query.filter(Alert.is_read == False)
    
    alerts = query.limit(limit).all()
    
    # Get student names for each alert
    alert_data = []
    for a in alerts:
        student = db.query(User).filter(User.id == a.user_id).first()
        alert_data.append({
            "id": a.id,
            "user_id": a.user_id,
            "student_name": student.full_name if student else "Unknown Student",
            "alert_type": a.alert_type,
            "risk_level": a.risk_level,
            "message": a.message,
            "is_read": a.is_read,
            "created_at": a.created_at
        })
    
    return {
        "counselor_id": current_user.id,
        "total_alerts": len(alert_data),
        "alerts": alert_data
    }

@router.put("/alerts/{alert_id}/read")
def mark_alert_read(
    alert_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db)
):
    """Mark alert as read"""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.is_read = True
    db.commit()
    
    return {"message": "Alert marked as read"}

# ==================== High-Risk Users ====================

@router.get("/high-risk-users")
def get_high_risk_users(
    current_user: User = Depends(verify_counselor),
    risk_level: str = "HIGH",
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get list of high-risk users"""
    
    # Get users with high-risk assessments in last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    
    high_risk_assessments = db.query(Assessment).filter(
        (Assessment.risk_level.in_([risk_level, "SEVERE"])) &
        (Assessment.created_at >= seven_days_ago)
    ).order_by(Assessment.created_at.desc()).limit(limit).all()
    
    # Get unique users - handle empty list
    user_ids = list(set(a.user_id for a in high_risk_assessments))
    
    user_data = []
    if user_ids:  # Only query if there are user IDs
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        
        for user in users:
            latest_assessment = next(
                (a for a in high_risk_assessments if a.user_id == user.id),
                None
            )
            if latest_assessment:
                user_data.append({
                    "user_id": user.id,
                    "name": user.full_name,
                    "email": user.email,
                    "risk_level": latest_assessment.risk_level,
                    "composite_score": latest_assessment.composite_score,
                    "last_assessment": latest_assessment.created_at
                })
    
    return {
        "counselor_id": current_user.id,
        "total_users": len(user_data),
        "high_risk_users": user_data
    }

# ==================== User Dashboard ====================

@router.get("/student/{user_id}/dashboard")
def get_student_dashboard(
    user_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db)
):
    """Get comprehensive dashboard for a student"""
    
    # Get user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get latest profile assessment
    profile = db.query(ProfileAssessment).filter(
        ProfileAssessment.user_id == user_id
    ).first()
    
    # Get latest DASS21
    dass21 = db.query(DASS21Assessment).filter(
        DASS21Assessment.user_id == user_id
    ).order_by(DASS21Assessment.created_at.desc()).first()
    
    # Get today's check-in
    today = datetime.utcnow().date()
    checkin = db.query(DailyCheckIn).filter(
        (DailyCheckIn.user_id == user_id) &
        (db.func.date(DailyCheckIn.created_at) == today)
    ).first()
    
    # Get latest risk assessment
    latest_assessment = db.query(Assessment).filter(
        Assessment.user_id == user_id
    ).order_by(Assessment.created_at.desc()).first()
    
    # Get check-in history (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_checkins = db.query(DailyCheckIn).filter(
        (DailyCheckIn.user_id == user_id) &
        (DailyCheckIn.created_at >= seven_days_ago)
    ).all()
    
    return {
        "user": {
            "id": user.id,
            "name": user.full_name,
            "email": user.email,
            "department": user.department,
            "year_of_study": user.year_of_study
        },
        "profile_assessment": {
            "profile_score": profile.profile_score if profile else None,
            "updated_at": profile.updated_at if profile else None
        } if profile else None,
        "dass21_assessment": {
            "depression_score": dass21.depression_score,
            "anxiety_score": dass21.anxiety_score,
            "stress_score": dass21.stress_score,
            "total_dass21_score": dass21.total_dass21_score,
            "depression_severity": dass21.depression_severity,
            "anxiety_severity": dass21.anxiety_severity,
            "stress_severity": dass21.stress_severity,
            "created_at": dass21.created_at
        } if dass21 else None,
        "today_checkin": {
            "mood": checkin.mood,
            "stress_level": checkin.stress_level,
            "anxiety_level": checkin.anxiety_level,
            "self_harm_thoughts": checkin.self_harm_thoughts,
            "created_at": checkin.created_at
        } if checkin else None,
        "latest_risk_assessment": {
            "composite_score": latest_assessment.composite_score,
            "risk_level": latest_assessment.risk_level,
            "needs_escalation": latest_assessment.needs_escalation,
            "recommendations": latest_assessment.recommendations,
            "created_at": latest_assessment.created_at
        } if latest_assessment else None,
        "recent_checkins_count": len(recent_checkins),
        "critical_alerts": {
            "self_harm_thoughts": sum(1 for c in recent_checkins if c.self_harm_thoughts),
            "negative_thoughts": sum(1 for c in recent_checkins if c.negative_thoughts),
            "substance_use": sum(1 for c in recent_checkins if c.substance_use_today)
        }
    }

# ==================== Counselor Sessions ====================

@router.post("/sessions")
def create_counselor_session(
    session_data: CounselorSessionCreate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db)
):
    """Create a new counselor session"""
    
    # Verify student exists
    user = db.query(User).filter(User.id == session_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Create session
    db_session = CounselorSession(
        user_id=session_data.user_id,
        counselor_id=current_user.id,
        session_type=session_data.session_type,
        status="in_progress",
        risk_level_at_escalation=session_data.risk_level_at_escalation,
        counselor_notes=session_data.counselor_notes
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return {
        "id": db_session.id,
        "user_id": session_data.user_id,
        "counselor_id": current_user.id,
        "status": db_session.status,
        "created_at": db_session.created_at
    }

@router.get("/sessions/{session_id}")
def get_counselor_session(
    session_id: int,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db)
):
    """Get counselor session details"""
    
    session = db.query(CounselorSession).filter(
        CounselorSession.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
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
        "updated_at": session.updated_at
    }

@router.put("/sessions/{session_id}")
def update_counselor_session(
    session_id: int,
    update_data: CounselorSessionUpdate,
    current_user: User = Depends(verify_counselor),
    db: Session = Depends(get_db)
):
    """Update counselor session"""
    
    session = db.query(CounselorSession).filter(
        CounselorSession.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Update fields
    if update_data.status:
        session.status = update_data.status
    if update_data.counselor_notes:
        session.counselor_notes = update_data.counselor_notes
    if update_data.intervention_type:
        session.intervention_type = update_data.intervention_type
    if update_data.outcome:
        session.outcome = update_data.outcome
    if update_data.follow_up_needed is not None:
        session.follow_up_needed = update_data.follow_up_needed
    if update_data.follow_up_date:
        session.follow_up_date = update_data.follow_up_date
    
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    
    return {"message": "Session updated successfully", "session_id": session.id}

@router.get("/sessions/user/{user_id}")
def get_user_sessions(
    user_id: int,
    current_user: User = Depends(verify_counselor),
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get all sessions for a user"""
    
    sessions = db.query(CounselorSession).filter(
        CounselorSession.user_id == user_id
    ).order_by(CounselorSession.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "counselor_id": s.counselor_id,
                "session_type": s.session_type,
                "status": s.status,
                "risk_level_at_escalation": s.risk_level_at_escalation,
                "created_at": s.created_at,
                "updated_at": s.updated_at
            }
            for s in sessions
        ]
    }

# ==================== Analytics ====================

@router.get("/analytics/summary")
def get_analytics_summary(
    current_user: User = Depends(verify_counselor),
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get analytics summary for all students"""
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all high-risk assessments in last N days
    high_risk = db.query(Assessment).filter(
        (Assessment.created_at >= since_date) &
        (Assessment.risk_level.in_(["HIGH", "SEVERE"]))
    ).all()
    
    # Get total students
    total_students = db.query(User).filter(
        User.role == UserRole.STUDENT
    ).count()
    
    # Get students with high risk in period
    at_risk_students = len(set(a.user_id for a in high_risk))
    
    # Get assessments by risk level
    risk_distribution = {}
    for assessment in high_risk:
        risk_level = assessment.risk_level
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
    
    # Get sessions
    sessions = db.query(CounselorSession).filter(
        CounselorSession.created_at >= since_date
    ).all()
    
    return {
        "period_days": days,
        "total_students": total_students,
        "at_risk_students": at_risk_students,
        "risk_percentage": round((at_risk_students / total_students * 100) if total_students > 0 else 0, 2),
        "high_risk_assessments": len(high_risk),
        "risk_distribution": risk_distribution,
        "counselor_sessions": len(sessions),
        "sessions_completed": len([s for s in sessions if s.status == "completed"])
    }


# ==================== Students ====================

@router.get("/students")
def get_students(
    current_user: User = Depends(verify_counselor),
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all students with their assessment data for counselor view"""
    
    # Get all students (role = student)
    query = db.query(User).filter(User.role == UserRole.STUDENT)
    
    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (User.full_name.ilike(search_term)) |
            (User.email.ilike(search_term)) |
            (User.username.ilike(search_term))
        )
    
    students = query.all()
    
    # Build response with assessment data
    student_data = []
    for student in students:
        # Get latest assessment
        latest_assessment = db.query(Assessment).filter(
            Assessment.user_id == student.id
        ).order_by(Assessment.created_at.desc()).first()
        
        # Get latest DASS21
        latest_dass21 = db.query(DASS21Assessment).filter(
            DASS21Assessment.user_id == student.id
        ).order_by(DASS21Assessment.created_at.desc()).first()
        
        # Get latest daily check-in
        latest_checkin = db.query(DailyCheckIn).filter(
            DailyCheckIn.user_id == student.id
        ).order_by(DailyCheckIn.created_at.desc()).first()
        
        risk_level = None
        risk_color = "green"
        if latest_assessment:
            risk_level = latest_assessment.risk_level
            if risk_level == "high":
                risk_color = "red"
            elif risk_level == "moderate":
                risk_color = "orange"
        
        # Apply risk level filter if specified
        if risk_level and risk_level != "none":
            if risk_level == "high" and risk_level != "high":
                continue
            elif risk_level == "moderate" and risk_level != "moderate":
                continue
        
        student_data.append({
            "id": student.id,
            "name": student.full_name,
            "email": student.email,
            "username": student.username,
            "risk_level": risk_level or "none",
            "risk_color": risk_color,
            "last_assessment": latest_assessment.created_at.isoformat() if latest_assessment else None,
            "last_assessment_details": {
                "depression_score": latest_dass21.depression_score if latest_dass21 else None,
                "anxiety_score": latest_dass21.anxiety_score if latest_dass21 else None,
                "stress_score": latest_dass21.stress_score if latest_dass21 else None,
            } if latest_dass21 else None,
            "last_checkin": latest_checkin.created_at.isoformat() if latest_checkin else None,
        })
    
    # Apply risk level filter if specified
    if risk_level and risk_level != "all":
        student_data = [s for s in student_data if s["risk_level"] == risk_level]
    
    return {
        "total": len(student_data),
        "students": student_data
    }
