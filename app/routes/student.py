"""
Student API Routes
Handles all student-specific endpoints including dashboard, check-ins, assessments
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
from app.database import get_db
from app.models.database_models import (
    User, ProfileAssessment, DASS21Assessment, DailyCheckIn,
    Assessment, Alert, CounselorSession
)
from app.routes.auth import get_current_user
from app.schemas import RiskAssessmentResponse

router = APIRouter(prefix="/api/student", tags=["Student"])

# ==================== DASHBOARD ====================

@router.get("/dashboard")
def get_student_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive student dashboard data
    
    Returns:
    {
        "user": {...},
        "current_risk_level": "HIGH",
        "risk_score": 75,
        "today_checkin": {...},
        "latest_dass21": {...},
        "profile_data": {...},
        "recent_checkins": [...],
        "recommendations": "...",
        "alerts": [...],
        "is_first_assessment": bool
    }
    """
    # Verify user is a student
    if current_user.role.value != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access student dashboard"
        )
    
    print(f"📊 Loading dashboard for user {current_user.id} ({current_user.email})")
    
    # Get user profile assessment
    profile_assessment = db.query(ProfileAssessment).filter(
        ProfileAssessment.user_id == current_user.id
    ).first()
    
    # Get latest DASS21 assessment
    latest_dass21 = db.query(DASS21Assessment).filter(
        DASS21Assessment.user_id == current_user.id
    ).order_by(DASS21Assessment.created_at.desc()).first()
    
    # Get today's check-in
    today = datetime.now().date()
    today_checkin = db.query(DailyCheckIn).filter(
        DailyCheckIn.user_id == current_user.id,
        func.date(DailyCheckIn.created_at) == today
    ).first()
    
    # Get last 7 days check-ins for trend
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_checkins = db.query(DailyCheckIn).filter(
        DailyCheckIn.user_id == current_user.id,
        DailyCheckIn.created_at >= seven_days_ago
    ).order_by(DailyCheckIn.created_at.desc()).all()
    
    # Get active alerts
    alerts = db.query(Alert).filter(
        Alert.user_id == current_user.id,
        Alert.is_read == False
    ).order_by(Alert.created_at.desc()).limit(5).all()
    
    # Get latest overall assessment
    latest_assessment = db.query(Assessment).filter(
        Assessment.user_id == current_user.id
    ).order_by(Assessment.created_at.desc()).first()
    
    # Calculate risk level
    risk_score = 0
    risk_level = "LOW"
    
    if latest_assessment:
        risk_score = int(latest_assessment.composite_score or 0)
        if risk_score >= 70:
            risk_level = "CRITICAL"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
    
    # Generate recommendations based on risk
    recommendations = generate_recommendations(
        risk_level, latest_dass21, profile_assessment, today_checkin
    )
    
    # Check if this is first assessment
    is_first_assessment = profile_assessment is None or latest_dass21 is None
    
    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": current_user.role.value
        },
        "current_risk_level": risk_level,
        "risk_score": risk_score,
        "today_checkin": {
            "id": today_checkin.id if today_checkin else None,
            "mood_score": today_checkin.mood if today_checkin else None,
            "stress_score": today_checkin.stress_level if today_checkin else None,
            "anxiety_score": today_checkin.anxiety_level if today_checkin else None,
            "sleep_quality": f"{today_checkin.sleep_hours} hrs" if today_checkin else None,
            "notes": today_checkin.notes if today_checkin else None,
            "check_in_date": today_checkin.created_at.date().isoformat() if today_checkin else None
        },
        "latest_dass21": {
            "id": latest_dass21.id if latest_dass21 else None,
            "depression_score": latest_dass21.depression_score if latest_dass21 else None,
            "anxiety_score": latest_dass21.anxiety_score if latest_dass21 else None,
            "stress_score": latest_dass21.stress_score if latest_dass21 else None,
            "depression_severity": latest_dass21.depression_severity if latest_dass21 else None,
            "anxiety_severity": latest_dass21.anxiety_severity if latest_dass21 else None,
            "stress_severity": latest_dass21.stress_severity if latest_dass21 else None,
            "total_score": latest_dass21.total_dass21_score if latest_dass21 else None,
            "assessment_date": latest_dass21.created_at.date().isoformat() if latest_dass21 else None
        },
        "profile_data": {
            "gpa": profile_assessment.gpa if profile_assessment else None,
            "attendance": profile_assessment.attendance if profile_assessment else None,
            "family_relationship_score": profile_assessment.family_relationship_score if profile_assessment else None,
            "communication_skills": profile_assessment.communication_skills if profile_assessment else None,
        },
        "recent_checkins": [
            {
                "id": c.id,
                "date": c.created_at.date().isoformat(),
                "check_in_date": c.created_at.date().isoformat(),
                "mood_score": c.mood,
                "stress_score": c.stress_level,
                "anxiety_score": c.anxiety_level,
                "sleep_quality": f"{c.sleep_hours} hrs",
                "notes": c.notes
            }
            for c in recent_checkins
        ],
        "recommendations": recommendations,
        "alerts": [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "alert_message": a.message,
                "risk_level": a.risk_level,
                "created_at": a.created_at.isoformat()
            }
            for a in alerts
        ],
        "is_first_assessment": is_first_assessment
    }


def generate_recommendations(risk_level: str, dass21, profile_assessment, today_checkin) -> str:
    """Generate personalized recommendations based on student data"""
    
    recommendations = []
    
    # Risk-based recommendations
    if risk_level == "CRITICAL":
        recommendations.append(
            "⚠️ Your mental health is concerning. Please contact the counselor immediately."
        )
    elif risk_level == "HIGH":
        recommendations.append(
            "Your stress levels are elevated. Consider scheduling a counselor session."
        )
    elif risk_level == "MEDIUM":
        recommendations.append(
            "Try stress management techniques like meditation or exercise."
        )
    
    # DASS21-based recommendations
    if dass21:
        if dass21.depression_score > 20:
            recommendations.append(
                "💭 Depression levels are high. Explore the Resource section for support."
            )
        if dass21.anxiety_score > 20:
            recommendations.append(
                "😰 Anxiety is elevated. Try breathing exercises and relaxation techniques."
            )
        if dass21.stress_score > 20:
            recommendations.append(
                "🔥 Stress is high. Break tasks into smaller steps and take regular breaks."
            )
    
    # Check-in based recommendations
    if today_checkin:
        if today_checkin.mood <= 2:
            recommendations.append(
                "😔 Your mood is low today. Try activities that bring you joy."
            )
        if today_checkin.sleep_hours and today_checkin.sleep_hours < 5:
            recommendations.append(
                "😴 Sleep quality is poor. Maintain a consistent sleep schedule."
            )
    
    # Profile-based recommendations
    if profile_assessment and profile_assessment.gpa < 2.5:
        recommendations.append(
            "📚 Consider academic support services to improve your GPA."
        )
    
    return " | ".join(recommendations) if recommendations else "Keep maintaining your wellness routine! 💪"


@router.get("/stats")
def get_student_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get student statistics and trends"""
    
    if current_user.role.value != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access this endpoint"
        )
    
    # Check-in streak
    recent_checkins = db.query(DailyCheckIn).filter(
        DailyCheckIn.user_id == current_user.id
    ).order_by(DailyCheckIn.check_in_date.desc()).limit(30).all()
    
    streak = 0
    today = datetime.now().date()
    
    for i, checkin in enumerate(recent_checkins):
        expected_date = today - timedelta(days=i)
        if checkin.check_in_date == expected_date:
            streak += 1
        else:
            break
    
    # Get all check-ins for month
    current_month_start = datetime.now().replace(day=1).date()
    current_month_checkins = db.query(DailyCheckIn).filter(
        DailyCheckIn.user_id == current_user.id,
        DailyCheckIn.created_at >= current_month_start
    ).count()
    
    # Get average mood for month
    avg_mood = db.query(
        DailyCheckIn.mood
    ).filter(
        DailyCheckIn.user_id == current_user.id,
        DailyCheckIn.created_at >= current_month_start
    ).all()
    
    avg_mood_score = (
        sum([m[0] for m in avg_mood]) / len(avg_mood) 
        if avg_mood else 0
    )
    
    return {
        "check_in_streak": streak,
        "current_month_checkins": current_month_checkins,
        "average_mood_score": round(avg_mood_score, 2),
        "assessments_completed": db.query(ProfileAssessment).filter(
            ProfileAssessment.user_id == current_user.id
        ).count() > 0
    }


@router.get("/resources")
def get_student_resources(
    current_user: User = Depends(get_current_user)
):
    """Get resources based on student's risk profile"""
    
    resources = {
        "crisis_hotlines": [
            {
                "name": "National Suicide Prevention Lifeline",
                "number": "988",
                "description": "24/7 confidential support"
            },
            {
                "name": "Crisis Text Line",
                "number": "Text HOME to 741741",
                "description": "Text-based crisis support"
            }
        ],
        "counseling_services": [
            {
                "name": "Campus Mental Health Center",
                "phone": "555-0123",
                "hours": "Monday-Friday 9AM-5PM",
                "location": "Building A, Room 205"
            }
        ],
        "coping_strategies": [
            {
                "title": "Deep Breathing Exercise",
                "description": "4-7-8 breathing: Inhale for 4, hold for 7, exhale for 8"
            },
            {
                "title": "Progressive Muscle Relaxation",
                "description": "Systematically tense and relax muscle groups"
            },
            {
                "title": "Grounding Technique (5-4-3-2-1)",
                "description": "Notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste"
            }
        ],
        "support_groups": [
            {
                "name": "Peer Support Circle",
                "frequency": "Weekly on Tuesday 6PM",
                "location": "Virtual - Zoom Link Provided"
            }
        ]
    }
    
    return resources


@router.get("/counselors")
def get_available_counselors(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of available counselors"""
    
    if current_user.role.value != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access this endpoint"
        )
    
    counselors = db.query(User).filter(
        User.role.value == "counselor",
        User.is_active == True
    ).all()
    
    return [
        {
            "id": c.id,
            "name": c.full_name,
            "email": c.email,
            "specialization": "Mental Health"  # This could come from a counselor profile table
        }
        for c in counselors
    ]
