from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List
from app.database import get_db
from app.models.database_models import User, DailyCheckIn, Assessment
from app.routes.auth import get_current_user
from app.utils.assessment_calculator import DailyCheckInCalculator, AssessmentAggregator
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/checkin", tags=["Daily Check-In"])

# Test endpoint to verify API is working
@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify checkin API is working"""
    return {"status": "ok", "message": "Checkin API is working"}

class DailyCheckInCreate(BaseModel):
    mood: int  # 1-5 scale
    mood_description: Optional[str] = None
    sleep_hours: float
    exercise_minutes: int = 0
    social_interaction: str  # None, Limited, Moderate, Good
    stress_level: int  # 1-10
    anxiety_level: int  # 1-10
    negative_thoughts: bool = False
    substance_use_today: bool = False
    self_harm_thoughts: bool = False
    notes: Optional[str] = None

class DailyCheckInResponse(BaseModel):
    id: int
    checkin_risk_score: float
    mood: int
    created_at: datetime

# Handle CORS preflight requests
@router.options("/today")
async def options_today():
    """Handle CORS preflight for POST /today"""
    return {}

@router.post("/today", response_model=dict)
def create_daily_checkin(
    checkin_data: DailyCheckInCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create today's daily check-in"""
    
    try:
        print(f"📋 Creating daily checkin for user {current_user.id}")
        print(f"📊 Mood: {checkin_data.mood}, Stress: {checkin_data.stress_level}")
        
        # Validate inputs
        if not (1 <= checkin_data.mood <= 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mood must be between 1 and 5"
            )
        
        # Create check-in record - simplified
        db_checkin = DailyCheckIn(
            user_id=current_user.id,
            mood=checkin_data.mood,
            mood_description=checkin_data.mood_description or "",
            sleep_hours=checkin_data.sleep_hours,
            exercise_minutes=checkin_data.exercise_minutes,
            social_interaction=checkin_data.social_interaction,
            stress_level=checkin_data.stress_level,
            anxiety_level=checkin_data.anxiety_level,
            negative_thoughts=checkin_data.negative_thoughts,
            substance_use_today=checkin_data.substance_use_today,
            self_harm_thoughts=checkin_data.self_harm_thoughts,
            notes=checkin_data.notes or ""
        )
        
        db.add(db_checkin)
        db.commit()
        db.refresh(db_checkin)
        
        print(f"✅ Daily checkin created: {db_checkin.id}")
        
        return {
            "id": db_checkin.id,
            "user_id": current_user.id,
            "mood": db_checkin.mood,
            "created_at": db_checkin.created_at.isoformat(),
            "message": "Daily check-in recorded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Checkin error: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save check-in: {str(e)}"
        )
    
    db.commit()
    db.refresh(db_checkin)
    
    return {
        "id": db_checkin.id,
        "user_id": current_user.id,
        "checkin_risk_score": daily_risk_score,
        "mood": db_checkin.mood,
        "sleep_hours": db_checkin.sleep_hours,
        "stress_level": db_checkin.stress_level,
        "self_harm_thoughts": db_checkin.self_harm_thoughts,
        "created_at": db_checkin.created_at,
        "message": "Daily check-in recorded successfully"
    }

@router.put("/today")
def update_today_checkin(
    checkin_data: DailyCheckInCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update today's check-in (only if it's the same day)"""
    
    try:
        today = datetime.utcnow().date()
        
        # Find today's checkin
        checkin = db.query(DailyCheckIn).filter(
            (DailyCheckIn.user_id == current_user.id) &
            (db.func.date(DailyCheckIn.created_at) == today)
        ).first()
        
        if not checkin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No check-in found for today"
            )
        
        # Update the checkin
        checkin.mood = checkin_data.mood
        checkin.mood_description = checkin_data.mood_description or ""
        checkin.sleep_hours = checkin_data.sleep_hours
        checkin.exercise_minutes = checkin_data.exercise_minutes
        checkin.social_interaction = checkin_data.social_interaction
        checkin.stress_level = checkin_data.stress_level
        checkin.anxiety_level = checkin_data.anxiety_level
        checkin.negative_thoughts = checkin_data.negative_thoughts
        checkin.substance_use_today = checkin_data.substance_use_today
        checkin.self_harm_thoughts = checkin_data.self_harm_thoughts
        checkin.notes = checkin_data.notes or ""
        
        db.commit()
        db.refresh(checkin)
        
        print(f"✅ Daily checkin updated: {checkin.id}")
        
        return {
            "id": checkin.id,
            "user_id": current_user.id,
            "mood": checkin.mood,
            "mood_description": checkin.mood_description,
            "sleep_hours": checkin.sleep_hours,
            "exercise_minutes": checkin.exercise_minutes,
            "social_interaction": checkin.social_interaction,
            "stress_level": checkin.stress_level,
            "anxiety_level": checkin.anxiety_level,
            "negative_thoughts": checkin.negative_thoughts,
            "substance_use_today": checkin.substance_use_today,
            "self_harm_thoughts": checkin.self_harm_thoughts,
            "notes": checkin.notes,
            "created_at": checkin.created_at.isoformat(),
            "message": "Check-in updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Update error: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update check-in: {str(e)}"
        )

@router.get("/today")
def get_today_checkin(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get today's check-in status"""
    
    today = datetime.utcnow().date()
    checkin = db.query(DailyCheckIn).filter(
        (DailyCheckIn.user_id == current_user.id) &
        (db.func.date(DailyCheckIn.created_at) == today)
    ).first()
    
    if not checkin:
        return {
            "checked_in": False,
            "message": "You haven't checked in today yet"
        }
    
    # Return checkin data
    return {
        "checked_in": True,
        "checkin": {
            "id": checkin.id,
            "mood": checkin.mood,
            "mood_description": checkin.mood_description,
            "sleep_hours": checkin.sleep_hours,
            "exercise_minutes": checkin.exercise_minutes,
            "social_interaction": checkin.social_interaction,
            "stress_level": checkin.stress_level,
            "anxiety_level": checkin.anxiety_level,
            "negative_thoughts": checkin.negative_thoughts,
            "substance_use_today": checkin.substance_use_today,
            "self_harm_thoughts": checkin.self_harm_thoughts,
            "notes": checkin.notes,
            "created_at": checkin.created_at.isoformat()
        }
    }

@router.get("/history")
def get_checkin_history(
    current_user: User = Depends(get_current_user),
    limit: int = 30,
    days: int = 90,
    db: Session = Depends(get_db)
):
    """Get check-in history"""
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    checkins = db.query(DailyCheckIn).filter(
        (DailyCheckIn.user_id == current_user.id) &
        (DailyCheckIn.created_at >= since_date)
    ).order_by(DailyCheckIn.created_at.desc()).limit(limit).all()
    
    history = []
    for checkin in checkins:
        checkin_dict = {
            "mood": checkin.mood,
            "sleep_hours": checkin.sleep_hours,
            "exercise_minutes": checkin.exercise_minutes,
            "social_interaction": checkin.social_interaction,
            "stress_level": checkin.stress_level,
            "anxiety_level": checkin.anxiety_level,
            "negative_thoughts": checkin.negative_thoughts,
            "substance_use_today": checkin.substance_use_today,
            "self_harm_thoughts": checkin.self_harm_thoughts
        }
        daily_risk_score = DailyCheckInCalculator.calculate(checkin_dict)
        
        history.append({
            "id": checkin.id,
            "mood": checkin.mood,
            "stress_level": checkin.stress_level,
            "anxiety_level": checkin.anxiety_level,
            "sleep_hours": checkin.sleep_hours,
            "exercise_minutes": checkin.exercise_minutes,
            "created_at": checkin.created_at.isoformat()
        })
    
    return {
        "user_id": current_user.id,
        "total_checkins": len(checkins),
        "records": history
    }

@router.get("/stats")
def get_checkin_stats(
    current_user: User = Depends(get_current_user),
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get check-in statistics"""
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    checkins = db.query(DailyCheckIn).filter(
        (DailyCheckIn.user_id == current_user.id) &
        (DailyCheckIn.created_at >= since_date)
    ).all()
    
    if not checkins:
        return {
            "user_id": current_user.id,
            "total_checkins": 0,
            "period_days": days,
            "message": "No check-ins found in this period"
        }
    
    # Calculate statistics
    avg_mood = sum(c.mood for c in checkins) / len(checkins)
    avg_stress = sum(c.stress_level for c in checkins) / len(checkins)
    avg_anxiety = sum(c.anxiety_level for c in checkins) / len(checkins)
    avg_sleep = sum(c.sleep_hours for c in checkins) / len(checkins)
    
    self_harm_count = sum(1 for c in checkins if c.self_harm_thoughts)
    negative_thoughts_count = sum(1 for c in checkins if c.negative_thoughts)
    substance_use_count = sum(1 for c in checkins if c.substance_use_today)
    
    return {
        "user_id": current_user.id,
        "period_days": days,
        "total_checkins": len(checkins),
        "average_mood": round(avg_mood, 2),
        "average_stress": round(avg_stress, 2),
        "average_anxiety": round(avg_anxiety, 2),
        "average_sleep_hours": round(avg_sleep, 2),
        "self_harm_thoughts_count": self_harm_count,
        "negative_thoughts_count": negative_thoughts_count,
        "substance_use_count": substance_use_count,
        "checkins_with_concerns": self_harm_count + negative_thoughts_count + substance_use_count
    }
