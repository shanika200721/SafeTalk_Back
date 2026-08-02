from __future__ import annotations

from datetime import date, datetime, timedelta
from itertools import count
from uuid import uuid4
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import (
    CounselorAssignment,
    DASS21Assessment,
    DailyCheckIn,
    JournalEntry,
    Resource,
    SupportContact,
    User,
)
from app.routes.auth import get_current_user
from app.services.profile_assessment import profile_status_payload
from app.ml.runtime.base import RuntimeInferenceError, RuntimeModelUnavailable
from app.ml.runtime.registry import predict_with_active_model
from app.services.consent import get_latest_consent, has_active_consent
from app.services.modalities import (
    MODEL_EVIDENCE,
    create_feature_snapshot,
    create_prediction,
    create_failed_prediction,
    create_unavailable_prediction,
)

router = APIRouter(prefix="/api/student", tags=["Student Wellness"])

_progress_id_counter = count(1)
_student_state: Dict[int, Dict[str, Any]] = {}


WELLNESS_CATEGORIES = [
    "Stress Relief",
    "Sleep",
    "Mindfulness",
    "Exam Preparation",
    "Motivation",
    "Relaxation",
    "Self Care",
    "Anxiety",
    "Daily Wellness",
]


VIDEOS = [
    {
        "id": "video-box-breathing",
        "title": "Box Breathing for Study Breaks",
        "thumbnail": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=900&q=80",
        "duration": "4:20",
        "category": "Stress Relief",
        "description": "A short approved guide for steady breathing between study sessions.",
        "provider": "Campus Wellness Team",
        "url": "https://www.youtube.com/embed/tEmt1Znux58",
        "approved": True,
    },
    {
        "id": "video-sleep-wind-down",
        "title": "Evening Wind-down Routine",
        "thumbnail": "https://images.unsplash.com/photo-1455642305367-68834a8eaae0?auto=format&fit=crop&w=900&q=80",
        "duration": "8:00",
        "category": "Sleep",
        "description": "A calm routine for putting study work down before sleep.",
        "provider": "Open Wellness Library",
        "url": "https://www.youtube.com/embed/ZToicYcHIOU",
        "approved": True,
    },
    {
        "id": "video-exam-reset",
        "title": "Exam Reset: One Manageable Step",
        "thumbnail": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=900&q=80",
        "duration": "5:35",
        "category": "Exam Preparation",
        "description": "A low-pressure reset before or after difficult coursework.",
        "provider": "Student Support Unit",
        "url": "https://www.youtube.com/embed/inpok4MKVLM",
        "approved": True,
    },
    {
        "id": "video-mindfulness-basics",
        "title": "Mindfulness Basics",
        "thumbnail": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "duration": "6:15",
        "category": "Mindfulness",
        "description": "A beginner-friendly explanation of present-moment attention.",
        "provider": "Approved Wellness Channel",
        "url": "https://www.youtube.com/embed/ssss7V1_eyA",
        "approved": True,
    },
]


BREATHING = [
    {"id": "box", "title": "Box Breathing", "pattern": [4, 4, 4, 4], "duration_minutes": 3, "category": "Stress Relief"},
    {"id": "478", "title": "4-7-8", "pattern": [4, 7, 8, 0], "duration_minutes": 4, "category": "Sleep"},
    {"id": "triangle", "title": "Triangle Breathing", "pattern": [4, 4, 4], "duration_minutes": 3, "category": "Relaxation"},
    {"id": "calm", "title": "Calm Breathing", "pattern": [5, 0, 5, 0], "duration_minutes": 5, "category": "Daily Wellness"},
    {"id": "exam-reset", "title": "Exam Reset", "pattern": [4, 2, 6, 0], "duration_minutes": 2, "category": "Exam Preparation"},
    {"id": "sleep-relaxation", "title": "Sleep Relaxation", "pattern": [4, 4, 8, 0], "duration_minutes": 10, "category": "Sleep"},
]


MEDITATIONS = [
    {"id": "med-2-ground", "title": "Two-Minute Grounding", "duration": "2 min", "category": "2 min", "description": "A brief arrival practice for busy moments."},
    {"id": "med-5-morning", "title": "Morning Check-in", "duration": "5 min", "category": "Morning", "description": "Start the day with a gentle body and mood scan."},
    {"id": "med-10-exam", "title": "Before an Exam", "duration": "10 min", "category": "Exam", "description": "Settle attention and choose one manageable next step."},
    {"id": "med-15-sleep", "title": "Sleep Wind-down", "duration": "15 min", "category": "Sleep", "description": "A longer quiet practice for the end of the day."},
    {"id": "med-5-relax", "title": "Relaxed Breathing", "duration": "5 min", "category": "Relax", "description": "Follow soft visual pacing and simple narration."},
]


AMBIENT_SOUNDS = [
    "Rain",
    "Ocean",
    "Forest",
    "Wind",
    "Fireplace",
    "White Noise",
    "Brown Noise",
    "Cafe",
]


ACTIVITIES = [
    {"id": "grounding-54321", "title": "5-4-3-2-1 grounding", "category": "Anxiety", "duration": "3 min"},
    {"id": "gratitude-journal", "title": "Gratitude journal", "category": "Daily Wellness", "duration": "5 min"},
    {"id": "affirmations", "title": "Positive affirmations", "category": "Motivation", "duration": "2 min"},
    {"id": "mood-reflection", "title": "Mood reflection", "category": "Self Care", "duration": "4 min"},
    {"id": "breathing-challenge", "title": "Breathing challenge", "category": "Stress Relief", "duration": "3 min"},
    {"id": "body-scan", "title": "Body scan", "category": "Relaxation", "duration": "6 min"},
    {"id": "kindness-exercise", "title": "Kindness exercise", "category": "Self Care", "duration": "3 min"},
]


FALLBACK_RESOURCES = [
    {
        "id": "article-exam-stress",
        "title": "Managing Exam Pressure",
        "type": "Articles",
        "category": "Exam Preparation",
        "description": "Practical planning and recovery ideas for exam weeks.",
        "url": None,
        "approved": True,
    },
    {
        "id": "exercise-grounding",
        "title": "5-4-3-2-1 Grounding Exercise",
        "type": "Exercises",
        "category": "Anxiety",
        "description": "A self-paced grounding activity using the senses.",
        "url": None,
        "approved": True,
    },
    {
        "id": "download-sleep-plan",
        "title": "Sleep Wind-down Checklist",
        "type": "Downloads",
        "category": "Sleep",
        "description": "A simple checklist for preparing for rest.",
        "url": None,
        "approved": True,
    },
    {
        "id": "campus-counseling",
        "title": "Campus Counseling Unit",
        "type": "Campus resources",
        "category": "Support",
        "description": "University support contact information from the campus directory.",
        "url": None,
        "approved": True,
    },
    {
        "id": "emergency-guidance",
        "title": "Emergency Guidance",
        "type": "Emergency guidance",
        "category": "Support",
        "description": "How to reach immediate support if you or someone nearby needs urgent help.",
        "url": None,
        "approved": True,
    },
]


class JournalEntryIn(BaseModel):
    entry_date: Optional[date] = None
    title: Optional[str] = Field(default=None, max_length=160)
    mood: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    content: str = Field(min_length=1, max_length=12000)
    share_with_counselor: bool = False
    ai_analysis_opt_in: bool = False


class FavoriteIn(BaseModel):
    favorite: bool = True


class CompletionIn(BaseModel):
    completed: bool = True


class ProgressIn(BaseModel):
    activity_type: str
    item_id: str
    minutes: int = Field(default=0, ge=0, le=240)
    completed: bool = True


class PreferencesIn(BaseModel):
    theme: Optional[str] = None
    accent_color: Optional[str] = None
    language: Optional[str] = None
    notifications: Optional[Dict[str, bool]] = None
    daily_reminder: Optional[Dict[str, Any]] = None
    large_text: Optional[bool] = None
    reduced_motion: Optional[bool] = None


def _role_value(user: User) -> str:
    return getattr(user.role, "value", user.role)


def _require_student(user: User) -> None:
    if _role_value(user) != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access wellness features",
        )


def _state(user_id: int) -> Dict[str, Any]:
    return _student_state.setdefault(
        user_id,
        {
            "favorites": {"resources": [], "videos": [], "meditation": [], "breathing": [], "sounds": []},
            "completed": {"resources": [], "videos": [], "meditation": [], "breathing": []},
            "recently_viewed": [],
            "progress_events": [],
            "preferences": {
                "theme": "system",
                "accent_color": "#0f9f9a",
                "language": "en",
                "notifications": {
                    "assessment_reminders": True,
                    "upcoming_follow_up": True,
                    "new_resources": True,
                    "breathing_reminder": False,
                    "meditation_reminder": False,
                },
                "daily_reminder": {"enabled": False, "time": "18:00"},
                "large_text": False,
                "reduced_motion": False,
            },
        },
    )


def _paginate(items: List[Dict[str, Any]], page: int, page_size: int) -> Dict[str, Any]:
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


def _matches(item: Dict[str, Any], search: Optional[str], category: Optional[str]) -> bool:
    if category and item.get("category", "").lower() != category.lower():
        return False
    if not search:
        return True
    needle = search.lower()
    return any(
        needle in str(item.get(field, "")).lower()
        for field in ("title", "description", "category", "provider", "type")
    )


def _set_membership(values: List[str], item_id: str, enabled: bool) -> List[str]:
    if enabled and item_id not in values:
        values.append(item_id)
    if not enabled and item_id in values:
        values.remove(item_id)
    return values


def _resource_items(db: Session, user: User) -> List[Dict[str, Any]]:
    approved_db_resources = (
        db.query(Resource)
        .filter(Resource.is_active == True, Resource.status == "approved")
        .order_by(Resource.created_at.desc())
        .limit(50)
        .all()
    )
    items = [
        {
            "id": f"db-resource-{resource.id}",
            "title": resource.title,
            "type": resource.resource_type.title(),
            "category": resource.category or "Daily Wellness",
            "description": resource.description or "",
            "url": resource.url,
            "phone": resource.phone,
            "approved": True,
        }
        for resource in approved_db_resources
    ]

    support_contacts = (
        db.query(SupportContact)
        .filter(SupportContact.student_visible == True, SupportContact.active == True)
        .order_by(SupportContact.priority.asc())
        .limit(20)
        .all()
    )
    items.extend(
        {
            "id": f"support-contact-{contact.id}",
            "title": contact.display_name,
            "type": "University contacts",
            "category": "Support",
            "description": contact.contact_type.replace("_", " ").title(),
            "phone": contact.telephone_number,
            "email": contact.email,
            "approved": bool(contact.verified or contact.student_visible),
        }
        for contact in support_contacts
    )

    return [*items, *FALLBACK_RESOURCES]


def _wellness_summary(db: Session, user: User) -> Dict[str, Any]:
    today = datetime.utcnow().date()
    latest_dass = (
        db.query(DASS21Assessment)
        .filter(DASS21Assessment.user_id == user.id)
        .order_by(DASS21Assessment.created_at.desc())
        .first()
    )
    latest_mood = (
        db.query(DailyCheckIn)
        .filter(DailyCheckIn.user_id == user.id)
        .order_by(DailyCheckIn.created_at.desc())
        .first()
    )
    profile_status = profile_status_payload(db, user)
    profile_done = profile_status["assessment_status"] in {"submitted", "completed"}
    dass_count = db.query(DASS21Assessment).filter(DASS21Assessment.user_id == user.id).count()
    mood_count = db.query(DailyCheckIn).filter(DailyCheckIn.user_id == user.id).count()
    assignment = (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.student_id == user.id, CounselorAssignment.active == True)
        .order_by(CounselorAssignment.assigned_date.desc())
        .first()
    )
    next_session = None
    state = _state(user.id)
    journal_count = db.query(JournalEntry).filter(JournalEntry.student_id == user.id, JournalEntry.deleted_at.is_(None)).count()
    progress_events = state["progress_events"]

    completed_assessments = int(profile_done) + dass_count + mood_count
    pending = []
    if not profile_done:
        pending.append("Profile assessment")
    if not latest_dass:
        pending.append("DASS-21 assessment")
    if not latest_mood or latest_mood.created_at.date() != today:
        pending.append("Mood check-in")

    return {
        "today_wellbeing": "Support Available" if latest_mood else "Mood Check-in Open",
        "mood_check_in": {
            "status": "Completed" if latest_mood and latest_mood.created_at.date() == today else "Available",
            "last_mood": latest_mood.mood if latest_mood else None,
            "last_date": latest_mood.created_at.date().isoformat() if latest_mood else None,
        },
        "last_activity": progress_events[-1] if progress_events else None,
        "assessment_progress": {
            "completed_assessments": completed_assessments,
            "pending_assessments": pending,
            "last_dass": latest_dass.created_at.date().isoformat() if latest_dass else None,
            "last_mood": latest_mood.mood if latest_mood else None,
        },
        "upcoming_follow_up": next_session,
        "counselor_assigned": bool(assignment),
        "support_available": True,
        "latest_wellness_activity": progress_events[-1] if progress_events else None,
        "activity_counts": {
            "breathing_sessions": sum(1 for event in progress_events if event["activity_type"] == "breathing"),
            "meditation": sum(1 for event in progress_events if event["activity_type"] == "meditation"),
            "resources_viewed": len(state["recently_viewed"]),
            "journal_entries": journal_count,
            "mood_checkins": mood_count,
        },
    }


@router.get("/wellness")
def get_wellness(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    summary = _wellness_summary(db, current_user)
    return {
        "summary": summary,
        "categories": WELLNESS_CATEGORIES,
        "shortcuts": {
            "breathing": "/breathing",
            "resources": "/resources",
            "safetalk": "/safetalk-bot",
        },
        "notifications": [
            {"id": "assessment-reminder", "type": "Assessment reminder", "message": "A check-in is available when you have a quiet moment."},
            {"id": "new-resources", "type": "New resources", "message": "New approved wellness resources are ready to browse."},
        ],
        "camera": {
            "available": False,
            "message": "Facial analysis is currently unavailable.",
            "consent_required": True,
        },
    }


@router.get("/resources")
def get_resources(
    search: Optional[str] = None,
    category: Optional[str] = None,
    resource_type: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=8, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    state = _state(current_user.id)
    items = [
        {**item, "favorite": item["id"] in state["favorites"]["resources"], "recently_viewed": item["id"] in state["recently_viewed"]}
        for item in _resource_items(db, current_user)
        if _matches(item, search, category)
        and (not resource_type or item.get("type", "").lower() == resource_type.lower())
        and item.get("approved")
    ]
    return {
        **_paginate(items, page, page_size),
        "categories": WELLNESS_CATEGORIES,
        "types": ["Articles", "Videos", "Exercises", "Downloads", "Campus resources", "Emergency guidance", "University contacts"],
        "favorites": state["favorites"]["resources"],
        "recently_viewed": state["recently_viewed"],
    }


@router.post("/resources/{resource_id}/favorite")
def favorite_resource(resource_id: str, payload: FavoriteIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    state = _state(current_user.id)
    _set_membership(state["favorites"]["resources"], resource_id, payload.favorite)
    return {"id": resource_id, "favorite": payload.favorite}


@router.post("/resources/{resource_id}/view")
def view_resource(resource_id: str, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    viewed = _state(current_user.id)["recently_viewed"]
    if resource_id in viewed:
        viewed.remove(resource_id)
    viewed.insert(0, resource_id)
    del viewed[10:]
    return {"id": resource_id, "recently_viewed": viewed}


@router.get("/videos")
def get_videos(
    search: Optional[str] = None,
    category: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=6, ge=1, le=30),
    current_user: User = Depends(get_current_user),
):
    _require_student(current_user)
    state = _state(current_user.id)
    items = [
        {
            **video,
            "favorite": video["id"] in state["favorites"]["videos"],
            "completed": video["id"] in state["completed"]["videos"],
            "autoplay": False,
        }
        for video in VIDEOS
        if video["approved"] and _matches(video, search, category)
    ]
    return {**_paginate(items, page, page_size), "categories": WELLNESS_CATEGORIES}


@router.post("/videos/{video_id}/favorite")
def favorite_video(video_id: str, payload: FavoriteIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    _set_membership(_state(current_user.id)["favorites"]["videos"], video_id, payload.favorite)
    return {"id": video_id, "favorite": payload.favorite}


@router.post("/videos/{video_id}/complete")
def complete_video(video_id: str, payload: CompletionIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    _set_membership(_state(current_user.id)["completed"]["videos"], video_id, payload.completed)
    return {"id": video_id, "completed": payload.completed}


@router.get("/breathing")
def get_breathing(current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    state = _state(current_user.id)
    return {
        "items": [
            {
                **item,
                "favorite": item["id"] in state["favorites"]["breathing"],
                "completed": item["id"] in state["completed"]["breathing"],
            }
            for item in BREATHING
        ]
    }


@router.post("/breathing/session")
def record_breathing_session(payload: ProgressIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    state = _state(current_user.id)
    _set_membership(state["completed"]["breathing"], payload.item_id, payload.completed)
    event = _progress_event(current_user.id, payload)
    return {"recorded": True, "event": event}


@router.get("/meditation")
def get_meditation(category: Optional[str] = None, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    state = _state(current_user.id)
    items = [
        {
            **item,
            "favorite": item["id"] in state["favorites"]["meditation"],
            "completed": item["id"] in state["completed"]["meditation"],
        }
        for item in MEDITATIONS
        if not category or item["category"].lower() == category.lower()
    ]
    return {"items": items, "categories": ["2 min", "5 min", "10 min", "15 min", "Morning", "Exam", "Sleep", "Relax"]}


@router.post("/meditation/{meditation_id}/favorite")
def favorite_meditation(meditation_id: str, payload: FavoriteIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    _set_membership(_state(current_user.id)["favorites"]["meditation"], meditation_id, payload.favorite)
    return {"id": meditation_id, "favorite": payload.favorite}


@router.post("/meditation/{meditation_id}/complete")
def complete_meditation(meditation_id: str, payload: CompletionIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    _set_membership(_state(current_user.id)["completed"]["meditation"], meditation_id, payload.completed)
    return {"id": meditation_id, "completed": payload.completed}


@router.get("/ambient-sounds")
def get_ambient_sounds(current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    state = _state(current_user.id)
    return {"items": [{"id": sound.lower().replace(" ", "-"), "title": sound, "favorite": sound in state["favorites"]["sounds"]} for sound in AMBIENT_SOUNDS]}


@router.get("/activities")
def get_activities(current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    return {"items": ACTIVITIES}


@router.get("/journal")
def get_journal(
    search: Optional[str] = None,
    tag: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    query = db.query(JournalEntry).filter(JournalEntry.student_id == current_user.id, JournalEntry.deleted_at.is_(None))
    if search:
        needle = search.lower()
        query = query.filter((JournalEntry.title.ilike(f"%{needle}%")) | (JournalEntry.mood_tag.ilike(f"%{needle}%")))
    if tag:
        rows = query.order_by(JournalEntry.entry_date.desc(), JournalEntry.created_at.desc()).all()
        entries = [entry for entry in rows if tag in (entry.tags_json or [])]
    else:
        total = query.count()
        entries = (
            query.order_by(JournalEntry.entry_date.desc(), JournalEntry.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
    return {
        "items": [_journal_summary(entry) for entry in entries],
        "page": page,
        "page_size": page_size,
        "total": len(entries) if tag else total,
        "private": True,
        "shared_only_when_explicit": True,
        "not_continuously_monitored": True,
    }


@router.post("/journal")
def create_journal_entry(payload: JournalEntryIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    now = datetime.utcnow()
    entry_date = datetime.combine(payload.entry_date or date.today(), datetime.min.time())
    latest_text_consent = get_latest_consent(db, current_user.id, "text_processing")
    entry = JournalEntry(
        journal_entry_id=f"journal_{uuid4().hex}",
        student_id=current_user.id,
        title=payload.title,
        body=payload.content,
        mood_tag=payload.mood,
        tags_json=payload.tags,
        entry_date=entry_date,
        ai_analysis_opt_in=payload.ai_analysis_opt_in,
        analysis_status="pending" if payload.ai_analysis_opt_in else "not_requested",
        analysis_consent_record_id=latest_text_consent.id if payload.ai_analysis_opt_in and latest_text_consent else None,
        shared_with_counselor=payload.share_with_counselor,
        shared_at=now if payload.share_with_counselor else None,
        metadata_json={
            "permissions": {
                "store_journal_privately": True,
                "analyze_journal_with_ai": payload.ai_analysis_opt_in,
                "share_journal_with_counselor": payload.share_with_counselor,
            },
            "not_continuously_monitored": True,
        },
        created_at=now,
        updated_at=now,
    )
    db.add(entry)
    db.flush()
    if payload.ai_analysis_opt_in:
        _analyze_journal_if_permitted(db, current_user, entry)
    db.commit()
    db.refresh(entry)
    return _journal_detail(entry)


@router.put("/journal/{entry_id}")
def update_journal_entry(entry_id: int, payload: JournalEntryIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    entry = _journal_or_404(db, entry_id, current_user)
    now = datetime.utcnow()
    entry.entry_date = datetime.combine(payload.entry_date or date.today(), datetime.min.time())
    entry.title = payload.title
    entry.body = payload.content
    entry.mood_tag = payload.mood
    entry.tags_json = payload.tags
    entry.shared_with_counselor = payload.share_with_counselor
    entry.shared_at = now if payload.share_with_counselor and not entry.shared_at else (entry.shared_at if payload.share_with_counselor else None)
    entry.ai_analysis_opt_in = payload.ai_analysis_opt_in
    entry.analysis_status = "pending" if payload.ai_analysis_opt_in else "not_requested"
    entry.updated_at = now
    latest_text_consent = get_latest_consent(db, current_user.id, "text_processing")
    entry.analysis_consent_record_id = latest_text_consent.id if payload.ai_analysis_opt_in and latest_text_consent else None
    if payload.ai_analysis_opt_in:
        _analyze_journal_if_permitted(db, current_user, entry)
    db.commit()
    db.refresh(entry)
    return _journal_detail(entry)


@router.get("/journal/{entry_id}")
def get_journal_entry(entry_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    return _journal_detail(_journal_or_404(db, entry_id, current_user))


@router.delete("/journal/{entry_id}")
def delete_journal_entry(entry_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    entry = _journal_or_404(db, entry_id, current_user)
    entry.deleted_at = datetime.utcnow()
    entry.updated_at = datetime.utcnow()
    db.commit()
    return {"deleted": True, "id": entry.id}


@router.get("/journal/export")
def export_journal(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.student_id == current_user.id, JournalEntry.deleted_at.is_(None))
        .order_by(JournalEntry.entry_date.desc(), JournalEntry.created_at.desc())
        .all()
    )
    return {
        "filename": f"wellness-journal-{current_user.id}.json",
        "private": True,
        "items": [_journal_detail(entry) for entry in entries],
    }


def _progress_event(user_id: int, payload: ProgressIn) -> Dict[str, Any]:
    event = {
        "id": next(_progress_id_counter),
        "activity_type": payload.activity_type,
        "item_id": payload.item_id,
        "minutes": payload.minutes,
        "completed": payload.completed,
        "created_at": datetime.utcnow().isoformat(),
    }
    _state(user_id)["progress_events"].append(event)
    return event


@router.post("/progress/track")
def track_progress(payload: ProgressIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    return {"recorded": True, "event": _progress_event(current_user.id, payload)}


@router.get("/progress")
def get_progress(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _require_student(current_user)
    events = _state(current_user.id)["progress_events"]
    now = datetime.utcnow()
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    def since(start: datetime) -> List[Dict[str, Any]]:
        return [event for event in events if datetime.fromisoformat(event["created_at"]) >= start]

    weekly = since(week_start)
    monthly = since(month_start)
    achievements = []
    if any(event["activity_type"] == "breathing" for event in weekly):
        achievements.append("You made space to breathe this week.")
    if any(event["activity_type"] == "meditation" for event in monthly):
        achievements.append("You returned to a meditation practice this month.")
    if db.query(JournalEntry).filter(JournalEntry.student_id == current_user.id, JournalEntry.deleted_at.is_(None)).count() > 0:
        achievements.append("You wrote a private reflection.")

    return {
        "weekly_activity": weekly,
        "monthly_activity": monthly,
        "achievements": achievements,
        "supportive_language": True,
        "competitive_scoring": False,
    }


def _journal_or_404(db: Session, entry_id: int, user: User) -> JournalEntry:
    entry = (
        db.query(JournalEntry)
        .filter(JournalEntry.id == entry_id, JournalEntry.student_id == user.id, JournalEntry.deleted_at.is_(None))
        .first()
    )
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal entry not found")
    return entry


def _journal_summary(entry: JournalEntry) -> Dict[str, Any]:
    preview = (entry.body or "").strip().replace("\n", " ")
    if len(preview) > 180:
        preview = preview[:177].rstrip() + "..."
    return {
        "id": entry.id,
        "journal_entry_id": entry.journal_entry_id,
        "title": entry.title,
        "entry_date": entry.entry_date.date().isoformat(),
        "mood": entry.mood_tag,
        "tags": entry.tags_json or [],
        "content_preview": preview,
        "share_with_counselor": entry.shared_with_counselor,
        "ai_analysis_opt_in": entry.ai_analysis_opt_in,
        "analysis_status": entry.analysis_status,
        "privacy": "student_shared" if entry.shared_with_counselor else "private",
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "not_continuously_monitored": True,
    }


def _journal_detail(entry: JournalEntry) -> Dict[str, Any]:
    return {**_journal_summary(entry), "content": entry.body}


def _analyze_journal_if_permitted(db: Session, user: User, entry: JournalEntry) -> None:
    if not entry.ai_analysis_opt_in:
        entry.analysis_status = "not_requested"
        return
    if not has_active_consent(db, user.id, "text_processing"):
        prediction = create_unavailable_prediction(
            db,
            user=user,
            modality="text",
            failure_code="CONSENT_REQUIRED",
            message="Journal analysis requires active text-processing consent.",
            source_type="journal_entry",
            source_record_id=entry.id,
            source_timestamp=entry.created_at,
        )
        entry.analysis_status = "unavailable"
        entry.metadata_json = {**(entry.metadata_json or {}), "latest_prediction_id": prediction.id}
        return
    if len((entry.body or "").strip()) < 20:
        prediction = create_unavailable_prediction(
            db,
            user=user,
            modality="text",
            failure_code="MINIMUM_CONTENT_NOT_MET",
            message="Journal entry was too short for optional text analysis.",
            source_type="journal_entry",
            source_record_id=entry.id,
            source_timestamp=entry.created_at,
        )
        entry.analysis_status = "unavailable"
        entry.metadata_json = {**(entry.metadata_json or {}), "latest_prediction_id": prediction.id}
        return

    snapshot = create_feature_snapshot(
        db,
        student_id=user.id,
        modality="text",
        source_type="journal_entry",
        source_record_id=entry.id,
        source_timestamp=entry.created_at,
        feature_schema_version=MODEL_EVIDENCE["text"]["schema"],
        preprocessing_version=MODEL_EVIDENCE["text"]["preprocessing"],
        features_json={"text_length": len(entry.body), "contains_raw_text": False, "source_type": "journal_entry"},
        metadata_json={"raw_text_stored_in_feature_snapshot": False, "not_continuously_monitored": True},
    )
    active_model = None
    try:
        active_model, result = predict_with_active_model(db, modality="text", payload={"text": entry.body})
        snapshot.feature_schema_version = active_model.feature_schema_version or snapshot.feature_schema_version
        snapshot.preprocessing_version = active_model.preprocessing_version or snapshot.preprocessing_version
        snapshot.features_json = result.features
        prediction = create_prediction(
            db,
            student_id=user.id,
            modality="text",
            status_value="succeeded",
            is_available=True,
            output_type="machine_learning",
            source_type="journal_entry",
            source_record_id=entry.id,
            source_timestamp=entry.created_at,
            feature_snapshot=snapshot,
            probability=result.probability,
            confidence=result.confidence,
            label=result.label,
            metadata_json={
                **result.metadata,
                "class_probabilities": result.probabilities,
                "source_type": "journal_entry",
                "selection_policy": "most_recent_valid_text_prediction",
                "student_facing": "Journal entries are not monitored continuously.",
            },
            model_registry=active_model,
        )
        entry.analysis_status = "completed"
    except RuntimeModelUnavailable:
        prediction = create_unavailable_prediction(
            db,
            user=user,
            modality="text",
            failure_code="MODEL_NOT_ACTIVE",
            message="The trained text modality model is not active in the runtime API.",
            source_type="journal_entry",
            source_record_id=entry.id,
            source_timestamp=entry.created_at,
            feature_snapshot=snapshot,
        )
        entry.analysis_status = "unavailable"
    except RuntimeInferenceError:
        prediction = create_failed_prediction(
            db,
            user=user,
            modality="text",
            failure_code="INFERENCE_FAILED",
            message="The active text runtime model could not safely analyze this journal entry.",
            source_type="journal_entry",
            source_record_id=entry.id,
            source_timestamp=entry.created_at,
            feature_snapshot=snapshot,
            model_registry=active_model,
        )
        entry.analysis_status = "failed"
    entry.metadata_json = {**(entry.metadata_json or {}), "latest_prediction_id": prediction.id}


@router.get("/preferences")
def get_preferences(current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    return _state(current_user.id)["preferences"]


@router.patch("/preferences")
def update_preferences(payload: PreferencesIn, current_user: User = Depends(get_current_user)):
    _require_student(current_user)
    preferences = _state(current_user.id)["preferences"]
    for key, value in payload.dict(exclude_unset=True).items():
        if isinstance(value, dict) and isinstance(preferences.get(key), dict):
            preferences[key].update(value)
        else:
            preferences[key] = value
    return preferences
