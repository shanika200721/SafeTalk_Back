from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

# ============= User Schemas =============
class UserRole(str, Enum):
    STUDENT = "student"
    COUNSELOR = "counselor"
    ADMIN = "admin"

class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    role: UserRole = UserRole.STUDENT
    department: Optional[str] = None
    year_of_study: Optional[int] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    year_of_study: Optional[int] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ============= Auth Schemas =============
class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional['User'] = None

class TokenData(BaseModel):
    username: Optional[str] = None

class Login(BaseModel):
    username: str
    password: str

# ============= Profile Assessment Schemas =============
class ProfileAssessmentBase(BaseModel):
    gpa: float = 0
    repeated_subjects: int = 0
    attendance: float = 100
    family_relationship_score: float = 10
    income_level: Optional[str] = None
    parents_employment: Optional[str] = None
    family_support: float = 5
    living_arrangement: Optional[str] = None
    employment_status: Optional[str] = None
    financial_stress: bool = False
    communication_skills: float = 5
    social_connection: float = 5
    sleep_pattern: str = "Regular"
    exercise_frequency: str = "Occasionally"
    substance_use: str = "None"

class ProfileAssessmentCreate(ProfileAssessmentBase):
    user_id: int

class ProfileAssessment(ProfileAssessmentBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# ============= Assessment Schemas =============
class ModalityScores(BaseModel):
    profile_score: float = 0
    mood_score: float = 0
    dass21_score: float = 0
    text_score: float = 0
    voice_score: float = 0
    face_score: float = 0
    behavioral_score: float = 0

class AssessmentBase(BaseModel):
    assessment_type: str
    scores: ModalityScores
    profile_data: Optional[Dict] = None

class AssessmentCreate(AssessmentBase):
    user_id: int
    das21_depression: Optional[float] = None
    dass21_anxiety: Optional[float] = None
    dass21_stress: Optional[float] = None

class AssessmentResponse(BaseModel):
    user_id: int
    assessment_type: str
    composite_score: float
    risk_level: str
    needs_escalation: bool
    recommendations: List[str]
    timestamp: str
    modality_breakdown: Dict

class Assessment(BaseModel):
    id: int
    user_id: int
    assessment_type: str
    composite_score: float
    risk_level: str
    needs_escalation: bool
    recommendations: Optional[List[str]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ============= DASS21 Assessment Schemas =============
class DASS21Request(BaseModel):
    responses: List[int] = Field(..., min_items=21, max_items=21)
    
    class Config:
        json_schema_extra = {
            "example": {
                "responses": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]
            }
        }

class DASS21Response(BaseModel):
    id: int
    user_id: int
    responses: List[int]
    depression_score: float
    anxiety_score: float
    stress_score: float
    total_dass21_score: float
    depression_severity: str
    anxiety_severity: str
    stress_severity: str
    created_at: datetime

    class Config:
        from_attributes = True

# ============= Risk Assessment Schemas =============
class RiskAssessmentRequest(BaseModel):
    user_id: int
    scores: ModalityScores
    profile_data: Optional[Dict] = None

class RiskAssessmentResponse(BaseModel):
    user_id: int
    composite_score: float
    risk_level: str
    needs_escalation: bool
    recommendations: List[str]
    timestamp: str
    modality_breakdown: Dict

# ============= Daily Checkin Schemas =============
class DailyCheckinCreate(BaseModel):
    mood_score: float = Field(..., ge=1, le=10)
    emotional_state: str
    notable_events: Optional[str] = None
    stressors: Optional[str] = None
    positive_moments: Optional[str] = None
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[str] = None
    exercise: bool = False
    ate_well: bool = False
    thoughts: Optional[str] = None
    has_negative_thoughts: bool = False
    coping_strategies_used: Optional[List[str]] = None
    support_needed: bool = False
    support_type: Optional[str] = None

class DailyCheckin(DailyCheckinCreate):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# ============= Chat History Schemas =============
class ChatMessageCreate(BaseModel):
    message_text: str
    sender: str = "user"

class ChatMessage(ChatMessageCreate):
    id: int
    user_id: int
    text_sentiment: Optional[str] = None
    risk_indicators: Optional[List[str]] = None
    text_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    message: str
    sentiment: Optional[str] = None
    risk_score: Optional[float] = None
    recommendations: Optional[List[str]] = None

# ============= Voice Analysis Schemas =============
class VoiceAnalysisCreate(BaseModel):
    audio_file_path: str
    duration_seconds: float

class VoiceAnalysisResponse(BaseModel):
    emotional_state: str
    stress_level: float
    risk_score: float
    pitch_variation: float
    energy_level: float
    speech_rate: float

# ============= Facial Analysis Schemas =============
class FacialAnalysisCreate(BaseModel):
    image_file_path: str

class FacialAnalysisResponse(BaseModel):
    emotion_detected: str
    emotion_confidence: float
    stress_level: float
    risk_score: float
    eye_gaze: Optional[str] = None

# ============= Risk Record Schemas =============
class RiskRecord(BaseModel):
    id: int
    user_id: int
    risk_level: str
    composite_score: float
    needs_escalation: bool
    created_at: datetime

    class Config:
        from_attributes = True

# ============= Resource Schemas =============
class ResourceBase(BaseModel):
    title: str
    category: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None

class ResourceCreate(ResourceBase):
    pass

class Resource(ResourceBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# ============= Emergency Contact Schemas =============
class EmergencyContact(BaseModel):
    id: int
    name: str
    country: str
    phone: str
    url: Optional[str] = None
    description: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True

# ============= Counselor Schemas =============
class CounselorNotesCreate(BaseModel):
    student_id: int
    notes: str
    observations: Optional[str] = None
    recommendations: Optional[List[str]] = None

class StudentBasicInfo(BaseModel):
    id: int
    username: str
    full_name: str
    email: str
    department: Optional[str]

class StudentWithLatestRisk(StudentBasicInfo):
    latest_risk_level: Optional[str] = None
    latest_assessment_date: Optional[datetime] = None

# ============= Pagination Schemas =============
class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    data: List[Dict]
