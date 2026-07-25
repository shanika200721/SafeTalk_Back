from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Any, Optional, List, Dict
from datetime import datetime
from enum import Enum

# ============= User Schemas =============
class UserRole(str, Enum):
    STUDENT = "student"
    COUNSELOR = "counselor"
    ADMIN = "admin"
    PSYCHIATRIST = "psychiatrist"

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

class UserLookup(BaseModel):
    id: int
    username: str
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: UserRole
    department: Optional[str] = None
    year_of_study: Optional[int] = None
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

# ============= Consent Schemas =============
class ConsentUpdate(BaseModel):
    is_granted: bool
    policy_version: str = Field(..., min_length=1, max_length=50)

class ConsentState(BaseModel):
    consent_type: str
    description: str
    is_granted: bool
    policy_version: Optional[str] = None
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ConsentRecordSchema(BaseModel):
    id: int
    consent_type: str
    is_granted: bool
    policy_version: str
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    source: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ============= Canonical Modality Contracts =============
class CanonicalModality(str, Enum):
    PROFILE = "profile"
    DASS21 = "dass21"
    MOOD = "mood"
    TEXT = "text"
    SPEECH = "speech"
    FACE = "face"
    BEHAVIORAL = "behavioral"


class PredictionStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    STALE = "stale"


class PredictionOutputType(str, Enum):
    RULE_BASED = "rule_based"
    HEURISTIC = "heuristic"
    MACHINE_LEARNING = "machine_learning"
    MANUAL = "manual"
    EXTERNALLY_SUPPLIED = "externally_supplied"


class DataQuality(BaseModel):
    status: str = "accepted"
    flags: List[str] = Field(default_factory=list)


class PredictionEvidence(BaseModel):
    available: bool
    source_type: Optional[str] = None
    source_id: Optional[int] = None
    feature_snapshot_id: Optional[int] = None


class PredictionModelEvidence(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    registry_id: Optional[int] = None


class ModalityPredictionResponse(BaseModel):
    prediction_id: int
    user_id: int
    modality: CanonicalModality
    status: PredictionStatus
    is_available: bool
    output_type: PredictionOutputType
    score: Optional[float] = Field(default=None, ge=0, le=100)
    probability: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    label: Optional[str] = None
    failure_code: Optional[str] = None
    failure_message_safe: Optional[str] = None
    source_timestamp: Optional[datetime] = None
    generated_at: datetime
    valid_until: Optional[datetime] = None
    model: PredictionModelEvidence
    preprocessing_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    data_quality: DataQuality
    evidence: PredictionEvidence
    consent_policy_version: Optional[str] = None
    clinical_use_boundary: str = "screening_support_only"
    limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityPredictionListResponse(BaseModel):
    user_id: int
    predictions: List[ModalityPredictionResponse]


class ModalityAvailabilityItem(BaseModel):
    modality: CanonicalModality
    implemented: bool
    runtime_model_active: bool
    contract_available: bool
    consent_required: str
    source_requirements: List[str]
    limitations: List[str]


class ModalityAvailabilityResponse(BaseModel):
    modalities: List[ModalityAvailabilityItem]


class ModelVerificationResponse(BaseModel):
    model_id: Optional[int] = None
    passed: bool
    failure_code: Optional[str] = None
    failure_message_safe: Optional[str] = None
    actual_hash: Optional[str] = None
    expected_hash: Optional[str] = None
    serializer: Optional[str] = None
    metadata_complete: bool = False
    smoke_test_status: str = "not_run"
    activation_eligible: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistryResponse(BaseModel):
    id: int
    model_name: str
    modality: str
    version: str
    framework: str
    artifact_path: str
    preprocessing_path: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    artifact_sha256: Optional[str] = None
    serializer: Optional[str] = None
    framework_version: Optional[str] = None
    preprocessing_version: Optional[str] = None
    label_mapping_version: Optional[str] = None
    status: str
    verification_status: Optional[str] = None
    verification_failure_code: Optional[str] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    is_active: bool
    model_card_path: Optional[str] = None
    intended_use: Optional[str] = None
    prohibited_use: Optional[str] = None
    limitations_json: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProfilePredictionRequest(BaseModel):
    source_assessment_id: Optional[int] = None
    profile_payload: Optional["ProfileAssessmentCreate"] = None
    strategy: str = Field(default="heuristic", pattern="^(heuristic|active_model)$")


class DASS21PredictionRequest(BaseModel):
    assessment_id: Optional[int] = None
    responses: Optional[List[int]] = Field(default=None, min_items=21, max_items=21)

    @field_validator("responses")
    @classmethod
    def validate_responses(cls, value):
        if value is None:
            return value
        if not all(0 <= item <= 3 for item in value):
            raise ValueError("Each DASS-21 response must be between 0 and 3")
        return value


class MoodPredictionRequest(BaseModel):
    checkin_id: Optional[int] = None
    days: Optional[int] = Field(default=None, ge=1, le=90)


class TextPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SpeechPredictionRequest(BaseModel):
    chat_message_id: Optional[int] = None
    upload_reference_id: Optional[str] = Field(default=None, max_length=120)

    @field_validator("upload_reference_id")
    @classmethod
    def reject_path_like_reference(cls, value):
        if value is None:
            return value
        if any(fragment in value for fragment in ("/", "\\", ":", "..")):
            raise ValueError("Use a secure upload reference, not a filesystem path")
        return value


class FacePredictionRequest(BaseModel):
    source_reference_id: Optional[str] = Field(default=None, max_length=120)

    @field_validator("source_reference_id")
    @classmethod
    def reject_path_like_reference(cls, value):
        if value is None:
            return value
        if any(fragment in value for fragment in ("/", "\\", ":", "..")):
            raise ValueError("Use a secure source reference, not a filesystem path")
        return value


class BehavioralPredictionRequest(BaseModel):
    pass

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
    metadata: Optional["DASS21Metadata"] = None

    class Config:
        from_attributes = True


class DASS21Metadata(BaseModel):
    questionnaire_version: Optional[str] = None
    item_mapping_version: Optional[str] = None
    scoring_version: Optional[str] = None
    score_multiplier: Optional[float] = None
    completed_item_count: Optional[int] = None
    is_complete: Optional[bool] = None
    scored_at: Optional[datetime] = None
    consent_policy_version: Optional[str] = None

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


class ControlledFusionAssessRequest(BaseModel):
    user_id: Optional[int] = None


class FusionVersionInfo(BaseModel):
    config_version: str
    config_hash: str
    threshold_version: str
    mapping_version: str
    coverage_policy_version: str
    staleness_policy_version: str


class FusionEvidence(BaseModel):
    configured_modalities: List[str]
    available_modalities: List[str]
    used_modalities: List[str]
    missing_modalities: List[str]
    excluded_modalities: List[Dict[str, Any]]
    base_weight_coverage: float
    effective_weight_total: float
    modality_count: int
    latest_source_timestamp: Optional[datetime] = None
    oldest_source_timestamp: Optional[datetime] = None
    evidence_window: Optional[Dict[str, Any]] = None
    coverage_category: str
    minimum_evidence_met: bool


class FusionInputEvidence(BaseModel):
    modality: str
    prediction_id: Optional[int] = None
    mapped_score: Optional[float] = None
    base_weight: Optional[float] = None
    effective_weight: Optional[float] = None
    included: bool = True
    exclusion_reason: Optional[str] = None
    source_timestamp: Optional[datetime] = None
    prediction_age_seconds: Optional[float] = None


class ControlledFusionAssessmentResponse(BaseModel):
    assessment_id: Optional[int] = None
    user_id: int
    status: str
    score: Optional[float] = Field(default=None, ge=0, le=1)
    risk_level: Optional[str] = None
    assessment_type: str = "screening_support"
    fusion: FusionVersionInfo
    evidence: FusionEvidence
    inputs: List[FusionInputEvidence]
    limitations: List[str]
    model_risk_level: Optional[str] = None
    model_score: Optional[float] = Field(default=None, ge=0, le=1)
    human_review_status: str = "not_requested"
    counselor_decision: Optional[Dict[str, Any]] = None
    counselor_override: Optional[Dict[str, Any]] = None
    alert_created: bool = False


class ControlledFusionConfigResponse(BaseModel):
    config_version: str
    config_hash: str
    modalities: List[str]
    base_weights: Dict[str, float]
    thresholds: Dict[str, float]
    threshold_version: str
    mapping_versions: Dict[str, str]
    staleness_policy_version: str
    staleness_windows_days: Dict[str, float]
    coverage_policy_version: str
    coverage_categories: Dict[str, Any]
    minimum_evidence_policy: Dict[str, Any]
    limitations: List[str]

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
