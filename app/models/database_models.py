from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.db.base import Base

class UserRole(str, enum.Enum):
    STUDENT = "student"
    COUNSELOR = "counselor"
    ADMIN = "admin"
    PSYCHIATRIST = "psychiatrist"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    role = Column(Enum(UserRole), default=UserRole.STUDENT)
    department = Column(String, nullable=True)
    year_of_study = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile_assessments = relationship("ProfileAssessment", back_populates="user")
    daily_checkins = relationship("DailyCheckIn", back_populates="user")
    assessments = relationship("Assessment", back_populates="user")
    counselor_sessions = relationship("CounselorSession", foreign_keys="[CounselorSession.user_id]", back_populates="user")
    feature_snapshots = relationship("FeatureSnapshot", back_populates="student")
    modality_predictions = relationship("ModalityPrediction", back_populates="student")
    risk_assessments = relationship("RiskAssessment", back_populates="student")

class ProfileAssessment(Base):
    __tablename__ = "profile_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Academic Information
    gpa = Column(Float, default=0)
    repeated_subjects = Column(Integer, default=0)
    attendance = Column(Float, default=100)
    academic_difficulty = Column(String, nullable=True)
    
    # Family Information
    family_relationship_score = Column(Float, default=10)
    income_level = Column(String, nullable=True)
    parents_employment = Column(String, nullable=True)
    family_support = Column(Float, default=5)
    
    # Living Situation
    living_arrangement = Column(String, nullable=True)
    employment_status = Column(String, nullable=True)
    financial_stress = Column(Boolean, default=False)
    
    # Behavioral & Social
    communication_skills = Column(Float, default=5)
    social_connection = Column(Float, default=5)
    sleep_pattern = Column(String, default="Regular")
    exercise_frequency = Column(String, default="Occasionally")
    substance_use = Column(String, default="None")
    
    # Calculated Risk Score
    profile_score = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="profile_assessments")

class DailyCheckIn(Base):
    __tablename__ = "daily_checkins"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    mood = Column(Integer)  # 1-5 scale
    mood_description = Column(String, nullable=True)
    sleep_hours = Column(Float)
    exercise_minutes = Column(Integer, default=0)
    social_interaction = Column(String)  # None, Limited, Moderate, Good
    stress_level = Column(Integer)  # 1-10 scale
    anxiety_level = Column(Integer)  # 1-10 scale
    negative_thoughts = Column(Boolean, default=False)
    substance_use_today = Column(Boolean, default=False)
    self_harm_thoughts = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="daily_checkins")

class DASS21Assessment(Base):
    __tablename__ = "dass21_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # DASS21 responses (0-3 scale for each item)
    responses = Column(JSON)  # Store all 21 responses
    
    # Calculated scores
    depression_score = Column(Float)
    anxiety_score = Column(Float)
    stress_score = Column(Float)
    total_dass21_score = Column(Float)
    
    # Severity classifications
    depression_severity = Column(String)  # Normal, Mild, Moderate, Severe, Extremely Severe
    anxiety_severity = Column(String)
    stress_severity = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    assessment_type = Column(String)  # profile, daily, dass21, multimodal, etc.
    
    # Multimodal scores
    profile_score = Column(Float, default=0)
    mood_score = Column(Float, default=0)
    dass21_score = Column(Float, default=0)
    text_score = Column(Float, default=0)
    voice_score = Column(Float, default=0)
    face_score = Column(Float, default=0)
    behavioral_score = Column(Float, default=0)
    
    # Composite results
    composite_score = Column(Float)
    risk_level = Column(String)  # LOW, MEDIUM, HIGH, SEVERE
    needs_escalation = Column(Boolean, default=False)
    recommendations = Column(JSON)  # Store recommendations as JSON
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="assessments")

class CounselorSession(Base):
    __tablename__ = "counselor_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    counselor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    session_type = Column(String)  # auto_escalated, scheduled, emergency, etc.
    status = Column(String, default="pending")  # pending, in_progress, completed
    risk_level_at_escalation = Column(String)
    
    # Session notes
    counselor_notes = Column(Text, nullable=True)
    intervention_type = Column(String, nullable=True)
    outcome = Column(String, nullable=True)
    
    # Follow-up
    follow_up_needed = Column(Boolean, default=False)
    follow_up_date = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="counselor_sessions")

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alerts_is_read", "is_read"),
        Index("ix_alerts_risk_level", "risk_level"),
        Index("ix_alerts_created_at", "created_at"),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    alert_type = Column(String)  # escalation, milestone, behavioral_change
    risk_level = Column(String)
    message = Column(Text)
    is_read = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    events = relationship("AlertEvent", back_populates="alert")

class AssessmentHistory(Base):
    __tablename__ = "assessment_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    assessment_type = Column(String)
    
    # Store snapshot of all scores
    data = Column(JSON)
    composite_score = Column(Float)
    risk_level = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Resource(Base):
    __tablename__ = "resources"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    category = Column(String)  # crisis, coping, therapy, medical, etc.
    description = Column(Text)
    url = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))  # Student or Counselor
    receiver_id = Column(Integer, ForeignKey("users.id"))  # Usually Counselor
    
    message = Column(Text)
    message_type = Column(String, default="text")  # text, image, file, etc.
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sender = relationship("User", foreign_keys=[sender_id])
    receiver = relationship("User", foreign_keys=[receiver_id])

class SafeTalkBotMessage(Base):
    __tablename__ = "safetalk_bot_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    user_message = Column(Text)
    bot_response = Column(Text)
    intent = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    crisis_level = Column(Integer, default=0)  # 0-10 severity scale
    response_details = Column(JSON, nullable=True)  # Stores techniques, alternatives, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("model_name", "modality", "version", name="uq_model_registry_name_modality_version"),
        Index("ix_model_registry_model_version", "model_name", "version"),
        Index(
            "uq_model_registry_one_active",
            "model_name",
            "modality",
            unique=True,
            postgresql_where=text("is_active = true"),
            sqlite_where=text("is_active = 1"),
        ),
        Index("ix_model_registry_modality", "modality"),
    )

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    modality = Column(String, nullable=False)
    version = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)
    preprocessing_path = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    feature_schema_version = Column(String, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    thresholds_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    predictions = relationship("ModalityPrediction", back_populates="model_registry")
    fused_risk_assessments = relationship("RiskAssessment", back_populates="fusion_model")


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    __table_args__ = (
        Index("ix_feature_snapshots_student_created", "student_id", "created_at"),
        Index("ix_feature_snapshots_modality", "modality"),
        Index("ix_feature_snapshots_source", "source_type", "source_record_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    source_type = Column(String, nullable=False)
    source_record_id = Column(Integer, nullable=False)
    modality = Column(String, nullable=False)
    features_json = Column(JSON, nullable=False)
    preprocessing_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="feature_snapshots")


class ModalityPrediction(Base):
    __tablename__ = "modality_predictions"
    __table_args__ = (
        CheckConstraint("probability >= 0 AND probability <= 1", name="ck_modality_predictions_probability_0_1"),
        CheckConstraint("score_0_100 >= 0 AND score_0_100 <= 100", name="ck_modality_predictions_score_0_100"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_modality_predictions_confidence_0_1"),
        Index("ix_modality_predictions_student_created", "student_id", "created_at"),
        Index("ix_modality_predictions_modality", "modality"),
        Index("ix_modality_predictions_source", "source_type", "source_record_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    modality = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_record_id = Column(Integer, nullable=False)
    model_registry_id = Column(Integer, ForeignKey("model_registry.id", ondelete="RESTRICT"), nullable=False)
    predicted_class = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    score_0_100 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    explanation_json = Column(JSON, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="modality_predictions")
    model_registry = relationship("ModelRegistry", back_populates="predictions")
    risk_inputs = relationship("RiskAssessmentInput", back_populates="modality_prediction")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    __table_args__ = (
        CheckConstraint("final_probability >= 0 AND final_probability <= 1", name="ck_risk_assessments_final_probability_0_1"),
        CheckConstraint("final_score >= 0 AND final_score <= 100", name="ck_risk_assessments_final_score_0_100"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_risk_assessments_confidence_0_1"),
        Index("ix_risk_assessments_student_created", "student_id", "created_at"),
        Index("ix_risk_assessments_risk_level", "risk_level"),
        Index("ix_risk_assessments_safety_override", "safety_override"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    fusion_model_id = Column(Integer, ForeignKey("model_registry.id", ondelete="RESTRICT"), nullable=True)
    final_probability = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    data_completeness = Column(JSON, nullable=True)
    safety_override = Column(Boolean, default=False, nullable=False)
    safety_override_reason = Column(Text, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="risk_assessments")
    fusion_model = relationship("ModelRegistry", back_populates="fused_risk_assessments")
    inputs = relationship("RiskAssessmentInput", back_populates="risk_assessment")


class RiskAssessmentInput(Base):
    __tablename__ = "risk_assessment_inputs"
    __table_args__ = (
        UniqueConstraint("risk_assessment_id", "modality_prediction_id", name="uq_risk_assessment_input_pair"),
        Index("ix_risk_assessment_inputs_assessment", "risk_assessment_id"),
        Index("ix_risk_assessment_inputs_prediction", "modality_prediction_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    risk_assessment_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=False)
    modality_prediction_id = Column(Integer, ForeignKey("modality_predictions.id"), nullable=False)

    risk_assessment = relationship("RiskAssessment", back_populates="inputs")
    modality_prediction = relationship("ModalityPrediction", back_populates="risk_inputs")


class AlertEvent(Base):
    __tablename__ = "alert_events"
    __table_args__ = (
        Index("ix_alert_events_alert_created", "alert_id", "created_at"),
        Index("ix_alert_events_new_status", "new_status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    old_status = Column(String, nullable=True)
    new_status = Column(String, nullable=False)
    changed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    alert = relationship("Alert", back_populates="events")
    changed_by_user = relationship("User")


class WorkerJob(Base):
    __tablename__ = "worker_jobs"
    __table_args__ = (
        Index("ix_worker_jobs_status", "status"),
        Index("ix_worker_jobs_source", "source_type", "source_record_id"),
        Index("ix_worker_jobs_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_record_id = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
