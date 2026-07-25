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
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=True)
    department = Column(String, nullable=True)
    year_of_study = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    suspended_at = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True, index=True)
    invitation_sent_at = Column(DateTime, nullable=True)
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
    consent_records = relationship("ConsentRecord", back_populates="user")
    university = relationship("University", back_populates="users")
    counselor_assignments_as_student = relationship(
        "CounselorAssignment",
        foreign_keys="CounselorAssignment.student_id",
        back_populates="student",
    )
    counselor_assignments_as_counselor = relationship(
        "CounselorAssignment",
        foreign_keys="CounselorAssignment.counselor_id",
        back_populates="counselor",
    )
    counselor_reviews_as_student = relationship(
        "CounselorReview",
        foreign_keys="CounselorReview.student_id",
        back_populates="student",
    )
    counselor_reviews_as_counselor = relationship(
        "CounselorReview",
        foreign_keys="CounselorReview.counselor_id",
        back_populates="counselor",
    )
    counselor_notes_as_student = relationship(
        "CounselorNote",
        foreign_keys="CounselorNote.student_id",
        back_populates="student",
    )
    counselor_notes_as_counselor = relationship(
        "CounselorNote",
        foreign_keys="CounselorNote.counselor_id",
        back_populates="counselor",
    )
    counselor_profile = relationship("CounselorProfile", back_populates="user", uselist=False)


class University(Base):
    __tablename__ = "universities"
    __table_args__ = (
        UniqueConstraint("university_code", "campus_name", name="uq_universities_code_campus"),
        Index("ix_universities_active", "active"),
        Index("ix_universities_code", "university_code"),
    )

    id = Column(Integer, primary_key=True, index=True)
    university_id = Column(String, unique=True, nullable=False, index=True)
    university_name = Column(String, nullable=False)
    university_code = Column(String, nullable=False)
    campus_name = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    district = Column(String, nullable=True)
    province = Column(String, nullable=True)
    general_phone = Column(String, nullable=True)
    counseling_unit_phone = Column(String, nullable=True)
    emergency_support_phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    website = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    users = relationship("User", back_populates="university")
    counselor_profiles = relationship("CounselorProfile", back_populates="university")
    support_contacts = relationship("SupportContact", back_populates="university")


class CounselorProfile(Base):
    __tablename__ = "counselor_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_counselor_profiles_user_id"),
        Index("ix_counselor_profiles_university_active", "university_id", "active"),
        Index("ix_counselor_profiles_availability", "availability_status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    counselor_profile_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=True)
    full_name = Column(String, nullable=False)
    professional_title = Column(String, nullable=True)
    qualification = Column(Text, nullable=True)
    specialization = Column(Text, nullable=True)
    registration_number = Column(String, nullable=True)
    office_name = Column(String, nullable=True)
    office_location = Column(Text, nullable=True)
    telephone_number = Column(String, nullable=True)
    whatsapp_number = Column(String, nullable=True)
    email = Column(String, nullable=True)
    available_days = Column(String, nullable=True)
    available_from = Column(String, nullable=True)
    available_until = Column(String, nullable=True)
    accepts_voice_calls = Column(Boolean, default=True, nullable=False)
    accepts_whatsapp_calls = Column(Boolean, default=False, nullable=False)
    accepts_whatsapp_messages = Column(Boolean, default=True, nullable=False)
    emergency_contact_enabled = Column(Boolean, default=False, nullable=False)
    profile_photo_reference = Column(String, nullable=True)
    languages_json = Column(JSON, nullable=True)
    availability_status = Column(String, default="available", nullable=False)
    approved = Column(Boolean, default=False, nullable=False)
    student_visible = Column(Boolean, default=True, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    admin_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="counselor_profile")
    university = relationship("University", back_populates="counselor_profiles")
    support_contacts = relationship("SupportContact", back_populates="counselor_profile")


class CounselorUniversityAssignment(Base):
    __tablename__ = "counselor_university_assignments"
    __table_args__ = (
        Index("ix_counselor_university_assignments_profile_active", "counselor_profile_id", "active"),
        Index("ix_counselor_university_assignments_university_active", "university_id", "active"),
    )

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(String, unique=True, nullable=False, index=True)
    counselor_profile_id = Column(Integer, ForeignKey("counselor_profiles.id"), nullable=False)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=False)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    assignment_reason = Column(Text, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    counselor_profile = relationship("CounselorProfile")
    university = relationship("University")
    assigned_by_user = relationship("User")

class ProfileAssessment(Base):
    __tablename__ = "profile_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    profile_assessment_id = Column(String, unique=True, nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    questionnaire_version = Column(String, nullable=True)
    status = Column(String, nullable=True, index=True)
    responses_json = Column(JSON, nullable=True)
    normalized_features_json = Column(JSON, nullable=True)
    preprocessing_version = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    stale_at = Column(DateTime, nullable=True)
    prediction_id = Column(Integer, ForeignKey("modality_predictions.id"), nullable=True)
    consent_record_id = Column(Integer, ForeignKey("consent_records.id"), nullable=True)
    source = Column(String, nullable=True)
    privacy_metadata_json = Column(JSON, nullable=True)
    
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
    prediction = relationship("ModalityPrediction", foreign_keys=[prediction_id])
    consent_record = relationship("ConsentRecord", foreign_keys=[consent_record_id])

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

    questionnaire_version = Column(String, nullable=True)
    item_mapping_version = Column(String, nullable=True)
    scoring_version = Column(String, nullable=True)
    score_multiplier = Column(Float, nullable=True)
    completed_item_count = Column(Integer, nullable=True)
    is_complete = Column(Boolean, nullable=True)
    scored_at = Column(DateTime, nullable=True)
    consent_policy_version = Column(String, nullable=True)
    
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
    resource_type = Column(String, default="article", nullable=False, index=True)
    description = Column(Text)
    url = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    status = Column(String, default="draft", nullable=False, index=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_logs"
    __table_args__ = (
        Index("ix_admin_audit_logs_created", "created_at"),
        Index("ix_admin_audit_logs_user_action", "user_id", "action"),
        Index("ix_admin_audit_logs_entity", "entity_type", "entity_id"),
        CheckConstraint(
            "status IN ('success', 'failed', 'blocked')",
            name="ck_admin_audit_logs_status",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=True)
    old_value_json = Column(JSON, nullable=True)
    new_value_json = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    status = Column(String, nullable=False, default="success")
    privacy_scope = Column(String, nullable=False, default="administrative_summary")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")


class SystemSetting(Base):
    __tablename__ = "system_settings"
    __table_args__ = (
        UniqueConstraint("section", "setting_key", name="uq_system_settings_section_key"),
        Index("ix_system_settings_section", "section"),
    )

    id = Column(Integer, primary_key=True, index=True)
    section = Column(String, nullable=False)
    setting_key = Column(String, nullable=False)
    setting_value = Column(JSON, nullable=True)
    value_type = Column(String, default="json", nullable=False)
    description = Column(Text, nullable=True)
    read_only = Column(Boolean, default=False, nullable=False)
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    updated_by_user = relationship("User")


class AdminReport(Base):
    __tablename__ = "admin_reports"
    __table_args__ = (
        Index("ix_admin_reports_type_created", "report_type", "created_at"),
        Index("ix_admin_reports_requested_by", "requested_by"),
    )

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, nullable=False, index=True)
    report_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    parameters_json = Column(JSON, nullable=True)
    summary_json = Column(JSON, nullable=True)
    export_formats_json = Column(JSON, nullable=True)
    status = Column(String, default="generated", nullable=False)
    requested_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    requested_by_user = relationship("User")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))  # Student or Counselor
    receiver_id = Column(Integer, ForeignKey("users.id"))  # Usually Counselor
    
    message = Column(Text)
    message_type = Column(String, default="text")  # text, image, file, etc.
    ai_analysis_requested = Column(Boolean, default=False, nullable=False)
    ai_analysis_status = Column(String, default="not_requested", nullable=False)
    ai_prediction_id = Column(Integer, ForeignKey("modality_predictions.id"), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sender = relationship("User", foreign_keys=[sender_id])
    receiver = relationship("User", foreign_keys=[receiver_id])
    ai_prediction = relationship("ModalityPrediction", foreign_keys=[ai_prediction_id])

class SafeTalkBotMessage(Base):
    __tablename__ = "safetalk_bot_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, ForeignKey("safetalk_conversations.id"), nullable=True)
    
    user_message = Column(Text)
    bot_response = Column(Text)
    intent = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    crisis_level = Column(Integer, default=0)  # 0-10 severity scale
    response_details = Column(JSON, nullable=True)  # Stores techniques, alternatives, etc.
    route = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    topic_label = Column(String, nullable=True)
    response_template_version = Column(String, nullable=True)
    response_policy_version = Column(String, nullable=True)
    response_variant_id = Column(String, nullable=True)
    safety_check_required = Column(Boolean, nullable=True)
    human_contact_recommended = Column(Boolean, nullable=True)
    safety_policy_version = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    conversation = relationship("SafeTalkConversation", back_populates="messages")


class SafeTalkConversation(Base):
    __tablename__ = "safetalk_conversations"
    __table_args__ = (
        Index("ix_safetalk_conversations_user_created", "user_id", "created_at"),
        Index("ix_safetalk_conversations_status", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    topic_label = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")
    context_state = Column(JSON, nullable=True)
    safety_policy_version = Column(String, nullable=True)
    response_policy_version = Column(String, nullable=True)
    legacy_response_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    archived_at = Column(DateTime, nullable=True)

    user = relationship("User")
    messages = relationship("SafeTalkBotMessage", back_populates="conversation")


class JournalEntry(Base):
    __tablename__ = "journal_entries"
    __table_args__ = (
        Index("ix_journal_entries_student_created", "student_id", "created_at"),
        Index("ix_journal_entries_student_entry_date", "student_id", "entry_date"),
        Index("ix_journal_entries_shared", "shared_with_counselor"),
    )

    id = Column(Integer, primary_key=True, index=True)
    journal_entry_id = Column(String, unique=True, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    body = Column(Text, nullable=False)
    mood_tag = Column(String, nullable=True)
    tags_json = Column(JSON, nullable=True)
    entry_date = Column(DateTime, nullable=False)
    ai_analysis_opt_in = Column(Boolean, default=False, nullable=False)
    analysis_status = Column(String, default="not_requested", nullable=False)
    analysis_consent_record_id = Column(Integer, ForeignKey("consent_records.id"), nullable=True)
    shared_with_counselor = Column(Boolean, default=False, nullable=False)
    shared_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    student = relationship("User")
    analysis_consent_record = relationship("ConsentRecord")


class ModelRuntimeHealth(Base):
    __tablename__ = "model_runtime_health"
    __table_args__ = (
        Index("ix_model_runtime_health_model_created", "model_registry_id", "created_at"),
        Index("ix_model_runtime_health_modality_state", "modality", "health_state"),
    )

    id = Column(Integer, primary_key=True, index=True)
    model_registry_id = Column(Integer, ForeignKey("model_registry.id"), nullable=True)
    modality = Column(String, nullable=False)
    health_state = Column(String, nullable=False, default="unknown")
    loader_status = Column(String, nullable=True)
    preprocessing_status = Column(String, nullable=True)
    smoke_test_status = Column(String, nullable=True)
    failure_reason = Column(Text, nullable=True)
    last_successful_inference_at = Column(DateTime, nullable=True)
    last_failed_inference_at = Column(DateTime, nullable=True)
    average_inference_duration_ms = Column(Float, nullable=True)
    predictions_last_24h = Column(Integer, default=0, nullable=False)
    fusion_eligible = Column(Boolean, default=False, nullable=False)
    details_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    model_registry = relationship("ModelRegistry")


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
        Index(
            "uq_model_registry_one_active_modality",
            "modality",
            unique=True,
            postgresql_where=text("is_active = true"),
            sqlite_where=text("is_active = 1"),
        ),
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
    artifact_sha256 = Column(String, nullable=True)
    serializer = Column(String, nullable=True)
    framework_version = Column(String, nullable=True)
    preprocessing_version = Column(String, nullable=True)
    label_mapping_version = Column(String, nullable=True)
    training_dataset_identifier = Column(String, nullable=True)
    training_split_identifier = Column(String, nullable=True)
    evaluation_report_identifier = Column(String, nullable=True)
    model_card_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="discovered", index=True)
    verification_status = Column(String, nullable=True)
    verification_checked_at = Column(DateTime, nullable=True)
    verification_failure_code = Column(String, nullable=True)
    verification_message = Column(Text, nullable=True)
    verification_json = Column(JSON, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    limitations_json = Column(JSON, nullable=True)
    intended_use = Column(Text, nullable=True)
    prohibited_use = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
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
        Index("ix_feature_snapshots_student_modality_source", "student_id", "modality", "source_type", "source_record_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    source_type = Column(String, nullable=False)
    source_record_id = Column(Integer, nullable=True)
    modality = Column(String, nullable=False)
    features_json = Column(JSON, nullable=False)
    preprocessing_version = Column(String, nullable=True)
    source_timestamp = Column(DateTime, nullable=True)
    feature_schema_version = Column(String, nullable=True)
    consent_policy_version = Column(String, nullable=True)
    data_quality_status = Column(String, nullable=True)
    data_quality_flags = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="feature_snapshots")
    predictions = relationship("ModalityPrediction", back_populates="feature_snapshot")


class ModalityPrediction(Base):
    __tablename__ = "modality_predictions"
    __table_args__ = (
        CheckConstraint("probability >= 0 AND probability <= 1", name="ck_modality_predictions_probability_0_1"),
        CheckConstraint("score_0_100 >= 0 AND score_0_100 <= 100", name="ck_modality_predictions_score_0_100"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_modality_predictions_confidence_0_1"),
        CheckConstraint(
            "modality IN ('profile', 'dass21', 'mood', 'text', 'speech', 'face', 'behavioral')",
            name="ck_modality_predictions_canonical_modality",
        ),
        CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed', 'unavailable', 'rejected', 'stale')",
            name="ck_modality_predictions_status",
        ),
        CheckConstraint(
            "output_type IN ('rule_based', 'heuristic', 'machine_learning', 'manual', 'externally_supplied')",
            name="ck_modality_predictions_output_type",
        ),
        Index("ix_modality_predictions_student_created", "student_id", "created_at"),
        Index("ix_modality_predictions_modality", "modality"),
        Index("ix_modality_predictions_source", "source_type", "source_record_id"),
        Index("ix_modality_predictions_student_modality_status", "student_id", "modality", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    modality = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_record_id = Column(Integer, nullable=True)
    feature_snapshot_id = Column(Integer, ForeignKey("feature_snapshots.id"), nullable=True)
    model_registry_id = Column(Integer, ForeignKey("model_registry.id", ondelete="RESTRICT"), nullable=True)
    predicted_class = Column(String, nullable=True)
    probability = Column(Float, nullable=True)
    score_0_100 = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="succeeded")
    is_available = Column(Boolean, nullable=False, default=True)
    failure_code = Column(String, nullable=True)
    failure_message_safe = Column(Text, nullable=True)
    source_timestamp = Column(DateTime, nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    valid_until = Column(DateTime, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    preprocessing_version = Column(String, nullable=True)
    feature_schema_version = Column(String, nullable=True)
    output_type = Column(String, nullable=False, default="machine_learning")
    label = Column(String, nullable=True)
    raw_output_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    consent_policy_version = Column(String, nullable=True)
    evidence_available = Column(Boolean, nullable=False, default=True)
    clinical_use_boundary = Column(String, nullable=False, default="screening_support_only")
    data_quality_status = Column(String, nullable=True)
    data_quality_flags = Column(JSON, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="modality_predictions")
    feature_snapshot = relationship("FeatureSnapshot", back_populates="predictions")
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
    final_probability = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="completed")
    assessment_type = Column(String, nullable=False, default="screening_support")
    model_score = Column(Float, nullable=True)
    model_risk_level = Column(String, nullable=True)
    fusion_config_version = Column(String, nullable=True)
    fusion_config_hash = Column(String, nullable=True)
    threshold_version = Column(String, nullable=True)
    mapping_version = Column(String, nullable=True)
    staleness_policy_version = Column(String, nullable=True)
    coverage_policy_version = Column(String, nullable=True)
    configured_modalities = Column(JSON, nullable=True)
    available_modalities = Column(JSON, nullable=True)
    used_modalities = Column(JSON, nullable=True)
    missing_modalities = Column(JSON, nullable=True)
    excluded_modalities = Column(JSON, nullable=True)
    evidence_coverage = Column(Float, nullable=True)
    coverage_category = Column(String, nullable=True)
    effective_weights = Column(JSON, nullable=True)
    latest_source_timestamp = Column(DateTime, nullable=True)
    oldest_source_timestamp = Column(DateTime, nullable=True)
    evidence_window_json = Column(JSON, nullable=True)
    limitations_json = Column(JSON, nullable=True)
    screening_only = Column(Boolean, default=True, nullable=False)
    model_output_only = Column(Boolean, default=True, nullable=False)
    human_review_status = Column(String, nullable=False, default="not_requested")
    counselor_decision = Column(JSON, nullable=True)
    counselor_override = Column(JSON, nullable=True)
    alert_created = Column(Boolean, default=False, nullable=False)
    data_completeness = Column(JSON, nullable=True)
    safety_override = Column(Boolean, default=False, nullable=False)
    safety_override_reason = Column(Text, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    student = relationship("User", back_populates="risk_assessments")
    fusion_model = relationship("ModelRegistry", back_populates="fused_risk_assessments")
    inputs = relationship("RiskAssessmentInput", back_populates="risk_assessment")


class CounselorAssignment(Base):
    __tablename__ = "counselor_assignments"
    __table_args__ = (
        Index("ix_counselor_assignments_student_active", "student_id", "active"),
        Index("ix_counselor_assignments_counselor_active", "counselor_id", "active"),
        Index("ix_counselor_assignments_assigned_date", "assigned_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(String, unique=True, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    counselor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    assignment_reason = Column(Text, nullable=True)
    assigned_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    end_date = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    student = relationship("User", foreign_keys=[student_id], back_populates="counselor_assignments_as_student")
    counselor = relationship("User", foreign_keys=[counselor_id], back_populates="counselor_assignments_as_counselor")
    assigned_by_user = relationship("User", foreign_keys=[assigned_by])


class CounselorReview(Base):
    __tablename__ = "counselor_reviews"
    __table_args__ = (
        CheckConstraint(
            "status IN ('NEW', 'UNDER_REVIEW', 'FOLLOW_UP_REQUIRED', 'REFERRED', 'CLOSED')",
            name="ck_counselor_reviews_status",
        ),
        Index("ix_counselor_reviews_student_status", "student_id", "status"),
        Index("ix_counselor_reviews_counselor_status", "counselor_id", "status"),
        Index("ix_counselor_reviews_assessment", "assessment_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(String, unique=True, nullable=False, index=True)
    assessment_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    counselor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="NEW", nullable=False)
    review_notes = Column(Text, nullable=True)
    decision = Column(Text, nullable=True)
    risk_judgement = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    assessment = relationship("RiskAssessment")
    student = relationship("User", foreign_keys=[student_id], back_populates="counselor_reviews_as_student")
    counselor = relationship("User", foreign_keys=[counselor_id], back_populates="counselor_reviews_as_counselor")


class CounselorNote(Base):
    __tablename__ = "counselor_notes"
    __table_args__ = (
        Index("ix_counselor_notes_student_active_created", "student_id", "active", "created_at"),
        Index("ix_counselor_notes_counselor_created", "counselor_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(String, unique=True, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    counselor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    note_text = Column(Text, nullable=False)
    note_type = Column(String, nullable=False, default="clinical")
    active = Column(Boolean, default=True, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    student = relationship("User", foreign_keys=[student_id], back_populates="counselor_notes_as_student")
    counselor = relationship("User", foreign_keys=[counselor_id], back_populates="counselor_notes_as_counselor")


class SupportContact(Base):
    __tablename__ = "support_contacts"
    __table_args__ = (
        CheckConstraint(
            "contact_type IN ('assigned_counselor', 'university_unit', 'university_fallback', 'system_fallback', 'crisis_service')",
            name="ck_support_contacts_type",
        ),
        Index("ix_support_contacts_university_active_priority", "university_id", "active", "priority"),
        Index("ix_support_contacts_type_active", "contact_type", "active"),
    )

    id = Column(Integer, primary_key=True, index=True)
    support_contact_id = Column(String, unique=True, nullable=False, index=True)
    university_id = Column(Integer, ForeignKey("universities.id"), nullable=True)
    counselor_profile_id = Column(Integer, ForeignKey("counselor_profiles.id"), nullable=True)
    contact_type = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    telephone_number = Column(String, nullable=True)
    whatsapp_number = Column(String, nullable=True)
    email = Column(String, nullable=True)
    available_days = Column(String, nullable=True)
    available_from = Column(String, nullable=True)
    available_until = Column(String, nullable=True)
    telephone_enabled = Column(Boolean, default=True, nullable=False)
    whatsapp_enabled = Column(Boolean, default=True, nullable=False)
    student_visible = Column(Boolean, default=True, nullable=False)
    emergency_service = Column(Boolean, default=False, nullable=False)
    verified = Column(Boolean, default=False, nullable=False)
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    priority = Column(Integer, default=100, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    university = relationship("University", back_populates="support_contacts")
    counselor_profile = relationship("CounselorProfile", back_populates="support_contacts")
    verified_by_user = relationship("User")


class SupportContactAction(Base):
    __tablename__ = "support_contact_actions"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ('support_panel_opened', 'telephone_action_selected', 'whatsapp_action_selected', 'number_copied', 'support_details_viewed')",
            name="ck_support_contact_actions_type",
        ),
        Index("ix_support_contact_actions_user_created", "user_id", "created_at"),
        Index("ix_support_contact_actions_contact_created", "support_contact_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    action_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    support_contact_id = Column(Integer, ForeignKey("support_contacts.id"), nullable=True)
    contact_type = Column(String, nullable=True)
    action_type = Column(String, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")
    support_contact = relationship("SupportContact")


class CounselorProfileAudit(Base):
    __tablename__ = "counselor_profile_audit"
    __table_args__ = (
        Index("ix_counselor_profile_audit_profile_created", "counselor_profile_id", "created_at"),
        Index("ix_counselor_profile_audit_changed_by", "changed_by_user_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String, unique=True, nullable=False, index=True)
    counselor_profile_id = Column(Integer, ForeignKey("counselor_profiles.id"), nullable=True)
    changed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    change_type = Column(String, nullable=False)
    changed_fields = Column(JSON, nullable=True)
    before_json = Column(JSON, nullable=True)
    after_json = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    counselor_profile = relationship("CounselorProfile")
    changed_by_user = relationship("User")


class RiskAssessmentInput(Base):
    __tablename__ = "risk_assessment_inputs"
    __table_args__ = (
        UniqueConstraint("risk_assessment_id", "modality_prediction_id", name="uq_risk_assessment_input_pair"),
        Index("ix_risk_assessment_inputs_assessment", "risk_assessment_id"),
        Index("ix_risk_assessment_inputs_prediction", "modality_prediction_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    risk_assessment_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=False)
    modality_prediction_id = Column(Integer, ForeignKey("modality_predictions.id"), nullable=True)
    modality = Column(String, nullable=True)
    source_score = Column(Float, nullable=True)
    mapped_score = Column(Float, nullable=True)
    base_weight = Column(Float, nullable=True)
    effective_weight = Column(Float, nullable=True)
    included = Column(Boolean, default=True, nullable=False)
    exclusion_reason = Column(Text, nullable=True)
    source_timestamp = Column(DateTime, nullable=True)
    prediction_age_seconds = Column(Float, nullable=True)
    mapping_version = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)

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


class ConsentRecord(Base):
    __tablename__ = "consent_records"
    __table_args__ = (
        CheckConstraint(
            "consent_type IN ('profile_processing', 'profile_data_storage', 'profile_model_processing', "
            "'dass21_processing', 'mood_processing', "
            "'text_processing', 'voice_processing', 'face_processing', 'behavioral_processing', "
            "'facial_capture', 'facial_model_processing', "
            "'counselor_escalation', 'research_data_use')",
            name="ck_consent_records_known_type",
        ),
        Index("ix_consent_records_user_type_created", "user_id", "consent_type", "created_at"),
        Index("ix_consent_records_type", "consent_type"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    consent_type = Column(String, nullable=False)
    is_granted = Column(Boolean, nullable=False, default=False)
    policy_version = Column(String, nullable=False)
    granted_at = Column(DateTime, nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="consent_records")
