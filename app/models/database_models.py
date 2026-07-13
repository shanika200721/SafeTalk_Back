from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

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
    depression_severity = Column(String)  # Normal, Mild, Moderate, Severe, Very Severe
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
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    alert_type = Column(String)  # escalation, milestone, behavioral_change
    risk_level = Column(String)
    message = Column(Text)
    is_read = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

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
