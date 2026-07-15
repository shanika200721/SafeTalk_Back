import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.database_models import (
    Alert,
    AlertEvent,
    ModalityPrediction,
    RiskAssessment,
    RiskAssessmentInput,
    User,
    UserRole,
)
from app.services.model_registry import (
    activate_model_version,
    get_active_model,
    register_model,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def create_student(db_session):
    student = User(
        email="student@example.com",
        username="student1",
        full_name="Student One",
        hashed_password="hashed",
        role=UserRole.STUDENT,
    )
    db_session.add(student)
    db_session.commit()
    db_session.refresh(student)
    return student


def test_database_session_creation(db_session):
    assert db_session.is_active
    assert db_session.query(User).count() == 0


def test_model_registry_creation_and_active_model_retrieval(db_session):
    first = register_model(
        db_session,
        model_name="text-risk",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path="models/text-risk/1.0.0/model.joblib",
        is_active=True,
    )
    second = register_model(
        db_session,
        model_name="text-risk",
        modality="text",
        version="1.1.0",
        framework="sklearn",
        artifact_path="models/text-risk/1.1.0/model.joblib",
    )
    activate_model_version(
        db_session,
        model_name="text-risk",
        modality="text",
        version="1.1.0",
    )
    db_session.commit()

    active = get_active_model(db_session, modality="text", model_name="text-risk")

    assert active.id == second.id
    assert active.version == "1.1.0"
    assert db_session.get(type(first), first.id).is_active is False


def test_modality_prediction_risk_assessment_and_input_insertion(db_session):
    student = create_student(db_session)
    model = register_model(
        db_session,
        model_name="daily-checkin-risk",
        modality="behavioral",
        version="1.0.0",
        framework="sklearn",
        artifact_path="models/behavioral/1.0.0/model.joblib",
        is_active=True,
    )
    db_session.commit()

    prediction = ModalityPrediction(
        student_id=student.id,
        modality="behavioral",
        source_type="daily_checkin",
        source_record_id=100,
        model_registry_id=model.id,
        predicted_class="moderate",
        probability=0.72,
        score_0_100=72,
        confidence=0.8,
        explanation_json={"signals": ["stress_level"]},
        processing_time_ms=12.5,
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assessment = RiskAssessment(
        student_id=student.id,
        final_probability=0.72,
        final_score=72,
        risk_level="moderate",
        confidence=0.8,
        data_completeness={"behavioral": True},
        safety_override=False,
        explanation_json={"inputs": 1},
    )
    db_session.add(assessment)
    db_session.commit()
    db_session.refresh(assessment)

    link = RiskAssessmentInput(
        risk_assessment_id=assessment.id,
        modality_prediction_id=prediction.id,
    )
    db_session.add(link)
    db_session.commit()

    assert db_session.query(ModalityPrediction).count() == 1
    assert db_session.query(RiskAssessment).count() == 1
    assert db_session.query(RiskAssessmentInput).count() == 1
    assert assessment.inputs[0].modality_prediction_id == prediction.id


def test_alert_event_insertion(db_session):
    student = create_student(db_session)
    alert = Alert(
        user_id=student.id,
        alert_type="escalation",
        risk_level="HIGH",
        message="High-risk check-in detected",
        is_read=False,
    )
    db_session.add(alert)
    db_session.commit()
    db_session.refresh(alert)

    event = AlertEvent(
        alert_id=alert.id,
        old_status="unread",
        new_status="read",
        changed_by_user_id=student.id,
        notes="Acknowledged in test",
    )
    db_session.add(event)
    db_session.commit()

    assert db_session.query(AlertEvent).count() == 1
    assert alert.events[0].new_status == "read"


def test_transaction_rollback(db_session):
    student = User(
        email="rollback@example.com",
        username="rollback",
        full_name="Rollback Student",
        hashed_password="hashed",
        role=UserRole.STUDENT,
    )
    db_session.add(student)
    db_session.flush()
    db_session.rollback()

    assert db_session.query(User).filter(User.username == "rollback").count() == 0


def test_model_version_uniqueness(db_session):
    register_model(
        db_session,
        model_name="voice-risk",
        modality="voice",
        version="1.0.0",
        framework="pytorch",
        artifact_path="models/voice/1.0.0/model.pt",
    )
    db_session.commit()

    with pytest.raises(IntegrityError):
        register_model(
            db_session,
            model_name="voice-risk",
            modality="voice",
            version="1.0.0",
            framework="pytorch",
            artifact_path="models/voice/1.0.0/duplicate.pt",
        )


def test_score_probability_and_confidence_constraints(db_session):
    student = create_student(db_session)
    model = register_model(
        db_session,
        model_name="face-risk",
        modality="face",
        version="1.0.0",
        framework="tensorflow",
        artifact_path="models/face/1.0.0/model.keras",
    )
    db_session.commit()

    invalid_prediction = ModalityPrediction(
        student_id=student.id,
        modality="face",
        source_type="facial_analysis",
        source_record_id=300,
        model_registry_id=model.id,
        predicted_class="high",
        probability=1.2,
        score_0_100=101,
        confidence=-0.1,
    )
    db_session.add(invalid_prediction)

    with pytest.raises(IntegrityError):
        db_session.commit()
