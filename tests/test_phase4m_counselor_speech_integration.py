from datetime import datetime
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.ml.runtime.base import RuntimePredictionResult
from app.models.database_models import ChatMessage, ConsentRecord, CounselorAssignment, ModalityPrediction, ModelRegistry, RiskAssessment, User, UserRole
from app.routes import chat as chat_routes
from app.security import hash_password


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
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
    monkeypatch.setattr(chat_routes, "UPLOAD_DIR", tmp_path / "uploaded_audio")
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_user(db_session, username, role=UserRole.STUDENT):
    user = User(
        email=f"{username}@example.com",
        username=username,
        full_name=f"{username.title()} User",
        hashed_password=hash_password("Password123!"),
        role=role,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def auth_headers(client, username):
    response = client.post("/api/auth/login", json={"username": username, "password": "Password123!"})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def assign(db_session, student, counselor):
    assignment = CounselorAssignment(
        assignment_id=f"asg-{student.id}-{counselor.id}",
        student_id=student.id,
        counselor_id=counselor.id,
        active=True,
    )
    db_session.add(assignment)
    db_session.commit()
    return assignment


def grant_voice_consent(db_session, student):
    record = ConsentRecord(
        user_id=student.id,
        consent_type="voice_processing",
        is_granted=True,
        policy_version="phase4m-test",
        granted_at=datetime.utcnow(),
        source="test",
    )
    db_session.add(record)
    db_session.commit()
    return record


def grant_text_consent(db_session, student, granted=True):
    record = ConsentRecord(
        user_id=student.id,
        consent_type="text_processing",
        is_granted=granted,
        policy_version="phase4v-test",
        granted_at=datetime.utcnow() if granted else None,
        withdrawn_at=None if granted else datetime.utcnow(),
        source="test",
    )
    db_session.add(record)
    db_session.commit()
    return record


def active_text_model(db_session):
    model = ModelRegistry(
        model_name="text-classification-logistic-regression",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/text/text-classification-logistic-regression/1.0.0/text-e8d74030dfff/pipeline.joblib",
        artifact_sha256="test",
        serializer="joblib",
        status="active",
        verification_status="passed",
        is_active=True,
        preprocessing_version="text-runtime-test",
        feature_schema_version="text-feature-test",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


def audio_file(content=b"RIFF0000WAVEdata", content_type="audio/wav"):
    return {"audio": ("voice.wav", BytesIO(content), content_type)}


def test_assigned_chat_lifecycle_ordering_and_read_state(client, db_session):
    student = create_user(db_session, "phase4m-student")
    counselor = create_user(db_session, "phase4m-counselor", UserRole.COUNSELOR)
    outsider = create_user(db_session, "phase4m-outsider", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)
    outsider_headers = auth_headers(client, outsider.username)

    first = client.post(
        "/api/chat/send",
        json={"receiver_id": counselor.id, "message": "Hello counselor", "message_type": "text"},
        headers=student_headers,
    )
    reply = client.post(
        "/api/chat/send",
        json={"receiver_id": student.id, "message": "Hello, I received this.", "message_type": "text"},
        headers=counselor_headers,
    )

    assert first.status_code == 200
    assert first.json()["delivery_status"] == "sent"
    assert reply.status_code == 200
    assert client.get(f"/api/chat/messages/{student.id}", headers=outsider_headers).status_code == 403

    opened = client.get(f"/api/chat/messages/{counselor.id}", headers=student_headers)
    assert opened.status_code == 200
    payload = opened.json()
    assert [item["message"] for item in payload] == ["Hello counselor", "Hello, I received this."]
    assert payload[1]["delivery_status"] == "read"
    assert payload[1]["read_at"] is not None

    conversations = client.get("/api/chat/conversations", headers=counselor_headers)
    assert conversations.status_code == 200
    row = conversations.json()[0]
    assert row["user_id"] == student.id
    assert row["conversation_id"] == f"direct:{student.id}:{counselor.id}"
    assert row["student"]["id"] == student.id
    assert row["latest_message_type"] == "text"


def test_student_text_chat_with_consent_creates_canonical_text_prediction_and_fusion(client, db_session, monkeypatch):
    student = create_user(db_session, "phase4v-text-student")
    counselor = create_user(db_session, "phase4v-text-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    consent = grant_text_consent(db_session, student)
    model = active_text_model(db_session)
    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)

    def fake_predict(db, *, modality, payload):
        assert modality == "text"
        assert payload["text"] == "I feel overwhelmed but I am safe tonight."
        return model, RuntimePredictionResult(
            label="normal",
            probability=0.18,
            confidence=0.82,
            probabilities={"normal": 0.82, "suicidal": 0.18},
            features={"text_length": len(payload["text"]), "contains_raw_text": False},
            metadata={"runtime_strategy": "active_model", "positive_class": "suicidal", "raw_text_stored": False},
        )

    monkeypatch.setattr(chat_routes, "predict_with_active_model", fake_predict)

    sent = client.post(
        "/api/chat/send",
        json={"receiver_id": counselor.id, "message": "I feel overwhelmed but I am safe tonight.", "message_type": "text"},
        headers=student_headers,
    )

    assert sent.status_code == 200
    body = sent.json()
    assert body["ai_analysis_requested"] is True
    assert body["ai_analysis_status"] == "succeeded"
    prediction = db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "text").one()
    assert body["ai_prediction_id"] == prediction.id
    assert prediction.status == "succeeded"
    assert prediction.source_type == "counselor_chat_text_message"
    assert prediction.source_record_id == body["id"]
    assert prediction.model_registry_id == model.id
    assert prediction.probability == pytest.approx(0.18)
    assert prediction.metadata_json["analysis_consent_record_id"] == consent.id
    assert prediction.metadata_json["student_authored"] is True
    assert prediction.metadata_json["bot_generated"] is False
    assert prediction.feature_snapshot.features_json["contains_raw_text"] is False
    assert "overwhelmed" not in str(prediction.feature_snapshot.features_json)

    assessments = db_session.query(RiskAssessment).all()
    assert len(assessments) == 1
    assert assessments[0].trigger_prediction_id == prediction.id
    assert assessments[0].trigger_source == "counselor_chat_text_analysis"

    detail = client.get(f"/api/counselor/student/{student.id}", headers=counselor_headers)
    assert detail.status_code == 200
    text_summary = next(item for item in detail.json()["model_component_summary"] if item["modality"] == "text")
    assert text_summary["status"] == "succeeded"
    assert text_summary["risk_percentage"] == pytest.approx(18.0)
    assert text_summary["evidence_id"] == prediction.id
    text_evidence = next(item for item in detail.json()["modality_evidence"] if item["modality"] == "text")
    assert text_evidence["source_type"] == "counselor_chat_text_message"
    assert text_evidence["model_version"] == "1.0.0"


def test_student_text_chat_without_text_consent_is_delivered_without_text_prediction(client, db_session):
    student = create_user(db_session, "phase4v-no-text-consent-student")
    counselor = create_user(db_session, "phase4v-no-text-consent-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    grant_text_consent(db_session, student, granted=False)
    student_headers = auth_headers(client, student.username)

    sent = client.post(
        "/api/chat/send",
        json={"receiver_id": counselor.id, "message": "This should be delivered only.", "message_type": "text"},
        headers=student_headers,
    )

    assert sent.status_code == 200
    assert sent.json()["ai_analysis_requested"] is False
    assert sent.json()["ai_analysis_status"] == "not_requested"
    assert db_session.query(ChatMessage).count() == 1
    assert db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "text").count() == 0
    assert db_session.query(RiskAssessment).count() == 0


def test_counselor_text_message_is_never_student_text_evidence(client, db_session):
    student = create_user(db_session, "phase4v-counselor-text-student")
    counselor = create_user(db_session, "phase4v-counselor-text-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    grant_text_consent(db_session, student)
    counselor_headers = auth_headers(client, counselor.username)

    sent = client.post(
        "/api/chat/send",
        json={"receiver_id": student.id, "message": "Counselor-authored private support note in chat.", "message_type": "text"},
        headers=counselor_headers,
    )

    assert sent.status_code == 200
    assert sent.json()["ai_analysis_requested"] is False
    assert db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "text").count() == 0


def test_voice_delivery_without_analysis_does_not_require_analysis_consent(client, db_session):
    student = create_user(db_session, "phase4m-voice-student")
    counselor = create_user(db_session, "phase4m-voice-counselor", UserRole.COUNSELOR)
    admin = create_user(db_session, "phase4m-admin", UserRole.ADMIN)
    assign(db_session, student, counselor)
    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)
    admin_headers = auth_headers(client, admin.username)

    delivered = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "false"},
        files=audio_file(),
        headers=student_headers,
    )

    assert delivered.status_code == 200
    body = delivered.json()
    assert body["ai_analysis_requested"] is False
    assert body["ai_analysis_status"] == "not_requested"
    assert db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "speech").count() == 0
    assert client.get(f"/api/chat/messages/{body['id']}/audio", headers=counselor_headers).status_code == 200
    assert client.get(f"/api/chat/messages/{body['id']}/audio", headers=admin_headers).status_code == 403


def test_voice_upload_accepts_browser_webm_opus_mime_parameters(client, db_session):
    student = create_user(db_session, "phase4m-webm-student")
    counselor = create_user(db_session, "phase4m-webm-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    student_headers = auth_headers(client, student.username)

    delivered = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "false"},
        files=audio_file(content=b"webm-opus-placeholder", content_type="audio/webm;codecs=opus"),
        headers=student_headers,
    )

    assert delivered.status_code == 200
    message = db_session.query(ChatMessage).filter(ChatMessage.message_type == "voice").one()
    assert message.metadata_json["original_mime_type"] == "audio/webm;codecs=opus"
    assert message.metadata_json["accepted_mime_type"] == "audio/webm"


def test_voice_analysis_requires_consent_and_remains_unavailable_until_runtime_verified(client, db_session):
    student = create_user(db_session, "phase4m-analysis-student")
    counselor = create_user(db_session, "phase4m-analysis-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    student_headers = auth_headers(client, student.username)

    blocked = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "true"},
        files=audio_file(),
        headers=student_headers,
    )
    assert blocked.status_code == 403
    assert db_session.query(ChatMessage).count() == 0

    consent = grant_voice_consent(db_session, student)
    delivered = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "true"},
        files=audio_file(),
        headers=student_headers,
    )
    assert delivered.status_code == 200
    body = delivered.json()
    assert body["ai_analysis_status"] == "unavailable"
    prediction = db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "speech").one()
    assert prediction.status == "unavailable"
    assert prediction.source_type == "counselor_chat_voice_message"
    assert prediction.source_record_id == body["id"]
    assert prediction.metadata_json["analysis_consent_record_id"] == consent.id
    assert prediction.metadata_json["student_is_audio_speaker"] is True


def test_counselor_voice_is_delivered_but_never_speech_evidence(client, db_session):
    student = create_user(db_session, "phase4m-cv-student")
    counselor = create_user(db_session, "phase4m-cv-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    counselor_headers = auth_headers(client, counselor.username)

    delivered = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(student.id), "analyze_emotional_tone": "true"},
        files=audio_file(),
        headers=counselor_headers,
    )

    assert delivered.status_code == 200
    assert delivered.json()["ai_analysis_requested"] is False
    assert delivered.json()["ai_analysis_status"] == "not_requested"
    assert db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "speech").count() == 0
