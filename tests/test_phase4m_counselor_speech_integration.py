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
from app.models.database_models import ChatMessage, ConsentRecord, CounselorAssignment, ModalityPrediction, User, UserRole
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
