from datetime import datetime, timedelta
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import (
    ChatMessage,
    ConsentRecord,
    CounselorAssignment,
    JournalEntry,
    ModalityPrediction,
    SafeTalkBotMessage,
    User,
    UserRole,
)
from app.security import hash_password
from app.services.fusion import run_controlled_fusion
from app.services.modalities import create_feature_snapshot, create_prediction, verify_owned_chat_message
from app.services.safetalk_safety import RESPONSE_POLICY_VERSION, render_safety_response, route_safetalk_message


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


def grant_consent(db_session, user, consent_type):
    record = ConsentRecord(
        user_id=user.id,
        consent_type=consent_type,
        is_granted=True,
        policy_version="phase4k-test",
        granted_at=datetime.utcnow(),
        source="test",
    )
    db_session.add(record)
    db_session.commit()
    return record


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


def test_safetalk_conversation_reopens_and_continues_without_mixing(client, db_session):
    student = create_user(db_session, "phase4k-talk")
    other = create_user(db_session, "phase4k-other")
    headers = auth_headers(client, student.username)
    other_headers = auth_headers(client, other.username)

    first = client.post("/api/bot/safetalk/conversations", json={}, headers=headers).json()["id"]
    second = client.post("/api/bot/safetalk/conversations", json={}, headers=headers).json()["id"]

    client.post(f"/api/bot/safetalk/conversations/{first}/messages", json={"message": "hlo"}, headers=headers)
    client.post(f"/api/bot/safetalk/conversations/{second}/messages", json={"message": "today is my birthday but no one wished me"}, headers=headers)
    client.post(f"/api/bot/safetalk/conversations/{first}/messages", json={"message": "im sad"}, headers=headers)

    opened_first = client.get(f"/api/bot/safetalk/conversations/{first}", headers=headers)
    opened_second = client.get(f"/api/bot/safetalk/conversations/{second}", headers=headers)
    denied = client.get(f"/api/bot/safetalk/conversations/{first}", headers=other_headers)

    assert opened_first.status_code == 200
    assert [item["user_message"] for item in opened_first.json()["messages"]] == ["hlo", "im sad"]
    assert [item["user_message"] for item in opened_second.json()["messages"]] == ["today is my birthday but no one wished me"]
    assert opened_first.json()["messages"][0]["response_policy_version"] == RESPONSE_POLICY_VERSION
    assert denied.status_code == 404

    conversations = client.get("/api/bot/safetalk/conversations", headers=headers).json()
    assert all("last_message_preview" in item for item in conversations)
    assert any(item["id"] == first and item["message_count"] == 2 for item in conversations)


def test_safetalk_response_quality_variants_and_crisis_priority():
    greeting = route_safetalk_message("hiii")
    birthday = route_safetalk_message("today is my birthday but no one wished me")
    unclear_one = render_safety_response(route_safetalk_message("..."), context_state={})
    unclear_two = render_safety_response(
        route_safetalk_message("???"),
        context_state={"previous_response_variant_id": unclear_one["variant_id"]},
    )
    crisis = render_safety_response(route_safetalk_message("I want to kill myself"))

    assert greeting.route == "greeting"
    assert birthday.topic_label == "Loneliness"
    assert "remembered or cared for" in render_safety_response(birthday)["message"].lower()
    assert unclear_one["message"] != unclear_two["message"]
    assert "safetalk cannot contact emergency services" in crisis["message"].lower()


def test_voice_delivery_is_separate_from_analysis_and_counselor_voice_excluded(client, db_session):
    student = create_user(db_session, "voice-student")
    counselor = create_user(db_session, "voice-counselor", UserRole.COUNSELOR)
    assign(db_session, student, counselor)
    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)

    delivered_without_analysis = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "false"},
        files={"audio": ("note.wav", BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
        headers=student_headers,
    )
    assert delivered_without_analysis.status_code == 200
    assert delivered_without_analysis.json()["ai_analysis_status"] == "not_requested"

    grant_consent(db_session, student, "voice_processing")
    no_analysis = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "false"},
        files={"audio": ("note.wav", BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
        headers=student_headers,
    )
    assert no_analysis.status_code == 200
    assert no_analysis.json()["ai_analysis_status"] == "not_requested"
    with_analysis = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(counselor.id), "analyze_emotional_tone": "true"},
        files={"audio": ("note.wav", BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
        headers=student_headers,
    )
    assert with_analysis.status_code == 200
    assert with_analysis.json()["ai_analysis_status"] == "unavailable"
    assert db_session.query(ModalityPrediction).filter(ModalityPrediction.modality == "speech").count() == 1

    counselor_message = ChatMessage(sender_id=counselor.id, receiver_id=student.id, message="voice_x.wav", message_type="voice")
    db_session.add(counselor_message)
    db_session.commit()
    with pytest.raises(Exception):
        verify_owned_chat_message(db_session, counselor_message.id, student)

    assert client.get(f"/api/chat/messages/{with_analysis.json()['id']}/audio", headers=counselor_headers).status_code in {200, 404}


def test_journal_persistence_privacy_and_opt_in_text_modality(client, db_session):
    student = create_user(db_session, "journal-phase4k")
    counselor = create_user(db_session, "journal-phase4k-c", UserRole.COUNSELOR)
    headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)

    created = client.post(
        "/api/student/journal",
        json={
            "title": "Private",
            "mood": "overwhelmed",
            "tags": ["study"],
            "content": "I feel overwhelmed by project work but I am writing this privately.",
            "ai_analysis_opt_in": True,
            "share_with_counselor": False,
        },
        headers=headers,
    )

    assert created.status_code == 200
    assert created.json()["privacy"] == "private"
    assert created.json()["analysis_status"] == "unavailable"
    assert db_session.query(JournalEntry).count() == 1
    prediction = db_session.query(ModalityPrediction).filter(ModalityPrediction.source_type == "journal_entry").first()
    assert prediction.modality == "text"
    assert prediction.status == "unavailable"

    listing = client.get("/api/student/journal", headers=headers).json()
    assert "content_preview" in listing["items"][0]
    assert "content" not in listing["items"][0]
    detail = client.get(f"/api/student/journal/{created.json()['id']}", headers=headers).json()
    assert "writing this privately" in detail["content"]
    assert client.get("/api/student/journal", headers=counselor_headers).status_code == 403


def test_runtime_status_admin_only_and_no_sensitive_content(client, db_session):
    admin = create_user(db_session, "runtime-admin", UserRole.ADMIN)
    student = create_user(db_session, "runtime-student")
    admin_headers = auth_headers(client, admin.username)
    student_headers = auth_headers(client, student.username)

    assert client.get("/api/admin/models/runtime-status", headers=student_headers).status_code == 403
    response = client.get("/api/admin/models/runtime-status", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()
    modalities = {item["modality"]: item for item in payload["runtime_status"]}
    assert modalities["behavioral"]["health_state"] == "unavailable"
    assert "sensitive source content" in payload["privacy"].lower()


def test_fusion_includes_only_eligible_speech_and_excludes_unverified_face(db_session):
    student = create_user(db_session, "fusion-phase4k")
    now = datetime.utcnow()
    for modality, score, source_type in [
        ("profile", 20.0, "profile_assessment"),
        ("dass21", 25.0, "dass21_assessment"),
        ("mood", 30.0, "daily_checkin"),
        ("speech", 80.0, "chat_voice_message"),
        ("face", 90.0, "face_capture"),
    ]:
        snapshot = create_feature_snapshot(
            db_session,
            student_id=student.id,
            modality=modality,
            source_type=source_type,
            source_record_id=None,
            source_timestamp=now,
            feature_schema_version=f"{modality}-schema-test",
            preprocessing_version=f"{modality}-pre-test",
            features_json={"test": True},
        )
        create_prediction(
            db_session,
            student_id=student.id,
            modality=modality,
            status_value="succeeded",
            is_available=True,
            output_type="heuristic" if modality == "profile" else "machine_learning",
            source_type=source_type,
            source_record_id=None,
            source_timestamp=now,
            feature_snapshot=snapshot,
            score=score,
            probability=score / 100,
            confidence=0.8,
            label="test",
            metadata_json={"fusion_eligible": modality == "speech"},
            model_name=f"{modality}-model",
            model_version="1.0.0",
            preprocessing_version=f"{modality}-pre-test",
            feature_schema_version=f"{modality}-schema-test",
            valid_for_hours=24,
        )
    db_session.commit()

    result = run_controlled_fusion(db_session, user_id=student.id, persist=False, assessment_time=now + timedelta(minutes=1))
    used = set(result["evidence"]["used_modalities"])
    excluded = {(item["modality"], item["reason"]) for item in result["evidence"]["excluded_modalities"]}

    assert "speech" in used
    assert ("face", "insufficient_model_reliability") in excluded
    assert "behavioral" in result["evidence"]["missing_modalities"] or any(item["modality"] == "behavioral" for item in result["evidence"]["excluded_modalities"])
