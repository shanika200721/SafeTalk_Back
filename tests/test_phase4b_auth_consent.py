import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.db.base import Base
from app.models.database_models import ConsentRecord, User, UserRole
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
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": "Password123!"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def grant_consent(client, headers, consent_type):
    response = client.put(
        f"/api/consents/{consent_type}",
        json={"is_granted": True, "policy_version": "1.0"},
        headers=headers,
    )
    assert response.status_code == 200
    return response


def audio_file(content=b"RIFF0000WAVEdata"):
    return {"audio": ("voice.wav", io.BytesIO(content), "audio/wav")}


def test_public_student_registration_succeeds_and_forces_student(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "student-new@example.com",
            "username": "student-new",
            "full_name": "Student New",
            "password": "Password123!",
            "role": "student",
        },
    )

    assert response.status_code == 200
    assert response.json()["role"] == "student"


@pytest.mark.parametrize("role", ["counselor", "psychiatrist", "admin"])
def test_public_staff_registration_is_rejected(client, role):
    response = client.post(
        "/api/auth/register",
        json={
            "email": f"{role}@example.com",
            "username": role,
            "full_name": f"{role} User",
            "password": "Password123!",
            "role": role,
        },
    )

    assert response.status_code == 400
    assert "Public registration is limited to student accounts" in response.json()["error"]


def test_user_by_id_requires_auth_and_allows_self_or_admin_only(client, db_session):
    student = create_user(db_session, "student-a")
    other_student = create_user(db_session, "student-b")
    admin = create_user(db_session, "admin", UserRole.ADMIN)
    counselor = create_user(db_session, "counselor", UserRole.COUNSELOR)

    assert client.get(f"/api/auth/users/{student.id}").status_code == 401

    student_headers = auth_headers(client, student.username)
    assert client.get(f"/api/auth/users/{student.id}", headers=student_headers).status_code == 200
    assert client.get(f"/api/auth/users/{other_student.id}", headers=student_headers).status_code == 403

    admin_headers = auth_headers(client, admin.username)
    assert client.get(f"/api/auth/users/{student.id}", headers=admin_headers).status_code == 200

    counselor_headers = auth_headers(client, counselor.username)
    assert client.get(f"/api/auth/users/{student.id}", headers=counselor_headers).status_code == 403


def test_consent_grant_withdraw_history_and_unknown_type(client, db_session):
    student = create_user(db_session, "consent-user")
    headers = auth_headers(client, student.username)

    current = client.get("/api/consents", headers=headers)
    assert current.status_code == 200
    assert current.json()["consents"]["voice_processing"]["is_granted"] is False

    grant = grant_consent(client, headers, "voice_processing")
    assert grant.json()["is_granted"] is True
    assert grant.json()["granted_at"] is not None

    withdraw = client.put(
        "/api/consents/voice_processing",
        json={"is_granted": False, "policy_version": "1.0"},
        headers=headers,
    )
    assert withdraw.status_code == 200
    assert withdraw.json()["withdrawn_at"] is not None

    history = client.get("/api/consents/history", headers=headers)
    assert history.status_code == 200
    assert len(history.json()) == 2

    unknown = client.put(
        "/api/consents/not_real",
        json={"is_granted": True, "policy_version": "1.0"},
        headers=headers,
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "UNKNOWN_CONSENT_TYPE"


def test_consent_is_current_user_only(client, db_session):
    first = create_user(db_session, "consent-first")
    second = create_user(db_session, "consent-second")
    headers = auth_headers(client, first.username)

    grant_consent(client, headers, "research_data_use")

    second_record_count = (
        db_session.query(ConsentRecord)
        .filter(
            ConsentRecord.user_id == second.id,
            ConsentRecord.consent_type == "research_data_use",
        )
        .count()
    )
    assert second_record_count == 0


def test_voice_upload_requires_consent_then_sender_and_receiver_can_stream(client, db_session):
    sender = create_user(db_session, "voice-sender")
    receiver = create_user(db_session, "voice-receiver")
    unrelated = create_user(db_session, "voice-unrelated")
    sender_headers = auth_headers(client, sender.username)
    receiver_headers = auth_headers(client, receiver.username)
    unrelated_headers = auth_headers(client, unrelated.username)

    blocked = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(receiver.id)},
        files=audio_file(),
        headers=sender_headers,
    )
    assert blocked.status_code == 403
    assert blocked.json()["error"]["code"] == "CONSENT_REQUIRED"

    grant_consent(client, sender_headers, "voice_processing")
    uploaded = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(receiver.id)},
        files=audio_file(),
        headers=sender_headers,
    )
    assert uploaded.status_code == 200
    message_id = uploaded.json()["id"]

    assert client.get(f"/api/chat/messages/{message_id}/audio").status_code == 401
    assert client.get(f"/api/chat/messages/{message_id}/audio", headers=unrelated_headers).status_code == 403
    assert client.get(f"/api/chat/messages/{message_id}/audio", headers=sender_headers).status_code == 200
    assert client.get(f"/api/chat/messages/{message_id}/audio", headers=receiver_headers).status_code == 200


def test_audio_filename_compat_route_rejects_path_traversal(client, db_session):
    user = create_user(db_session, "traversal-user")
    headers = auth_headers(client, user.username)

    response = client.get("/api/chat/audio/..%5Csecret.wav", headers=headers)
    assert response.status_code == 403


def test_voice_upload_rejects_invalid_mime_and_oversize(client, db_session):
    sender = create_user(db_session, "voice-validate-sender")
    receiver = create_user(db_session, "voice-validate-receiver")
    headers = auth_headers(client, sender.username)
    grant_consent(client, headers, "voice_processing")

    invalid_mime = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(receiver.id)},
        files={"audio": ("voice.wav", io.BytesIO(b"data"), "text/plain")},
        headers=headers,
    )
    assert invalid_mime.status_code == 400

    too_large = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(receiver.id)},
        files=audio_file(b"0" * (chat_routes.MAX_AUDIO_BYTES + 1)),
        headers=headers,
    )
    assert too_large.status_code == 413


def test_dass21_submission_requires_consent(client, db_session):
    student = create_user(db_session, "dass-consent")
    headers = auth_headers(client, student.username)
    payload = {"responses": [0, 1, 2] * 7}

    blocked = client.post("/api/assessments/dass21", json=payload, headers=headers)
    assert blocked.status_code == 403
    assert blocked.json()["error"]["code"] == "CONSENT_REQUIRED"

    grant_consent(client, headers, "dass21_processing")
    allowed = client.post("/api/assessments/dass21", json=payload, headers=headers)
    assert allowed.status_code == 200


def test_consent_table_initializes_additively(db_session):
    inspector = inspect(db_session.bind)
    assert "consent_records" in inspector.get_table_names()
    assert db_session.query(User).count() == 0
