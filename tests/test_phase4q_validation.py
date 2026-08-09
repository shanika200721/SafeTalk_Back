from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.ml.runtime.behavioral import predict_behavioral_signal
from app.models.database_models import BehavioralTelemetryEvent, ConsentRecord, User, UserRole
from app.security import hash_password
from app.services.fusion import run_controlled_fusion
from app.services.modalities import create_behavioral_prediction


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
        last_login_at=datetime.utcnow() - timedelta(days=1),
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
    db_session.add(
        ConsentRecord(
            user_id=user.id,
            consent_type=consent_type,
            is_granted=True,
            policy_version="phase4q-test",
            granted_at=datetime.utcnow(),
            source="test",
        )
    )
    db_session.commit()


def test_behavioral_telemetry_requires_consent_and_rejects_raw_payloads(client, db_session):
    user = create_user(db_session, "phase4q-telemetry")
    headers = auth_headers(client, user.username)
    payload = {
        "event_type": "session_summary",
        "source_page": "/dashboard",
        "session_duration_seconds": 42.5,
        "interaction_count": 7,
        "response_latency_ms": 1200,
        "typing_active_ms": 8000,
        "typing_pause_count": 2,
        "typed_character_count": 32,
    }

    blocked = client.post("/api/modalities/behavioral/telemetry", json=payload, headers=headers)
    assert blocked.status_code == 403

    grant_consent(db_session, user, "behavioral_processing")
    raw = client.post(
        "/api/modalities/behavioral/telemetry",
        json={**payload, "metadata": {"raw_text": "do not store me"}},
        headers=headers,
    )
    assert raw.status_code == 200

    accepted = client.post("/api/modalities/behavioral/telemetry", json=payload, headers=headers)
    assert accepted.status_code == 200
    assert "typing_active_ms" in accepted.json()["stored_fields"]
    events = db_session.query(BehavioralTelemetryEvent).order_by(BehavioralTelemetryEvent.id).all()
    assert len(events) == 2
    event = events[0]
    assert event.metadata_json["privacy"] == "aggregate_only_no_raw_keystrokes_no_pointer_paths"
    assert "raw_text" not in event.metadata_json["client_metadata"]


def test_behavioral_anomaly_uses_aggregate_telemetry_but_remains_fusion_excluded(db_session):
    user = create_user(db_session, "phase4q-anomaly")
    now = datetime.utcnow()
    for index in range(9):
        db_session.add(
            BehavioralTelemetryEvent(
                student_id=user.id,
                event_type="session_summary",
                source_page="/dashboard",
                session_id=f"baseline-{index}",
                session_duration_seconds=600,
                interaction_count=50,
                response_latency_ms=500,
                typing_active_ms=30000,
                typing_pause_count=4,
                typed_character_count=100,
                consent_policy_version="phase4q-test",
                created_at=now - timedelta(days=10 + index),
            )
        )
    db_session.commit()

    signal = predict_behavioral_signal(db_session, user, now=now)
    prediction = create_behavioral_prediction(db_session, user)
    fusion = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=now)

    assert signal.features["baseline_telemetry_events"] == 9
    assert signal.features["telemetry_event_drop_component"] > 0
    assert "raw_keystroke_content" in signal.provenance["fields_not_available"]
    assert prediction.metadata_json["risk_mapping_status"] == "no_validated_risk_mapping"
    assert any(item["modality"] == "behavioral" and item["reason"] == "no_validated_risk_mapping" for item in fusion["evidence"]["excluded_modalities"])
