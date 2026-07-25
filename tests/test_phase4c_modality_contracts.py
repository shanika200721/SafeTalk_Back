import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import (
    ChatMessage,
    DASS21Assessment,
    DailyCheckIn,
    FeatureSnapshot,
    ModalityPrediction,
    User,
    UserRole,
)
from app.routes import chat as chat_routes
from app.schemas import (
    CanonicalModality,
    FacePredictionRequest,
    ModalityPredictionResponse,
    PredictionStatus,
    SpeechPredictionRequest,
    TextPredictionRequest,
)
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


def grant_consent(client, headers, consent_type):
    response = client.put(
        f"/api/consents/{consent_type}",
        json={"is_granted": True, "policy_version": "1.0"},
        headers=headers,
    )
    assert response.status_code == 200


def grant_modalities(client, headers, *consent_types):
    for consent_type in consent_types:
        grant_consent(client, headers, consent_type)


def audio_file():
    return {"audio": ("voice.wav", io.BytesIO(b"RIFF0000WAVEdata"), "audio/wav")}


def profile_payload(user_id):
    return {
        "user_id": user_id,
        "gpa": 2.2,
        "repeated_subjects": 1,
        "attendance": 75,
        "family_relationship_score": 6,
        "family_support": 4,
        "financial_stress": True,
        "communication_skills": 3,
        "social_connection": 3,
        "sleep_pattern": "Irregular",
        "exercise_frequency": "Rarely",
        "substance_use": "None",
    }


def checkin_payload():
    return {
        "mood": 2,
        "sleep_hours": 5,
        "exercise_minutes": 0,
        "social_interaction": "Limited",
        "stress_level": 8,
        "anxiety_level": 7,
        "negative_thoughts": True,
        "substance_use_today": False,
        "self_harm_thoughts": False,
        "notes": "private note",
    }


def test_schema_contracts_reject_unknown_modality_and_path_like_inputs():
    with pytest.raises(ValueError):
        CanonicalModality("voice")

    with pytest.raises(ValueError):
        SpeechPredictionRequest(upload_reference_id="C:\\private\\voice.wav")

    with pytest.raises(ValueError):
        FacePredictionRequest(source_reference_id="../image.png")

    response = ModalityPredictionResponse(
        prediction_id=1,
        user_id=1,
        modality="text",
        status=PredictionStatus.UNAVAILABLE,
        is_available=False,
        output_type="machine_learning",
        score=None,
        probability=None,
        confidence=None,
        generated_at="2026-07-25T00:00:00",
        model={},
        data_quality={"status": "not_evaluated", "flags": []},
        evidence={"available": False, "source_type": "direct_text_payload"},
    )
    assert response.score is None


def test_text_request_length_limit():
    with pytest.raises(ValueError):
        TextPredictionRequest(text="")
    with pytest.raises(ValueError):
        TextPredictionRequest(text="x" * 5001)


def test_availability_reports_runtime_truth(client, db_session):
    user = create_user(db_session, "availability")
    headers = auth_headers(client, user.username)

    response = client.get("/api/modalities/availability", headers=headers)

    assert response.status_code == 200
    modalities = {item["modality"]: item for item in response.json()["modalities"]}
    assert modalities["profile"]["contract_available"] is True
    assert modalities["text"]["runtime_model_active"] is False
    assert modalities["speech"]["runtime_model_active"] is False
    assert modalities["face"]["runtime_model_active"] is False
    assert modalities["behavioral"]["runtime_model_active"] is False


def test_legacy_profile_submission_creates_heuristic_prediction(client, db_session):
    user = create_user(db_session, "profile-user")
    headers = auth_headers(client, user.username)
    grant_modalities(client, headers, "profile_processing")

    response = client.post("/api/assessments/profile", json=profile_payload(user.id), headers=headers)

    assert response.status_code == 200
    prediction = db_session.query(ModalityPrediction).filter_by(student_id=user.id, modality="profile").one()
    assert prediction.status == "succeeded"
    assert prediction.output_type == "heuristic"
    assert prediction.probability is None
    assert prediction.confidence is None
    assert prediction.score_0_100 is not None


def test_dass21_metadata_stored_scoring_unchanged_and_rule_based_prediction(client, db_session):
    user = create_user(db_session, "dass-user")
    headers = auth_headers(client, user.username)
    grant_modalities(client, headers, "dass21_processing")
    responses = [0, 1, 2] * 7

    response = client.post("/api/assessments/dass21", json={"responses": responses}, headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_dass21_score"] == 42
    assert payload["metadata"]["questionnaire_version"] == "DASS-21"
    assert payload["metadata"]["scoring_version"] == "1.0.0"
    assert payload["metadata"]["score_multiplier"] == 2.0
    assert payload["metadata"]["completed_item_count"] == 21
    assert payload["metadata"]["is_complete"] is True

    stored = db_session.query(DASS21Assessment).filter_by(user_id=user.id).one()
    assert stored.total_dass21_score == 42
    prediction = db_session.query(ModalityPrediction).filter_by(student_id=user.id, modality="dass21").one()
    assert prediction.output_type == "rule_based"
    assert prediction.status == "succeeded"


def test_old_dass21_records_still_predict_with_backfilled_metadata(client, db_session):
    user = create_user(db_session, "old-dass-user")
    headers = auth_headers(client, user.username)
    grant_modalities(client, headers, "dass21_processing")
    old = DASS21Assessment(
        user_id=user.id,
        responses=[0] * 21,
        depression_score=0,
        anxiety_score=0,
        stress_score=0,
        total_dass21_score=0,
        depression_severity="Normal",
        anxiety_severity="Normal",
        stress_severity="Normal",
    )
    db_session.add(old)
    db_session.commit()
    db_session.refresh(old)

    response = client.post("/api/modalities/dass21/predict", json={"assessment_id": old.id}, headers=headers)

    assert response.status_code == 200
    db_session.refresh(old)
    assert old.scoring_version == "1.0.0"
    assert response.json()["score"] == 0


def test_mood_prediction_references_owned_checkin_and_blocks_other_user(client, db_session):
    owner = create_user(db_session, "mood-owner")
    other = create_user(db_session, "mood-other")
    owner_headers = auth_headers(client, owner.username)
    other_headers = auth_headers(client, other.username)
    grant_modalities(client, owner_headers, "mood_processing")
    grant_modalities(client, other_headers, "mood_processing")

    created = client.post("/api/checkin/today", json=checkin_payload(), headers=owner_headers)
    assert created.status_code == 200
    checkin_id = created.json()["id"]

    blocked = client.post("/api/modalities/mood/predict", json={"checkin_id": checkin_id}, headers=other_headers)
    assert blocked.status_code == 403

    allowed = client.post("/api/modalities/mood/predict", json={"checkin_id": checkin_id}, headers=owner_headers)
    assert allowed.status_code == 200
    assert allowed.json()["evidence"]["source_id"] == checkin_id
    assert allowed.json()["output_type"] == "heuristic"


def test_text_prediction_is_unavailable_requires_consent_and_does_not_return_raw_text(client, db_session):
    sender = create_user(db_session, "text-sender")
    receiver = create_user(db_session, "text-receiver")
    headers = auth_headers(client, sender.username)

    chat = client.post(
        "/api/chat/send",
        json={"receiver_id": receiver.id, "message": "ordinary direct message"},
        headers=headers,
    )
    assert chat.status_code == 200

    blocked = client.post("/api/modalities/text/predict", json={"text": "sensitive raw text"}, headers=headers)
    assert blocked.status_code == 403

    grant_modalities(client, headers, "text_processing")
    response = client.post("/api/modalities/text/predict", json={"text": "sensitive raw text"}, headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["score"] is None
    assert payload["failure_code"] == "MODEL_NOT_ACTIVE"
    assert "sensitive raw text" not in str(payload)

    snapshot = db_session.query(FeatureSnapshot).filter_by(student_id=sender.id, modality="text").one()
    assert "sensitive raw text" not in str(snapshot.features_json)


def test_speech_unavailable_blocks_unauthorized_audio_and_never_returns_raw_path(client, db_session):
    sender = create_user(db_session, "speech-sender")
    receiver = create_user(db_session, "speech-receiver")
    unrelated = create_user(db_session, "speech-unrelated")
    sender_headers = auth_headers(client, sender.username)
    unrelated_headers = auth_headers(client, unrelated.username)
    grant_modalities(client, sender_headers, "voice_processing")
    grant_modalities(client, unrelated_headers, "voice_processing")

    uploaded = client.post(
        "/api/chat/send-voice",
        data={"receiver_id": str(receiver.id)},
        files=audio_file(),
        headers=sender_headers,
    )
    assert uploaded.status_code == 200
    message_id = uploaded.json()["id"]

    blocked = client.post("/api/modalities/speech/predict", json={"chat_message_id": message_id}, headers=unrelated_headers)
    assert blocked.status_code == 403

    response = client.post("/api/modalities/speech/predict", json={"chat_message_id": message_id}, headers=sender_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["score"] is None
    assert payload["failure_code"] == "MODEL_NOT_ACTIVE"
    assert ".wav" not in str(payload)


def test_face_and_behavioral_return_honest_unavailable_contracts(client, db_session):
    user = create_user(db_session, "inactive-modalities")
    headers = auth_headers(client, user.username)
    grant_modalities(client, headers, "face_processing", "behavioral_processing")

    face = client.post("/api/modalities/face/predict", json={}, headers=headers)
    behavioral = client.post("/api/modalities/behavioral/predict", json={}, headers=headers)

    assert face.status_code == 200
    assert face.json()["status"] == "unavailable"
    assert face.json()["score"] is None
    assert face.json()["failure_code"] == "MODEL_NOT_ACTIVE"

    assert behavioral.status_code == 200
    assert behavioral.json()["status"] == "unavailable"
    assert behavioral.json()["score"] is None
    assert behavioral.json()["failure_code"] == "MODALITY_NOT_VALIDATED"


def test_prediction_retrieval_self_admin_and_generic_counselor_denied(client, db_session):
    student = create_user(db_session, "retrieve-student")
    other = create_user(db_session, "retrieve-other")
    admin = create_user(db_session, "retrieve-admin", UserRole.ADMIN)
    counselor = create_user(db_session, "retrieve-counselor", UserRole.COUNSELOR)
    student_headers = auth_headers(client, student.username)
    other_headers = auth_headers(client, other.username)
    admin_headers = auth_headers(client, admin.username)
    counselor_headers = auth_headers(client, counselor.username)
    grant_modalities(client, student_headers, "profile_processing")

    created = client.post("/api/assessments/profile", json=profile_payload(student.id), headers=student_headers)
    assert created.status_code == 200
    prediction = db_session.query(ModalityPrediction).filter_by(student_id=student.id, modality="profile").one()

    assert client.get(f"/api/modalities/predictions/{prediction.id}", headers=student_headers).status_code == 200
    assert client.get(f"/api/modalities/predictions/{prediction.id}", headers=admin_headers).status_code == 200
    assert client.get(f"/api/modalities/predictions/{prediction.id}", headers=other_headers).status_code == 403
    assert client.get(f"/api/modalities/predictions/{prediction.id}", headers=counselor_headers).status_code == 403
    assert client.get(f"/api/modalities/predictions?user_id={student.id}", headers=counselor_headers).status_code == 403


def test_phase4c_schema_columns_initialize_additively(db_session):
    inspector = inspect(db_session.bind)
    dass_columns = {column["name"] for column in inspector.get_columns("dass21_assessments")}
    prediction_columns = {column["name"] for column in inspector.get_columns("modality_predictions")}
    snapshot_columns = {column["name"] for column in inspector.get_columns("feature_snapshots")}

    assert {"scoring_version", "completed_item_count", "is_complete", "consent_policy_version"} <= dass_columns
    assert {"status", "is_available", "failure_code", "output_type", "feature_snapshot_id"} <= prediction_columns
    assert {"source_timestamp", "feature_schema_version", "data_quality_status"} <= snapshot_columns
