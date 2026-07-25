import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import FeatureSnapshot, ModelRegistry, ModalityPrediction, User, UserRole
from app.security import hash_password
from app.services import model_registry as registry_service
from app.services.model_registry import (
    activate_model_version,
    apply_verification_result,
    calculate_sha256,
    register_or_update_candidate,
    verify_model_artifact,
)


PROFILE_ARTIFACT = (
    "ml_models/profile/profile-depression-random-forest/1.0.0/"
    "profile-minimal_contextual-66e36ed73f40/pipeline.joblib"
)
TEXT_ARTIFACT = (
    "ml_models/text/text-classification-logistic-regression/1.0.0/"
    "text-e8d74030dfff/pipeline.joblib"
)
SPEECH_ARTIFACT = (
    "ml_models/speech/speech-emotion-random-forest/1.0.0/"
    "speech-99cdbe8dbb57/pipeline.joblib"
)
FACE_ARTIFACT = (
    "ml_models/face/face-emotion-random-forest/1.0.0/"
    "face-image_statistics-b9d5c76172fc/pipeline.joblib"
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


def create_user(db_session, username, role=UserRole.STUDENT, year_of_study=None):
    user = User(
        email=f"{username}@example.com",
        username=username,
        full_name=f"{username.title()} User",
        hashed_password=hash_password("Password123!"),
        role=role,
        year_of_study=year_of_study,
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


def profile_payload(user_id):
    return {
        "user_id": user_id,
        "gpa": 2.4,
        "repeated_subjects": 1,
        "attendance": 80,
        "family_relationship_score": 6,
        "family_support": 4,
        "financial_stress": True,
        "communication_skills": 3,
        "social_connection": 3,
        "sleep_pattern": "Irregular",
        "exercise_frequency": "Rarely",
        "substance_use": "None",
    }


def register_verified_active(db_session, artifact_path):
    model = register_or_update_candidate(db_session, artifact_path)
    result = verify_model_artifact(model)
    apply_verification_result(db_session, model, result)
    assert result.passed is True
    activate_model_version(
        db_session,
        model_name=model.model_name,
        modality=model.modality,
        version=model.version,
    )
    db_session.commit()
    db_session.refresh(model)
    return model


def test_phase4d_model_registry_columns_initialize_additively(db_session):
    columns = {column["name"] for column in inspect(db_session.bind).get_columns("model_registry")}

    assert {
        "artifact_sha256",
        "serializer",
        "framework_version",
        "verification_status",
        "verification_json",
        "approved_at",
        "model_card_path",
        "status",
    } <= columns


def test_verify_selected_profile_and_text_artifacts_pass():
    profile = ModelRegistry(
        model_name="profile-depression-random-forest",
        modality="profile",
        version="1.0.0",
        framework="sklearn",
        artifact_path=PROFILE_ARTIFACT,
        serializer="joblib",
    )
    text = ModelRegistry(
        model_name="text-classification-logistic-regression",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path=TEXT_ARTIFACT,
        serializer="joblib",
    )

    assert verify_model_artifact(profile).passed is True
    assert verify_model_artifact(text).passed is True


def test_speech_and_face_load_but_are_not_activation_eligible():
    speech = ModelRegistry(
        model_name="speech-emotion-random-forest",
        modality="speech",
        version="1.0.0",
        framework="sklearn",
        artifact_path=SPEECH_ARTIFACT,
        serializer="joblib",
    )
    face = ModelRegistry(
        model_name="face-emotion-random-forest",
        modality="face",
        version="1.0.0",
        framework="sklearn",
        artifact_path=FACE_ARTIFACT,
        serializer="joblib",
    )

    speech_result = verify_model_artifact(speech)
    face_result = verify_model_artifact(face)

    assert speech_result.passed is False
    assert speech_result.failure_code == "PREPROCESSOR_MISSING"
    assert speech_result.activation_eligible is False
    assert face_result.passed is False
    assert face_result.failure_code == "PREPROCESSOR_MISSING"
    assert face_result.activation_eligible is False


def test_hash_metadata_serializer_and_activation_gates_fail_closed(db_session, tmp_path, monkeypatch):
    text = ModelRegistry(
        model_name="text-classification-logistic-regression",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path=TEXT_ARTIFACT,
        artifact_sha256="0" * 64,
        serializer="joblib",
    )
    assert verify_model_artifact(text).failure_code == "HASH_MISMATCH"

    model = register_or_update_candidate(db_session, TEXT_ARTIFACT)
    with pytest.raises(ValueError):
        activate_model_version(
            db_session,
            model_name=model.model_name,
            modality=model.modality,
            version=model.version,
        )
    db_session.rollback()

    approved_root = tmp_path / "ml_models"
    approved_root.mkdir()
    monkeypatch.setattr(registry_service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(registry_service, "APPROVED_MODEL_ROOT", approved_root)

    unsupported_path = approved_root / "model.pkl"
    unsupported_path.write_bytes(b"not a supported runtime artifact")
    unsupported = ModelRegistry(
        model_name="unsupported",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path=str(unsupported_path),
        serializer="pickle",
    )
    assert verify_model_artifact(unsupported).failure_code == "UNSUPPORTED_SERIALIZER"

    missing_metadata_path = approved_root / "pipeline.joblib"
    joblib.dump({"not": "a classifier"}, missing_metadata_path)
    missing_metadata = ModelRegistry(
        model_name="missing-metadata",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path=str(missing_metadata_path),
        artifact_sha256=calculate_sha256(missing_metadata_path),
        serializer="joblib",
    )
    assert verify_model_artifact(missing_metadata).failure_code == "METADATA_MISSING"


def test_only_one_active_model_per_modality_is_allowed(db_session):
    first = ModelRegistry(
        model_name="first-text",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/text/first/pipeline.joblib",
        status="active",
        verification_status="passed",
        is_active=True,
    )
    second = ModelRegistry(
        model_name="second-text",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/text/second/pipeline.joblib",
        status="active",
        verification_status="passed",
        is_active=True,
    )
    db_session.add(first)
    db_session.add(second)

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_admin_model_routes_discover_verify_activate_and_require_admin(client, db_session):
    student = create_user(db_session, "model-student")
    admin = create_user(db_session, "model-admin", UserRole.ADMIN)
    student_headers = auth_headers(client, student.username)
    admin_headers = auth_headers(client, admin.username)

    assert client.get("/api/models", headers=student_headers).status_code == 403

    discovered = client.post("/api/models/discover", headers=admin_headers)
    assert discovered.status_code == 200
    text_model = next(item for item in discovered.json() if item["modality"] == "text")

    verified = client.post(f"/api/models/{text_model['id']}/verify", headers=admin_headers)
    assert verified.status_code == 200
    assert verified.json()["passed"] is True

    activated = client.post(f"/api/models/{text_model['id']}/activate", headers=admin_headers)
    assert activated.status_code == 200
    assert activated.json()["is_active"] is True


def test_text_prediction_uses_active_model_without_storing_raw_text(client, db_session):
    model = register_verified_active(db_session, TEXT_ARTIFACT)
    user = create_user(db_session, "text-ml")
    headers = auth_headers(client, user.username)
    grant_consent(client, headers, "text_processing")

    response = client.post(
        "/api/modalities/text/predict",
        json={"text": "I feel sad but I am safe and asking for support"},
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["output_type"] == "machine_learning"
    assert payload["model"]["registry_id"] == model.id
    assert payload["score"] is None
    assert payload["probability"] is not None
    assert "I feel sad" not in str(payload)

    snapshot = db_session.query(FeatureSnapshot).filter_by(student_id=user.id, modality="text").one()
    prediction = db_session.query(ModalityPrediction).filter_by(student_id=user.id, modality="text").one()
    assert snapshot.features_json["contains_raw_text"] is False
    assert "I feel sad" not in str(snapshot.features_json)
    assert prediction.model_registry_id == model.id


def test_profile_active_model_strategy_is_distinct_from_default_heuristic(client, db_session):
    model = register_verified_active(db_session, PROFILE_ARTIFACT)
    user = create_user(db_session, "profile-ml", year_of_study=1)
    headers = auth_headers(client, user.username)
    grant_consent(client, headers, "profile_processing")

    heuristic = client.post(
        "/api/modalities/profile/predict",
        json={"profile_payload": profile_payload(user.id)},
        headers=headers,
    )
    active = client.post(
        "/api/modalities/profile/predict",
        json={"strategy": "active_model", "profile_payload": profile_payload(user.id)},
        headers=headers,
    )

    assert heuristic.status_code == 200
    assert active.status_code == 200
    assert heuristic.json()["output_type"] == "heuristic"
    assert heuristic.json()["score"] is not None
    assert active.json()["output_type"] == "machine_learning"
    assert active.json()["model"]["registry_id"] == model.id
    assert active.json()["probability"] is not None
