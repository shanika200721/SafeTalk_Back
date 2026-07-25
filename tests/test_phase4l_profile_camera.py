from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import CounselorAssignment, ModalityPrediction, ProfileAssessment, User, UserRole
from app.security import hash_password


def _db_session_fixture():
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


import pytest


@pytest.fixture()
def db_session():
    yield from _db_session_fixture()


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
        year_of_study=2,
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


def grant(client, headers, consent_type):
    response = client.put(
        f"/api/consents/{consent_type}",
        json={"is_granted": True, "policy_version": "1.0"},
        headers=headers,
    )
    assert response.status_code == 200


def valid_responses():
    return {
        "current_year_of_study": "year 2",
        "academic_workload": "heavy",
        "financial_strain": "moderate",
        "family_support": "some",
        "friend_support": "some",
        "living_situation": "hostel",
        "preferred_support_channel": "online",
        "self_reported_anxiety": "no",
        "self_reported_panic_attack": "no",
        "academic_performance_category": "average",
        "social_isolation": "sometimes",
        "previous_counseling_support": "prefer_not_to_say",
    }


def assign(db_session, student, counselor):
    assignment = CounselorAssignment(
        assignment_id=f"phase4l-{student.id}-{counselor.id}",
        student_id=student.id,
        counselor_id=counselor.id,
        assigned_by=counselor.id,
        assignment_reason="Phase 4L summary test",
        active=True,
    )
    db_session.add(assignment)
    db_session.commit()
    return assignment


def test_questionnaire_contract_and_validation(client, db_session):
    student = create_user(db_session, "phase4l-student")
    headers = auth_headers(client, student.username)
    grant(client, headers, "profile_data_storage")

    questions = client.get("/api/student/profile-assessment/questions", headers=headers)
    assert questions.status_code == 200
    payload = questions.json()
    assert payload["questionnaire_version"] == "profile_assessment_v2"
    assert payload["model_feature_order"] == ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
    assert {item["category"] for item in payload["questions"]} >= {"academic_context", "financial_context", "support_preferences"}

    invalid = client.post(
        "/api/student/profile-assessment/draft",
        json={"questionnaire_version": "profile_assessment_v2", "responses": {"unknown": "yes"}},
        headers=headers,
    )
    assert invalid.status_code == 400

    draft = client.post(
        "/api/student/profile-assessment/draft",
        json={"questionnaire_version": "profile_assessment_v2", "responses": {"current_year_of_study": "year 2"}},
        headers=headers,
    )
    assert draft.status_code == 200
    assert draft.json()["status"] == "draft"


def test_submit_persists_versioned_assessment_without_fake_score(client, db_session):
    student = create_user(db_session, "phase4l-submit")
    headers = auth_headers(client, student.username)
    grant(client, headers, "profile_data_storage")
    grant(client, headers, "profile_model_processing")

    response = client.post(
        "/api/student/profile-assessment/submit",
        json={"questionnaire_version": "profile_assessment_v2", "responses": valid_responses()},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["prediction_status"] in {"unavailable", "succeeded"}

    assessment = db_session.query(ProfileAssessment).filter(ProfileAssessment.user_id == student.id).one()
    assert assessment.questionnaire_version == "profile_assessment_v2"
    assert assessment.responses_json["previous_counseling_support"] == "prefer_not_to_say"
    assert assessment.normalized_features_json["feature_order"] == ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
    prediction = db_session.query(ModalityPrediction).filter(ModalityPrediction.id == assessment.prediction_id).one()
    assert prediction.score_0_100 is None
    assert prediction.source_type == "profile_assessment"


def test_counselor_summary_is_assignment_bound_and_not_raw_dump(client, db_session):
    student = create_user(db_session, "phase4l-summary-student")
    counselor = create_user(db_session, "phase4l-summary-counselor", UserRole.COUNSELOR)
    other = create_user(db_session, "phase4l-summary-other", UserRole.COUNSELOR)
    student_headers = auth_headers(client, student.username)
    grant(client, student_headers, "profile_data_storage")
    client.post(
        "/api/student/profile-assessment/submit",
        json={"questionnaire_version": "profile_assessment_v2", "responses": valid_responses()},
        headers=student_headers,
    )
    assign(db_session, student, counselor)

    other_headers = auth_headers(client, other.username)
    denied = client.get(f"/api/counselor/student/{student.id}/profile-summary", headers=other_headers)
    assert denied.status_code == 403

    counselor_headers = auth_headers(client, counselor.username)
    allowed = client.get(f"/api/counselor/student/{student.id}/profile-summary", headers=counselor_headers)
    assert allowed.status_code == 200
    text = str(allowed.json())
    assert "previous_counseling_support" not in text
    assert "prefer_not_to_say" not in text
    assert "financial_strain" in text


def test_admin_aggregate_and_facial_status_are_privacy_preserving(client, db_session):
    admin = create_user(db_session, "phase4l-admin", UserRole.ADMIN)
    student = create_user(db_session, "phase4l-camera-student")
    student_headers = auth_headers(client, student.username)
    admin_headers = auth_headers(client, admin.username)
    grant(client, student_headers, "facial_capture")

    face = client.get("/api/student/facial-analysis/status", headers=student_headers)
    assert face.status_code == 200
    payload = face.json()
    assert payload["runtime_state"] == "inactive"
    assert payload["consent"]["facial_capture"] is True
    assert "score" not in payload

    stats = client.get("/api/admin/profile-assessments/statistics", headers=admin_headers)
    assert stats.status_code == 200
    assert stats.json()["raw_responses_included"] is False
