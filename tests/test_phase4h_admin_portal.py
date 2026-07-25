from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import (
    AdminAuditLog,
    AdminReport,
    CounselorAssignment,
    CounselorNote,
    DASS21Assessment,
    ModelRegistry,
    Resource,
    SafeTalkBotMessage,
    University,
    User,
    UserRole,
)
from app.security import hash_password


def db_session_fixture():
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
    yield from db_session_fixture()


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


def create_user(db_session, username, role=UserRole.STUDENT, university=None):
    user = User(
        email=f"{username}@example.com",
        username=username,
        full_name=f"{username.title()} User",
        hashed_password=hash_password("Password123!"),
        role=role,
        university_id=university.id if university else None,
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


def test_admin_dashboard_is_admin_only_and_summary_only(client, db_session):
    admin = create_user(db_session, "phase4h-admin", UserRole.ADMIN)
    student = create_user(db_session, "phase4h-student", UserRole.STUDENT)
    counselor = create_user(db_session, "phase4h-counselor", UserRole.COUNSELOR)
    db_session.add(
        DASS21Assessment(
            user_id=student.id,
            responses=[3] * 21,
            depression_score=21,
            anxiety_score=21,
            stress_score=21,
            total_dass21_score=63,
            is_complete=True,
        )
    )
    db_session.add(
        SafeTalkBotMessage(
            user_id=student.id,
            user_message="raw private safetalk text",
            bot_response="raw private bot response",
            crisis_level=0,
        )
    )
    db_session.add(
        CounselorNote(
            note_id="note-phase4h",
            student_id=student.id,
            counselor_id=counselor.id,
            note_text="private counselor note",
        )
    )
    db_session.commit()

    student_headers = auth_headers(client, student.username)
    admin_headers = auth_headers(client, admin.username)

    assert client.get("/api/admin/dashboard", headers=student_headers).status_code == 403

    response = client.get("/api/admin/dashboard", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_administrators"] == 1
    assert payload["total_students"] == 1
    assert "raw private safetalk text" not in str(payload)
    assert "private counselor note" not in str(payload)
    assert "[3, 3" not in str(payload)


def test_admin_user_university_assignment_and_audit_flow(client, db_session):
    admin = create_user(db_session, "manager", UserRole.ADMIN)
    headers = auth_headers(client, admin.username)

    university_response = client.post(
        "/api/admin/universities",
        json={"university_name": "Phase 4H University", "university_code": "P4H", "district": "Colombo"},
        headers=headers,
    )
    assert university_response.status_code == 200
    university_id = university_response.json()["id"]

    counselor_response = client.post(
        "/api/admin/users",
        json={
            "username": "managed-counselor",
            "email": "managed-counselor@example.com",
            "password": "Password123!",
            "full_name": "Managed Counselor",
            "role": "counselor",
            "university_id": university_id,
        },
        headers=headers,
    )
    student_response = client.post(
        "/api/admin/users",
        json={
            "username": "managed-student",
            "email": "managed-student@example.com",
            "password": "Password123!",
            "full_name": "Managed Student",
            "role": "student",
            "university_id": university_id,
        },
        headers=headers,
    )
    assert counselor_response.status_code == 200
    assert student_response.status_code == 200

    assignment = client.post(
        f"/api/admin/users/{student_response.json()['id']}/assign-counselor",
        json={"counselor_id": counselor_response.json()["id"], "assignment_reason": "phase4h test"},
        headers=headers,
    )
    assert assignment.status_code == 200
    assert db_session.query(CounselorAssignment).count() == 1

    users = client.get("/api/admin/users?role=student", headers=headers)
    universities = client.get("/api/admin/universities", headers=headers)
    audit = client.get("/api/admin/audit", headers=headers)

    assert users.status_code == 200
    assert users.json()["users"][0]["assigned_counselor"] == "Managed Counselor"
    assert universities.json()["universities"][0]["students"] == 1
    assert audit.status_code == 200
    assert any(item["action"] == "assign_counselor" for item in audit.json()["audit_logs"])


def test_admin_model_activation_requires_verified_model(client, db_session):
    admin = create_user(db_session, "model-admin-4h", UserRole.ADMIN)
    headers = auth_headers(client, admin.username)
    model = ModelRegistry(
        model_name="phase4h-text",
        modality="text",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/text/fake/pipeline.joblib",
        status="discovered",
        verification_status=None,
        is_active=False,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    blocked = client.post(f"/api/admin/models/{model.id}/activate", headers=headers)
    assert blocked.status_code == 400

    model.status = "verified"
    model.verification_status = "passed"
    db_session.commit()
    activated = client.post(f"/api/admin/models/{model.id}/activate", headers=headers)

    assert activated.status_code == 200
    assert activated.json()["active"] is True


def test_admin_resources_reports_settings_and_exports(client, db_session):
    admin = create_user(db_session, "platform-admin", UserRole.ADMIN)
    headers = auth_headers(client, admin.username)

    resource = client.post(
        "/api/admin/resources",
        json={
            "title": "Grounding Exercise",
            "category": "coping",
            "resource_type": "exercise",
            "description": "Summary-only exercise metadata",
        },
        headers=headers,
    )
    assert resource.status_code == 200
    approved = client.post(f"/api/admin/resources/{resource.json()['id']}/approve", headers=headers)
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    settings = client.get("/api/admin/settings", headers=headers)
    assert settings.status_code == 200
    assert "feature_flags" in settings.json()["sections"]

    report = client.post("/api/admin/reports", json={"report_type": "usage_summary"}, headers=headers)
    export = client.get(f"/api/admin/reports/{report.json()['id']}/export?format=csv", headers=headers)

    assert report.status_code == 200
    assert export.status_code == 200
    assert "metric,value" in export.text
    assert db_session.query(Resource).count() == 1
    assert db_session.query(AdminReport).count() == 1
    assert db_session.query(AdminAuditLog).count() >= 2
