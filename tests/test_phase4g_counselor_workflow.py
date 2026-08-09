from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import (
    CounselorAssignment,
    CounselorNote,
    DASS21Assessment,
    DailyCheckIn,
    RiskAssessment,
    User,
    UserRole,
)
from app.security import hash_password


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
        department="Computing",
        year_of_study=4,
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


def assign(db_session, student, counselor, active=True):
    assignment = CounselorAssignment(
        assignment_id=f"asg-{student.id}-{counselor.id}-{active}",
        student_id=student.id,
        counselor_id=counselor.id,
        assigned_by=counselor.id,
        assignment_reason="Phase 4G test assignment",
        active=active,
    )
    db_session.add(assignment)
    db_session.commit()
    db_session.refresh(assignment)
    return assignment


def seed_student_signals(db_session, student, risk_level="HIGH", score=76):
    now = datetime.utcnow()
    risk = RiskAssessment(
        student_id=student.id,
        final_probability=score / 100,
        final_score=score,
        risk_level=risk_level,
        confidence=0.82,
        assessment_type="screening_support",
        model_score=score,
        model_risk_level=risk_level,
        evidence_coverage=0.67,
        coverage_category="partial",
        available_modalities=["dass21", "mood"],
        used_modalities=["dass21", "mood"],
        missing_modalities=["text", "speech", "face", "behavioral"],
        screening_only=True,
        model_output_only=True,
    )
    db_session.add(risk)
    dass = DASS21Assessment(
        user_id=student.id,
        responses=[1] * 21,
        depression_score=18,
        anxiety_score=16,
        stress_score=20,
        total_dass21_score=54,
        depression_severity="Moderate",
        anxiety_severity="Severe",
        stress_severity="Moderate",
        is_complete=True,
        created_at=now - timedelta(days=1),
    )
    checkin = DailyCheckIn(
        user_id=student.id,
        mood=2,
        mood_description="Low",
        sleep_hours=5.5,
        exercise_minutes=10,
        social_interaction="Limited",
        stress_level=8,
        anxiety_level=7,
        negative_thoughts=True,
        substance_use_today=False,
        self_harm_thoughts=False,
        notes="Exam pressure",
        created_at=now,
    )
    db_session.add_all([dass, checkin])
    db_session.commit()
    db_session.refresh(risk)
    return risk


def setup_people(db_session):
    admin = create_user(db_session, "admin4g", UserRole.ADMIN)
    counselor = create_user(db_session, "counselor4g", UserRole.COUNSELOR)
    other_counselor = create_user(db_session, "othercounselor4g", UserRole.COUNSELOR)
    student = create_user(db_session, "student4g", UserRole.STUDENT)
    other_student = create_user(db_session, "otherstudent4g", UserRole.STUDENT)
    assign(db_session, student, counselor)
    risk = seed_student_signals(db_session, student)
    seed_student_signals(db_session, other_student, risk_level="LOW", score=22)
    return admin, counselor, other_counselor, student, other_student, risk


def test_counselor_dashboard_and_student_list_are_assignment_scoped(client, db_session):
    _, counselor, _, student, other_student, _ = setup_people(db_session)

    headers = auth_headers(client, counselor.username)
    dashboard = client.get("/api/counselor/dashboard", headers=headers)
    students = client.get("/api/counselor/students", headers=headers)

    assert dashboard.status_code == 200
    assert dashboard.json()["assigned_students"] == 1
    assert dashboard.json()["charts"]["trend"]
    assert students.status_code == 200
    ids = [row["id"] for row in students.json()["students"]]
    assert student.id in ids
    assert other_student.id not in ids


def test_unassigned_counselor_is_forbidden_from_sensitive_student_endpoints(client, db_session):
    _, _, other_counselor, student, _, _ = setup_people(db_session)

    headers = auth_headers(client, other_counselor.username)
    response = client.get(f"/api/counselor/student/{student.id}", headers=headers)
    timeline = client.get(f"/api/counselor/student/{student.id}/timeline", headers=headers)
    note = client.post(
        "/api/counselor/notes",
        json={"student_id": student.id, "note_text": "Should not be created"},
        headers=headers,
    )

    assert response.status_code == 403
    assert timeline.status_code == 403
    assert note.status_code == 403


def test_admin_can_access_all_students_and_student_is_denied_counselor_routes(client, db_session):
    admin, _, _, student, other_student, _ = setup_people(db_session)

    admin_headers = auth_headers(client, admin.username)
    student_headers = auth_headers(client, student.username)

    admin_students = client.get("/api/counselor/students", headers=admin_headers)
    admin_detail = client.get(f"/api/counselor/student/{other_student.id}", headers=admin_headers)
    student_attempt = client.get("/api/counselor/students", headers=student_headers)

    assert admin_students.status_code == 200
    assert {row["id"] for row in admin_students.json()["students"]} == {student.id, other_student.id}
    assert admin_detail.status_code == 200
    assert student_attempt.status_code == 403


def test_review_lifecycle_does_not_overwrite_model_output(client, db_session):
    _, counselor, _, student, _, risk = setup_people(db_session)
    original_model_score = risk.model_score
    original_model_risk_level = risk.model_risk_level
    headers = auth_headers(client, counselor.username)

    created = client.post(
        "/api/counselor/reviews",
        json={
            "student_id": student.id,
            "assessment_id": risk.id,
            "status": "NEW",
            "review_notes": "Initial review",
        },
        headers=headers,
    )
    assert created.status_code == 200
    review_id = created.json()["id"]

    updated = client.patch(
        f"/api/counselor/reviews/{review_id}",
        json={
            "status": "FOLLOW_UP_REQUIRED",
            "decision": "Schedule follow-up",
            "risk_judgement": "Counselor judgement recorded separately",
        },
        headers=headers,
    )

    db_session.refresh(risk)
    assert updated.status_code == 200
    assert updated.json()["status"] == "FOLLOW_UP_REQUIRED"
    assert risk.model_score == original_model_score
    assert risk.model_risk_level == original_model_risk_level
    assert risk.human_review_status == "not_requested"


def test_counselor_notes_are_secure_and_history_is_retained(client, db_session):
    _, counselor, _, student, _, _ = setup_people(db_session)
    headers = auth_headers(client, counselor.username)

    created = client.post(
        "/api/counselor/notes",
        json={"student_id": student.id, "note_text": "Private counselor note", "note_type": "follow_up"},
        headers=headers,
    )
    assert created.status_code == 200
    note_id = created.json()["id"]

    updated = client.patch(
        f"/api/counselor/notes/{note_id}",
        json={"note_text": "Updated private note", "active": False},
        headers=headers,
    )
    detail = client.get(f"/api/counselor/student/{student.id}", headers=headers)

    assert updated.status_code == 200
    assert updated.json()["active"] is False
    assert detail.status_code == 200
    assert len(detail.json()["notes"]) == 1
    stored_note = db_session.query(CounselorNote).filter(CounselorNote.id == note_id).first()
    assert stored_note is not None
    assert stored_note.active is False


def test_student_reports_generate_json_csv_and_pdf(client, db_session):
    _, counselor, _, student, _, _ = setup_people(db_session)
    headers = auth_headers(client, counselor.username)

    json_report = client.get(f"/api/counselor/student/{student.id}/reports", headers=headers)
    csv_report = client.get(f"/api/counselor/student/{student.id}/reports?format=csv", headers=headers)
    pdf_report = client.get(f"/api/counselor/student/{student.id}/reports?format=pdf", headers=headers)

    assert json_report.status_code == 200
    assert json_report.json()["model_disclaimer"]
    assert json_report.json()["timeline"]
    assert csv_report.status_code == 200
    assert "model_disclaimer" in csv_report.text
    assert pdf_report.status_code == 200
    assert pdf_report.content.startswith(b"%PDF")


def test_admin_assignment_route_preserves_assignment_history(client, db_session):
    admin = create_user(db_session, "assignmentadmin4g", UserRole.ADMIN)
    counselor = create_user(db_session, "assignmentcounselor4g", UserRole.COUNSELOR)
    next_counselor = create_user(db_session, "assignmentnext4g", UserRole.COUNSELOR)
    student = create_user(db_session, "assignmentstudent4g", UserRole.STUDENT)
    headers = auth_headers(client, admin.username)

    first = client.post(
        "/api/counselor/assignments",
        json={"student_id": student.id, "counselor_id": counselor.id, "assignment_reason": "Initial"},
        headers=headers,
    )
    second = client.post(
        "/api/counselor/assignments",
        json={"student_id": student.id, "counselor_id": next_counselor.id, "assignment_reason": "Transfer"},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assignments = db_session.query(CounselorAssignment).filter(CounselorAssignment.student_id == student.id).all()
    assert len(assignments) == 2
    assert sum(1 for assignment in assignments if assignment.active) == 1
    assert any(assignment.end_date is not None for assignment in assignments)
