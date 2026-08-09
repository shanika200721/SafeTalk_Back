from datetime import datetime

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import (
    Alert,
    CounselorAssignment,
    CounselorProfile,
    CounselorProfileAudit,
    RiskAssessment,
    SupportContact,
    SupportContactAction,
    University,
    User,
    UserRole,
)
from app.security import hash_password
from app.services.support_contacts import normalize_e164, select_support_contact, telephone_uri, whatsapp_uri


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
        seed_fallback(session)
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


def create_university(db_session, code="UOC"):
    university = University(
        university_id=f"uni-{code.lower()}",
        university_name=f"{code} University",
        university_code=code,
        campus_name="Main Campus",
        counseling_unit_phone="+94705584634",
        active=True,
    )
    db_session.add(university)
    db_session.commit()
    db_session.refresh(university)
    return university


def create_profile(db_session, counselor, university, *, active=True, approved=True, phone="+94705584634"):
    profile = CounselorProfile(
        counselor_profile_id=f"cprof-{counselor.id}",
        user_id=counselor.id,
        university_id=university.id,
        full_name=counselor.full_name,
        professional_title="Counselor",
        telephone_number=phone,
        whatsapp_number=phone,
        available_days="Monday-Friday",
        available_from="09:00",
        available_until="17:00",
        accepts_voice_calls=True,
        accepts_whatsapp_messages=True,
        languages_json=["English", "Sinhala"],
        availability_status="available",
        approved=approved,
        active=active,
    )
    db_session.add(profile)
    db_session.commit()
    db_session.refresh(profile)
    return profile


def assign_student(db_session, student, counselor, active=True):
    assignment = CounselorAssignment(
        assignment_id=f"asg-{student.id}-{counselor.id}-{active}",
        student_id=student.id,
        counselor_id=counselor.id,
        assigned_by=counselor.id,
        active=active,
    )
    db_session.add(assignment)
    db_session.commit()
    return assignment


def seed_fallback(db_session):
    fallback = SupportContact(
        support_contact_id="support-fallback-safetalk-v1",
        contact_type="system_fallback",
        display_name="SafeTalk Counselor Support",
        telephone_number="+94705584634",
        whatsapp_number="+94705584634",
        telephone_enabled=True,
        whatsapp_enabled=True,
        student_visible=True,
        emergency_service=False,
        verified=True,
        verified_at=datetime.utcnow(),
        active=True,
        priority=1000,
    )
    db_session.add(fallback)
    db_session.commit()


def test_number_validation_and_safe_uri_generation():
    assert normalize_e164("+94705584634") == "+94705584634"
    assert normalize_e164("+94 70 558 4634") == "+94705584634"
    assert telephone_uri("+94705584634") == "tel:+94705584634"
    assert whatsapp_uri("+94705584634").startswith("https://wa.me/94705584634")
    assert "+94705584634" not in whatsapp_uri("+94705584634").split("wa.me/")[1].split("?")[0]

    for value in ["94705584634", "+94ABC5584634", "javascript:alert(1)", "+94/705584634"]:
        with pytest.raises(HTTPException):
            normalize_e164(value, required=True)


def test_contact_selection_priority_and_scope(db_session):
    university = create_university(db_session, "UOA")
    other_university = create_university(db_session, "UOB")
    student = create_user(db_session, "supportstudent", UserRole.STUDENT, university)
    counselor = create_user(db_session, "supportcounselor", UserRole.COUNSELOR, university)
    other_counselor = create_user(db_session, "otherunisupportcounselor", UserRole.COUNSELOR, other_university)
    create_profile(db_session, counselor, university)
    create_profile(db_session, other_counselor, other_university, phone="+94715584634")
    assign_student(db_session, student, counselor)
    assign_student(db_session, student, other_counselor)

    selected, _ = select_support_contact(db_session, student)
    assert selected["contact_type"] == "assigned_counselor"
    assert selected["display_name"] == counselor.full_name
    assert selected["telephone_uri"] == "tel:+94705584634"
    assert "id" not in selected
    assert selected["emergency_service"] is False


def test_inactive_assigned_counselor_excluded_and_university_then_system_fallback_selected(db_session):
    university = create_university(db_session, "UOC")
    student = create_user(db_session, "fallbackstudent", UserRole.STUDENT, university)
    counselor = create_user(db_session, "inactivecounselor", UserRole.COUNSELOR, university)
    create_profile(db_session, counselor, university, active=False)
    assign_student(db_session, student, counselor)
    unit = SupportContact(
        support_contact_id="support-university-unit",
        university_id=university.id,
        contact_type="university_unit",
        display_name="University Counseling Unit",
        telephone_number="+94705584634",
        whatsapp_number="+94705584634",
        verified=True,
        active=True,
        student_visible=True,
        priority=10,
    )
    db_session.add(unit)
    db_session.commit()

    selected, _ = select_support_contact(db_session, student)
    assert selected["contact_type"] == "university_unit"

    unit.active = False
    db_session.commit()
    selected, _ = select_support_contact(db_session, student)
    assert selected["contact_type"] == "system_fallback"
    assert selected["display_name"] == "SafeTalk Counselor Support"


def test_student_support_routes_are_read_only_and_actions_are_privacy_minimized(client, db_session):
    university = create_university(db_session, "UOD")
    student = create_user(db_session, "routestudent", UserRole.STUDENT, university)
    headers = auth_headers(client, student.username)

    contact = client.get("/api/support/contact", headers=headers)
    action = client.post("/api/support/actions", json={"action_type": "telephone_action_selected"}, headers=headers)

    assert contact.status_code == 200
    assert contact.json()["contact_type"] == "system_fallback"
    assert contact.json()["telephone_uri"] == "tel:+94705584634"
    assert action.status_code == 200
    assert action.json()["counselor_contacted"] is False
    assert action.json()["emergency_services_contacted"] is False
    assert db_session.query(SupportContactAction).count() == 1
    assert db_session.query(Alert).count() == 0
    assert db_session.query(RiskAssessment).count() == 0


def test_counselor_can_update_only_permitted_profile_fields(client, db_session):
    university = create_university(db_session, "UOE")
    counselor = create_user(db_session, "selfprofilecounselor", UserRole.COUNSELOR, university)
    profile = create_profile(db_session, counselor, university, phone="+94705584634")
    headers = auth_headers(client, counselor.username)

    response = client.patch(
        "/api/counselor/profile",
        json={
            "telephone_number": "+94 71 558 4634",
            "whatsapp_number": "+94 72 558 4634",
            "available_from": "10:00",
            "available_until": "16:00",
            "languages_json": ["English", "Tamil"],
            "university_id": None,
            "role": "admin",
        },
        headers=headers,
    )

    db_session.refresh(profile)
    db_session.refresh(counselor)
    assert response.status_code == 200
    assert response.json()["telephone_number"] == "+94715584634"
    assert response.json()["whatsapp_number"] == "+94725584634"
    assert profile.university_id == university.id
    assert counselor.role == UserRole.COUNSELOR
    assert db_session.query(CounselorProfileAudit).count() == 1


def test_counselor_cannot_edit_another_profile_and_student_cannot_edit_contacts(client, db_session):
    university = create_university(db_session, "UOF")
    counselor = create_user(db_session, "owncounselor", UserRole.COUNSELOR, university)
    other = create_user(db_session, "othercounselor", UserRole.COUNSELOR, university)
    student = create_user(db_session, "contacteditstudent", UserRole.STUDENT, university)
    create_profile(db_session, counselor, university)
    other_profile = create_profile(db_session, other, university, phone="+94715584634")

    counselor_headers = auth_headers(client, counselor.username)
    student_headers = auth_headers(client, student.username)

    student_response = client.patch("/api/counselor/profile", json={"telephone_number": "+94715584634"}, headers=student_headers)
    admin_route_response = client.patch(
        f"/api/admin/counselors/{other_profile.id}",
        json={"telephone_number": "+94725584634"},
        headers=counselor_headers,
    )

    assert student_response.status_code == 403
    assert admin_route_response.status_code == 403


def test_admin_can_manage_university_counselor_and_support_contact(client, db_session):
    admin = create_user(db_session, "directoryadmin", UserRole.ADMIN)
    headers = auth_headers(client, admin.username)

    university = client.post(
        "/api/admin/universities",
        json={
            "university_name": "Example University",
            "university_code": "EXU",
            "counseling_unit_phone": "+94 70 558 4634",
        },
        headers=headers,
    )
    assert university.status_code == 200
    university_id = university.json()["id"]

    counselor = client.post(
        "/api/admin/counselors",
        json={
            "username": "managedcounselor",
            "email": "managedcounselor@example.com",
            "password": "Password123!",
            "full_name": "Managed Counselor",
            "university_id": university_id,
            "telephone_number": "+94705584634",
            "whatsapp_number": "+94 70 558 4634",
            "available_from": "09:00",
            "available_until": "17:00",
        },
        headers=headers,
    )
    support_contact = client.post(
        "/api/admin/support-contacts",
        json={
            "university_id": university_id,
            "contact_type": "university_unit",
            "display_name": "Example Counseling Unit",
            "telephone_number": "+94 70 558 4634",
            "whatsapp_number": "+94705584634",
            "emergency_service": False,
        },
        headers=headers,
    )

    assert counselor.status_code == 200
    assert counselor.json()["telephone_number"] == "+94705584634"
    assert counselor.json()["registration_number"] is None
    assert support_contact.status_code == 200
    assert support_contact.json()["telephone_uri"] == "tel:+94705584634"
    assert support_contact.json()["whatsapp_uri"].startswith("https://wa.me/94705584634")
