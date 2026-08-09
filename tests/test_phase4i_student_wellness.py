from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import pytest

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import Resource, User, UserRole
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


def test_wellness_routes_are_student_authorized(client, db_session):
    student = create_user(db_session, "phase4i-student", UserRole.STUDENT)
    counselor = create_user(db_session, "phase4i-counselor", UserRole.COUNSELOR)

    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)

    assert client.get("/api/student/wellness", headers=student_headers).status_code == 200
    assert client.get("/api/student/wellness", headers=counselor_headers).status_code == 403


def test_journal_is_private_and_user_isolated(client, db_session):
    student_a = create_user(db_session, "journal-a", UserRole.STUDENT)
    student_b = create_user(db_session, "journal-b", UserRole.STUDENT)
    counselor = create_user(db_session, "journal-counselor", UserRole.COUNSELOR)
    headers_a = auth_headers(client, student_a.username)
    headers_b = auth_headers(client, student_b.username)
    counselor_headers = auth_headers(client, counselor.username)

    created = client.post(
        "/api/student/journal",
        json={"mood": "steady", "tags": ["sleep"], "content": "private phase 4i note"},
        headers=headers_a,
    )
    assert created.status_code == 200
    assert created.json()["privacy"] == "private"

    own = client.get("/api/student/journal", headers=headers_a)
    other = client.get("/api/student/journal", headers=headers_b)
    counselor_read = client.get("/api/student/journal", headers=counselor_headers)

    assert "private phase 4i note" in str(own.json())
    assert "private phase 4i note" not in str(other.json())
    assert counselor_read.status_code == 403


def test_resources_are_approved_searchable_and_favoritable(client, db_session):
    student = create_user(db_session, "resource-student", UserRole.STUDENT)
    headers = auth_headers(client, student.username)
    db_session.add(
        Resource(
            title="Approved Sleep Article",
            category="Sleep",
            resource_type="Articles",
            description="Approved sleep support",
            status="approved",
            is_active=True,
        )
    )
    db_session.add(
        Resource(
            title="Draft Hidden Resource",
            category="Sleep",
            resource_type="Articles",
            description="Should not appear",
            status="draft",
            is_active=True,
        )
    )
    db_session.commit()

    response = client.get("/api/student/resources?search=sleep", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert any(item["title"] == "Approved Sleep Article" for item in payload["items"])
    assert "Draft Hidden Resource" not in str(payload)

    favorite = client.post("/api/student/resources/db-resource-1/favorite", json={"favorite": True}, headers=headers)
    viewed = client.post("/api/student/resources/db-resource-1/view", headers=headers)
    assert favorite.json()["favorite"] is True
    assert "db-resource-1" in viewed.json()["recently_viewed"]


def test_videos_have_no_autoplay_and_track_favorites_completion(client, db_session):
    student = create_user(db_session, "video-student", UserRole.STUDENT)
    headers = auth_headers(client, student.username)

    videos = client.get("/api/student/videos", headers=headers)
    assert videos.status_code == 200
    first_id = videos.json()["items"][0]["id"]
    assert all(item["approved"] is True for item in videos.json()["items"])
    assert all(item["autoplay"] is False for item in videos.json()["items"])

    favorite = client.post(f"/api/student/videos/{first_id}/favorite", json={"favorite": True}, headers=headers)
    completed = client.post(f"/api/student/videos/{first_id}/complete", json={"completed": True}, headers=headers)
    assert favorite.json()["favorite"] is True
    assert completed.json()["completed"] is True


def test_progress_preferences_and_accessibility_metadata(client, db_session):
    student = create_user(db_session, "progress-student", UserRole.STUDENT)
    headers = auth_headers(client, student.username)

    tracked = client.post(
        "/api/student/progress/track",
        json={"activity_type": "breathing", "item_id": "box", "minutes": 3, "completed": True},
        headers=headers,
    )
    assert tracked.status_code == 200

    progress = client.get("/api/student/progress", headers=headers)
    preferences = client.patch(
        "/api/student/preferences",
        json={"theme": "dark", "large_text": True, "reduced_motion": True},
        headers=headers,
    )
    wellness = client.get("/api/student/wellness", headers=headers)

    assert progress.json()["competitive_scoring"] is False
    assert any("breathe" in item.lower() for item in progress.json()["achievements"])
    assert preferences.json()["theme"] == "dark"
    assert preferences.json()["large_text"] is True
    assert preferences.json()["reduced_motion"] is True
    assert wellness.json()["camera"]["available"] is False
    assert wellness.json()["camera"]["message"] == "Facial analysis is currently unavailable."
