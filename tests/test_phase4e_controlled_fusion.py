from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.db.base import Base
from app.main import app
from app.models.database_models import Alert, Assessment, FeatureSnapshot, ModalityPrediction, RiskAssessment, RiskAssessmentInput, User, UserRole
from app.security import hash_password
from app.services.fusion import controlled_fusion_config, run_controlled_fusion


FROZEN_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


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


def add_prediction(
    db_session,
    user,
    modality,
    *,
    score=None,
    probability=None,
    output_type="heuristic",
    status_value="succeeded",
    is_available=True,
    source_time=None,
    created_offset=0,
    evidence=True,
    model_version="1.0.0",
    preprocessing_version="runtime-v1",
):
    source_time = source_time or (FROZEN_NOW - timedelta(hours=1))
    created_at = source_time + timedelta(minutes=created_offset)
    snapshot = None
    if evidence:
        snapshot = FeatureSnapshot(
            student_id=user.id,
            modality=modality,
            source_type=f"{modality}_source",
            source_record_id=100 + created_offset,
            source_timestamp=source_time.replace(tzinfo=None),
            feature_schema_version=f"{modality}-schema-v1",
            preprocessing_version=preprocessing_version,
            features_json={"stored_raw_payload": False},
            consent_policy_version="1.0",
        )
        db_session.add(snapshot)
        db_session.flush()
    prediction = ModalityPrediction(
        student_id=user.id,
        modality=modality,
        source_type=f"{modality}_source",
        source_record_id=100 + created_offset,
        feature_snapshot_id=snapshot.id if snapshot else None,
        predicted_class="suicidal" if modality == "text" else "screening_signal",
        probability=probability,
        score_0_100=score,
        status=status_value,
        is_available=is_available,
        source_timestamp=source_time.replace(tzinfo=None),
        generated_at=created_at.replace(tzinfo=None),
        valid_until=(FROZEN_NOW + timedelta(days=2)).replace(tzinfo=None),
        model_name=f"{modality}-model",
        model_version=model_version,
        preprocessing_version=preprocessing_version,
        feature_schema_version=f"{modality}-schema-v1",
        output_type=output_type,
        evidence_available=evidence,
        consent_policy_version="1.0",
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)
    return prediction


def seed_four_valid_modalities(db_session, user):
    add_prediction(db_session, user, "profile", score=40, output_type="heuristic")
    add_prediction(db_session, user, "dass21", score=50, output_type="rule_based")
    add_prediction(db_session, user, "mood", score=20, output_type="heuristic")
    add_prediction(db_session, user, "text", probability=0.80, output_type="machine_learning")


def test_config_uses_approved_v2_weights_thresholds_and_stable_hash():
    config = controlled_fusion_config()

    assert config["base_weights"] == {
        "dass21": 0.25,
        "face": 0.12,
        "mood": 0.15,
        "profile": 0.1,
        "speech": 0.13,
        "text": 0.25,
    }
    assert config["thresholds"] == {"high_severe": 0.7, "low_moderate": 0.32, "moderate_high": 0.52}
    assert config["minimum_evidence_policy"]["minimum_modality_count"] == 2
    assert config["staleness_windows_days"]["mood"] == 3
    assert config["coverage_categories"]["moderate"]["min_inclusive"] == 0.5
    assert config["config_hash"] == controlled_fusion_config()["config_hash"]


def test_normalization_weight_renormalization_and_repeatability(db_session):
    user = create_user(db_session, "student")
    seed_four_valid_modalities(db_session, user)

    first = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=FROZEN_NOW)
    second = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=FROZEN_NOW)

    expected = ((0.40 * 0.10) + (0.50 * 0.25) + (0.20 * 0.15) + (0.80 * 0.25)) / 0.75
    assert first["score"] == pytest.approx(expected)
    assert second["score"] == first["score"]
    assert first["risk_level"] == "high"
    assert first["evidence"]["base_weight_coverage"] == pytest.approx(0.75)
    assert first["evidence"]["coverage_category"] == "moderate"
    assert sum(item["effective_weight"] for item in first["inputs"]) == pytest.approx(1.0)
    assert "speech" in first["evidence"]["missing_modalities"]
    assert "face" in first["evidence"]["missing_modalities"]


def test_failed_stale_unavailable_and_missing_values_are_excluded_not_zeroed(db_session):
    user = create_user(db_session, "missingness")
    add_prediction(db_session, user, "profile", score=60, output_type="heuristic")
    add_prediction(db_session, user, "dass21", score=60, output_type="rule_based")
    add_prediction(db_session, user, "text", probability=0.90, output_type="machine_learning", status_value="failed", is_available=False)
    add_prediction(
        db_session,
        user,
        "mood",
        score=100,
        output_type="heuristic",
        source_time=FROZEN_NOW - timedelta(days=5),
    )

    result = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=FROZEN_NOW)

    assert result["score"] == pytest.approx(0.60)
    assert result["evidence"]["used_modalities"] == ["dass21", "profile"]
    reasons = {item["reason"] for item in result["evidence"]["excluded_modalities"]}
    assert "status_failed" in reasons
    assert "prediction_stale" in reasons
    assert result["score"] != pytest.approx(((0.60 * 0.10) + (0.60 * 0.25)) / 1.0)


def test_insufficient_evidence_persists_without_fake_zero_score(db_session):
    user = create_user(db_session, "insufficient")
    add_prediction(db_session, user, "profile", score=90, output_type="heuristic")

    result = run_controlled_fusion(db_session, user_id=user.id, persist=True, assessment_time=FROZEN_NOW)

    assessment = db_session.query(RiskAssessment).one()
    assert result["status"] == "insufficient_evidence"
    assert result["score"] is None
    assert result["risk_level"] is None
    assert assessment.final_probability is None
    assert assessment.final_score is None
    assert assessment.risk_level is None
    assert assessment.alert_created is False
    assert db_session.query(Assessment).count() == 0
    assert db_session.query(Alert).count() == 0


def test_persistence_records_inputs_versions_and_no_alerts(db_session):
    user = create_user(db_session, "persisted")
    seed_four_valid_modalities(db_session, user)
    add_prediction(db_session, user, "text", probability=0.95, output_type="externally_supplied", created_offset=10)

    result = run_controlled_fusion(db_session, user_id=user.id, persist=True, assessment_time=FROZEN_NOW)

    assessment = db_session.query(RiskAssessment).one()
    links = db_session.query(RiskAssessmentInput).order_by(RiskAssessmentInput.modality).all()
    assert result["assessment_id"] == assessment.id
    assert assessment.fusion_config_version == "controlled-late-fusion-v2"
    assert assessment.threshold_version == "v2"
    assert assessment.mapping_version == "runtime-mapping-v1"
    assert assessment.human_review_status == "not_requested"
    assert assessment.counselor_decision is None
    assert assessment.counselor_override is None
    assert assessment.alert_created is False
    included_links = [link for link in links if link.included]
    excluded_links = [link for link in links if not link.included]
    assert [link.modality for link in included_links] == ["dass21", "mood", "profile", "text"]
    assert [link.exclusion_reason for link in excluded_links] == ["output_type_not_allowed:externally_supplied"]
    assert sum(link.effective_weight for link in included_links) == pytest.approx(1.0)
    assert any(item["included"] is False for item in result["inputs"])
    assert db_session.query(Assessment).count() == 0
    assert db_session.query(Alert).count() == 0


def test_staleness_boundary_and_invalid_profile_output_type(db_session):
    user = create_user(db_session, "boundary")
    add_prediction(db_session, user, "profile", score=50, output_type="heuristic")
    add_prediction(db_session, user, "profile", probability=0.99, output_type="externally_supplied", created_offset=5)
    add_prediction(db_session, user, "dass21", score=50, output_type="rule_based", source_time=FROZEN_NOW - timedelta(days=30))
    add_prediction(db_session, user, "mood", score=50, output_type="heuristic", source_time=FROZEN_NOW - timedelta(days=3, seconds=1))

    result = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=FROZEN_NOW)

    assert result["status"] == "completed"
    assert result["evidence"]["used_modalities"] == ["dass21", "profile"]
    reasons = [item["reason"] for item in result["evidence"]["excluded_modalities"]]
    assert "output_type_not_allowed:externally_supplied" in reasons
    assert "prediction_stale" in reasons


def test_fusion_authorization_and_legacy_route_deprecation(client, db_session):
    student = create_user(db_session, "selfstudent")
    other = create_user(db_session, "otherstudent")
    counselor = create_user(db_session, "counselor", role=UserRole.COUNSELOR)
    student_headers = auth_headers(client, student.username)
    counselor_headers = auth_headers(client, counselor.username)

    own = client.post("/api/fusion/assess", json={}, headers=student_headers)
    assert own.status_code == 200
    assert own.json()["user_id"] == student.id

    blocked = client.post("/api/fusion/assess", json={"user_id": other.id}, headers=student_headers)
    assert blocked.status_code == 403

    counselor_blocked = client.post("/api/fusion/assess", json={"user_id": student.id}, headers=counselor_headers)
    assert counselor_blocked.status_code == 403

    legacy = client.post(
        "/api/assessments/risk-assessment",
        json={"user_id": student.id, "scores": {"profile_score": 0}},
        headers=student_headers,
    )
    assert legacy.status_code == 410
    assert legacy.json()["error"]["code"] == "LEGACY_RISK_ASSESSMENT_DEPRECATED"
