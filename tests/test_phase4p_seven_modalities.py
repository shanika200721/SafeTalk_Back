from __future__ import annotations

import base64
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.ml.preprocessing.face.constants import CANONICAL_EMOTION_LABELS, FACE_STATISTIC_COLUMNS
from app.ml.runtime.behavioral import predict_behavioral_signal
from app.ml.runtime.face import FACE_RISK_MAPPING_STATUS, FaceRuntimeLoader
from app.ml.runtime.face_detector import FaceDetectionResult
from app.models.database_models import DailyCheckIn, JournalEntry, ModelRegistry, User, UserRole
from app.security import hash_password
from app.services.fusion import run_controlled_fusion
from app.services.modalities import create_behavioral_prediction, create_mood_prediction_for_checkin
from app.utils.dass21_calculator import DASS21Calculator


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


def create_user(db_session, username="phase4p-user"):
    user = User(
        email=f"{username}@example.com",
        username=username,
        full_name="Phase 4P User",
        hashed_password=hash_password("Password123!"),
        role=UserRole.STUDENT,
        is_active=True,
        last_login_at=datetime.utcnow() - timedelta(days=1),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_dass21_known_sample_mapping_and_normalization():
    responses = [0] * 21
    result = DASS21Calculator.calculate(responses)

    assert result["depression_score"] == 0
    assert result["anxiety_score"] == 0
    assert result["stress_score"] == 0
    assert result["depression_severity"] == "Normal"
    assert DASS21Calculator.calculate_dass21_risk_score(result) == 0

    maximum = DASS21Calculator.calculate([3] * 21)
    assert maximum["total_dass21_score"] == 126
    assert DASS21Calculator.calculate_dass21_risk_score(maximum) == 100


def test_mood_signal_uses_real_checkin_fields_and_trend(db_session):
    user = create_user(db_session, "phase4p-mood")
    start = datetime.utcnow() - timedelta(days=9)
    for index, mood in enumerate([5, 5, 4, 4, 3, 2, 2, 1]):
        db_session.add(
            DailyCheckIn(
                user_id=user.id,
                mood=mood,
                sleep_hours=5,
                exercise_minutes=0,
                social_interaction="Limited",
                stress_level=8,
                anxiety_level=7,
                negative_thoughts=index >= 6,
                substance_use_today=False,
                self_harm_thoughts=False,
                created_at=start + timedelta(days=index),
            )
        )
    db_session.commit()
    latest = db_session.query(DailyCheckIn).order_by(DailyCheckIn.created_at.desc()).first()

    prediction = create_mood_prediction_for_checkin(db_session, latest)

    assert prediction.status == "succeeded"
    assert prediction.output_type == "heuristic"
    assert prediction.score_0_100 > 50
    assert prediction.feature_snapshot.features_json["worsening_trend"] > 0
    assert prediction.metadata_json["components"]["repeated_low_mood_component"] > 0


def test_behavioral_anomaly_uses_only_persisted_activity_metadata(db_session):
    user = create_user(db_session, "phase4p-behavior")
    now = datetime.utcnow()
    for days_ago in range(10, 25, 2):
        db_session.add(
            JournalEntry(
                journal_entry_id=f"journal-{days_ago}",
                student_id=user.id,
                body="private activity metadata only",
                entry_date=now - timedelta(days=days_ago),
                created_at=now - timedelta(days=days_ago),
                updated_at=now - timedelta(days=days_ago),
            )
        )
    db_session.commit()

    signal = predict_behavioral_signal(db_session, user, now=now)
    prediction = create_behavioral_prediction(db_session, user)

    assert signal.data_sufficiency in {"sufficient_personal_baseline", "limited_personal_baseline"}
    assert "typing_speed_cpm" in signal.provenance["fields_not_available"]
    assert prediction.status == "succeeded"
    assert prediction.metadata_json["fusion_eligible"] is False
    assert prediction.metadata_json["exclusion_reason"] == "no_validated_risk_mapping"

    fusion = run_controlled_fusion(db_session, user_id=user.id, persist=False, assessment_time=now)
    assert any(item["modality"] == "behavioral" and item["reason"] == "no_validated_risk_mapping" for item in fusion["evidence"]["excluded_modalities"])


class FakeFaceModel:
    classes_ = np.asarray(CANONICAL_EMOTION_LABELS)

    def predict(self, frame):
        assert list(frame.columns) == list(FACE_STATISTIC_COLUMNS)
        return np.asarray(["neutral"])

    def predict_proba(self, frame):
        probabilities = np.full((1, len(CANONICAL_EMOTION_LABELS)), 0.05, dtype=np.float64)
        neutral_index = list(CANONICAL_EMOTION_LABELS).index("neutral")
        probabilities[0, neutral_index] = 0.65
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def _face_data_url() -> str:
    image = Image.new("RGB", (96, 96), color=(120, 130, 140))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_face_runtime_outputs_emotion_probabilities_and_fusion_exclusion(monkeypatch):
    registry = ModelRegistry(
        id=101,
        model_name="face-emotion-random-forest",
        modality="face",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/face/face-emotion-random-forest/1.0.0/face-image_statistics-b9d5c76172fc/pipeline.joblib",
        artifact_sha256="unused-in-monkeypatched-loader",
        serializer="joblib",
        preprocessing_version="face-runtime-v1",
        feature_schema_version="1.0.0",
        is_active=True,
        status="active",
        verification_status="passed",
    )
    loader = FaceRuntimeLoader()
    monkeypatch.setattr(loader, "load_model", lambda model: FakeFaceModel())
    monkeypatch.setattr(
        "app.ml.runtime.face.detect_single_face",
        lambda path: FaceDetectionResult(
            status="one_face_detected",
            face_count=1,
            bounding_boxes=[{"x": 8, "y": 8, "width": 80, "height": 80}],
        ),
    )

    result = loader.predict(registry, {"image_data_url": _face_data_url()})

    assert result.label == "neutral"
    assert set(result.probabilities) == set(CANONICAL_EMOTION_LABELS)
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-9) == 1.0
    assert result.metadata["feature_shape"] == [1, 5]
    assert result.metadata["face_detection_status"] == "one_face_detected"
    assert result.metadata["fusion_status"] == FACE_RISK_MAPPING_STATUS
    assert result.metadata["fusion_eligible"] is False
