from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.runtime.face import FacePreprocessingError, extract_verified_face_features
from app.ml.runtime.face_detector import FaceDetectionError, FaceDetectorUnavailable, FaceDetectionResult, detect_single_face
from app.ml.runtime.speech_preprocessor import ffmpeg_executable
from app.runtime_health import check_environment, operational_checks


def _image_path(tmp_path: Path, color=(120, 130, 140)) -> Path:
    image = Image.new("RGB", (96, 96), color=color)
    path = tmp_path / "face.jpg"
    image.save(path, format="JPEG")
    return path


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def test_health_and_readiness_endpoints_expose_safe_state():
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert "SECRET_KEY" not in str(ready.json())
    assert "checks" in ready.json()
    assert ready.status_code in {200, 503}


def test_operational_checks_include_runtime_dependencies():
    payload = operational_checks()

    assert "ffmpeg" in payload["checks"]
    assert "face_detector" in payload["checks"]
    assert "database" in payload["checks"]
    assert payload["status"] in {"ready", "not_ready"}


def test_environment_check_accepts_configured_cors_origin_list(monkeypatch):
    monkeypatch.setattr("app.runtime_health.settings.ENVIRONMENT", "staging")
    monkeypatch.setattr("app.runtime_health.settings.DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setattr("app.runtime_health.settings.SECRET_KEY", "test-secret")
    monkeypatch.setattr("app.runtime_health.settings.CORS_ORIGINS", ["http://localhost:8080"])
    monkeypatch.setattr("app.runtime_health.settings.MODEL_ROOT", "/app/ml_models")
    monkeypatch.setattr("app.runtime_health.settings.UPLOAD_ROOT", "/app/backend/uploaded_audio")
    monkeypatch.setattr("app.runtime_health.settings.FFMPEG_BINARY", "/usr/bin/ffmpeg")

    payload = check_environment()

    assert payload["status"] == "ok"
    assert payload["missing"] == []
    assert payload["unsafe"] == []


def test_ffmpeg_resolver_uses_configured_binary(monkeypatch, tmp_path: Path):
    executable = tmp_path / "ffmpeg"
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr("app.ml.runtime.speech_preprocessor.settings.FFMPEG_BINARY", str(executable))

    assert ffmpeg_executable() == str(executable)


def test_face_runtime_rejects_unsupported_content_type(tmp_path: Path):
    path = _image_path(tmp_path)

    with pytest.raises(FacePreprocessingError):
        from app.ml.runtime.face import FaceRuntimeLoader

        FaceRuntimeLoader().predict(None, {"image_data_url": _data_url(path), "content_type": "image/gif"})


def test_face_features_fail_closed_when_detector_unavailable(monkeypatch, tmp_path: Path):
    path = _image_path(tmp_path)

    def unavailable(_path):
        raise FaceDetectorUnavailable("missing detector")

    monkeypatch.setattr("app.ml.runtime.face.detect_single_face", unavailable)

    with pytest.raises(FacePreprocessingError, match="fails closed"):
        extract_verified_face_features(path)


def test_face_features_fail_closed_when_no_face_detected(monkeypatch, tmp_path: Path):
    path = _image_path(tmp_path)

    def no_face(_path):
        raise FaceDetectionError("Face detector rejected image: no_face_detected.")

    monkeypatch.setattr("app.ml.runtime.face.detect_single_face", no_face)

    with pytest.raises(FacePreprocessingError, match="no_face_detected"):
        extract_verified_face_features(path)


def test_face_features_fail_closed_for_invalid_image(tmp_path: Path):
    path = tmp_path / "invalid.jpg"
    path.write_bytes(b"not an image")

    with pytest.raises(FacePreprocessingError, match="unreadable|corrupt"):
        extract_verified_face_features(path)


def test_face_detector_timeout_fails_closed(monkeypatch, tmp_path: Path):
    path = _image_path(tmp_path)

    def slow_detector(_path):
        time.sleep(0.2)
        return FaceDetectionResult(status="one_face_detected", face_count=1, bounding_boxes=[])

    monkeypatch.setattr("app.ml.runtime.face_detector.settings.FACE_DETECTOR_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr("app.ml.runtime.face_detector._detect_faces", slow_detector)

    with pytest.raises(FaceDetectionError, match="timed out"):
        detect_single_face(path)


def test_face_detector_internal_failure_is_controlled(monkeypatch, tmp_path: Path):
    path = _image_path(tmp_path)

    def failing_detector(_path):
        raise FaceDetectorUnavailable("forced unavailable")

    monkeypatch.setattr("app.ml.runtime.face_detector._detect_faces", failing_detector)

    with pytest.raises(FaceDetectorUnavailable):
        detect_single_face(path)


def test_face_features_accept_single_detected_face(monkeypatch, tmp_path: Path):
    path = _image_path(tmp_path)
    monkeypatch.setattr(
        "app.ml.runtime.face.detect_single_face",
        lambda _path: FaceDetectionResult(
            status="one_face_detected",
            face_count=1,
            bounding_boxes=[{"x": 4, "y": 4, "width": 80, "height": 80}],
        ),
    )

    features, metadata = extract_verified_face_features(path)

    assert set(features) == {"mean_intensity", "std_intensity", "contrast", "edge_density", "entropy"}
    assert metadata["face_detection_status"] == "one_face_detected"
