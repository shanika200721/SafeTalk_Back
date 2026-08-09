from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from app.ml.preprocessing.speech.constants import CANONICAL_EMOTION_LABELS, FEATURE_COLUMNS
from app.ml.runtime.speech import SPEECH_RISK_MAPPING_STATUS, SpeechRuntimeLoader
from app.ml.runtime.speech_preprocessor import (
    SpeechPreprocessingError,
    assert_speech_feature_contract,
    extract_verified_speech_features,
    ffmpeg_executable,
    validate_speech_audio_quality,
)
from app.models.database_models import ModalityPrediction, ModelRegistry
from app.services.fusion import _check_prediction


def _write_tone(path: Path, *, seconds: float = 1.2, amplitude: float = 0.2) -> Path:
    sample_rate = 16000
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (amplitude * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
    wavfile.write(path, sample_rate, audio)
    return path


class FakeSpeechModel:
    classes_ = np.asarray(CANONICAL_EMOTION_LABELS)

    def predict(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return np.asarray(["neutral"])

    def predict_proba(self, frame):
        probabilities = np.full((1, len(CANONICAL_EMOTION_LABELS)), 0.05, dtype=np.float64)
        neutral_index = list(CANONICAL_EMOTION_LABELS).index("neutral")
        probabilities[0, neutral_index] = 0.65
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return probabilities


def _registry() -> ModelRegistry:
    return ModelRegistry(
        id=99,
        model_name="speech-emotion-random-forest",
        modality="speech",
        version="1.0.0",
        framework="sklearn",
        artifact_path="ml_models/speech/speech-emotion-random-forest/1.0.0/speech-99cdbe8dbb57/pipeline.joblib",
        artifact_sha256="unused-in-monkeypatched-loader",
        serializer="joblib",
        preprocessing_version="speech-runtime-v1",
        feature_schema_version="1.0.0",
        is_active=True,
        status="active",
        verification_status="passed",
    )


def test_feature_contract_order_shape_finite_and_deterministic(tmp_path: Path):
    path = _write_tone(tmp_path / "voice.wav")

    first = extract_verified_speech_features(path, content_type="audio/wav")
    second = extract_verified_speech_features(path, content_type="audio/wav")

    assert first.as_2d_array().shape == (1, 44)
    assert first.feature_order == list(FEATURE_COLUMNS)
    assert np.all(np.isfinite(first.values))
    assert first.values == second.values
    assert_speech_feature_contract(first)


def test_audio_quality_rejects_silent_and_short_recordings(tmp_path: Path):
    short = _write_tone(tmp_path / "short.wav", seconds=0.2)
    silent = _write_tone(tmp_path / "silent.wav", amplitude=0.0)

    assert validate_speech_audio_quality(short, content_type="audio/wav").status == "too_short"
    assert validate_speech_audio_quality(silent, content_type="audio/wav").status == "silent"


def test_corrupt_audio_rejected(tmp_path: Path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a wav")

    quality = validate_speech_audio_quality(path, content_type="audio/wav")

    assert quality.status == "corrupt"
    with pytest.raises(SpeechPreprocessingError):
        extract_verified_speech_features(path, content_type="audio/wav")


def test_browser_webm_fails_closed_when_decoder_is_unavailable(tmp_path: Path):
    path = tmp_path / "browser_voice_test.webm"
    path.write_bytes(b"\x1a\x45\xdf\xa3not-a-real-webm")

    quality = validate_speech_audio_quality(path, content_type="audio/webm;codecs=opus")

    if ffmpeg_executable() is None:
        assert quality.status == "unsupported"
    else:
        assert quality.status == "corrupt"


def test_browser_webm_opus_decodes_to_44_feature_contract_when_ffmpeg_available(tmp_path: Path):
    executable = ffmpeg_executable()
    if executable is None:
        pytest.skip("FFmpeg is not available on PATH for browser WebM/Opus validation")

    wav_path = _write_tone(tmp_path / "browser_origin_source.wav", seconds=1.5)
    webm_path = tmp_path / "browser_origin_audio.webm"
    completed = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-c:a",
            "libopus",
            str(webm_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    quality = validate_speech_audio_quality(webm_path, content_type="audio/webm;codecs=opus")
    vector = extract_verified_speech_features(webm_path, content_type="audio/webm;codecs=opus")

    assert quality.status == "accepted"
    assert quality.sample_rate == 16000
    assert vector.as_2d_array().shape == (1, 44)
    assert vector.feature_order == list(FEATURE_COLUMNS)


def test_speech_model_predict_and_predict_proba_smoke(tmp_path: Path, monkeypatch):
    path = _write_tone(tmp_path / "voice.wav")
    loader = SpeechRuntimeLoader()
    monkeypatch.setattr(loader, "load_model", lambda registry: FakeSpeechModel())

    result = loader.predict(_registry(), {"path": str(path), "content_type": "audio/wav"})

    assert result.label == "neutral"
    assert set(result.probabilities) == set(CANONICAL_EMOTION_LABELS)
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-9) == 1.0
    assert result.metadata["feature_shape"] == [1, 44]
    assert result.metadata["fusion_status"] == SPEECH_RISK_MAPPING_STATUS
    assert result.metadata["fusion_eligible"] is False
    assert result.score is None


def test_speech_prediction_excluded_without_approved_risk_mapping():
    prediction = ModalityPrediction(
        id=77,
        student_id=1,
        modality="speech",
        status="succeeded",
        is_available=True,
        evidence_available=True,
        output_type="machine_learning",
        probability=0.65,
        confidence=0.65,
        model_name="speech-emotion-random-forest",
        model_version="1.0.0",
        preprocessing_version="speech-runtime-v1",
        feature_schema_version="1.0.0",
        consent_policy_version="phase4o-test",
        metadata_json={"fusion_eligible": False, "fusion_status": SPEECH_RISK_MAPPING_STATUS},
        model_registry=_registry(),
    )

    selected, excluded = _check_prediction(prediction, 1, prediction.generated_at or prediction.created_at)

    assert selected is None
    assert excluded.reason == SPEECH_RISK_MAPPING_STATUS
