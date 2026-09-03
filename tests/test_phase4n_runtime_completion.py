from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from app.ml.preprocessing.speech.constants import FEATURE_COLUMNS
from app.ml.runtime.speech_preprocessor import SpeechPreprocessingError, extract_verified_speech_features


def test_speech_preprocessor_outputs_exact_44_feature_contract(tmp_path: Path):
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "browser-representative.wav"
    wavfile.write(path, sample_rate, audio)

    vector = extract_verified_speech_features(path, content_type="audio/wav")

    assert vector.shape == (44,)
    assert vector.feature_order == list(FEATURE_COLUMNS)
    assert len(vector.as_2d_array().shape) == 2
    assert vector.as_2d_array().shape == (1, 44)
    assert all(np.isfinite(vector.values))


def test_speech_preprocessor_rejects_unverified_webm_contract(tmp_path: Path):
    path = tmp_path / "browser-recording.webm"
    path.write_bytes(b"\x1a\x45\xdf\xa3not-a-real-fixture")

    with pytest.raises(SpeechPreprocessingError, match="WebM"):
        extract_verified_speech_features(path, content_type="audio/webm")
