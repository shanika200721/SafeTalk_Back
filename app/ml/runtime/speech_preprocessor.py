from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.ml.preprocessing.speech.audio_io import load_wav_audio, write_generated_wav
from app.ml.preprocessing.speech.constants import FEATURE_COLUMNS, SPEECH_FEATURE_SCHEMA_VERSION, SPEECH_PREPROCESSING_VERSION
from app.ml.preprocessing.speech.features import extract_acoustic_features


class SpeechPreprocessingError(ValueError):
    """Raised when audio cannot be transformed into the verified speech feature contract."""


TRAINING_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.75
MAX_DURATION_SECONDS = 120.0
MIN_RMS_ENERGY = 0.001
MAX_CLIPPING_RATIO = 0.02


@dataclass(frozen=True)
class SpeechFeatureVector:
    feature_order: list[str]
    values: list[float]
    features: dict[str, float]
    warnings: list[str]
    preprocessing_version: str = SPEECH_PREPROCESSING_VERSION
    feature_schema_version: str = SPEECH_FEATURE_SCHEMA_VERSION

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values),)

    def as_2d_array(self) -> np.ndarray:
        return np.asarray([self.values], dtype=np.float64)


@dataclass(frozen=True)
class SpeechQualityResult:
    status: str
    flags: list[str]
    duration_seconds: float | None = None
    sample_rate: int | None = None
    waveform_length: int | None = None
    rms_energy: float | None = None
    peak_amplitude: float | None = None
    clipping_ratio: float | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def ffmpeg_executable() -> str | None:
    configured = (settings.FFMPEG_BINARY or "ffmpeg").strip()
    if not configured:
        return None
    configured_path = Path(configured)
    if configured_path.is_absolute() or configured_path.parent != Path("."):
        return str(configured_path) if configured_path.exists() and configured_path.is_file() else None
    return shutil.which(configured)


def ffmpeg_version() -> dict[str, object]:
    executable = ffmpeg_executable()
    if not executable:
        return {"available": False, "executable": None, "version": None}
    try:
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return {"available": False, "executable": executable, "version": None}
    first_line = (completed.stdout or completed.stderr or "").splitlines()
    return {
        "available": completed.returncode == 0,
        "executable": executable,
        "version": first_line[0][:120] if first_line else None,
    }


def _is_browser_container(path: Path, content_type: str | None) -> bool:
    normalized_content_type = (content_type or "").lower().split(";")[0].strip()
    return path.suffix.lower() == ".webm" or normalized_content_type in {"audio/webm", "video/webm"}


def decode_browser_audio_to_wav(
    path: str | Path,
    *,
    content_type: str | None = None,
    target_sample_rate: int = TRAINING_SAMPLE_RATE,
) -> Path:
    source = Path(path)
    if not _is_browser_container(source, content_type):
        return source
    executable = ffmpeg_executable()
    if not executable:
        raise SpeechPreprocessingError(
            "Browser WebM/Opus decoding requires FFmpeg on PATH; speech analysis fails closed when it is unavailable."
        )
    if not source.exists() or not source.is_file():
        raise SpeechPreprocessingError("Speech audio source file does not exist.")

    target = Path(tempfile.NamedTemporaryFile(prefix="safetalk_speech_", suffix=".wav", delete=False).name)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(int(target_sample_rate)),
        "-f",
        "wav",
        str(target),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise SpeechPreprocessingError("Browser audio decode failed.") from exc
    if completed.returncode != 0:
        target.unlink(missing_ok=True)
        message = (completed.stderr or completed.stdout or "unsupported browser audio").strip()
        raise SpeechPreprocessingError(f"Browser audio decode failed: {message[:180]}")
    return target


def validate_speech_audio_quality(
    path: str | Path,
    *,
    content_type: str | None = None,
    target_sample_rate: int = TRAINING_SAMPLE_RATE,
) -> SpeechQualityResult:
    decoded_path: Path | None = None
    try:
        source = Path(path)
        decoded_path = decode_browser_audio_to_wav(source, content_type=content_type, target_sample_rate=target_sample_rate)
        sample_rate, audio = load_wav_audio(decoded_path, mono=True, target_sample_rate=target_sample_rate)
    except SpeechPreprocessingError as exc:
        message = str(exc).lower()
        if "ffmpeg" in message or "unsupported" in message:
            return SpeechQualityResult("unsupported", [str(exc)])
        return SpeechQualityResult("corrupt", [str(exc)])
    except Exception as exc:
        return SpeechQualityResult("corrupt", [f"decode/load failed: {exc.__class__.__name__}"])
    finally:
        if decoded_path and decoded_path != Path(path):
            decoded_path.unlink(missing_ok=True)

    flags: list[str] = []
    if audio.size == 0:
        return SpeechQualityResult("corrupt", ["empty waveform"], sample_rate=sample_rate, waveform_length=0)

    clean = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    duration = float(clean.size / sample_rate)
    rms = float(np.sqrt(np.mean(clean**2)))
    peak = float(np.max(np.abs(clean)))
    clipping = float(np.mean(np.abs(clean) >= 0.999))

    status = "accepted"
    if duration < MIN_DURATION_SECONDS:
        status = "too_short"
        flags.append("duration_below_minimum")
    elif duration > MAX_DURATION_SECONDS:
        status = "low_quality"
        flags.append("duration_above_maximum")
    if rms < MIN_RMS_ENERGY or peak < MIN_RMS_ENERGY:
        status = "silent"
        flags.append("silent_or_near_silent")
    if clipping > MAX_CLIPPING_RATIO:
        status = "low_quality"
        flags.append("excessive_clipping")

    return SpeechQualityResult(
        status=status,
        flags=flags,
        duration_seconds=duration,
        sample_rate=sample_rate,
        waveform_length=int(clean.size),
        rms_energy=rms,
        peak_amplitude=peak,
        clipping_ratio=clipping,
    )


def assert_speech_feature_contract(vector: SpeechFeatureVector) -> None:
    if vector.shape != (44,):
        raise SpeechPreprocessingError(f"Speech feature vector must contain 44 values; got {vector.shape[0]}.")
    if vector.feature_order != list(FEATURE_COLUMNS):
        raise SpeechPreprocessingError("Speech feature order does not match the approved 44-feature contract.")
    values = np.asarray(vector.values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise SpeechPreprocessingError("Speech feature vector contains non-finite values.")


def extract_verified_speech_features(path: str | Path, *, content_type: str | None = None) -> SpeechFeatureVector:
    """Return the exact 44-feature vector used by the selected speech artifact.

    Browser WebM/Opus is decoded only through an explicit FFmpeg runtime dependency.
    When FFmpeg is not available, analysis fails closed rather than fabricating
    WAV-equivalent features.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    normalized_content_type = (content_type or "").lower()
    decoded_path: Path | None = None
    if _is_browser_container(source, normalized_content_type):
        decoded_path = decode_browser_audio_to_wav(source, content_type=content_type)
        source = decoded_path
        suffix = source.suffix.lower()
        normalized_content_type = "audio/wav"
    if suffix not in {".wav", ".wave"} and normalized_content_type not in {"", "audio/wav", "audio/wave", "audio/x-wav"}:
        raise SpeechPreprocessingError("Only WAV audio is currently verified for speech feature extraction.")
    if not source.exists() or not source.is_file():
        raise SpeechPreprocessingError("Speech audio source file does not exist.")

    normalized_path: Path | None = None
    try:
        try:
            sample_rate, audio = load_wav_audio(source, mono=True, target_sample_rate=TRAINING_SAMPLE_RATE)
        except Exception as exc:
            raise SpeechPreprocessingError(f"Speech audio decode failed: {exc.__class__.__name__}") from exc
        normalized_path = Path(tempfile.NamedTemporaryFile(prefix="safetalk_speech_normalized_", suffix=".wav", delete=False).name)
        write_generated_wav(normalized_path, sample_rate, audio, overwrite=True)
        source = normalized_path
        features, warnings = extract_acoustic_features(source)
    finally:
        if normalized_path:
            normalized_path.unlink(missing_ok=True)
        if decoded_path:
            decoded_path.unlink(missing_ok=True)
    failure_warnings = [item for item in warnings if str(item).startswith("feature extraction failed")]
    if failure_warnings:
        raise SpeechPreprocessingError("; ".join(failure_warnings))

    missing = [name for name in FEATURE_COLUMNS if name not in features]
    if missing:
        raise SpeechPreprocessingError(f"Speech feature extraction missed required features: {missing}")

    values = [float(features[name]) for name in FEATURE_COLUMNS]
    if len(values) != 44:
        raise SpeechPreprocessingError(f"Speech feature vector must contain 44 values; got {len(values)}.")
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise SpeechPreprocessingError("Speech feature vector contains non-finite values.")

    vector = SpeechFeatureVector(
        feature_order=list(FEATURE_COLUMNS),
        values=values,
        features={name: float(features[name]) for name in FEATURE_COLUMNS},
        warnings=list(warnings),
    )
    assert_speech_feature_contract(vector)
    return vector


def write_test_wav_fixture(path: str | Path, *, sample_rate: int = TRAINING_SAMPLE_RATE, seconds: float = 1.5) -> Path:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (0.18 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
    return write_generated_wav(path, sample_rate, audio, overwrite=True)
