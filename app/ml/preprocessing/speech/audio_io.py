"""Read-only audio metadata and loading helpers."""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from app.ml.preprocessing.speech.constants import OPTIONAL_AUDIO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS
from app.ml.preprocessing.speech.schemas import SpeechAudioMetadata


def audio_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_audio_metadata(path: str | Path) -> SpeechAudioMetadata:
    source = Path(path)
    extension = source.suffix.lower()
    warnings: list[str] = []
    size = source.stat().st_size if source.exists() else 0
    if not source.exists():
        return SpeechAudioMetadata(
            file_format=extension.lstrip(".") or "<none>",
            file_size_bytes=0,
            readable=False,
            validation_warnings=["file does not exist"],
        )
    if size == 0:
        return SpeechAudioMetadata(
            file_format=extension.lstrip(".") or "<none>",
            file_size_bytes=0,
            readable=False,
            validation_warnings=["zero-byte audio file"],
        )
    if extension in OPTIONAL_AUDIO_EXTENSIONS:
        return SpeechAudioMetadata(
            file_format=extension.lstrip("."),
            file_size_bytes=size,
            readable=False,
            validation_warnings=["mp3/flac metadata is not enabled without optional lightweight dependencies"],
        )
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        return SpeechAudioMetadata(
            file_format=extension.lstrip(".") or "<none>",
            file_size_bytes=size,
            readable=False,
            validation_warnings=[f"unsupported audio extension: {extension or '<none>'}"],
        )
    try:
        with wave.open(str(source), "rb") as handle:
            frame_count = int(handle.getnframes())
            sample_rate = int(handle.getframerate())
            channel_count = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
        duration = frame_count / float(sample_rate) if sample_rate > 0 else 0.0
        if duration <= 0:
            warnings.append("non-positive duration")
        return SpeechAudioMetadata(
            duration_seconds=round(float(duration), 9),
            sample_rate=sample_rate,
            channel_count=channel_count,
            sample_width=sample_width,
            frame_count=frame_count,
            file_format="wav",
            file_size_bytes=size,
            readable=True,
            validation_warnings=warnings,
        )
    except Exception as exc:
        return SpeechAudioMetadata(
            file_format="wav",
            file_size_bytes=size,
            readable=False,
            validation_warnings=[f"unreadable WAV: {exc.__class__.__name__}"],
        )


def _integer_audio_to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float64, copy=False)
    elif np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        audio = data.astype(np.float64) / float(scale)
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0)


def load_wav_audio(path: str | Path, *, mono: bool = True, target_sample_rate: int | None = None) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(str(path))
    if sample_rate <= 0:
        raise ValueError("WAV sample rate must be positive")
    audio = _integer_audio_to_float(np.asarray(data))
    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)
    elif audio.ndim > 2:
        raise ValueError("Unsupported audio dimensionality")
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        divisor = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(audio, target_sample_rate // divisor, sample_rate // divisor)
        sample_rate = target_sample_rate
    return int(sample_rate), np.asarray(audio, dtype=np.float64)


def write_generated_wav(path: str | Path, sample_rate: int, audio: np.ndarray, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite generated audio: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    wavfile.write(str(target), int(sample_rate), (clipped * 32767.0).astype(np.int16))
    return target

