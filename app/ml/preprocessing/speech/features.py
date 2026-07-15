"""Deterministic lightweight acoustic feature extraction."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fftpack import dct

from app.ml.common.schemas import FeatureDefinition, FeatureSchema, Modality
from app.ml.preprocessing.speech.audio_io import load_wav_audio
from app.ml.preprocessing.speech.constants import (
    FEATURE_COLUMNS,
    MFCC_COUNT,
    SPEECH_FEATURE_SCHEMA_VERSION,
    SPEECH_PREPROCESSING_VERSION,
)


def _frame_audio(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    count = 1 + (len(audio) - frame_length) // hop_length
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=(count, frame_length), strides=strides).copy()
    return frames * np.hanning(frame_length)


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0, 0.0
    return float(clean.mean()), float(clean.std(ddof=0))


def _power_spectrum(frames: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    return np.maximum(spectrum, 1e-12)


def _spectral_features(frames: np.ndarray, sample_rate: int) -> dict[str, float]:
    spectrum = _power_spectrum(frames)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    weights = spectrum.sum(axis=1)
    centroid = (spectrum * freqs).sum(axis=1) / np.maximum(weights, 1e-12)
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * spectrum).sum(axis=1) / np.maximum(weights, 1e-12))
    cumulative = np.cumsum(spectrum, axis=1)
    thresholds = 0.85 * cumulative[:, -1]
    rolloff = np.array([freqs[np.searchsorted(cumulative[idx], thresholds[idx], side="left").clip(max=len(freqs) - 1)] for idx in range(len(frames))])
    geometric = np.exp(np.mean(np.log(spectrum), axis=1))
    arithmetic = np.mean(spectrum, axis=1)
    flatness = geometric / np.maximum(arithmetic, 1e-12)
    result: dict[str, float] = {}
    for prefix, values in {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }.items():
        mean, std = _safe_mean_std(values)
        result[f"{prefix}_mean"] = mean
        result[f"{prefix}_std"] = std
    result["spectral_flatness_mean"] = _safe_mean_std(flatness)[0]
    return result


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft_bins: int, n_filters: int = 26) -> np.ndarray:
    low_mel = _hz_to_mel(np.array([0.0]))[0]
    high_mel = _hz_to_mel(np.array([sample_rate / 2.0]))[0]
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft_bins * 2 - 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_filters, n_fft_bins), dtype=np.float64)
    for idx in range(1, n_filters + 1):
        left, center, right = bins[idx - 1], bins[idx], bins[idx + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for bin_idx in range(left, min(center, n_fft_bins)):
            filters[idx - 1, bin_idx] = (bin_idx - left) / max(center - left, 1)
        for bin_idx in range(center, min(right, n_fft_bins)):
            filters[idx - 1, bin_idx] = (right - bin_idx) / max(right - center, 1)
    return filters


def _mfcc_features(frames: np.ndarray, sample_rate: int) -> dict[str, float]:
    spectrum = _power_spectrum(frames) ** 2
    filters = _mel_filterbank(sample_rate, spectrum.shape[1])
    energies = np.maximum(np.dot(spectrum, filters.T), 1e-12)
    coeffs = dct(np.log(energies), type=2, axis=1, norm="ortho")[:, :MFCC_COUNT]
    result: dict[str, float] = {}
    for idx in range(MFCC_COUNT):
        mean, std = _safe_mean_std(coeffs[:, idx])
        result[f"mfcc_{idx + 1:02d}_mean"] = mean
        result[f"mfcc_{idx + 1:02d}_std"] = std
    return result


def _pitch_features(frames: np.ndarray, sample_rate: int) -> tuple[dict[str, float], list[str]]:
    pitches: list[float] = []
    min_lag = max(1, int(sample_rate / 400.0))
    max_lag = max(min_lag + 1, int(sample_rate / 75.0))
    warnings: list[str] = []
    for frame in frames:
        frame = frame - frame.mean()
        energy = float(np.sqrt(np.mean(frame**2)))
        if energy < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if corr[0] <= 1e-12 or max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment) + min_lag)
        confidence = corr[lag] / corr[0]
        if confidence >= 0.3:
            pitches.append(sample_rate / lag)
    if not pitches:
        warnings.append("pitch unavailable")
    mean, std = _safe_mean_std(np.asarray(pitches, dtype=np.float64))
    return {"pitch_mean": mean, "pitch_std": std, "voiced_frame_ratio": float(len(pitches) / max(len(frames), 1))}, warnings


def extract_acoustic_features(path: str | Path) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    try:
        sample_rate, audio = load_wav_audio(path, mono=True)
    except Exception as exc:
        return {name: 0.0 for name in FEATURE_COLUMNS}, [f"feature extraction failed: {exc.__class__.__name__}"]
    if audio.size == 0:
        return {name: 0.0 for name in FEATURE_COLUMNS}, ["empty audio array"]
    audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    duration = float(audio.size / sample_rate)
    frames = _frame_audio(audio, sample_rate)
    raw_frames = np.lib.stride_tricks.sliding_window_view(audio if len(audio) >= frames.shape[1] else np.pad(audio, (0, frames.shape[1] - len(audio))), frames.shape[1])[:: max(1, int(round(sample_rate * 0.010)))]
    if len(raw_frames) != len(frames):
        raw_frames = frames
    zero_crossings = np.mean(np.diff(np.signbit(raw_frames), axis=1), axis=1)
    rms = np.sqrt(np.mean(raw_frames**2, axis=1))
    zcr_mean, zcr_std = _safe_mean_std(zero_crossings)
    rms_mean, rms_std = _safe_mean_std(rms)
    features: dict[str, float] = {
        "duration_seconds": duration,
        "zero_crossing_rate_mean": zcr_mean,
        "zero_crossing_rate_std": zcr_std,
        "rms_energy_mean": rms_mean,
        "rms_energy_std": rms_std,
        "dynamic_range": float(np.percentile(audio, 95) - np.percentile(audio, 5)),
    }
    features.update(_spectral_features(frames, sample_rate))
    features.update(_mfcc_features(frames, sample_rate))
    pitch, pitch_warnings = _pitch_features(raw_frames, sample_rate)
    warnings.extend(pitch_warnings)
    features.update(pitch)
    silence_threshold = max(1e-4, float(np.max(rms)) * 0.02 if rms.size else 1e-4)
    silent = rms <= silence_threshold
    features["silence_ratio"] = float(np.mean(silent)) if silent.size else 0.0
    pause_count = 0
    in_pause = False
    for item in silent.tolist():
        if item and not in_pause:
            pause_count += 1
            in_pause = True
        elif not item:
            in_pause = False
    features["pause_count"] = float(pause_count)
    clean = {name: float(np.nan_to_num(features.get(name, 0.0), nan=0.0, posinf=0.0, neginf=0.0)) for name in FEATURE_COLUMNS}
    return clean, warnings


def build_speech_feature_schema(*, dataset_name: str = "speech-emotion", dataset_version: str = "v1") -> FeatureSchema:
    definitions = []
    descriptions = {
        "duration_seconds": ("float", False, 0, None, "seconds"),
        "zero_crossing_rate_mean": ("float", False, 0, 1, "ratio"),
        "zero_crossing_rate_std": ("float", False, 0, 1, "ratio"),
        "rms_energy_mean": ("float", False, 0, 1, "normalized amplitude"),
        "rms_energy_std": ("float", False, 0, 1, "normalized amplitude"),
        "spectral_centroid_mean": ("float", False, 0, None, "Hz"),
        "spectral_centroid_std": ("float", False, 0, None, "Hz"),
        "spectral_bandwidth_mean": ("float", False, 0, None, "Hz"),
        "spectral_bandwidth_std": ("float", False, 0, None, "Hz"),
        "spectral_rolloff_mean": ("float", False, 0, None, "Hz"),
        "spectral_rolloff_std": ("float", False, 0, None, "Hz"),
        "spectral_flatness_mean": ("float", False, 0, None, "ratio"),
        "pitch_mean": ("float", True, 0, None, "Hz"),
        "pitch_std": ("float", True, 0, None, "Hz"),
        "voiced_frame_ratio": ("float", False, 0, 1, "ratio"),
        "silence_ratio": ("float", False, 0, 1, "ratio"),
        "pause_count": ("float", False, 0, None, "count"),
        "dynamic_range": ("float", False, 0, 2, "normalized amplitude"),
    }
    for idx in range(1, MFCC_COUNT + 1):
        descriptions[f"mfcc_{idx:02d}_mean"] = ("float", False, None, None, "coefficient")
        descriptions[f"mfcc_{idx:02d}_std"] = ("float", False, 0, None, "coefficient")
    for name in FEATURE_COLUMNS:
        dtype, nullable, minimum, maximum, units = descriptions[name]
        definitions.append(
            FeatureDefinition(
                name=name,
                dtype=dtype,
                description=f"{name}; deterministic acoustic feature, units: {units}; no transcription or semantic content extraction.",
                source_columns=["audio_waveform"],
                nullable=nullable,
                minimum=minimum,
                maximum=maximum,
                preprocessing_step="scipy/numpy lightweight acoustic feature extraction",
            )
        )
    return FeatureSchema(
        schema_name="speech-emotion-acoustic-features",
        feature_schema_version=SPEECH_FEATURE_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        preprocessing_version=SPEECH_PREPROCESSING_VERSION,
        modality=Modality.VOICE,
        features=definitions,
        target_columns=["canonical_emotion_label"],
        excluded_columns=["corpus_name", "source_file", "speaker_id", "safe_speaker_key", "record_id"],
        created_at=datetime.now(timezone.utc),
        notes=(
            "Feature schema is non-clinical. Corpus name, source filename, and speaker keys are excluded from predictive features by default. "
            "No model training, inference, transcription, alerting, or treatment recommendation is part of this schema."
        ),
    )

