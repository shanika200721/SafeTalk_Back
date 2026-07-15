"""Constants for read-only speech emotion preprocessing."""

from __future__ import annotations

SPEECH_PREPROCESSING_VERSION = "1.0.0"
SPEECH_FEATURE_SCHEMA_VERSION = "1.0.0"
SPEECH_LABEL_MAPPING_VERSION = "1.0.0"
SPEECH_CORPUS_MAPPING_VERSION = "1.0.0"

DATASET_NAME = "speech-emotion"
DATASET_VERSION = "v1"

CORPUS_CREMA = "CREMA"
CORPUS_RAVDESS = "RAVDESS"
CORPUS_SAVEE = "SAVEE"
CORPUS_TESS = "TESS"
CONFIRMED_CORPORA = (CORPUS_CREMA, CORPUS_RAVDESS, CORPUS_SAVEE, CORPUS_TESS)

SUPPORTED_AUDIO_EXTENSIONS = {".wav"}
OPTIONAL_AUDIO_EXTENSIONS = {".mp3", ".flac"}

CANONICAL_EMOTION_LABELS = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)

RECORD_ID_PREFIX = "speech-v1-rec"
SAFE_SPEAKER_KEY_PREFIX = "speech-v1-spk"

BASE_FEATURE_COLUMNS = (
    "duration_seconds",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "rms_energy_mean",
    "rms_energy_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "pitch_mean",
    "pitch_std",
    "voiced_frame_ratio",
    "silence_ratio",
    "pause_count",
    "dynamic_range",
)

MFCC_COUNT = 13
MFCC_FEATURE_COLUMNS = tuple(f"mfcc_{idx:02d}_{stat}" for idx in range(1, MFCC_COUNT + 1) for stat in ("mean", "std"))
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MFCC_FEATURE_COLUMNS

