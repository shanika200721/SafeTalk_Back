"""Constants and feature policy for the Speech baseline."""

from __future__ import annotations

SPEECH_MODEL_FAMILY_VERSION = "1.0.0"
SPEECH_BASELINE_EXPERIMENT_VERSION = "1.0.0"
SPEECH_FEATURE_PIPELINE_VERSION = "1.0.0"

SPEECH_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SPEECH_RECORD_ID_COLUMN = "record_id"
SPEECH_TARGET_COLUMN = "canonical_emotion_label"
SPEECH_SPEAKER_KEY_COLUMN = "safe_speaker_key"
SPEECH_CORPUS_COLUMN = "corpus_name"

DEFAULT_FEATURES = "generated/preprocessing/speech/v1/speech_features.csv"
DEFAULT_CANONICAL_MANIFEST = "generated/preprocessing/speech/v1/speech_canonical_manifest.csv"
DEFAULT_FEATURE_SCHEMA = "generated/preprocessing/speech/v1/speech_feature_schema.json"
DEFAULT_PREPROCESSING_REPORT = "generated/preprocessing/speech/v1/speech_preprocessing_report.json"
DEFAULT_DUPLICATE_MANIFEST = "generated/preprocessing/speech/v1/speech_duplicate_manifest.json"
DEFAULT_CORPUS_SUMMARY = "generated/preprocessing/speech/v1/speech_corpus_summary.json"
DEFAULT_SPLIT_MANIFEST = "generated/manifests/splits/speech/v1/speech_split_manifest.json"
DEFAULT_SPLIT_ASSIGNMENTS = "generated/manifests/splits/speech/v1/speech_split_assignments.csv"
DEFAULT_SPEAKER_ISOLATION_REPORT = "generated/manifests/splits/speech/v1/speech_speaker_isolation_report.json"
DEFAULT_DUPLICATE_ISOLATION_REPORT = "generated/manifests/splits/speech/v1/speech_duplicate_isolation_report.json"
DEFAULT_CORPUS_DISTRIBUTION = "generated/manifests/splits/speech/v1/speech_corpus_distribution.json"
DEFAULT_FINGERPRINT_DIR = "generated/manifests/fingerprints/speech"
DEFAULT_REPORT_DIR = "generated/reports/speech_baseline/v1"
DEFAULT_MODEL_ROOT = "ml_models"

CORE_ACOUSTIC_FEATURES = [
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
]
PITCH_PROSODY_FEATURES = [
    "pitch_mean",
    "pitch_std",
    "voiced_frame_ratio",
    "silence_ratio",
    "pause_count",
    "dynamic_range",
]
MFCC_FEATURES = [f"mfcc_{index:02d}_{stat}" for index in range(1, 14) for stat in ("mean", "std")]
FEATURE_SETS = {
    "core_acoustic": CORE_ACOUSTIC_FEATURES,
    "core_pitch_prosody": [*CORE_ACOUSTIC_FEATURES, *PITCH_PROSODY_FEATURES],
    "full_acoustic": [*CORE_ACOUSTIC_FEATURES, *PITCH_PROSODY_FEATURES, *MFCC_FEATURES],
}

PROHIBITED_FEATURES = {
    SPEECH_RECORD_ID_COLUMN,
    SPEECH_TARGET_COLUMN,
    SPEECH_SPEAKER_KEY_COLUMN,
    SPEECH_CORPUS_COLUMN,
    "speaker_id",
    "source_file",
    "source_path",
    "source_split",
    "audio_relative_path",
    "original_audio_hash",
    "audio_hash",
    "file_format",
    "sample_rate",
    "channel_count",
    "sample_width",
    "frame_count",
    "file_size_bytes",
    "readable",
    "validation_warnings",
    "feature_extraction_warnings",
}

REQUIRED_MODEL_CARD_DISCLAIMER = "This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system."

