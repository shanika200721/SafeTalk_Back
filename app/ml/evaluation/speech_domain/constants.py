"""Constants for Speech leave-one-corpus-out domain evaluation."""

from __future__ import annotations

SPEECH_DOMAIN_EVALUATION_VERSION = "1.0.0"
SPEECH_LOCO_POLICY_VERSION = "1.0.0"

CORPORA = ["CREMA", "RAVDESS", "SAVEE", "TESS"]
DEFAULT_RANDOM_SEED = 42
DEFAULT_VALIDATION_SPEAKER_FRACTION = 0.2

DEFAULT_FEATURES = "generated/preprocessing/speech/v1/speech_features.csv"
DEFAULT_CANONICAL_MANIFEST = "generated/preprocessing/speech/v1/speech_canonical_manifest.csv"
DEFAULT_FEATURE_SCHEMA = "generated/preprocessing/speech/v1/speech_feature_schema.json"
DEFAULT_CORPUS_MAPPING = "ml-research/configs/speech.corpus_mapping.v1.json"
DEFAULT_LABEL_POLICY = "ml-research/configs/speech.domain_label_policy.v1.json"
DEFAULT_FINGERPRINT_DIR = "generated/manifests/fingerprints/speech"
DEFAULT_EVALUATION_MANIFEST_DIR = "generated/manifests/evaluation/speech_domain/v1"
DEFAULT_REPORT_DIR = "generated/reports/speech_domain_shift/v1"
DEFAULT_MODEL_ROOT = "ml_models/speech-domain-evaluation"
DEFAULT_POOLED_BASELINE = "generated/reports/speech_baseline/v1/speech_baseline_summary.json"

DIAGNOSTIC_ONLY_WARNING = "Diagnostic only; not an emotion model, not registered, not active, and not for production use."

