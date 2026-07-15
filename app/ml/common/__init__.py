"""Common ML utilities for SafeTalk Phase 2."""

__all__ = [
    "DATASET_FINGERPRINT_VERSION",
    "dataset_config_hash",
    "fingerprint_dataset",
    "fingerprint_source_path",
    "load_dataset_fingerprint",
    "load_dataset_config",
    "load_feature_schema",
    "load_preprocessing_config",
    "load_split_manifest",
    "save_schema_json",
    "save_dataset_fingerprint",
    "verify_dataset_fingerprint",
]


def __getattr__(name):
    serialization_exports = {
        "load_dataset_config",
        "load_dataset_fingerprint",
        "load_feature_schema",
        "load_preprocessing_config",
        "load_split_manifest",
        "save_schema_json",
    }
    fingerprinting_exports = {
        "dataset_config_hash",
        "fingerprint_dataset",
        "fingerprint_source_path",
        "save_dataset_fingerprint",
        "verify_dataset_fingerprint",
    }
    if name == "DATASET_FINGERPRINT_VERSION":
        from app.ml.common.hashing import DATASET_FINGERPRINT_VERSION

        return DATASET_FINGERPRINT_VERSION
    if name in serialization_exports:
        from app.ml.common import serialization

        return getattr(serialization, name)
    if name in fingerprinting_exports:
        from app.ml.common import fingerprinting

        return getattr(fingerprinting, name)
    raise AttributeError(name)
