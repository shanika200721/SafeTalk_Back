"""Lightweight deterministic image statistics for facial preprocessing."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.common.schemas import FeatureDefinition, FeatureSchema
from app.ml.preprocessing.face.constants import (
    DATASET_NAME,
    DATASET_VERSION,
    FACE_FEATURE_SCHEMA_VERSION,
    FACE_PREPROCESSING_VERSION,
    FACE_STATISTIC_COLUMNS,
)


def build_face_feature_schema() -> FeatureSchema:
    features = [
        FeatureDefinition(
            name="mean_intensity",
            dtype="float",
            description="Mean grayscale intensity in [0, 255]; engineering statistic only.",
            source_columns=["image pixels"],
            nullable=False,
            minimum=0,
            maximum=255,
            preprocessing_step="optional_lightweight_statistics",
        ),
        FeatureDefinition(
            name="std_intensity",
            dtype="float",
            description="Standard deviation of grayscale intensity; engineering statistic only.",
            source_columns=["image pixels"],
            nullable=False,
            minimum=0,
            maximum=255,
            preprocessing_step="optional_lightweight_statistics",
        ),
        FeatureDefinition(
            name="contrast",
            dtype="float",
            description="P95 minus P5 grayscale intensity contrast; engineering statistic only.",
            source_columns=["image pixels"],
            nullable=False,
            minimum=0,
            maximum=255,
            preprocessing_step="optional_lightweight_statistics",
        ),
        FeatureDefinition(
            name="edge_density",
            dtype="float",
            description="Simple gradient edge-density estimate in [0, 1]; not a landmark or identity feature.",
            source_columns=["image pixels"],
            nullable=False,
            minimum=0,
            maximum=1,
            preprocessing_step="optional_lightweight_statistics",
        ),
        FeatureDefinition(
            name="entropy",
            dtype="float",
            description="Grayscale histogram entropy estimate; engineering statistic only.",
            source_columns=["image pixels"],
            nullable=False,
            minimum=0,
            maximum=8,
            preprocessing_step="optional_lightweight_statistics",
        ),
    ]
    return FeatureSchema(
        schema_name="face-emotion-lightweight-statistics-v1",
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        preprocessing_version=FACE_PREPROCESSING_VERSION,
        modality="face",
        features=features,
        target_columns=["canonical_emotion_label"],
        excluded_columns=["record_id", "source_split", "image_relative_path", "image_hash", "safe_subject_key", "original_label"],
        created_at=datetime.now(timezone.utc),
        notes=(
            "Schema defines metadata and optional deterministic image statistics only. "
            "No CNN fitting, pretrained download, embeddings, landmarks, or face recognition are included."
        ),
    )


def face_feature_schema_payload() -> dict:
    payload = build_face_feature_schema().to_safe_dict()
    payload.update(
        {
            "raw_normalized_pixel_tensor_contract": {
                "status": "future_input_contract_only",
                "default_shape": [48, 48, 1],
                "normalization": "scale pixel values to [0, 1] only in later model pipelines",
            },
            "future_cnn_input_specification": {
                "allowed": "future research only",
                "model_training": "not performed by preprocessing",
                "pretrained_downloads": "not allowed in this step",
            },
            "excluded_metadata": ["source_split", "filename", "image_hash", "safe_subject_key"],
            "subject_key_handling": "safe_subject_key is unavailable unless documented filename IDs exist; never use it as a predictive feature.",
            "non_clinical_status": "Facial emotion statistics are not depression, suicide-risk, crisis, alert, or treatment features.",
            "feature_columns": list(FACE_STATISTIC_COLUMNS),
        }
    )
    return payload


def extract_lightweight_image_statistics(path: str | Path) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    try:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
        if arr.size == 0:
            raise ValueError("empty image array")
        gx = np.abs(np.diff(arr, axis=1))
        gy = np.abs(np.diff(arr, axis=0))
        edge_density = float(((gx > 20).mean() + (gy > 20).mean()) / 2.0) if gx.size and gy.size else 0.0
        hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=False)
        probs = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
        entropy = float(-np.sum([p * math.log2(p) for p in probs if p > 0]))
        values = {
            "mean_intensity": float(np.mean(arr)),
            "std_intensity": float(np.std(arr)),
            "contrast": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
            "edge_density": edge_density,
            "entropy": entropy,
        }
        return {key: float(round(value, 6)) for key, value in values.items()}, warnings
    except Exception as exc:
        warnings.append(f"image statistics unavailable: {exc.__class__.__name__}")
        return {name: 0.0 for name in FACE_STATISTIC_COLUMNS}, warnings

