from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.preprocessing.face.constants import CANONICAL_EMOTION_LABELS, FACE_STATISTIC_COLUMNS
from app.ml.preprocessing.face.features import extract_lightweight_image_statistics
from app.ml.preprocessing.face.image_io import extract_image_metadata
from app.ml.runtime.base import BaseRuntimeLoader, RuntimeInferenceError, RuntimePredictionResult
from app.models.database_models import ModelRegistry


FACE_RISK_MAPPING_STATUS = "excluded_low_reliability_no_validated_risk_mapping"
FACE_RUNTIME_LIMITATION = (
    "Facial emotion was analyzed as an experimental signal only and is excluded from fusion because "
    "the selected artifact has low test reliability and no approved suicide-risk mapping."
)


class FacePreprocessingError(ValueError):
    """Raised when a face image cannot be transformed into the runtime feature contract."""


def _label_mapping_for_model(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    return [str(item) for item in classes] if classes is not None else []


def _validate_labels(labels: list[str]) -> None:
    expected = set(CANONICAL_EMOTION_LABELS)
    actual = set(labels)
    if actual != expected:
        raise RuntimeInferenceError(
            f"Face model labels do not match the approved emotion set. Expected {sorted(expected)}, got {sorted(actual)}."
        )


def _decode_data_url(data_url: str) -> Path:
    match = re.match(r"^data:image/(?P<kind>jpeg|jpg|png);base64,(?P<payload>[A-Za-z0-9+/=\s]+)$", data_url.strip())
    if not match:
        raise FacePreprocessingError("Face image payload must be a JPEG or PNG data URL.")
    raw = base64.b64decode(match.group("payload"), validate=True)
    if not raw:
        raise FacePreprocessingError("Face image payload is empty.")
    if len(raw) > 2 * 1024 * 1024:
        raise FacePreprocessingError("Face image payload exceeds the maximum accepted size.")
    suffix = ".jpg" if match.group("kind") in {"jpeg", "jpg"} else ".png"
    target = Path(tempfile.NamedTemporaryFile(prefix="safetalk_face_", suffix=suffix, delete=False).name)
    target.write_bytes(raw)
    return target


def _resolve_image_source(payload: dict[str, Any]) -> tuple[Path, bool, str]:
    data_url = payload.get("image_data_url")
    if data_url:
        return _decode_data_url(str(data_url)), True, "browser_data_url"
    path = payload.get("path") or payload.get("image_path")
    if not path:
        raise FacePreprocessingError("Face runtime requires an explicit captured image payload.")
    return Path(path), False, "file_path"


def extract_verified_face_features(path: str | Path) -> tuple[dict[str, float], dict[str, Any]]:
    source = Path(path)
    metadata = extract_image_metadata(source)
    if not metadata.readable:
        raise FacePreprocessingError("Face image is unreadable or corrupt.")
    if metadata.width is None or metadata.height is None or metadata.width < 48 or metadata.height < 48:
        raise FacePreprocessingError("Face image dimensions are below the minimum runtime size.")
    if metadata.file_size_bytes > 2 * 1024 * 1024:
        raise FacePreprocessingError("Face image exceeds the maximum accepted size.")

    features, warnings = extract_lightweight_image_statistics(source)
    missing = [name for name in FACE_STATISTIC_COLUMNS if name not in features]
    if missing:
        raise FacePreprocessingError(f"Face feature extraction missed required features: {missing}")
    ordered = {name: float(features[name]) for name in FACE_STATISTIC_COLUMNS}
    if not np.all(np.isfinite(np.asarray(list(ordered.values()), dtype=np.float64))):
        raise FacePreprocessingError("Face feature vector contains non-finite values.")

    return ordered, {
        "image_width": metadata.width,
        "image_height": metadata.height,
        "color_mode": metadata.color_mode,
        "file_format": metadata.file_format,
        "file_size_bytes": metadata.file_size_bytes,
        "image_validation_warnings": metadata.validation_warnings,
        "feature_warnings": warnings,
        "face_detection_status": "not_available_no_detector_dependency",
        "face_detection_limitation": "No OpenCV/landmark detector dependency is configured; runtime treats the explicit capture as the candidate face crop.",
    }


class FaceRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        image_path, temporary, source_kind = _resolve_image_source(payload)
        try:
            features, metadata = extract_verified_face_features(image_path)
            model = self.load_model(registry_entry)
            labels = _label_mapping_for_model(model)
            _validate_labels(labels)
            frame = pd.DataFrame([features], columns=list(FACE_STATISTIC_COLUMNS))
            try:
                prediction = model.predict(frame)
            except Exception as exc:
                raise RuntimeInferenceError("Face model prediction failed.") from exc
            label = str(prediction[0])
            if label not in set(CANONICAL_EMOTION_LABELS):
                raise RuntimeInferenceError("Face model returned an unsupported emotion label.")

            probabilities: dict[str, float] = {}
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    raw = np.asarray(model.predict_proba(frame), dtype=np.float64)
                except Exception as exc:
                    raise RuntimeInferenceError("Face model probability prediction failed.") from exc
                if raw.shape != (1, len(labels)):
                    raise RuntimeInferenceError("Face model probability output shape is invalid.")
                if not np.all(np.isfinite(raw)) or np.any(raw < 0):
                    raise RuntimeInferenceError("Face model probabilities contain invalid values.")
                total = float(raw.sum())
                if not np.isclose(total, 1.0, atol=1e-5):
                    raise RuntimeInferenceError("Face model probabilities do not sum to 1.")
                probabilities = {emotion: float(raw[0, index]) for index, emotion in enumerate(labels)}
                confidence = probabilities.get(label)

            return RuntimePredictionResult(
                label=label,
                probability=confidence,
                confidence=confidence,
                probabilities=probabilities,
                features=features,
                metadata={
                    **metadata,
                    "source_kind": source_kind,
                    "feature_order": list(FACE_STATISTIC_COLUMNS),
                    "feature_shape": [1, len(FACE_STATISTIC_COLUMNS)],
                    "label_mapping": labels,
                    "technical_status": "technically_active_experimental",
                    "research_reliability": "low",
                    "fusion_status": FACE_RISK_MAPPING_STATUS,
                    "fusion_eligible": False,
                    "normalized_score": None,
                    "risk_mapping_version": None,
                    "limitation": FACE_RUNTIME_LIMITATION,
                },
            )
        finally:
            if temporary:
                image_path.unlink(missing_ok=True)
