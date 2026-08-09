from typing import Any

import numpy as np
import pandas as pd

from app.ml.preprocessing.speech.constants import CANONICAL_EMOTION_LABELS, FEATURE_COLUMNS
from app.ml.runtime.base import BaseRuntimeLoader, RuntimeInferenceError, RuntimeModelUnavailable, RuntimePredictionResult
from app.ml.runtime.speech_preprocessor import (
    SpeechPreprocessingError,
    assert_speech_feature_contract,
    extract_verified_speech_features,
    validate_speech_audio_quality,
)
from app.models.database_models import ModelRegistry


SPEECH_RISK_MAPPING_STATUS = "excluded_no_approved_risk_mapping"
SPEECH_RUNTIME_LIMITATION = (
    "Voice emotion was analyzed, but it is not included in the final fused screening score because "
    "the project does not currently have an approved emotion-to-risk mapping."
)


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
            f"Speech model labels do not match the approved emotion set. Expected {sorted(expected)}, got {sorted(actual)}."
        )


def confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


class SpeechRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        audio_path = payload.get("path") or payload.get("audio_path")
        if not audio_path:
            raise RuntimeInferenceError("Speech runtime requires an audio path.")
        content_type = payload.get("content_type")
        quality = validate_speech_audio_quality(audio_path, content_type=content_type)
        if not quality.accepted:
            raise SpeechPreprocessingError(f"Speech audio quality rejected: {quality.status}")

        vector = extract_verified_speech_features(audio_path, content_type=content_type)
        assert_speech_feature_contract(vector)
        model = self.load_model(registry_entry)
        labels = _label_mapping_for_model(model)
        _validate_labels(labels)

        frame = pd.DataFrame([vector.features], columns=list(FEATURE_COLUMNS))
        try:
            prediction = model.predict(frame)
        except Exception as exc:
            raise RuntimeInferenceError("Speech model prediction failed.") from exc
        label = str(prediction[0])
        if label not in set(CANONICAL_EMOTION_LABELS):
            raise RuntimeInferenceError("Speech model returned an unsupported emotion label.")

        probabilities: dict[str, float] = {}
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                raw = np.asarray(model.predict_proba(frame), dtype=np.float64)
            except Exception as exc:
                raise RuntimeInferenceError("Speech model probability prediction failed.") from exc
            if raw.shape != (1, len(labels)):
                raise RuntimeInferenceError("Speech model probability output shape is invalid.")
            if not np.all(np.isfinite(raw)) or np.any(raw < 0):
                raise RuntimeInferenceError("Speech model probabilities contain invalid values.")
            total = float(raw.sum())
            if not np.isclose(total, 1.0, atol=1e-5):
                raise RuntimeInferenceError("Speech model probabilities do not sum to 1.")
            probabilities = {emotion: float(raw[0, index]) for index, emotion in enumerate(labels)}
            confidence = probabilities.get(label)

        return RuntimePredictionResult(
            label=label,
            probability=confidence,
            confidence=confidence,
            probabilities=probabilities,
            features=vector.features,
            metadata={
                "feature_order": vector.feature_order,
                "feature_shape": [1, 44],
                "warnings": vector.warnings,
                "data_quality_status": quality.status,
                "data_quality_flags": quality.flags,
                "quality": {
                    "duration_seconds": quality.duration_seconds,
                    "sample_rate": quality.sample_rate,
                    "waveform_length": quality.waveform_length,
                    "rms_energy": quality.rms_energy,
                    "peak_amplitude": quality.peak_amplitude,
                    "clipping_ratio": quality.clipping_ratio,
                },
                "confidence_band": confidence_band(confidence),
                "label_mapping": labels,
                "fusion_status": SPEECH_RISK_MAPPING_STATUS,
                "fusion_eligible": False,
                "normalized_score": None,
                "risk_mapping_version": None,
                "limitation": SPEECH_RUNTIME_LIMITATION,
            },
        )
