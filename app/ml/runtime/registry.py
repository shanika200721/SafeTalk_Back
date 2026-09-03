from typing import Any
import time

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.ml.runtime.base import RuntimeModelUnavailable, RuntimePredictionResult
from app.ml.runtime.face import FaceRuntimeLoader
from app.ml.runtime.profile import ProfileRuntimeLoader
from app.ml.runtime.speech import SpeechRuntimeLoader
from app.ml.runtime.text import TextRuntimeLoader
from app.models.database_models import ModelRegistry
from app.services.model_registry import get_active_model


_LOADERS = {
    "profile": ProfileRuntimeLoader(),
    "text": TextRuntimeLoader(),
    "speech": SpeechRuntimeLoader(),
    "face": FaceRuntimeLoader(),
}

logger = get_logger("app.model_runtime")


def predict_with_active_model(
    db: Session,
    *,
    modality: str,
    payload: dict[str, Any],
) -> tuple[ModelRegistry, RuntimePredictionResult]:
    model = get_active_model(db, modality=modality)
    if model is None:
        raise RuntimeModelUnavailable("No active runtime model is registered for this modality.")

    loader = _LOADERS.get(modality)
    if loader is None:
        raise RuntimeModelUnavailable("This modality is not approved for runtime model inference.")

    start = time.perf_counter()
    try:
        result = loader.predict(model, payload)
    except RuntimeModelUnavailable:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.warning(
            "model_inference_completed",
            extra={
                "modality": modality,
                "model_name": model.model_name,
                "model_version": model.version,
                "runtime_result": "unavailable",
                "failure_code": "MODEL_UNAVAILABLE",
                "duration_ms": duration_ms,
            },
        )
        raise
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.warning(
            "model_inference_completed",
            extra={
                "modality": modality,
                "model_name": model.model_name,
                "model_version": model.version,
                "runtime_result": "failed",
                "failure_code": exc.__class__.__name__,
                "duration_ms": duration_ms,
            },
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "model_inference_completed",
        extra={
            "modality": modality,
            "model_name": model.model_name,
            "model_version": model.version,
            "runtime_result": "succeeded",
            "duration_ms": duration_ms,
            "fusion_status": result.metadata.get("fusion_status"),
        },
    )
    return model, result
