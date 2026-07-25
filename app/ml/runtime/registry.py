from typing import Any

from sqlalchemy.orm import Session

from app.ml.runtime.base import RuntimeModelUnavailable, RuntimePredictionResult
from app.ml.runtime.profile import ProfileRuntimeLoader
from app.ml.runtime.text import TextRuntimeLoader
from app.models.database_models import ModelRegistry
from app.services.model_registry import get_active_model


_LOADERS = {
    "profile": ProfileRuntimeLoader(),
    "text": TextRuntimeLoader(),
}


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

    return model, loader.predict(model, payload)
