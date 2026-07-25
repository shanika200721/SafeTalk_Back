from typing import Any

from app.ml.runtime.base import BaseRuntimeLoader, RuntimeModelUnavailable, RuntimePredictionResult
from app.models.database_models import ModelRegistry


class FaceRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        raise RuntimeModelUnavailable("Face runtime inference is not approved in Phase 4D.")
