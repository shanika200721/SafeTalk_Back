from typing import Any

from app.ml.runtime.base import BaseRuntimeLoader, RuntimeModelUnavailable, RuntimePredictionResult
from app.models.database_models import ModelRegistry


class SpeechRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        raise RuntimeModelUnavailable("Speech runtime inference is not approved in Phase 4D.")
