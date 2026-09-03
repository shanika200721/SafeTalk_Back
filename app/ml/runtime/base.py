from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib

from app.models.database_models import ModelRegistry
from app.services.model_registry import (
    _require_approved_artifact_path,
    _resolve_repo_path,
    calculate_sha256,
)


class RuntimeModelUnavailable(Exception):
    """Raised when no verified active runtime model can be used."""


class RuntimeInferenceError(Exception):
    """Raised when an approved model cannot safely produce a prediction."""


@dataclass
class RuntimePredictionResult:
    label: str
    probability: Optional[float] = None
    confidence: Optional[float] = None
    score: Optional[float] = None
    probabilities: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRuntimeLoader:
    _cache: dict[tuple[int, str], Any] = {}

    def load_model(self, registry_entry: ModelRegistry):
        if not registry_entry.is_active:
            raise RuntimeModelUnavailable("Runtime model is not active.")
        if registry_entry.verification_status != "passed" or registry_entry.status != "active":
            raise RuntimeModelUnavailable("Runtime model has not passed governance verification.")
        if registry_entry.serializer != "joblib":
            raise RuntimeModelUnavailable("Runtime model serializer is not supported.")

        artifact_path = _resolve_repo_path(registry_entry.artifact_path)
        try:
            _require_approved_artifact_path(artifact_path)
        except ValueError as exc:
            raise RuntimeModelUnavailable("Runtime model artifact is not in an approved directory.") from exc

        if not artifact_path.exists():
            raise RuntimeModelUnavailable("Runtime model artifact is missing.")

        actual_hash = calculate_sha256(artifact_path)
        if registry_entry.artifact_sha256 and actual_hash != registry_entry.artifact_sha256:
            raise RuntimeModelUnavailable("Runtime model artifact hash verification failed.")

        cache_key = (registry_entry.id, actual_hash)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            loaded = joblib.load(Path(artifact_path))
        except Exception as exc:
            raise RuntimeInferenceError("Runtime model could not be loaded.") from exc

        self._cache[cache_key] = loaded
        return loaded

    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        raise NotImplementedError
