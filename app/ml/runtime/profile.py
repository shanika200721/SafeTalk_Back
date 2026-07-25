from typing import Any

import pandas as pd

from app.ml.runtime.base import BaseRuntimeLoader, RuntimeInferenceError, RuntimePredictionResult
from app.models.database_models import ModelRegistry


def _normalize_year_of_study(value: Any) -> str:
    if value is None:
        return "year 1"
    if isinstance(value, int):
        return f"year {max(1, min(4, value))}"
    normalized = str(value).strip().lower()
    if normalized in {"1", "2", "3", "4"}:
        return f"year {normalized}"
    if normalized.startswith("year "):
        return normalized
    return normalized or "year 1"


class ProfileRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        loaded = self.load_model(registry_entry)
        year_of_study = _normalize_year_of_study(payload.get("year_of_study"))
        frame = pd.DataFrame([{"year_of_study": year_of_study}])

        try:
            prediction = loaded.predict(frame)
            label = str(prediction[0])
            probabilities: dict[str, float] = {}
            if hasattr(loaded, "predict_proba"):
                proba = loaded.predict_proba(frame)[0]
                classes = [str(item) for item in getattr(loaded, "classes_", [])]
                if not classes and hasattr(loaded, "named_steps"):
                    for step in reversed(loaded.named_steps.values()):
                        step_classes = getattr(step, "classes_", None)
                        if step_classes is not None:
                            classes = [str(item) for item in step_classes]
                            break
                probabilities = {classes[index]: float(value) for index, value in enumerate(proba)}
        except Exception as exc:
            raise RuntimeInferenceError("Profile runtime prediction failed.") from exc

        return RuntimePredictionResult(
            label=label,
            probability=probabilities.get("yes"),
            confidence=max(probabilities.values()) if probabilities else None,
            probabilities=probabilities,
            features={"year_of_study": year_of_study},
            metadata={
                "runtime_strategy": "active_model",
                "positive_class": "yes",
                "probability_semantics": "probability_of_self_reported_depression_yes",
            },
        )
