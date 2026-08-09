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


def _normalize_yes_no(value: Any, *, default: str = "no") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"yes", "no"}:
        return normalized
    if normalized in {"y", "true", "1"}:
        return "yes"
    if normalized in {"n", "false", "0"}:
        return "no"
    return default


class ProfileRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, Any]) -> RuntimePredictionResult:
        loaded = self.load_model(registry_entry)
        year_of_study = _normalize_year_of_study(payload.get("year_of_study"))
        row = {"year_of_study": year_of_study}
        metadata_features = []
        feature_schema = (registry_entry.metadata_json or {}).get("feature_schema") or {}
        selected_features = feature_schema.get("selected_features") or []
        if not selected_features and registry_entry.feature_schema_version == "profile-v2-feature-contract":
            selected_features = ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
        if "self_reported_anxiety" in selected_features or "self_reported_anxiety" in payload:
            row["self_reported_anxiety"] = _normalize_yes_no(payload.get("self_reported_anxiety"))
        if "self_reported_panic_attack" in selected_features or "self_reported_panic_attack" in payload:
            row["self_reported_panic_attack"] = _normalize_yes_no(payload.get("self_reported_panic_attack"))
        metadata_features = list(row)
        frame = pd.DataFrame([row])

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
            features=row,
            metadata={
                "runtime_strategy": "active_model",
                "positive_class": "yes",
                "probability_semantics": "probability_of_self_reported_depression_yes",
                "input_features": metadata_features,
            },
        )
