from app.ml.runtime.base import BaseRuntimeLoader, RuntimeInferenceError, RuntimePredictionResult
from app.models.database_models import ModelRegistry


class TextRuntimeLoader(BaseRuntimeLoader):
    def predict(self, registry_entry: ModelRegistry, payload: dict[str, str]) -> RuntimePredictionResult:
        loaded = self.load_model(registry_entry)
        text = payload.get("text") or ""

        try:
            prediction = loaded.predict([text])
            label = str(prediction[0])
            probabilities: dict[str, float] = {}
            if hasattr(loaded, "predict_proba"):
                proba = loaded.predict_proba([text])[0]
                classes = [str(item) for item in getattr(loaded, "classes_", [])]
                if not classes and hasattr(loaded, "named_steps"):
                    for step in reversed(loaded.named_steps.values()):
                        step_classes = getattr(step, "classes_", None)
                        if step_classes is not None:
                            classes = [str(item) for item in step_classes]
                            break
                probabilities = {classes[index]: float(value) for index, value in enumerate(proba)}
        except Exception as exc:
            raise RuntimeInferenceError("Text runtime prediction failed.") from exc

        return RuntimePredictionResult(
            label=label,
            probability=probabilities.get("suicidal"),
            confidence=max(probabilities.values()) if probabilities else None,
            probabilities=probabilities,
            features={"text_length": len(text), "contains_raw_text": False},
            metadata={
                "runtime_strategy": "active_model",
                "positive_class": "suicidal",
                "probability_semantics": "probability_of_suicidal_class",
                "raw_text_stored": False,
            },
        )
