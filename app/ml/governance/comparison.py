"""Safe unimodal comparison without invalid clinical ranking."""

from __future__ import annotations

from typing import Any, Dict, List

from app.ml.governance.readiness import assess_deployment_readiness, assess_evidence_strength
from app.ml.governance.schemas import UnimodalComparisonRecord

COMPARABILITY_WARNING = (
    "Metrics are not directly comparable: Profile predicts self-reported depression, "
    "Text predicts four text categories, Speech predicts acted emotion, Face predicts "
    "facial emotion, none predicts an authoritative suicide-risk outcome, and higher "
    "macro F1 does not mean greater clinical usefulness."
)


def _get(summary: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = summary
    for key in keys:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def build_unimodal_comparison(summaries: Dict[str, Dict[str, Any]]) -> List[UnimodalComparisonRecord]:
    rows = []
    definitions = {
        "profile": {
            "task": "binary self-reported depression classification",
            "label_type": "self_reported_depression",
            "feature_type": "minimal contextual profile features",
            "metric_key": ("test_metrics", "f1"),
            "scope": "very small dataset with 15-record test set",
            "safety": "all-positive behavior makes recall misleading",
            "fn": _get(summaries.get("profile", {}), "test_metrics", "false_negatives", default=0),
        },
        "text": {
            "task": "four-class mental-health text classification",
            "label_type": "weak social-media text categories",
            "feature_type": "normalized TF-IDF text",
            "metric_key": ("test_metrics", "macro_f1"),
            "scope": "incomplete author grouping and shortcut terms",
            "safety": "448 suicidal false negatives",
            "fn": _get(summaries.get("text", {}), "test_metrics", "suicidal_class", "false_negatives", default=448),
        },
        "speech": {
            "task": "acted speech emotion classification",
            "label_type": "acted emotion",
            "feature_type": "deterministic acoustic statistics",
            "metric_key": ("test_metrics", "macro_f1"),
            "scope": "strong corpus domain shift",
            "safety": "emotion errors are not suicide-risk false negatives",
            "fn": None,
        },
        "face": {
            "task": "facial emotion classification",
            "label_type": "facial emotion",
            "feature_type": "deterministic image statistics",
            "metric_key": ("test_metrics", "macro_f1"),
            "scope": "bounded subset only; reviewer independence unverified",
            "safety": "emotion errors are not suicide-risk false negatives",
            "fn": None,
        },
    }
    for modality, definition in definitions.items():
        summary = summaries.get(modality, {})
        test_metrics = summary.get("test_metrics", {})
        selected = summary.get("selected_candidate", {})
        train_count = summary.get("train_count")
        validation_count = summary.get("validation_count")
        test_count = summary.get("test_count")
        dataset_size = sum(x for x in (train_count, validation_count, test_count) if isinstance(x, int))
        metric = _get(summary, *definition["metric_key"])
        rows.append(
            UnimodalComparisonRecord(
                modality=modality,
                task=definition["task"],
                label_type=definition["label_type"],
                dataset_size=dataset_size or None,
                train_count=train_count,
                validation_count=validation_count,
                test_count=test_count,
                feature_type=definition["feature_type"],
                selected_estimator=selected.get("estimator_type", "not_applicable"),
                test_primary_metric=metric,
                test_macro_f1=test_metrics.get("macro_f1") or (test_metrics.get("f1") if modality == "profile" else None),
                test_balanced_accuracy=test_metrics.get("balanced_accuracy"),
                safety_relevant_metric=definition["safety"],
                false_negative_count=definition["fn"],
                domain_shift_evaluated=modality == "speech",
                scope_limitation=definition["scope"],
                comparability_warning=COMPARABILITY_WARNING,
            )
        )
    return rows


def comparison_metadata() -> Dict[str, Any]:
    return {
        "ranking_prohibited": True,
        "comparability_warning": COMPARABILITY_WARNING,
        "deployment_readiness": {modality: assess_deployment_readiness(modality).value for modality in ("profile", "text", "speech", "face")},
        "evidence_strength": {modality: assess_evidence_strength(modality).value for modality in ("profile", "text", "speech", "face")},
    }
