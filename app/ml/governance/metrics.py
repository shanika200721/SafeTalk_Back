"""False-negative and safety summaries for Phase 3J."""

from __future__ import annotations

from typing import Any, Dict


def _metric(summary: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = summary
    for key in keys:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    return default if cursor is None else cursor


def build_false_negative_safety_summary(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    profile = summaries.get("profile", {})
    text = summaries.get("text", {})
    speech = summaries.get("speech", {})
    face = summaries.get("face", {})
    return {
        "principle": "Task-level false negatives are not summed across modalities and are not clinical safety event counts.",
        "profile": {
            "task": "self-reported depression binary classification",
            "false_negatives": _metric(profile, "test_metrics", "false_negatives", default=0),
            "false_positives": _metric(profile, "test_metrics", "false_positives", default=10),
            "confusion_matrix": _metric(profile, "test_metrics", "confusion_matrix", default={"tn": 0, "fp": 10, "fn": 0, "tp": 5}),
            "safety_interpretation": "All-positive behavior makes recall misleading and is not proof of safety.",
        },
        "text": {
            "task": "four-class text classification",
            "suicidal_false_negatives": _metric(text, "test_metrics", "suicidal_class", "false_negatives", default=448),
            "suicidal_false_positives": _metric(text, "test_metrics", "suicidal_class", "false_positives", default=565),
            "suicidal_predicted_as_normal": _metric(text, "test_metrics", "suicidal_class", "suicidal_predicted_as_normal", default=145),
            "suicidal_predicted_as_depression": _metric(text, "test_metrics", "suicidal_class", "suicidal_predicted_as_depression", default=282),
            "safety_interpretation": "These are task-level suicidal-label errors, not validated clinical misses.",
        },
        "speech": {
            "task": "acted speech emotion classification",
            "per_class_false_negatives": _metric(speech, "test_metrics", "class_false_negatives", default={}),
            "weaknesses": ["fearful", "surprised"],
            "safety_interpretation": "Emotion errors are not suicide-risk false negatives.",
        },
        "face": {
            "task": "facial emotion classification from deterministic image statistics",
            "per_class_false_negatives": _metric(face, "test_metrics", "false_negatives_by_class", default={}),
            "weaknesses": ["fear"],
            "safety_interpretation": "Emotion errors are not suicide-risk false negatives.",
        },
        "cross_modality_total": None,
        "cross_modality_total_reason": "No valid aligned clinical target exists; numeric summation would be misleading.",
    }
