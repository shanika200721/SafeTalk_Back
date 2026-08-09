"""Safety policy checks for synthetic pilot protocol records."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def validate_safety_events(events: Sequence[Any]) -> Dict[str, Any]:
    errors = []
    for event in events:
        if not event.referred_to_human:
            errors.append(f"{event.safety_event_id}: high-risk protocol events require human referral")
        if "model" in event.detected_by.lower() or "prediction" in event.event_source.lower():
            errors.append(f"{event.safety_event_id}: model-based safety detection is prohibited")
        if not event.immediate_action:
            errors.append(f"{event.safety_event_id}: immediate_action is required")
    return {"valid": not errors, "errors": errors, "event_count": len(events)}


def safety_summary(events: Sequence[Any]) -> Dict[str, Any]:
    return {
        "valid": validate_safety_events(events)["valid"],
        "safety_event_count": len(events),
        "autonomous_model_intervention": False,
        "human_review_required": True,
        "hotline_information_finalized": False,
    }
