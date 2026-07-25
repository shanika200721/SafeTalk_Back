from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


SAFETY_POLICY_VERSION = "safetalk-safety-v1"
RESPONSE_POLICY_VERSION = "safetalk_response_policy_v2"
CONTEXT_WINDOW_MINUTES = 60
LIMITATIONS = [
    "SafeTalk is not a replacement for professional or emergency care.",
    "SafeTalk cannot contact emergency services or guarantee an immediate human response.",
]

ROUTING_ORDER = [
    "imminent_self_harm",
    "explicit_suicidal_intent",
    "possible_self_harm_or_crisis",
    "severe_distress",
    "emotional_disclosure",
    "coping_support_request",
    "mental_health_information",
    "positive_or_stable",
    "greeting",
    "gratitude_or_closing",
    "unclear_or_other",
]

TOPIC_LABELS = {
    "greeting": "Greeting",
    "positive_or_stable": "General support",
    "emotional_disclosure": "Low mood",
    "coping_support_request": "Coping exercise",
    "mental_health_information": "Information",
    "severe_distress": "General support",
    "possible_self_harm_or_crisis": "Safety support",
    "explicit_suicidal_intent": "Safety support",
    "imminent_self_harm": "Safety support",
    "gratitude_or_closing": "General support",
    "unclear_or_other": "Other",
}

YES_WORDS = {"yes", "yeah", "yep", "ya", "i am", "i do", "true"}
NO_WORDS = {"no", "nope", "nah", "not now", "i am not", "i'm not", "i dont", "i don't"}


@dataclass(frozen=True)
class SafetyRoute:
    route: str
    severity: str
    matched_rules: list[str]
    requires_direct_safety_question: bool
    recommend_immediate_support: bool
    allow_normal_conversation: bool
    safe_response_template_id: str
    internal_reason_code: str
    classifier_used: bool = False
    classifier_result: Optional[dict[str, Any]] = None
    limitations: list[str] = None
    topic_label: str = "Other"

    def to_internal_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "severity": self.severity,
            "matched_rules": self.matched_rules,
            "requires_direct_safety_question": self.requires_direct_safety_question,
            "recommend_immediate_support": self.recommend_immediate_support,
            "allow_normal_conversation": self.allow_normal_conversation,
            "safe_response_template_id": self.safe_response_template_id,
            "internal_reason_code": self.internal_reason_code,
            "classifier_used": self.classifier_used,
            "classifier_result": self.classifier_result,
            "limitations": self.limitations or LIMITATIONS,
            "topic_label": self.topic_label,
        }


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "safetalk"


def load_templates() -> dict[str, Any]:
    with (_data_dir() / "templates" / "safety_v1.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_crisis_resources() -> dict[str, Any]:
    with (_data_dir() / "resources" / "crisis_resources_v1.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _compact(text: str) -> str:
    compact = re.sub(r"[^a-z0-9']+", " ", _normalize(text)).strip()
    compact = re.sub(r"\bh+i+\b", "hi", compact)
    compact = re.sub(r"\bhe+y+\b", "hey", compact)
    compact = re.sub(r"\bhell+o+\b", "hello", compact)
    return compact


def _has_any(text: str, phrases: list[str]) -> Optional[str]:
    padded = f" {text} "
    for phrase in phrases:
        if f" {phrase} " in padded or phrase in text:
            return phrase
    return None


def _is_simple_exact(text: str, options: set[str]) -> bool:
    compact = _compact(text)
    return compact in options


def _contains_dangerous_language(text: str) -> bool:
    danger_phrases = [
        "kill myself",
        "end my life",
        "end it all",
        "want to die",
        "hurt myself",
        "self harm",
        "suicide",
        "overdose",
        "pills",
        "jump",
        "hang myself",
        "better off dead",
    ]
    return _has_any(_compact(text), danger_phrases) is not None


def _is_negated_self_harm(text: str) -> bool:
    compact = _compact(text)
    negations = [
        "not suicidal",
        "not going to kill myself",
        "do not want to kill myself",
        "don't want to kill myself",
        "do not want to hurt myself",
        "don't want to hurt myself",
        "not thinking about suicide",
        "not thinking of suicide",
        "no suicidal thoughts",
    ]
    return _has_any(compact, negations) is not None


def _is_quoted_or_third_party(text: str) -> bool:
    compact = _compact(text)
    third_party = [
        "my friend said",
        "my friend told me",
        "someone said",
        "i read",
        "in an article",
        "the phrase",
        "quote",
        "quoted",
    ]
    return _has_any(compact, third_party) is not None and _contains_dangerous_language(compact)


def _recent_context(context_state: Optional[dict[str, Any]], now: datetime) -> dict[str, Any]:
    if not context_state:
        return {}
    timestamp = context_state.get("crisis_message_at") or context_state.get("updated_at")
    if not timestamp:
        return context_state
    try:
        parsed = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return {}
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    if now - parsed > timedelta(minutes=CONTEXT_WINDOW_MINUTES):
        return {}
    return context_state


def _route(
    route: str,
    severity: str,
    template_id: str,
    reason: str,
    matched_rules: list[str],
    *,
    safety_question: bool = False,
    immediate: bool = False,
    allow_normal: bool = True,
    classifier_used: bool = False,
    classifier_result: Optional[dict[str, Any]] = None,
    topic_label: Optional[str] = None,
) -> SafetyRoute:
    return SafetyRoute(
        route=route,
        severity=severity,
        matched_rules=matched_rules,
        requires_direct_safety_question=safety_question,
        recommend_immediate_support=immediate,
        allow_normal_conversation=allow_normal,
        safe_response_template_id=template_id,
        internal_reason_code=reason,
        classifier_used=classifier_used,
        classifier_result=classifier_result,
        limitations=LIMITATIONS,
        topic_label=topic_label or TOPIC_LABELS[route],
    )


def route_safetalk_message(
    message: str,
    *,
    context_state: Optional[dict[str, Any]] = None,
    classifier_signal: Optional[dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> SafetyRoute:
    now = now or datetime.utcnow()
    text = _normalize(message)
    compact = _compact(text)
    recent_context = _recent_context(context_state, now)
    classifier_used = classifier_signal is not None

    if not compact:
        return _route("unclear_or_other", "none", "unclear_or_other_v1", "empty_input", [])

    if recent_context.get("unresolved_safety_question") and _is_simple_exact(text, YES_WORDS):
        return _route(
            "explicit_suicidal_intent",
            "crisis",
            "context_safety_yes_v1",
            "contextual_safety_confirmation",
            ["context_yes_after_safety_question"],
            safety_question=True,
            immediate=True,
            allow_normal=False,
        )
    if recent_context.get("unresolved_safety_question") and _is_simple_exact(text, NO_WORDS):
        return _route(
            "emotional_disclosure",
            "low",
            "context_safety_no_v1",
            "contextual_safety_denial",
            ["context_no_after_safety_question"],
            topic_label="General support",
        )

    if _is_quoted_or_third_party(text):
        return _route(
            "unclear_or_other",
            "medium",
            "quoted_risk_clarification_v1",
            "quoted_or_third_party_risk_language",
            ["quoted_or_third_party"],
            safety_question=True,
            allow_normal=False,
        )

    if _is_negated_self_harm(text):
        return _route(
            "emotional_disclosure",
            "low",
            "context_safety_no_v1",
            "negated_self_harm_language",
            ["self_harm_negation"],
            topic_label="General support",
        )

    imminent = [
        "i have a plan",
        "i have the pills",
        "pills with me",
        "already took",
        "i took something",
        "standing on a roof",
        "on a roof",
        "will do it tonight",
        "do it tonight",
        "tonight",
        "right now",
        "goodbye",
        "jump off",
    ]
    match = _has_any(compact, imminent)
    if match and (_contains_dangerous_language(compact) or match in {"already took", "i took something"}):
        return _route(
            "imminent_self_harm",
            "imminent",
            "imminent_self_harm_v1",
            "imminent_plan_or_means",
            [match],
            immediate=True,
            allow_normal=False,
        )

    explicit = [
        "want to kill myself",
        "going to kill myself",
        "kill myself",
        "want to die",
        "want to end my life",
        "going to end my life",
        "end my life",
        "thinking about suicide",
        "hurt myself",
        "want to hurt myself",
        "cut myself",
    ]
    match = _has_any(compact, explicit)
    if match:
        return _route(
            "explicit_suicidal_intent",
            "crisis",
            "explicit_suicidal_intent_v1",
            "explicit_self_harm_or_suicide_language",
            [match],
            safety_question=True,
            immediate=True,
            allow_normal=False,
        )

    possible = [
        "wish i could disappear",
        "want to disappear",
        "do not want to be here",
        "don't want to be here",
        "life is not worth it",
        "better without me",
        "better off without me",
        "better off dead",
        "no reason to live",
        "not worth living",
    ]
    match = _has_any(compact, possible)
    if match:
        return _route(
            "possible_self_harm_or_crisis",
            "high",
            "possible_self_harm_or_crisis_v1",
            "possible_self_harm_language",
            [match],
            safety_question=True,
            immediate=True,
            allow_normal=False,
        )

    severe = [
        "cannot handle this anymore",
        "can't handle this anymore",
        "cannot do this anymore",
        "can't do this anymore",
        "everything feels hopeless",
        "feel hopeless",
        "completely trapped",
        "feel trapped",
        "unbearable",
    ]
    match = _has_any(compact, severe)
    if match:
        return _route(
            "severe_distress",
            "medium",
            "severe_distress_v1",
            "severe_distress_language",
            [match],
            safety_question=True,
            allow_normal=False,
        )

    info = ["what is anxiety", "what is dass 21", "what is dass21", "what does depression mean", "what is depression", "what does anxiety mean"]
    match = _has_any(compact, info)
    if match:
        return _route("mental_health_information", "none", "mental_health_information_v1", "information_request", [match])

    coping = [
        "how can i calm down",
        "help me sleep",
        "cannot sleep",
        "cant sleep",
        "can't sleep",
        "what can i do about exam stress",
        "exam stress",
        "breathing exercise",
        "panic before presentation",
        "calm down",
    ]
    match = _has_any(compact, coping)
    if match:
        topic = "Sleep support" if "sleep" in match else "Exam stress" if "exam" in compact else "Coping exercise"
        return _route("coping_support_request", "low", "coping_support_request_v1", "coping_support_request", [match], topic_label=topic)

    emotional = [
        "i feel sad",
        "feel sad",
        "im sad",
        "i'm sad",
        "i feel lonely",
        "feel lonely",
        "i am lonely",
        "i'm lonely",
        "nobody care",
        "nobody cares",
        "no one cares",
        "no one wished",
        "birthday but no one wished",
        "i am worried",
        "i'm worried",
        "feel overwhelmed",
        "i feel overwhelmed",
        "i am stressed",
        "i'm stressed",
    ]
    match = _has_any(compact, emotional)
    if match:
        topic = (
            "Exam stress"
            if "exam" in compact
            else "Loneliness"
            if any(item in compact for item in ("lonely", "nobody care", "nobody cares", "no one cares", "no one wished", "birthday"))
            else "Anxiety support"
            if "worried" in match
            else "Low mood"
        )
        return _route("emotional_disclosure", "low", "emotional_disclosure_v1", "emotional_disclosure", [match], topic_label=topic)

    positive_exact = {"happy", "okay", "ok", "fine", "good"}
    positive = ["i feel happy", "today was good", "i am okay", "i'm okay", "i passed my exam", "had a good day", "i am happy"]
    match = _has_any(compact, positive)
    if match or compact in positive_exact:
        return _route("positive_or_stable", "none", "positive_or_stable_v1", "positive_or_stable", [match or compact])

    greetings = {"hi", "hlo", "hii", "hiii", "hello", "hey", "good morning", "good evening", "good afternoon"}
    if compact in greetings:
        return _route("greeting", "none", "greeting_v1", "greeting", [compact])

    closing = {"thank you", "thanks", "bye", "goodbye", "good night"}
    if compact in closing:
        return _route("gratitude_or_closing", "none", "gratitude_or_closing_v1", "gratitude_or_closing", [compact])

    if classifier_signal and classifier_signal.get("label") in {"suicidal", "suicidal_ideation"} and classifier_signal.get("confidence", 0) >= 0.9:
        return _route(
            "severe_distress",
            "medium",
            "severe_distress_v1",
            "ambiguous_high_classifier_signal",
            ["classifier_supporting_signal"],
            safety_question=True,
            allow_normal=False,
            classifier_used=classifier_used,
            classifier_result={"label": classifier_signal.get("label")},
        )

    return _route(
        "unclear_or_other",
        "none",
        "unclear_or_other_v1",
        "unclear_or_other",
        [],
        classifier_used=classifier_used,
        classifier_result={"label": classifier_signal.get("label")} if classifier_signal else None,
    )


def _variant_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        "greeting": [
            {
                "id": "greeting_v2_a",
                "message": "Hi. I'm here with you. What feels most important to talk about right now?",
                "follow_up": None,
            },
            {
                "id": "greeting_v2_b",
                "message": "Hello. I'm glad you came in. How has today been treating you so far?",
                "follow_up": None,
            },
        ],
        "emotional_disclosure": [
            {
                "id": "emotional_v2_a",
                "message": "That sounds heavy to carry today.",
                "follow_up": "Are you feeling more sad, lonely, stressed, or something else right now?",
            },
            {
                "id": "emotional_v2_b",
                "message": "I can hear that this is affecting you personally, not just in the background.",
                "follow_up": "What happened before that feeling became strongest?",
            },
            {
                "id": "emotional_v2_c",
                "message": "That sounds painful, especially when you were hoping to feel remembered or cared for.",
                "follow_up": "Are you feeling more lonely, disappointed, or both right now?",
                "topics": ["Loneliness"],
            },
        ],
        "coping_support_request": [
            {
                "id": "coping_v2_a",
                "message": "Let's make this smaller for the next few minutes.",
                "follow_up": "Try one slow breath in, one longer breath out, then tell me what feels most urgent.",
            },
            {
                "id": "coping_v2_b",
                "message": "Sleep difficulty can feel worse when your mind keeps working after your body is tired.",
                "follow_up": "Would it help to try a short wind-down step now?",
                "topics": ["Sleep support"],
            },
            {
                "id": "coping_v2_c",
                "message": "Exam pressure can make everything feel immediate.",
                "follow_up": "What is the smallest study or recovery step you can take in the next 10 minutes?",
                "topics": ["Exam stress"],
            },
        ],
        "positive_or_stable": [
            {
                "id": "positive_v2_a",
                "message": "I'm glad there is something steady or positive here.",
                "follow_up": "What helped make today feel a little better?",
            },
            {
                "id": "positive_v2_b",
                "message": "That is worth noticing.",
                "follow_up": "Do you want to mark what went well so you can return to it later?",
            },
        ],
        "gratitude_or_closing": [
            {
                "id": "closing_v2_a",
                "message": "You're welcome. I'm glad you checked in.",
                "follow_up": None,
            },
            {
                "id": "closing_v2_b",
                "message": "Take care of yourself today. You can come back whenever you want to talk.",
                "follow_up": None,
            },
        ],
        "mental_health_information": [
            {
                "id": "info_v2_a",
                "message": "I can explain that in a general way, but I can't diagnose you.",
                "follow_up": "Are you asking because of something you are experiencing, or just trying to understand the term?",
            }
        ],
        "unclear_or_other": [
            {
                "id": "unclear_v2_a",
                "message": "I want to understand you better.",
                "follow_up": "Could you say that another way, even in a few words?",
            },
            {
                "id": "unclear_v2_b",
                "message": "I'm not fully sure what you mean yet.",
                "follow_up": "Are you talking about how you feel, something that happened, or what you need next?",
            },
            {
                "id": "unclear_v2_c",
                "message": "I may be missing the main point.",
                "follow_up": "What part should I pay attention to first?",
            },
        ],
    }


def _select_variant(route: SafetyRoute, context: Optional[dict[str, Any]]) -> dict[str, Any]:
    variants = _variant_catalog().get(route.route) or []
    topic_specific = [item for item in variants if route.topic_label in item.get("topics", [])]
    generic = [item for item in variants if not item.get("topics")]
    candidates = topic_specific or generic or variants
    if not candidates:
        return {}
    previous_variant = (context or {}).get("previous_response_variant_id")
    for item in candidates:
        if item["id"] != previous_variant:
            return item
    return candidates[0]


def render_safety_response(route: SafetyRoute, context_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    template_payload = load_templates()
    template = template_payload["templates"][route.safe_response_template_id]
    resources = load_crisis_resources()
    is_crisis = route.route in {
        "possible_self_harm_or_crisis",
        "explicit_suicidal_intent",
        "imminent_self_harm",
        "severe_distress",
    }
    variant = {} if is_crisis else _select_variant(route, context_state)
    resource_actions = list(template.get("resource_actions") or [])
    if route.recommend_immediate_support:
        resource_actions.append(
            {
                "type": "fallback",
                "label": resources["fallback"]["label"],
                "emergency_designation": resources["fallback"]["emergency_designation"],
            }
        )
    return {
        "message": variant.get("message") or template["response"],
        "follow_up": variant.get("follow_up") if variant else template.get("follow_up"),
        "recommended_actions": template.get("recommended_actions") or [],
        "resource_actions": resource_actions,
        "template_version": template_payload["version"],
        "response_policy_version": RESPONSE_POLICY_VERSION,
        "variant_id": variant.get("id") or route.safe_response_template_id,
    }


def make_context_state(route: SafetyRoute, *, previous: Optional[dict[str, Any]] = None, now: Optional[datetime] = None) -> dict[str, Any]:
    now = now or datetime.utcnow()
    state = {
        "version": "safetalk-context-v1",
        "previous_route": route.route,
        "unresolved_safety_question": route.requires_direct_safety_question,
        "recent_crisis_state": route.route in {"possible_self_harm_or_crisis", "explicit_suicidal_intent", "imminent_self_harm"},
        "recent_user_denial_or_confirmation": None,
        "previous_response_variant_id": previous.get("previous_response_variant_id") if previous else None,
        "crisis_message_at": None,
        "updated_at": now.isoformat(),
        "context_window_minutes": CONTEXT_WINDOW_MINUTES,
    }
    if previous and previous.get("recent_crisis_state") and not state["recent_crisis_state"]:
        crisis_at = previous.get("crisis_message_at")
        state["crisis_message_at"] = crisis_at
        state["recent_crisis_state"] = bool(_recent_context(previous, now).get("recent_crisis_state"))
    if route.route in {"possible_self_harm_or_crisis", "explicit_suicidal_intent", "imminent_self_harm"}:
        state["crisis_message_at"] = now.isoformat()
    if route.internal_reason_code == "contextual_safety_confirmation":
        state["recent_user_denial_or_confirmation"] = "confirmation"
    if route.internal_reason_code == "contextual_safety_denial":
        state["recent_user_denial_or_confirmation"] = "denial"
        state["unresolved_safety_question"] = False
    return state


def make_title(user_message: str, topic_label: str) -> str:
    if topic_label == "Greeting":
        return "Greeting"
    compact = re.sub(r"\s+", " ", (user_message or "").strip())
    if not compact:
        return topic_label
    if len(compact) > 48:
        compact = compact[:45].rstrip() + "..."
    return compact[0].upper() + compact[1:]
