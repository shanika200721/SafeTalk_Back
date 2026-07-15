"""Authoritative DASS-21 scoring functions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from app.ml.preprocessing.dass21.constants import (
    DASS21_EXPECTED_ITEMS,
    DASS21_ITEM_MAPPING,
    DASS21_SCORING_VERSION,
    DASS21_SEVERITY_THRESHOLDS,
    ITEM_MULTIPLIER,
    QUESTIONNAIRE_VERSION,
    RESPONSE_MAX,
    RESPONSE_MIN,
    RESPONSE_SCALE_0_3,
    SUBSCALE_MAX_MULTIPLIED_SCORE,
    SUBSCALES,
)
from app.ml.preprocessing.dass21.schemas import (
    DASS21Responses,
    DASS21ScoreResult,
    DASS21SubscaleScore,
)


def _normalize_question_id(question_id: Any) -> str:
    if isinstance(question_id, int):
        if 1 <= question_id <= 21:
            return f"Q{question_id}"
        raise ValueError(f"Question number out of range: {question_id}")
    text = str(question_id).strip().upper()
    if text.isdigit():
        return _normalize_question_id(int(text))
    if text.startswith("Q") and text[1:].isdigit():
        number = int(text[1:])
        if number >= 1:
            return f"Q{number}"
    raise ValueError(f"Invalid DASS-21 question identifier: {question_id}")


def _coerce_response_mapping(responses: Mapping[Any, Any] | Iterable[tuple[Any, Any]]) -> dict[str, int]:
    if isinstance(responses, Mapping):
        iterable = responses.items()
    else:
        iterable = responses

    result: dict[str, int] = {}
    duplicates: list[str] = []
    for raw_question_id, raw_value in iterable:
        question_id = _normalize_question_id(raw_question_id)
        if question_id in result:
            duplicates.append(question_id)
        try:
            result[question_id] = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Response for {question_id} must be an integer") from exc

    if duplicates:
        raise ValueError(f"Duplicate DASS-21 question identifiers: {sorted(set(duplicates))}")
    return result


def convert_frontend_payload(payload: Mapping[str, Any] | list[int] | tuple[int, ...]) -> DASS21Responses:
    """Convert the current frontend/API payload into typed DASS-21 responses."""

    responses = payload.get("responses") if isinstance(payload, Mapping) else payload
    if not isinstance(responses, (list, tuple)):
        raise ValueError("Frontend DASS-21 payload must contain a list of responses")
    if len(responses) != 21:
        raise ValueError("DASS-21 frontend payload requires exactly 21 responses")
    mapping = {f"Q{index + 1}": value for index, value in enumerate(responses)}
    return DASS21Responses(responses=validate_dass21_responses(mapping))


def validate_dass21_responses(
    responses: Mapping[Any, Any] | Iterable[tuple[Any, Any]],
    *,
    allow_unknown: bool = False,
    require_complete: bool = True,
) -> dict[str, int]:
    """Validate response identifiers and the verified 0-3 response range."""

    response_map = _coerce_response_mapping(responses)
    expected = set(DASS21_EXPECTED_ITEMS)
    actual = set(response_map)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual, key=lambda item: int(item[1:]))

    if unknown and not allow_unknown:
        raise ValueError(f"Unknown DASS-21 question identifiers: {unknown}")
    if missing and require_complete:
        raise ValueError(f"Missing DASS-21 question responses: {missing}")

    for question_id, value in response_map.items():
        if question_id not in expected and allow_unknown:
            continue
        if value < RESPONSE_MIN or value > RESPONSE_MAX:
            raise ValueError(
                f"Response for {question_id} must be between {RESPONSE_MIN} and {RESPONSE_MAX}"
            )

    return {question_id: response_map[question_id] for question_id in DASS21_EXPECTED_ITEMS if question_id in response_map}


def calculate_subscale_raw_scores(responses: Mapping[Any, Any]) -> dict[str, int]:
    response_map = validate_dass21_responses(responses)
    return {
        subscale: sum(response_map[question_id] for question_id in items)
        for subscale, items in DASS21_ITEM_MAPPING.items()
    }


def apply_dass21_multiplier(raw_score: int, *, already_multiplied: bool = False) -> int:
    """Apply the DASS-21 x2 multiplier unless the value is already final-score scale."""

    return int(raw_score) if already_multiplied else int(raw_score) * ITEM_MULTIPLIER


def _classify_severity(subscale: str, multiplied_score: int) -> str:
    if subscale not in DASS21_SEVERITY_THRESHOLDS:
        raise ValueError(f"Unknown DASS-21 subscale: {subscale}")
    for severity, (minimum, maximum) in DASS21_SEVERITY_THRESHOLDS[subscale].items():
        if minimum <= multiplied_score <= maximum:
            return severity
    return "extremely_severe"


def classify_depression_severity(multiplied_score: int) -> str:
    return _classify_severity("depression", multiplied_score)


def classify_anxiety_severity(multiplied_score: int) -> str:
    return _classify_severity("anxiety", multiplied_score)


def classify_stress_severity(multiplied_score: int) -> str:
    return _classify_severity("stress", multiplied_score)


def normalize_subscale_score(multiplied_score: int) -> float:
    """Return a non-clinical engineering normalization on a 0-100 scale."""

    bounded = max(0, min(SUBSCALE_MAX_MULTIPLIED_SCORE, int(multiplied_score)))
    return round((bounded / SUBSCALE_MAX_MULTIPLIED_SCORE) * 100, 2)


def _build_subscale_score(subscale: str, raw_score: int) -> DASS21SubscaleScore:
    multiplied = apply_dass21_multiplier(raw_score)
    return DASS21SubscaleScore(
        raw_score=raw_score,
        multiplied_score=multiplied,
        severity=_classify_severity(subscale, multiplied),
        normalized_score_0_100=normalize_subscale_score(multiplied),
    )


def score_dass21(
    responses: DASS21Responses | Mapping[Any, Any] | Iterable[tuple[Any, Any]],
    *,
    allow_incomplete: bool = False,
    scored_at: datetime | None = None,
) -> DASS21ScoreResult:
    """Score a complete DASS-21 response set without imputing missing answers."""

    if isinstance(responses, DASS21Responses):
        response_map = responses.responses
        questionnaire_version = responses.questionnaire_version
    else:
        response_map = _coerce_response_mapping(responses)
        questionnaire_version = QUESTIONNAIRE_VERSION

    validated = validate_dass21_responses(response_map, require_complete=not allow_incomplete)
    missing_count = 21 - len(validated)
    if missing_count:
        raise ValueError("Incomplete DASS-21 scoring is not supported without explicit imputation, which is disabled")

    raw_scores = calculate_subscale_raw_scores(validated)
    scored_time = scored_at or datetime.now(timezone.utc)
    if scored_time.tzinfo is None or scored_time.tzinfo.utcoffset(scored_time) is None:
        raise ValueError("scored_at must be timezone-aware")

    return DASS21ScoreResult(
        scoring_version=DASS21_SCORING_VERSION,
        questionnaire_version=questionnaire_version,
        depression=_build_subscale_score("depression", raw_scores["depression"]),
        anxiety=_build_subscale_score("anxiety", raw_scores["anxiety"]),
        stress=_build_subscale_score("stress", raw_scores["stress"]),
        answered_item_count=len(validated),
        missing_item_count=missing_count,
        is_complete=missing_count == 0,
        validation_warnings=[],
        scored_at=scored_time.astimezone(timezone.utc),
    )


def convert_database_record(record: Any) -> DASS21ScoreResult:
    """Convert an existing database record without double multiplying stored totals."""

    responses = getattr(record, "responses", None)
    if responses:
        return score_dass21(convert_frontend_payload(list(responses)))

    required = ("depression_score", "anxiety_score", "stress_score")
    if not all(hasattr(record, field) for field in required):
        raise ValueError("Database record must contain responses or DASS-21 subscale score fields")

    subscale_scores = {}
    for subscale in SUBSCALES:
        multiplied = int(getattr(record, f"{subscale}_score"))
        raw = int(multiplied / ITEM_MULTIPLIER)
        subscale_scores[subscale] = DASS21SubscaleScore(
            raw_score=raw,
            multiplied_score=apply_dass21_multiplier(multiplied, already_multiplied=True),
            severity=_classify_severity(subscale, multiplied),
            normalized_score_0_100=normalize_subscale_score(multiplied),
        )

    return DASS21ScoreResult(
        depression=subscale_scores["depression"],
        anxiety=subscale_scores["anxiety"],
        stress=subscale_scores["stress"],
        answered_item_count=0,
        missing_item_count=21,
        is_complete=False,
        validation_warnings=[
            "Converted from stored subscale totals; individual responses were not available.",
            "Stored totals were treated as already multiplied final scores.",
        ],
    )


def explain_dass21_result(result: DASS21ScoreResult) -> dict[str, Any]:
    """Return a privacy-safe explanation without raw answers or risk classification."""

    return {
        "scoring_version": result.scoring_version,
        "questionnaire_version": result.questionnaire_version,
        "response_scale": RESPONSE_SCALE_0_3,
        "multiplier": ITEM_MULTIPLIER,
        "subscales": {
            "depression": result.depression.to_safe_dict(),
            "anxiety": result.anxiety.to_safe_dict(),
            "stress": result.stress.to_safe_dict(),
        },
        "normalized_score_warning": (
            "normalized_score_0_100 is a non-clinical engineering output and is not a diagnosis."
        ),
        "risk_warning": "DASS-21 alone does not calculate suicide-risk level, alerts, or treatment recommendations.",
    }


def to_legacy_api_dict(result: DASS21ScoreResult, *, include_responses: list[int] | None = None) -> dict[str, Any]:
    """Return existing API field names while preserving authoritative scoring."""

    def label(value: str) -> str:
        return value.replace("_", " ").title()

    payload = {
        "depression_score": result.depression.multiplied_score,
        "anxiety_score": result.anxiety.multiplied_score,
        "stress_score": result.stress.multiplied_score,
        "total_dass21_score": (
            result.depression.multiplied_score
            + result.anxiety.multiplied_score
            + result.stress.multiplied_score
        ),
        "depression_severity": label(result.depression.severity),
        "anxiety_severity": label(result.anxiety.severity),
        "stress_severity": label(result.stress.severity),
    }
    if include_responses is not None:
        payload["responses"] = include_responses
    return payload
