from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.ml.runtime.base import RuntimeInferenceError, RuntimeModelUnavailable
from app.ml.runtime.registry import predict_with_active_model
from app.models.database_models import (
    CounselorAssignment,
    FeatureSnapshot,
    ModalityPrediction,
    ProfileAssessment,
    User,
    UserRole,
)
from app.services.consent import CURRENT_POLICY_VERSION, get_latest_consent, has_active_consent
from app.services.modalities import (
    SCREENING_LIMITATION,
    create_failed_prediction,
    create_feature_snapshot,
    create_prediction,
    create_unavailable_prediction,
    trigger_fusion_for_prediction,
)


QUESTIONNAIRE_VERSION = "profile_assessment_v2"
PREPROCESSING_VERSION = "profile-runtime-v2"
FEATURE_SCHEMA_VERSION = "profile-v2-feature-contract"
ENCODING_VERSION = "profile-v2-categorical-raw"
REFRESH_AFTER_DAYS = 365
MODEL_FEATURE_ORDER = ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
PREFER_NOT_TO_SAY = "prefer_not_to_say"


def option(value: str, label: str) -> dict[str, str]:
    return {"value": value, "label": label}


QUESTIONS: list[dict[str, Any]] = [
    {
        "question_id": "age_range",
        "category": "basic_student_information",
        "step": "About You",
        "display_order": 10,
        "label": "Age range",
        "help_text": "Used only as broad background context.",
        "response_type": "single_choice",
        "options": [option("under_18", "Under 18"), option("18_20", "18-20"), option("21_24", "21-24"), option("25_plus", "25 or older"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "hidden",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "gender_option",
        "category": "basic_student_information",
        "step": "About You",
        "display_order": 20,
        "label": "Gender",
        "help_text": "Optional. This system does not infer gender.",
        "response_type": "single_choice",
        "options": [option("woman", "Woman"), option("man", "Man"), option("non_binary", "Non-binary"), option("self_describe", "Self describe"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "sensitive_optional",
        "counselor_summary_visibility": "hidden",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "preferred_language",
        "category": "basic_student_information",
        "step": "About You",
        "display_order": 30,
        "label": "Preferred language for support",
        "help_text": "Helps tailor support communication.",
        "response_type": "single_choice",
        "options": [option("english", "English"), option("sinhala", "Sinhala"), option("tamil", "Tamil"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "current_year_of_study",
        "category": "academic_context",
        "step": "Academic Life",
        "display_order": 110,
        "label": "Current year of study",
        "help_text": "This is part of the verified profile model contract.",
        "response_type": "single_choice",
        "options": [option("year 1", "Year 1"), option("year 2", "Year 2"), option("year 3", "Year 3"), option("year 4", "Year 4")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "model_feature",
        "model_feature": True,
        "feature_name": "year_of_study",
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "standard",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "academic_performance_category",
        "category": "academic_context",
        "step": "Academic Life",
        "display_order": 120,
        "label": "Approximate academic performance",
        "help_text": "Use a broad category; exact grades are not requested.",
        "response_type": "single_choice",
        "options": [option("doing_well", "Doing well"), option("average", "Average"), option("struggling", "Struggling"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "academic_workload",
        "category": "academic_context",
        "step": "Academic Life",
        "display_order": 130,
        "label": "Perceived academic workload",
        "help_text": "How manageable does your workload feel lately?",
        "response_type": "single_choice",
        "options": [option("manageable", "Manageable"), option("heavy", "Heavy"), option("overwhelming", "Overwhelming"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "recent_academic_setback",
        "category": "academic_context",
        "step": "Academic Life",
        "display_order": 140,
        "label": "Recent academic setback",
        "help_text": "For example, failed exam, delayed assignment, or difficulty attending classes.",
        "response_type": "single_choice",
        "options": [option("no", "No"), option("yes", "Yes"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "financial_strain",
        "category": "financial_context",
        "step": "Financial Situation",
        "display_order": 210,
        "label": "Current financial strain",
        "help_text": "No exact income is requested.",
        "response_type": "single_choice",
        "options": [option("low", "Low"), option("moderate", "Moderate"), option("high", "High"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "employment_while_studying",
        "category": "financial_context",
        "step": "Financial Situation",
        "display_order": 220,
        "label": "Employment while studying",
        "help_text": "Optional context about time and financial pressure.",
        "response_type": "single_choice",
        "options": [option("none", "Not employed"), option("part_time", "Part-time"), option("full_time", "Full-time"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "family_support",
        "category": "family_and_social_support",
        "step": "Family and Social Support",
        "display_order": 310,
        "label": "Perceived family support",
        "help_text": "Choose the closest current description.",
        "response_type": "single_choice",
        "options": [option("strong", "Strong"), option("some", "Some"), option("limited", "Limited"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "friend_support",
        "category": "family_and_social_support",
        "step": "Family and Social Support",
        "display_order": 320,
        "label": "Perceived friend support",
        "help_text": "Choose the closest current description.",
        "response_type": "single_choice",
        "options": [option("strong", "Strong"), option("some", "Some"), option("limited", "Limited"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "social_isolation",
        "category": "family_and_social_support",
        "step": "Family and Social Support",
        "display_order": 330,
        "label": "Feeling socially isolated",
        "help_text": "This is background context, not a diagnosis.",
        "response_type": "single_choice",
        "options": [option("rarely", "Rarely"), option("sometimes", "Sometimes"), option("often", "Often"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "living_situation",
        "category": "living_situation",
        "step": "Living and Lifestyle",
        "display_order": 410,
        "label": "Current living situation",
        "help_text": "No home address is requested.",
        "response_type": "single_choice",
        "options": [option("with_family", "With family"), option("hostel", "Hostel"), option("boarding", "Boarding"), option("alone", "Alone"), option("shared", "Shared accommodation"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "sleep_duration",
        "category": "lifestyle_and_daily_routine",
        "step": "Living and Lifestyle",
        "display_order": 420,
        "label": "Typical sleep duration",
        "help_text": "Approximate range is enough.",
        "response_type": "single_choice",
        "options": [option("less_than_5", "Less than 5 hours"), option("5_6", "5-6 hours"), option("7_8", "7-8 hours"), option("more_than_8", "More than 8 hours"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "sleep_quality",
        "category": "lifestyle_and_daily_routine",
        "step": "Living and Lifestyle",
        "display_order": 430,
        "label": "Sleep quality",
        "help_text": "Choose the closest recent pattern.",
        "response_type": "single_choice",
        "options": [option("good", "Good"), option("mixed", "Mixed"), option("poor", "Poor"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "university_adjustment",
        "category": "university_and_cultural_context",
        "step": "University Experience",
        "display_order": 510,
        "label": "Adjustment to university life",
        "help_text": "This helps personalize support options.",
        "response_type": "single_choice",
        "options": [option("settled", "Settled"), option("adjusting", "Still adjusting"), option("difficult", "Difficult"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "stigma_seeking_help",
        "category": "university_and_cultural_context",
        "step": "University Experience",
        "display_order": 520,
        "label": "Stigma around seeking mental-health help",
        "help_text": "Optional context about barriers to support.",
        "response_type": "single_choice",
        "options": [option("low", "Low"), option("moderate", "Moderate"), option("high", "High"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "self_reported_anxiety",
        "category": "personal_wellbeing_history",
        "step": "University Experience",
        "display_order": 610,
        "label": "Recently experienced significant anxiety",
        "help_text": "Optional self-report used by the verified profile model only when answered yes or no.",
        "response_type": "single_choice",
        "options": [option("no", "No"), option("yes", "Yes"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "model_feature",
        "model_feature": True,
        "feature_name": "self_reported_anxiety",
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "sensitive_optional",
        "counselor_summary_visibility": "limited",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "self_reported_panic_attack",
        "category": "personal_wellbeing_history",
        "step": "University Experience",
        "display_order": 620,
        "label": "Recently experienced panic attacks",
        "help_text": "Optional self-report used by the verified profile model only when answered yes or no.",
        "response_type": "single_choice",
        "options": [option("no", "No"), option("yes", "Yes"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "model_feature",
        "model_feature": True,
        "feature_name": "self_reported_panic_attack",
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "sensitive_optional",
        "counselor_summary_visibility": "limited",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "previous_counseling_support",
        "category": "personal_wellbeing_history",
        "step": "University Experience",
        "display_order": 630,
        "label": "Previously sought counseling support",
        "help_text": "Optional. Detailed clinical history is not requested.",
        "response_type": "single_choice",
        "options": [option("no", "No"), option("yes", "Yes"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "sensitive_optional",
        "counselor_summary_visibility": "hidden",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "preferred_support_channel",
        "category": "support_preferences",
        "step": "Support Preferences",
        "display_order": 710,
        "label": "Preferred support channel",
        "help_text": "This does not increase any risk score.",
        "response_type": "single_choice",
        "options": [option("online", "Online"), option("in_person", "In person"), option("phone", "Phone"), option("no_preference", "No preference")],
        "required": True,
        "allow_prefer_not_to_say": False,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
    {
        "question_id": "comfortable_contacting_counselor",
        "category": "support_preferences",
        "step": "Support Preferences",
        "display_order": 720,
        "label": "Comfortable contacting a counselor",
        "help_text": "Helps counselors understand support barriers.",
        "response_type": "single_choice",
        "options": [option("yes", "Yes"), option("unsure", "Unsure"), option("no", "No"), option(PREFER_NOT_TO_SAY, "Prefer not to say")],
        "required": False,
        "allow_prefer_not_to_say": True,
        "field_role": "contextual_only",
        "model_feature": False,
        "feature_name": None,
        "encoding_version": ENCODING_VERSION,
        "sensitivity_class": "private",
        "counselor_summary_visibility": "summary",
        "admin_visibility": "aggregate",
        "active": True,
        "questionnaire_version": QUESTIONNAIRE_VERSION,
    },
]


CATEGORY_LABELS = {
    "basic_student_information": "Basic Student Information",
    "academic_context": "Academic Context",
    "financial_context": "Financial Context",
    "family_and_social_support": "Family and Social Support",
    "living_situation": "Living Situation",
    "lifestyle_and_daily_routine": "Lifestyle and Daily Routine",
    "university_and_cultural_context": "University and Cultural Context",
    "personal_wellbeing_history": "Personal Wellbeing History",
    "support_preferences": "Support Preferences",
    "optional_sensitive_context": "Optional Sensitive Context",
}


QUESTION_BY_ID = {question["question_id"]: question for question in QUESTIONS}
ASSESSMENT_TERMINAL_STATUSES = {"submitted", "completed"}
ASSESSMENT_SOURCE = "phase4l_student_profile_assessment"


def questionnaire_contract() -> dict[str, Any]:
    steps = []
    seen = set()
    for question in sorted(QUESTIONS, key=lambda item: item["display_order"]):
        if question["step"] not in seen:
            seen.add(question["step"])
            steps.append(question["step"])
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "title": "Profile Assessment",
        "description": (
            "Background information for personalized screening support. "
            "This is not a diagnostic questionnaire, not a replacement for DASS-21, "
            "and not an emergency assessment."
        ),
        "refresh_policy": {"days": REFRESH_AFTER_DAYS, "reason": "annual_refresh_or_material_context_change"},
        "steps": steps + ["Review and Submit"],
        "categories": [{"id": key, "label": label} for key, label in CATEGORY_LABELS.items()],
        "questions": sorted(QUESTIONS, key=lambda item: item["display_order"]),
        "model_feature_order": MODEL_FEATURE_ORDER,
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def _reject_html(value: Any, question_id: str) -> None:
    if isinstance(value, str) and any(fragment in value.lower() for fragment in ("<script", "</", "javascript:")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsafe content in {question_id}")


def validate_responses(responses: dict[str, Any], *, require_required: bool) -> dict[str, Any]:
    if not isinstance(responses, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="responses must be an object")
    unknown = sorted(set(responses) - set(QUESTION_BY_ID))
    if unknown:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "UNKNOWN_QUESTION_ID", "questions": unknown})

    cleaned: dict[str, Any] = {}
    for question_id, value in responses.items():
        question = QUESTION_BY_ID[question_id]
        _reject_html(value, question_id)
        if value in (None, ""):
            continue
        if question["response_type"] == "single_choice":
            allowed = {item["value"] for item in question["options"]}
            if value not in allowed:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "INVALID_OPTION", "question_id": question_id})
        elif question["response_type"] == "short_text":
            if not isinstance(value, str) or len(value) > 300:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "INVALID_TEXT", "question_id": question_id})
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "UNSUPPORTED_RESPONSE_TYPE", "question_id": question_id})
        cleaned[question_id] = value

    if require_required:
        missing = [
            question["question_id"]
            for question in QUESTIONS
            if question["required"] and question["question_id"] not in cleaned
        ]
        if missing:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "REQUIRED_FIELDS_MISSING", "questions": missing})
    return cleaned


def preprocess_profile_responses(responses: dict[str, Any]) -> dict[str, Any]:
    validation_errors: list[dict[str, str]] = []
    features: dict[str, str] = {}
    feature_map: list[dict[str, Any]] = []

    for feature_name in MODEL_FEATURE_ORDER:
        question = next(item for item in QUESTIONS if item.get("feature_name") == feature_name)
        raw = responses.get(question["question_id"])
        if raw in (None, "", PREFER_NOT_TO_SAY):
            validation_errors.append({"feature_name": feature_name, "reason": "missing_or_prefer_not_to_say"})
            encoded = None
        elif feature_name == "year_of_study":
            encoded = normalize_year_of_study(raw)
        elif feature_name in {"self_reported_anxiety", "self_reported_panic_attack"}:
            encoded = normalize_yes_no(raw)
        else:
            encoded = raw
        if encoded is not None:
            features[feature_name] = encoded
        feature_map.append(
            {
                "question_id": question["question_id"],
                "raw_response": raw,
                "encoded_feature": encoded,
                "model_feature": feature_name,
                "encoding_version": ENCODING_VERSION,
            }
        )

    eligible = not validation_errors and list(features) == MODEL_FEATURE_ORDER
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": MODEL_FEATURE_ORDER,
        "features": features,
        "feature_map": feature_map,
        "validation_errors": validation_errors,
        "eligible_for_model": eligible,
        "contextual_only": {
            key: value
            for key, value in responses.items()
            if key in QUESTION_BY_ID and not QUESTION_BY_ID[key].get("model_feature")
        },
    }


def normalize_year_of_study(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "year1", "year 1"}:
        return "year 1"
    if text in {"2", "year2", "year 2"}:
        return "year 2"
    if text in {"3", "year3", "year 3"}:
        return "year 3"
    if text in {"4", "year4", "year 4"}:
        return "year 4"
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid year of study for profile preprocessing")


def normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "no"}:
        return text
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid binary self-report response")


def latest_phase4l_assessment(db: Session, user_id: int) -> ProfileAssessment | None:
    return (
        db.query(ProfileAssessment)
        .filter(
            ProfileAssessment.user_id == user_id,
            ProfileAssessment.questionnaire_version == QUESTIONNAIRE_VERSION,
            or_(ProfileAssessment.source == ASSESSMENT_SOURCE, ProfileAssessment.source.is_(None)),
        )
        .order_by(ProfileAssessment.updated_at.desc(), ProfileAssessment.id.desc())
        .first()
    )


def assessment_completion_status(assessment: ProfileAssessment | None) -> str:
    if not assessment:
        return "not_started"
    if assessment.stale_at and assessment.stale_at <= datetime.utcnow() and assessment.submitted_at:
        return "needs_update"
    if assessment.submitted_at or assessment.completed_at or assessment.status in ASSESSMENT_TERMINAL_STATUSES:
        return "completed"
    if assessment.status == "draft":
        return "draft"
    return assessment.status or "draft"


def prediction_status(assessment: ProfileAssessment | None) -> str:
    if not assessment:
        return "unavailable"
    if assessment.prediction:
        return assessment.prediction.status or "pending"
    if assessment.status == "processing":
        return "pending"
    return "unavailable"


def completion_percentage(assessment: ProfileAssessment | None) -> int:
    if not assessment:
        return 0
    if assessment_completion_status(assessment) in {"submitted", "completed", "needs_update", "stale"}:
        return 100
    responses = assessment.responses_json or {}
    if not QUESTIONS:
        return 0
    return min(100, round((len(responses) / len(QUESTIONS)) * 100))


def profile_status_payload(db: Session, user: User) -> dict[str, Any]:
    assessment = latest_phase4l_assessment(db, user.id)
    if not assessment:
        return {
            "assessment_status": "not_started",
            "status": "not_started",
            "message": "Profile assessment not completed",
            "questionnaire_version": QUESTIONNAIRE_VERSION,
            "assessment_id": None,
            "profile_assessment_id": None,
            "submitted_at": None,
            "completed_at": None,
            "updated_at": None,
            "stale_at": None,
            "completion_percentage": 0,
            "started_at": None,
            "can_continue": False,
            "can_update": False,
            "update_recommended": False,
            "prediction_status": "unavailable",
            "prediction_available": False,
        }

    stale = bool(assessment.stale_at and assessment.stale_at <= datetime.utcnow())
    assessment_status = assessment_completion_status(assessment)
    prediction = assessment.prediction
    return {
        "assessment_status": assessment_status,
        "status": assessment_status,
        "message": "Profile assessment not completed" if assessment_status in {"not_started", "draft"} else "Profile assessment completed.",
        "questionnaire_version": assessment.questionnaire_version,
        "assessment_id": assessment.id,
        "profile_assessment_id": assessment.profile_assessment_id,
        "started_at": assessment.started_at,
        "submitted_at": assessment.submitted_at,
        "completed_at": assessment.completed_at,
        "stale_at": assessment.stale_at,
        "updated_at": assessment.updated_at,
        "completion_percentage": completion_percentage(assessment),
        "can_continue": assessment_status == "draft",
        "can_update": assessment_status in {"completed", "needs_update", "stale"},
        "update_recommended": stale,
        "prediction_id": assessment.prediction_id,
        "prediction_status": prediction_status(assessment),
        "prediction_available": bool(prediction and prediction.status == "succeeded" and prediction.is_available),
    }


def create_or_update_draft(db: Session, user: User, responses: dict[str, Any], *, assessment_id: int | None = None) -> ProfileAssessment:
    cleaned = validate_responses(responses, require_required=False)
    now = datetime.utcnow()
    assessment = None
    if assessment_id:
        assessment = db.query(ProfileAssessment).filter(ProfileAssessment.id == assessment_id).first()
        if not assessment or assessment.user_id != user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile assessment draft not found")
    if assessment is None:
        assessment = latest_phase4l_assessment(db, user.id)
    if assessment is None or assessment.status in {"submitted", "processing", "completed"}:
        assessment = ProfileAssessment(
            profile_assessment_id=f"profile-4l-{uuid4().hex[:18]}",
            user_id=user.id,
            questionnaire_version=QUESTIONNAIRE_VERSION,
            source=ASSESSMENT_SOURCE,
            started_at=now,
            created_at=now,
        )
        db.add(assessment)
        db.flush()
    assessment.status = "draft"
    assessment.responses_json = cleaned
    assessment.updated_at = now
    assessment.privacy_metadata_json = privacy_metadata()
    db.flush()
    return assessment


def submit_profile_assessment(db: Session, user: User, responses: dict[str, Any], *, assessment_id: int | None = None) -> ProfileAssessment:
    storage_consent = get_latest_consent(db, user.id, "profile_data_storage") or get_latest_consent(db, user.id, "profile_processing")
    if not storage_consent or not storage_consent.is_granted or storage_consent.withdrawn_at is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "CONSENT_REQUIRED", "consent_type": "profile_data_storage"},
        )
    cleaned = validate_responses(responses, require_required=True)
    assessment = create_or_update_draft(db, user, cleaned, assessment_id=assessment_id)
    now = datetime.utcnow()
    normalized = preprocess_profile_responses(cleaned)
    assessment.status = "completed"
    assessment.submitted_at = now
    assessment.completed_at = now
    assessment.updated_at = now
    assessment.normalized_features_json = normalized
    assessment.preprocessing_version = PREPROCESSING_VERSION
    assessment.consent_record_id = storage_consent.id
    assessment.stale_at = now + timedelta(days=REFRESH_AFTER_DAYS)
    db.flush()
    db.commit()
    db.refresh(assessment)

    try:
        prediction = process_profile_prediction(db, user, assessment, normalized)
    except Exception:
        prediction = create_failed_prediction(
            db,
            user=user,
            modality="profile",
            failure_code="PROFILE_PROCESSING_UNEXPECTED",
            message="Profile assessment was saved, but profile inference could not safely complete.",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=assessment.submitted_at,
        )
    assessment.prediction_id = prediction.id if prediction else None
    assessment.status = "completed"
    assessment.updated_at = datetime.utcnow()
    db.flush()
    if prediction:
        trigger_fusion_for_prediction(db, prediction, trigger_source="profile_assessment_submission", actor=user)
    return assessment


def process_profile_prediction(
    db: Session,
    user: User,
    assessment: ProfileAssessment,
    normalized: dict[str, Any],
) -> ModalityPrediction:
    source_time = assessment.submitted_at or assessment.updated_at or assessment.created_at
    if not has_active_consent(db, user.id, "profile_model_processing") and not has_active_consent(db, user.id, "profile_processing"):
        return create_unavailable_prediction(
            db,
            user=user,
            modality="profile",
            failure_code="PROFILE_MODEL_CONSENT_REQUIRED",
            message="Profile assessment was saved, but model processing consent is not active.",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
        )
    if not normalized.get("eligible_for_model"):
        return create_unavailable_prediction(
            db,
            user=user,
            modality="profile",
            failure_code="PROFILE_FEATURES_INCOMPLETE",
            message="Profile assessment was saved, but optional model fields were skipped or unavailable.",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
        )

    snapshot: FeatureSnapshot | None = None
    active_model = None
    try:
        active_model, result = predict_with_active_model(db, modality="profile", payload=normalized["features"])
        snapshot = create_feature_snapshot(
            db,
            student_id=user.id,
            modality="profile",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
            feature_schema_version=active_model.feature_schema_version or FEATURE_SCHEMA_VERSION,
            preprocessing_version=active_model.preprocessing_version or PREPROCESSING_VERSION,
            features_json=result.features,
            metadata_json={
                "questionnaire_version": QUESTIONNAIRE_VERSION,
                "feature_order": MODEL_FEATURE_ORDER,
                "feature_map": normalized["feature_map"],
                "raw_responses_stored_in_prediction": False,
            },
        )
        return create_prediction(
            db,
            student_id=user.id,
            modality="profile",
            status_value="succeeded",
            is_available=True,
            output_type="machine_learning",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
            feature_snapshot=snapshot,
            probability=result.probability,
            confidence=result.confidence,
            label=result.label,
            metadata_json={
                **result.metadata,
                "class_probabilities": result.probabilities,
                "questionnaire_version": QUESTIONNAIRE_VERSION,
                "student_facing_label": "Background profile screening evidence",
                "limitations": [SCREENING_LIMITATION, "Profile data does not diagnose suicide risk."],
            },
            model_registry=active_model,
            valid_for_hours=REFRESH_AFTER_DAYS * 24,
        )
    except RuntimeModelUnavailable:
        return create_unavailable_prediction(
            db,
            user=user,
            modality="profile",
            failure_code="MODEL_NOT_ACTIVE",
            message="Profile assessment was saved, but no verified active profile model is available.",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
        )
    except RuntimeInferenceError:
        return create_failed_prediction(
            db,
            user=user,
            modality="profile",
            failure_code="PROFILE_INFERENCE_FAILED",
            message="Profile assessment was saved, but profile inference could not safely complete.",
            source_type="profile_assessment",
            source_record_id=assessment.id,
            source_timestamp=source_time,
            feature_snapshot=snapshot,
            model_registry=active_model,
        )


def privacy_metadata() -> dict[str, Any]:
    return {
        "raw_response_access": "student_only_by_default",
        "admin_raw_access": "denied_by_default",
        "counselor_visibility": "field_level_summary_only",
        "optional_sensitive_policy": "optional_prefer_not_to_say_not_suspicious",
        "policy_version": CURRENT_POLICY_VERSION,
    }


def assessment_summary(assessment: ProfileAssessment | None, *, counselor_view: bool = False) -> dict[str, Any]:
    if not assessment:
        return {"status": "not_started", "message": "Profile assessment not completed"}
    responses = assessment.responses_json or {}
    visible = {}
    for question in QUESTIONS:
        visibility = question["counselor_summary_visibility"]
        if counselor_view and visibility == "hidden":
            continue
        value = responses.get(question["question_id"])
        if value == PREFER_NOT_TO_SAY and counselor_view:
            continue
        if value is not None and visibility in {"summary", "limited"}:
            visible[question["question_id"]] = {
                "category": question["category"],
                "label": question["label"],
                "value": value,
                "visibility": visibility,
            }
    prediction = assessment.prediction
    contribution = None
    if prediction and prediction.risk_inputs:
        latest_input = sorted(prediction.risk_inputs, key=lambda item: item.id)[-1]
        contribution = {
            "included": latest_input.included,
            "base_weight": latest_input.base_weight,
            "effective_weight": latest_input.effective_weight,
            "mapped_score": latest_input.mapped_score,
            "exclusion_reason": latest_input.exclusion_reason,
        }
    return {
        "status": assessment.status,
        "assessment_id": assessment.id,
        "profile_assessment_id": assessment.profile_assessment_id,
        "questionnaire_version": assessment.questionnaire_version,
        "submitted_at": assessment.submitted_at,
        "completed_at": assessment.completed_at,
        "stale_at": assessment.stale_at,
        "summary": visible,
        "prediction": {
            "available": bool(prediction and prediction.status == "succeeded" and prediction.is_available),
            "status": prediction.status if prediction else "unavailable",
            "model_version": prediction.model_version if prediction else None,
            "preprocessing_version": assessment.preprocessing_version,
            "limitations": ["Profile summary is screening support only, not a diagnosis."],
        },
        "fusion_contribution": contribution,
    }


def counselor_can_access_student(db: Session, counselor: User, student_id: int) -> bool:
    if counselor.role == UserRole.ADMIN:
        return True
    if counselor.role not in {UserRole.COUNSELOR, UserRole.PSYCHIATRIST}:
        return False
    return bool(
        db.query(CounselorAssignment)
        .filter(
            CounselorAssignment.student_id == student_id,
            CounselorAssignment.counselor_id == counselor.id,
            CounselorAssignment.active.is_(True),
        )
        .first()
    )


def admin_statistics(db: Session) -> dict[str, Any]:
    rows = (
        db.query(ProfileAssessment.status, func.count(ProfileAssessment.id))
        .filter(ProfileAssessment.questionnaire_version == QUESTIONNAIRE_VERSION)
        .group_by(ProfileAssessment.status)
        .all()
    )
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "counts_by_status": {status_value or "legacy": count for status_value, count in rows},
        "total_phase4l_assessments": sum(count for _, count in rows),
        "raw_responses_included": False,
    }
