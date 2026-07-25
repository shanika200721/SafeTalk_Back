from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.ml.preprocessing.dass21.constants import (
    DASS21_FEATURE_SCHEMA_VERSION,
    DASS21_ITEM_MAPPING_VERSION,
    DASS21_SCORING_VERSION,
    ITEM_MULTIPLIER,
    QUESTIONNAIRE_VERSION,
)
from app.models.database_models import (
    ChatMessage,
    DASS21Assessment,
    DailyCheckIn,
    FeatureSnapshot,
    ModalityPrediction,
    ModelRegistry,
    ProfileAssessment,
    User,
    UserRole,
)
from app.schemas import (
    CanonicalModality,
    DataQuality,
    ModalityPredictionResponse,
    PredictionEvidence,
    PredictionModelEvidence,
    PredictionOutputType,
    PredictionStatus,
)
from app.services.consent import CURRENT_POLICY_VERSION
from app.utils.assessment_calculator import DailyCheckInCalculator, ProfileRiskCalculator
from app.utils.dass21_calculator import DASS21Calculator


SCREENING_LIMITATION = "This is a screening-support signal and not a clinical diagnosis."

CANONICAL_MODALITIES = (
    "profile",
    "dass21",
    "mood",
    "text",
    "speech",
    "face",
    "behavioral",
)

MODALITY_CONSENT_TYPES = {
    "profile": "profile_processing",
    "dass21": "dass21_processing",
    "mood": "mood_processing",
    "text": "text_processing",
    "speech": "voice_processing",
    "face": "face_processing",
    "behavioral": "behavioral_processing",
}

MODEL_EVIDENCE = {
    "profile": {"name": "profile-heuristic", "version": "1.0.0", "preprocessing": "profile-runtime-v1", "schema": "profile-feature-v1"},
    "dass21": {"name": "dass21-rule-scoring", "version": DASS21_SCORING_VERSION, "preprocessing": "dass21-runtime-v1", "schema": DASS21_FEATURE_SCHEMA_VERSION},
    "mood": {"name": "daily-checkin-heuristic", "version": "1.0.0", "preprocessing": "mood-runtime-v1", "schema": "mood-feature-v1"},
    "text": {"name": None, "version": None, "preprocessing": "text-contract-v1", "schema": "text-feature-v1"},
    "speech": {"name": None, "version": None, "preprocessing": "speech-contract-v1", "schema": "speech-feature-v1"},
    "face": {"name": None, "version": None, "preprocessing": "face-contract-v1", "schema": "face-feature-v1"},
    "behavioral": {"name": None, "version": None, "preprocessing": "behavioral-contract-v1", "schema": "behavioral-feature-v1"},
}


def can_access_own_prediction(user: User, prediction: ModalityPrediction) -> bool:
    return prediction.student_id == user.id


def can_admin_access_prediction(user: User, prediction: ModalityPrediction) -> bool:
    return user.role == UserRole.ADMIN


def can_counselor_access_student(counselor: User, student_id: int) -> bool:
    """Phase 4C placeholder: deny generic counselor prediction access by default."""
    return False


def authorize_prediction_read(user: User, prediction: ModalityPrediction) -> None:
    if can_access_own_prediction(user, prediction) or can_admin_access_prediction(user, prediction):
        return
    if user.role in {UserRole.COUNSELOR, UserRole.PSYCHIATRIST} and can_counselor_access_student(user, prediction.student_id):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You are not authorized to access this prediction",
    )


def create_feature_snapshot(
    db: Session,
    *,
    student_id: int,
    modality: str,
    source_type: str,
    source_record_id: Optional[int],
    source_timestamp: Optional[datetime],
    feature_schema_version: str,
    preprocessing_version: str,
    features_json: dict,
    data_quality_status: str = "accepted",
    data_quality_flags: Optional[list[str]] = None,
    metadata_json: Optional[dict] = None,
) -> FeatureSnapshot:
    snapshot = FeatureSnapshot(
        student_id=student_id,
        modality=modality,
        source_type=source_type,
        source_record_id=source_record_id,
        source_timestamp=source_timestamp,
        feature_schema_version=feature_schema_version,
        preprocessing_version=preprocessing_version,
        features_json=features_json,
        data_quality_status=data_quality_status,
        data_quality_flags=data_quality_flags or [],
        consent_policy_version=CURRENT_POLICY_VERSION,
        metadata_json=metadata_json or {},
    )
    db.add(snapshot)
    db.flush()
    return snapshot


def create_prediction(
    db: Session,
    *,
    student_id: int,
    modality: str,
    status_value: str,
    is_available: bool,
    output_type: str,
    source_type: str,
    source_record_id: Optional[int],
    source_timestamp: Optional[datetime],
    feature_snapshot: Optional[FeatureSnapshot] = None,
    score: Optional[float] = None,
    probability: Optional[float] = None,
    confidence: Optional[float] = None,
    label: Optional[str] = None,
    failure_code: Optional[str] = None,
    failure_message_safe: Optional[str] = None,
    raw_output_json: Optional[dict] = None,
    metadata_json: Optional[dict] = None,
    model_registry: Optional[ModelRegistry] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    preprocessing_version: Optional[str] = None,
    feature_schema_version: Optional[str] = None,
    valid_for_hours: Optional[int] = 24,
    data_quality_status: str = "accepted",
    data_quality_flags: Optional[list[str]] = None,
) -> ModalityPrediction:
    model = MODEL_EVIDENCE[modality]
    resolved_model_name = model_name or (model_registry.model_name if model_registry else model["name"])
    resolved_model_version = model_version or (model_registry.version if model_registry else model["version"])
    resolved_preprocessing_version = (
        preprocessing_version
        or (model_registry.preprocessing_version if model_registry else None)
        or model["preprocessing"]
    )
    resolved_feature_schema_version = (
        feature_schema_version
        or (model_registry.feature_schema_version if model_registry else None)
        or model["schema"]
    )
    generated_at = datetime.utcnow()
    prediction = ModalityPrediction(
        student_id=student_id,
        modality=modality,
        source_type=source_type,
        source_record_id=source_record_id,
        source_timestamp=source_timestamp,
        feature_snapshot_id=feature_snapshot.id if feature_snapshot else None,
        status=status_value,
        is_available=is_available,
        output_type=output_type,
        score_0_100=score,
        probability=probability,
        confidence=confidence,
        predicted_class=label,
        label=label,
        failure_code=failure_code,
        failure_message_safe=failure_message_safe,
        generated_at=generated_at,
        valid_until=generated_at + timedelta(hours=valid_for_hours) if valid_for_hours else None,
        model_registry_id=model_registry.id if model_registry else None,
        model_name=resolved_model_name,
        model_version=resolved_model_version,
        preprocessing_version=resolved_preprocessing_version,
        feature_schema_version=resolved_feature_schema_version,
        raw_output_json=raw_output_json,
        metadata_json=metadata_json or {},
        consent_policy_version=CURRENT_POLICY_VERSION,
        evidence_available=is_available and feature_snapshot is not None,
        clinical_use_boundary="screening_support_only",
        data_quality_status=data_quality_status,
        data_quality_flags=data_quality_flags or [],
        explanation_json={"limitations": [SCREENING_LIMITATION]},
    )
    db.add(prediction)
    db.flush()
    return prediction


def prediction_to_response(prediction: ModalityPrediction) -> ModalityPredictionResponse:
    return ModalityPredictionResponse(
        prediction_id=prediction.id,
        user_id=prediction.student_id,
        modality=CanonicalModality(prediction.modality),
        status=PredictionStatus(prediction.status),
        is_available=prediction.is_available,
        output_type=PredictionOutputType(prediction.output_type),
        score=prediction.score_0_100,
        probability=prediction.probability,
        confidence=prediction.confidence,
        label=prediction.label or prediction.predicted_class,
        failure_code=prediction.failure_code,
        failure_message_safe=prediction.failure_message_safe,
        source_timestamp=prediction.source_timestamp,
        generated_at=prediction.generated_at or prediction.created_at,
        valid_until=prediction.valid_until,
        model=PredictionModelEvidence(
            name=prediction.model_name,
            version=prediction.model_version,
            registry_id=prediction.model_registry_id,
        ),
        preprocessing_version=prediction.preprocessing_version,
        feature_schema_version=prediction.feature_schema_version,
        data_quality=DataQuality(
            status=prediction.data_quality_status or "accepted",
            flags=prediction.data_quality_flags or [],
        ),
        evidence=PredictionEvidence(
            available=prediction.evidence_available,
            source_type=prediction.source_type,
            source_id=prediction.source_record_id,
            feature_snapshot_id=prediction.feature_snapshot_id,
        ),
        consent_policy_version=prediction.consent_policy_version,
        clinical_use_boundary=prediction.clinical_use_boundary,
        limitations=[SCREENING_LIMITATION],
        metadata=prediction.metadata_json or {},
    )


def profile_label(score: float) -> str:
    if score >= 70:
        return "high_signal"
    if score >= 40:
        return "elevated_signal"
    return "low_signal"


def create_profile_prediction_for_assessment(db: Session, assessment: ProfileAssessment) -> ModalityPrediction:
    profile_features = {
        "gpa": assessment.gpa,
        "repeated_subjects": assessment.repeated_subjects,
        "attendance": assessment.attendance,
        "family_relationship_score": assessment.family_relationship_score,
        "family_support": assessment.family_support,
        "financial_stress": assessment.financial_stress,
        "communication_skills": assessment.communication_skills,
        "social_connection": assessment.social_connection,
        "sleep_pattern": assessment.sleep_pattern,
        "exercise_frequency": assessment.exercise_frequency,
        "substance_use": assessment.substance_use,
    }
    snapshot = create_feature_snapshot(
        db,
        student_id=assessment.user_id,
        modality="profile",
        source_type="profile_assessment",
        source_record_id=assessment.id,
        source_timestamp=assessment.updated_at or assessment.created_at,
        feature_schema_version=MODEL_EVIDENCE["profile"]["schema"],
        preprocessing_version=MODEL_EVIDENCE["profile"]["preprocessing"],
        features_json=profile_features,
    )
    score = float(assessment.profile_score or 0)
    return create_prediction(
        db,
        student_id=assessment.user_id,
        modality="profile",
        status_value="succeeded",
        is_available=True,
        output_type="heuristic",
        source_type="profile_assessment",
        source_record_id=assessment.id,
        source_timestamp=assessment.updated_at or assessment.created_at,
        feature_snapshot=snapshot,
        score=score,
        label=profile_label(score),
        metadata_json={"heuristic_source": "ProfileRiskCalculator"},
    )


def dass21_metadata_dict(assessment: DASS21Assessment) -> dict:
    return {
        "questionnaire_version": assessment.questionnaire_version,
        "item_mapping_version": assessment.item_mapping_version,
        "scoring_version": assessment.scoring_version,
        "score_multiplier": assessment.score_multiplier,
        "completed_item_count": assessment.completed_item_count,
        "is_complete": assessment.is_complete,
        "scored_at": assessment.scored_at,
        "consent_policy_version": assessment.consent_policy_version,
    }


def populate_dass21_metadata(assessment: DASS21Assessment) -> None:
    completed_item_count = len(assessment.responses or [])
    assessment.questionnaire_version = QUESTIONNAIRE_VERSION
    assessment.item_mapping_version = DASS21_ITEM_MAPPING_VERSION
    assessment.scoring_version = DASS21_SCORING_VERSION
    assessment.score_multiplier = float(ITEM_MULTIPLIER)
    assessment.completed_item_count = completed_item_count
    assessment.is_complete = completed_item_count == 21
    assessment.scored_at = datetime.utcnow()
    assessment.consent_policy_version = CURRENT_POLICY_VERSION


def create_dass21_prediction_for_assessment(db: Session, assessment: DASS21Assessment) -> ModalityPrediction:
    features = {
        "completed_item_count": assessment.completed_item_count,
        "is_complete": assessment.is_complete,
        "depression_score": assessment.depression_score,
        "anxiety_score": assessment.anxiety_score,
        "stress_score": assessment.stress_score,
        "total_dass21_score": assessment.total_dass21_score,
        "depression_severity": assessment.depression_severity,
        "anxiety_severity": assessment.anxiety_severity,
        "stress_severity": assessment.stress_severity,
    }
    snapshot = create_feature_snapshot(
        db,
        student_id=assessment.user_id,
        modality="dass21",
        source_type="dass21_assessment",
        source_record_id=assessment.id,
        source_timestamp=assessment.scored_at or assessment.created_at,
        feature_schema_version=DASS21_FEATURE_SCHEMA_VERSION,
        preprocessing_version=MODEL_EVIDENCE["dass21"]["preprocessing"],
        features_json=features,
    )
    score = DASS21Calculator.calculate_dass21_risk_score(
        {"total_dass21_score": assessment.total_dass21_score}
    )
    return create_prediction(
        db,
        student_id=assessment.user_id,
        modality="dass21",
        status_value="succeeded",
        is_available=True,
        output_type="rule_based",
        source_type="dass21_assessment",
        source_record_id=assessment.id,
        source_timestamp=assessment.scored_at or assessment.created_at,
        feature_snapshot=snapshot,
        score=score,
        label="dass21_screening_score",
        metadata_json={
            "subscales": features,
            "normalization": "legacy_total_dass21_score_divided_by_126",
            "questionnaire_version": QUESTIONNAIRE_VERSION,
            "item_mapping_version": DASS21_ITEM_MAPPING_VERSION,
            "scoring_version": DASS21_SCORING_VERSION,
        },
    )


def create_mood_prediction_for_checkin(db: Session, checkin: DailyCheckIn) -> ModalityPrediction:
    features = {
        "mood": checkin.mood,
        "sleep_hours": checkin.sleep_hours,
        "exercise_minutes": checkin.exercise_minutes,
        "social_interaction": checkin.social_interaction,
        "stress_level": checkin.stress_level,
        "anxiety_level": checkin.anxiety_level,
        "negative_thoughts": checkin.negative_thoughts,
        "substance_use_today": checkin.substance_use_today,
        "self_harm_thoughts": checkin.self_harm_thoughts,
    }
    score = DailyCheckInCalculator.calculate(features)
    snapshot = create_feature_snapshot(
        db,
        student_id=checkin.user_id,
        modality="mood",
        source_type="daily_checkin",
        source_record_id=checkin.id,
        source_timestamp=checkin.created_at,
        feature_schema_version=MODEL_EVIDENCE["mood"]["schema"],
        preprocessing_version=MODEL_EVIDENCE["mood"]["preprocessing"],
        features_json=features,
    )
    return create_prediction(
        db,
        student_id=checkin.user_id,
        modality="mood",
        status_value="succeeded",
        is_available=True,
        output_type="heuristic",
        source_type="daily_checkin",
        source_record_id=checkin.id,
        source_timestamp=checkin.created_at,
        feature_snapshot=snapshot,
        score=score,
        label=profile_label(score),
        metadata_json={"heuristic_source": "DailyCheckInCalculator"},
    )


def create_unavailable_prediction(
    db: Session,
    *,
    user: User,
    modality: str,
    failure_code: str,
    message: str,
    source_type: str,
    source_record_id: Optional[int] = None,
    source_timestamp: Optional[datetime] = None,
    feature_snapshot: Optional[FeatureSnapshot] = None,
) -> ModalityPrediction:
    return create_prediction(
        db,
        student_id=user.id,
        modality=modality,
        status_value="unavailable",
        is_available=False,
        output_type="machine_learning",
        source_type=source_type,
        source_record_id=source_record_id,
        source_timestamp=source_timestamp,
        feature_snapshot=feature_snapshot,
        failure_code=failure_code,
        failure_message_safe=message,
        valid_for_hours=None,
        data_quality_status="not_evaluated",
        metadata_json={"runtime_model_active": False},
    )


def create_failed_prediction(
    db: Session,
    *,
    user: User,
    modality: str,
    failure_code: str,
    message: str,
    source_type: str,
    source_record_id: Optional[int] = None,
    source_timestamp: Optional[datetime] = None,
    feature_snapshot: Optional[FeatureSnapshot] = None,
    model_registry: Optional[ModelRegistry] = None,
) -> ModalityPrediction:
    return create_prediction(
        db,
        student_id=user.id,
        modality=modality,
        status_value="failed",
        is_available=False,
        output_type="machine_learning",
        source_type=source_type,
        source_record_id=source_record_id,
        source_timestamp=source_timestamp,
        feature_snapshot=feature_snapshot,
        failure_code=failure_code,
        failure_message_safe=message,
        model_registry=model_registry,
        valid_for_hours=None,
        data_quality_status="not_evaluated",
        metadata_json={"runtime_model_active": bool(model_registry)},
    )


def verify_owned_chat_message(db: Session, message_id: int, user: User) -> ChatMessage:
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message or message.message_type != "voice":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice source not found")
    if message.sender_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the student's own voice messages can be used as speech evidence")
    return message


def availability_contract() -> list[dict]:
    return [
        {
            "modality": "profile",
            "implemented": True,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "profile_processing",
            "source_requirements": ["existing profile assessment or validated profile payload"],
            "limitations": ["Heuristic only.", SCREENING_LIMITATION],
        },
        {
            "modality": "dass21",
            "implemented": True,
            "runtime_model_active": True,
            "contract_available": True,
            "consent_required": "dass21_processing",
            "source_requirements": ["stored DASS-21 assessment or complete 21-item response payload"],
            "limitations": ["Rule-based scoring only.", SCREENING_LIMITATION],
        },
        {
            "modality": "mood",
            "implemented": True,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "mood_processing",
            "source_requirements": ["owned daily check-in record"],
            "limitations": ["Heuristic check-in signal only.", SCREENING_LIMITATION],
        },
        {
            "modality": "text",
            "implemented": False,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "text_processing",
            "source_requirements": ["explicit user-submitted text for analysis"],
            "limitations": ["Trained text runtime model is not active.", "SafeTalk chat is separate and is not used as this modality model."],
        },
        {
            "modality": "speech",
            "implemented": True,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "voice_processing",
            "source_requirements": ["authorized voice chat message id or secure upload reference"],
            "limitations": ["Voice-message delivery is separate from analysis.", "Runtime speech model is not active."],
        },
        {
            "modality": "face",
            "implemented": False,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "face_processing",
            "source_requirements": ["future secure face source reference"],
            "limitations": ["Backend face inference is not active.", "Frontend random emotion values are not accepted."],
        },
        {
            "modality": "behavioral",
            "implemented": False,
            "runtime_model_active": False,
            "contract_available": False,
            "consent_required": "behavioral_processing",
            "source_requirements": [],
            "limitations": ["Behavioral modality is not validated and never returns a success score in Phase 4C."],
        },
    ]
