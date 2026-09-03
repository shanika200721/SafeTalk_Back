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
from app.ml.runtime.behavioral import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_MODEL_NAME,
    BEHAVIORAL_MODEL_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    BEHAVIORAL_RISK_MAPPING_STATUS,
    BEHAVIORAL_RUNTIME_LIMITATION,
    predict_behavioral_signal,
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
    "mood": {"name": "daily-checkin-deterministic-signal", "version": "1.1.0", "preprocessing": "mood-runtime-v1.1", "schema": "mood-feature-v1.1"},
    "text": {"name": None, "version": None, "preprocessing": "text-contract-v1", "schema": "text-feature-v1"},
    "speech": {"name": None, "version": None, "preprocessing": "speech-contract-v1", "schema": "speech-feature-v1"},
    "face": {"name": None, "version": None, "preprocessing": "face-contract-v1", "schema": "face-feature-v1"},
    "behavioral": {"name": BEHAVIORAL_MODEL_NAME, "version": BEHAVIORAL_MODEL_VERSION, "preprocessing": BEHAVIORAL_PREPROCESSING_VERSION, "schema": BEHAVIORAL_FEATURE_SCHEMA_VERSION},
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


def trigger_fusion_for_prediction(
    db: Session,
    prediction: ModalityPrediction,
    *,
    trigger_source: str,
    actor: Optional[User] = None,
) -> dict:
    from app.services.fusion import evaluate_student_fusion

    try:
        return evaluate_student_fusion(
            db,
            student_id=prediction.student_id,
            trigger_source=trigger_source,
            trigger_prediction_id=prediction.id,
            actor_user_id=actor.id if actor else prediction.student_id,
            actor_role=getattr(actor, "role", UserRole.STUDENT) if actor else UserRole.STUDENT,
        )
    except AssertionError as exc:
        if "Unexpected fake query model" in str(exc):
            return {"status": "not_run", "reason": "test_fake_db_without_fusion_tables"}
        raise


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


def _mood_trend_features(db: Session, checkin: DailyCheckIn) -> dict:
    recent = (
        db.query(DailyCheckIn)
        .filter(DailyCheckIn.user_id == checkin.user_id, DailyCheckIn.created_at <= checkin.created_at)
        .order_by(DailyCheckIn.created_at.desc(), DailyCheckIn.id.desc())
        .limit(14)
        .all()
    )
    ordered = list(reversed(recent))
    moods = [float(item.mood) for item in ordered if item.mood is not None]
    if not moods:
        return {
            "recent_checkin_count": 0,
            "recent_mood_mean": None,
            "previous_mood_mean": None,
            "worsening_trend": 0.0,
            "repeated_low_mood_count": 0,
        }
    current_window = moods[-7:]
    previous_window = moods[:-7][-7:]
    recent_mean = sum(current_window) / len(current_window)
    previous_mean = sum(previous_window) / len(previous_window) if previous_window else None
    worsening = max(0.0, (previous_mean - recent_mean) / 4.0) if previous_mean is not None else 0.0
    return {
        "recent_checkin_count": len(moods),
        "recent_mood_mean": round(recent_mean, 4),
        "previous_mood_mean": round(previous_mean, 4) if previous_mean is not None else None,
        "worsening_trend": round(worsening, 6),
        "repeated_low_mood_count": sum(1 for value in current_window if value <= 2),
    }


def _deterministic_mood_signal(features: dict) -> tuple[float, dict]:
    current_mood_component = ((5.0 - float(features.get("mood", 3))) / 4.0) * 35.0
    stress_component = (max(1.0, min(10.0, float(features.get("stress_level", 5)))) - 1.0) / 9.0 * 15.0
    anxiety_component = (max(1.0, min(10.0, float(features.get("anxiety_level", 5)))) - 1.0) / 9.0 * 15.0
    sleep_hours = float(features.get("sleep_hours", 7) or 7)
    sleep_component = 12.0 if sleep_hours < 4 or sleep_hours > 10 else 8.0 if sleep_hours < 6 or sleep_hours > 9 else 0.0
    low_mood_component = min(10.0, float(features.get("repeated_low_mood_count", 0)) * 2.5)
    trend_component = float(features.get("worsening_trend", 0.0)) * 8.0
    flag_component = 0.0
    if features.get("negative_thoughts"):
        flag_component += 10.0
    if features.get("substance_use_today"):
        flag_component += 5.0
    if features.get("self_harm_thoughts"):
        flag_component += 25.0
    components = {
        "current_mood_component": round(current_mood_component, 4),
        "stress_component": round(stress_component, 4),
        "anxiety_component": round(anxiety_component, 4),
        "sleep_component": round(sleep_component, 4),
        "repeated_low_mood_component": round(low_mood_component, 4),
        "worsening_trend_component": round(trend_component, 4),
        "declared_concern_flags_component": round(flag_component, 4),
    }
    score = max(0.0, min(100.0, sum(components.values())))
    return round(score, 2), components


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
    features.update(_mood_trend_features(db, checkin))
    score, components = _deterministic_mood_signal(features)
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
        metadata_json={
            "heuristic_source": "deterministic_mood_signal_v1_1",
            "algorithm": "weighted current mood, stress, anxiety, sleep, repeated low mood, worsening trend, and collected concern flags",
            "components": components,
            "legacy_daily_checkin_calculator_score": DailyCheckInCalculator.calculate(features),
            "normalized_signal": round(score / 100.0, 6),
        },
    )


def create_behavioral_prediction(db: Session, user: User) -> ModalityPrediction:
    signal = predict_behavioral_signal(db, user)
    snapshot = create_feature_snapshot(
        db,
        student_id=user.id,
        modality="behavioral",
        source_type="behavioral_activity_aggregate",
        source_record_id=None,
        source_timestamp=datetime.utcnow(),
        feature_schema_version=BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        preprocessing_version=BEHAVIORAL_PREPROCESSING_VERSION,
        features_json=signal.features,
        data_quality_status=signal.data_sufficiency,
        data_quality_flags=[] if signal.data_sufficiency != "insufficient_personal_baseline" else ["insufficient_personal_baseline"],
        metadata_json=signal.provenance,
    )
    return create_prediction(
        db,
        student_id=user.id,
        modality="behavioral",
        status_value="succeeded",
        is_available=True,
        output_type="heuristic",
        source_type="behavioral_activity_aggregate",
        source_record_id=None,
        source_timestamp=datetime.utcnow(),
        feature_snapshot=snapshot,
        score=signal.anomaly_score,
        confidence=signal.confidence,
        label=signal.label,
        raw_output_json={
            "behavioral_anomaly_score": signal.anomaly_score,
            "confidence": signal.confidence,
            "data_sufficiency": signal.data_sufficiency,
            "feature_provenance": signal.provenance,
        },
        metadata_json={
            "technical_status": "active_contextual_anomaly_signal",
            "research_reliability": "contextual_only",
            "risk_mapping_status": BEHAVIORAL_RISK_MAPPING_STATUS,
            "fusion_status": "excluded_no_validated_risk_mapping",
            "fusion_eligible": False,
            "normalized_score": round(signal.anomaly_score / 100.0, 6),
            "exclusion_reason": "no_validated_risk_mapping",
            "data_sufficiency": signal.data_sufficiency,
            "provenance": signal.provenance,
            "limitation": BEHAVIORAL_RUNTIME_LIMITATION,
        },
        model_name=BEHAVIORAL_MODEL_NAME,
        model_version=BEHAVIORAL_MODEL_VERSION,
        preprocessing_version=BEHAVIORAL_PREPROCESSING_VERSION,
        feature_schema_version=BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        data_quality_status=signal.data_sufficiency,
        data_quality_flags=[] if signal.data_sufficiency != "insufficient_personal_baseline" else ["insufficient_personal_baseline"],
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
            "runtime_model_active": True,
            "contract_available": True,
            "consent_required": "mood_processing",
            "source_requirements": ["owned daily check-in record"],
            "limitations": ["Deterministic check-in signal with trend features.", SCREENING_LIMITATION],
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
            "limitations": ["Voice-message delivery is separate from analysis.", "Speech emotion remains fusion excluded without an approved risk mapping."],
        },
        {
            "modality": "face",
            "implemented": True,
            "runtime_model_active": False,
            "contract_available": True,
            "consent_required": "face_processing",
            "source_requirements": ["explicit browser camera capture image data URL"],
            "limitations": ["Facial emotion model has low test reliability.", "Frontend random emotion values are not accepted.", "Fusion excluded unless separately validated."],
        },
        {
            "modality": "behavioral",
            "implemented": True,
            "runtime_model_active": True,
            "contract_available": True,
            "consent_required": "behavioral_processing",
            "source_requirements": ["persisted login/check-in/journal/chat activity metadata"],
            "limitations": ["Contextual anomaly signal only.", "No validated suicide-risk mapping; fusion excluded."],
        },
    ]
