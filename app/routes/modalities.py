from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import (
    BehavioralTelemetryEvent,
    DASS21Assessment,
    DailyCheckIn,
    FeatureSnapshot,
    ModalityPrediction,
    ProfileAssessment,
    User,
)
from app.routes.auth import get_current_user
from app.schemas import (
    BehavioralPredictionRequest,
    BehavioralTelemetryRequest,
    BehavioralTelemetryResponse,
    CanonicalModality,
    DASS21PredictionRequest,
    FacePredictionRequest,
    ModalityAvailabilityResponse,
    ModalityPredictionListResponse,
    ModalityPredictionResponse,
    MoodPredictionRequest,
    ProfilePredictionRequest,
    SpeechPredictionRequest,
    TextPredictionRequest,
)
from app.ml.runtime.base import RuntimeInferenceError, RuntimeModelUnavailable
from app.ml.runtime.face import FacePreprocessingError
from app.ml.runtime.registry import predict_with_active_model
from app.ml.runtime.speech import SPEECH_RISK_MAPPING_STATUS, SPEECH_RUNTIME_LIMITATION
from app.ml.runtime.speech_preprocessor import SpeechPreprocessingError, validate_speech_audio_quality
from app.services.model_registry import get_active_model
from app.services.consent import CURRENT_POLICY_VERSION, require_active_consent
from app.services.modalities import (
    MODALITY_CONSENT_TYPES,
    MODEL_EVIDENCE,
    SCREENING_LIMITATION,
    availability_contract,
    authorize_prediction_read,
    create_prediction,
    create_dass21_prediction_for_assessment,
    create_behavioral_prediction,
    create_failed_prediction,
    create_feature_snapshot,
    create_mood_prediction_for_checkin,
    create_profile_prediction_for_assessment,
    create_unavailable_prediction,
    populate_dass21_metadata,
    prediction_to_response,
    trigger_fusion_for_prediction,
    verify_owned_chat_message,
)
from app.utils.assessment_calculator import ProfileRiskCalculator
from app.utils.dass21_calculator import DASS21Calculator


router = APIRouter(prefix="/api/modalities", tags=["Modalities"])
AUDIO_UPLOAD_DIR = Path("uploaded_audio")


def _require_modality_consent(db: Session, user: User, modality: str) -> None:
    require_active_consent(
        db,
        user,
        MODALITY_CONSENT_TYPES[modality],
        f"creating {modality} modality predictions",
    )


def _require_student_creator(user: User) -> None:
    if user.role.value != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can create modality predictions through this route",
        )


def _latest_owned_profile(db: Session, user: User) -> ProfileAssessment:
    assessment = (
        db.query(ProfileAssessment)
        .filter(ProfileAssessment.user_id == user.id)
        .order_by(ProfileAssessment.updated_at.desc(), ProfileAssessment.created_at.desc())
        .first()
    )
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile assessment not found")
    return assessment


def _get_owned_profile(db: Session, assessment_id: int, user: User) -> ProfileAssessment:
    assessment = db.query(ProfileAssessment).filter(ProfileAssessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile assessment not found")
    if assessment.user_id != user.id and user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Profile source is not authorized")
    return assessment


def _get_owned_dass21(db: Session, assessment_id: int, user: User) -> DASS21Assessment:
    assessment = db.query(DASS21Assessment).filter(DASS21Assessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DASS-21 assessment not found")
    if assessment.user_id != user.id and user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="DASS-21 source is not authorized")
    return assessment


def _get_owned_checkin(db: Session, checkin_id: int, user: User) -> DailyCheckIn:
    checkin = db.query(DailyCheckIn).filter(DailyCheckIn.id == checkin_id).first()
    if not checkin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Daily check-in not found")
    if checkin.user_id != user.id and user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Daily check-in source is not authorized")
    return checkin


@router.post("/profile/predict", response_model=ModalityPredictionResponse)
def predict_profile(
    request: ProfilePredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "profile")

    if request.source_assessment_id:
        assessment = _get_owned_profile(db, request.source_assessment_id, current_user)
    elif request.profile_payload:
        payload = request.profile_payload
        if payload.user_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot predict for another user")
        profile_dict = payload.dict()
        profile_score = ProfileRiskCalculator.calculate(profile_dict)
        assessment = ProfileAssessment(**profile_dict, profile_score=profile_score)
        db.add(assessment)
        db.flush()
    else:
        assessment = _latest_owned_profile(db, current_user)

    if request.strategy == "active_model":
        snapshot = None
        active_model = None
        source_timestamp = assessment.updated_at or assessment.created_at
        try:
            active_model, result = predict_with_active_model(
                db,
                modality="profile",
                payload={"year_of_study": current_user.year_of_study},
            )
            snapshot = create_feature_snapshot(
                db,
                student_id=assessment.user_id,
                modality="profile",
                source_type="profile_assessment",
                source_record_id=assessment.id,
                source_timestamp=source_timestamp,
                feature_schema_version=active_model.feature_schema_version or MODEL_EVIDENCE["profile"]["schema"],
                preprocessing_version=active_model.preprocessing_version or MODEL_EVIDENCE["profile"]["preprocessing"],
                features_json=result.features,
                metadata_json={"source_fields_used": ["year_of_study"]},
            )
            prediction = create_prediction(
                db,
                student_id=assessment.user_id,
                modality="profile",
                status_value="succeeded",
                is_available=True,
                output_type="machine_learning",
                source_type="profile_assessment",
                source_record_id=assessment.id,
                source_timestamp=source_timestamp,
                feature_snapshot=snapshot,
                probability=result.probability,
                confidence=result.confidence,
                label=result.label,
                metadata_json={**result.metadata, "class_probabilities": result.probabilities},
                model_registry=active_model,
            )
        except RuntimeModelUnavailable:
            prediction = create_unavailable_prediction(
                db,
                user=current_user,
                modality="profile",
                failure_code="MODEL_NOT_ACTIVE",
                message="The trained profile runtime model is not active; heuristic profile scoring remains available.",
                source_type="profile_assessment",
                source_record_id=assessment.id,
                source_timestamp=source_timestamp,
            )
        except RuntimeInferenceError:
            prediction = create_failed_prediction(
                db,
                user=current_user,
                modality="profile",
                failure_code="INFERENCE_FAILED",
                message="The active profile runtime model could not safely produce a prediction.",
                source_type="profile_assessment",
                source_record_id=assessment.id,
                source_timestamp=source_timestamp,
                feature_snapshot=snapshot,
                model_registry=active_model,
            )
    else:
        prediction = create_profile_prediction_for_assessment(db, assessment)
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_profile_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/dass21/predict", response_model=ModalityPredictionResponse)
def predict_dass21(
    request: DASS21PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "dass21")

    if request.assessment_id:
        assessment = _get_owned_dass21(db, request.assessment_id, current_user)
        if not assessment.scoring_version:
            populate_dass21_metadata(assessment)
    elif request.responses is not None:
        result = DASS21Calculator.calculate(request.responses)
        assessment = DASS21Assessment(
            user_id=current_user.id,
            responses=request.responses,
            depression_score=result["depression_score"],
            anxiety_score=result["anxiety_score"],
            stress_score=result["stress_score"],
            total_dass21_score=result["total_dass21_score"],
            depression_severity=result["depression_severity"],
            anxiety_severity=result["anxiety_severity"],
            stress_severity=result["stress_severity"],
        )
        populate_dass21_metadata(assessment)
        db.add(assessment)
        db.flush()
    else:
        assessment = (
            db.query(DASS21Assessment)
            .filter(DASS21Assessment.user_id == current_user.id)
            .order_by(DASS21Assessment.created_at.desc())
            .first()
        )
        if not assessment:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DASS-21 assessment not found")
        if not assessment.scoring_version:
            populate_dass21_metadata(assessment)

    prediction = create_dass21_prediction_for_assessment(db, assessment)
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_dass21_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/mood/predict", response_model=ModalityPredictionResponse)
def predict_mood(
    request: MoodPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "mood")
    if request.checkin_id:
        checkin = _get_owned_checkin(db, request.checkin_id, current_user)
    else:
        checkin = (
            db.query(DailyCheckIn)
            .filter(DailyCheckIn.user_id == current_user.id)
            .order_by(DailyCheckIn.created_at.desc())
            .first()
        )
        if not checkin:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Daily check-in not found")

    prediction = create_mood_prediction_for_checkin(db, checkin)
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_mood_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/text/predict", response_model=ModalityPredictionResponse)
def predict_text(
    request: TextPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "text")
    snapshot = create_feature_snapshot(
        db,
        student_id=current_user.id,
        modality="text",
        source_type="direct_text_payload",
        source_record_id=None,
        source_timestamp=None,
        feature_schema_version=MODEL_EVIDENCE["text"]["schema"],
        preprocessing_version=MODEL_EVIDENCE["text"]["preprocessing"],
        features_json={"text_length": len(request.text), "contains_raw_text": False},
    )
    active_model = None
    try:
        active_model, result = predict_with_active_model(
            db,
            modality="text",
            payload={"text": request.text},
        )
        snapshot.feature_schema_version = active_model.feature_schema_version or snapshot.feature_schema_version
        snapshot.preprocessing_version = active_model.preprocessing_version or snapshot.preprocessing_version
        snapshot.features_json = result.features
        snapshot.metadata_json = {"raw_text_stored": False}
        prediction = create_prediction(
            db,
            student_id=current_user.id,
            modality="text",
            status_value="succeeded",
            is_available=True,
            output_type="machine_learning",
            source_type="direct_text_payload",
            source_record_id=None,
            source_timestamp=None,
            feature_snapshot=snapshot,
            probability=result.probability,
            confidence=result.confidence,
            label=result.label,
            metadata_json={**result.metadata, "class_probabilities": result.probabilities},
            model_registry=active_model,
        )
    except RuntimeModelUnavailable:
        prediction = create_unavailable_prediction(
            db,
            user=current_user,
            modality="text",
            failure_code="MODEL_NOT_ACTIVE",
            message="The trained text modality model is not active in the runtime API.",
            source_type="direct_text_payload",
            feature_snapshot=snapshot,
        )
    except RuntimeInferenceError:
        prediction = create_failed_prediction(
            db,
            user=current_user,
            modality="text",
            failure_code="INFERENCE_FAILED",
            message="The active text runtime model could not safely produce a prediction.",
            source_type="direct_text_payload",
            feature_snapshot=snapshot,
            model_registry=active_model,
        )
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_text_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/speech/predict", response_model=ModalityPredictionResponse)
def predict_speech(
    request: SpeechPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "speech")
    source_record_id = None
    source_timestamp = None
    source_type = "secure_voice_reference"
    snapshot = None
    audio_path = None
    content_type = "audio/wav"

    if request.chat_message_id is not None:
        message = verify_owned_chat_message(db, request.chat_message_id, current_user)
        source_record_id = message.id
        source_timestamp = message.created_at
        source_type = "chat_voice_message"
        audio_path = AUDIO_UPLOAD_DIR / Path(message.message or "").name
        content_type = (message.metadata_json or {}).get("original_mime_type") or content_type
        snapshot = create_feature_snapshot(
            db,
            student_id=current_user.id,
            modality="speech",
            source_type=source_type,
            source_record_id=message.id,
            source_timestamp=message.created_at,
            feature_schema_version=MODEL_EVIDENCE["speech"]["schema"],
            preprocessing_version=MODEL_EVIDENCE["speech"]["preprocessing"],
            features_json={"source_kind": "authorized_chat_message", "raw_path_returned": False},
        )
    elif request.upload_reference_id:
        audio_path = AUDIO_UPLOAD_DIR / request.upload_reference_id
        source_type = "secure_voice_upload_reference"

    active_model = get_active_model(db, modality="speech")
    if not active_model:
        prediction = create_unavailable_prediction(
            db,
            user=current_user,
            modality="speech",
            failure_code="MODEL_NOT_ACTIVE",
            message="The speech runtime model is not active; voice storage is not speech analysis.",
            source_type=source_type,
            source_record_id=source_record_id,
            source_timestamp=source_timestamp,
            feature_snapshot=snapshot,
        )
    elif audio_path is None or not audio_path.exists():
        prediction = create_failed_prediction(
            db,
            user=current_user,
            modality="speech",
            failure_code="AUDIO_SOURCE_NOT_FOUND",
            message="Speech analysis requires an authorized stored audio source.",
            source_type=source_type,
            source_record_id=source_record_id,
            source_timestamp=source_timestamp,
            feature_snapshot=snapshot,
            model_registry=active_model,
        )
    else:
        quality = validate_speech_audio_quality(audio_path, content_type=content_type)
        if not quality.accepted:
            prediction = create_prediction(
                db,
                student_id=current_user.id,
                modality="speech",
                status_value="failed",
                is_available=False,
                output_type="machine_learning",
                source_type=source_type,
                source_record_id=source_record_id,
                source_timestamp=source_timestamp,
                feature_snapshot=snapshot,
                failure_code="AUDIO_QUALITY_REJECTED",
                failure_message_safe="Speech analysis was unavailable because the audio did not meet quality or decoder requirements.",
                raw_output_json={"quality": quality.__dict__},
                metadata_json={"fusion_eligible": False, "fusion_status": "excluded_audio_quality", "raw_path_returned": False},
                model_registry=active_model,
                valid_for_hours=None,
                data_quality_status=quality.status,
                data_quality_flags=quality.flags,
            )
        else:
            try:
                _, result = predict_with_active_model(
                    db,
                    modality="speech",
                    payload={"path": str(audio_path), "content_type": content_type},
                )
                if snapshot is None:
                    snapshot = create_feature_snapshot(
                        db,
                        student_id=current_user.id,
                        modality="speech",
                        source_type=source_type,
                        source_record_id=source_record_id,
                        source_timestamp=source_timestamp,
                        feature_schema_version=active_model.feature_schema_version or MODEL_EVIDENCE["speech"]["schema"],
                        preprocessing_version=active_model.preprocessing_version or MODEL_EVIDENCE["speech"]["preprocessing"],
                        features_json=result.features,
                        metadata_json={"raw_path_returned": False, "feature_shape": result.metadata.get("feature_shape")},
                    )
                else:
                    snapshot.features_json = result.features
                    snapshot.metadata_json = {**(snapshot.metadata_json or {}), "feature_shape": result.metadata.get("feature_shape")}
                prediction = create_prediction(
                    db,
                    student_id=current_user.id,
                    modality="speech",
                    status_value="succeeded",
                    is_available=True,
                    output_type="machine_learning",
                    source_type=source_type,
                    source_record_id=source_record_id,
                    source_timestamp=source_timestamp,
                    feature_snapshot=snapshot,
                    probability=result.probability,
                    confidence=result.confidence,
                    label=result.label,
                    raw_output_json={"emotion_label": result.label, "class_probabilities": result.probabilities},
                    metadata_json={
                        **result.metadata,
                        "class_probabilities": result.probabilities,
                        "fusion_status": SPEECH_RISK_MAPPING_STATUS,
                        "fusion_eligible": False,
                        "limitation": SPEECH_RUNTIME_LIMITATION,
                        "raw_path_returned": False,
                    },
                    model_registry=active_model,
                    data_quality_status=result.metadata.get("data_quality_status") or "accepted",
                    data_quality_flags=result.metadata.get("data_quality_flags") or [],
                )
            except (RuntimeModelUnavailable, RuntimeInferenceError, SpeechPreprocessingError):
                prediction = create_failed_prediction(
                    db,
                    user=current_user,
                    modality="speech",
                    failure_code="SPEECH_RUNTIME_FAILED",
                    message="The active speech runtime model could not safely produce a prediction.",
                    source_type=source_type,
                    source_record_id=source_record_id,
                    source_timestamp=source_timestamp,
                    feature_snapshot=snapshot,
                    model_registry=active_model,
                )
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_speech_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/face/predict", response_model=ModalityPredictionResponse)
def predict_face(
    request: FacePredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "face")
    active_model = get_active_model(db, modality="face")
    source_type = "browser_face_capture" if request.image_data_url else ("secure_face_reference" if request.source_reference_id else "no_face_source")
    if not active_model:
        prediction = create_unavailable_prediction(
            db,
            user=current_user,
            modality="face",
            failure_code="MODEL_NOT_ACTIVE",
            message="Backend face inference is not active. Frontend random emotion output is not accepted as evidence.",
            source_type=source_type,
        )
    elif not request.image_data_url:
        prediction = create_failed_prediction(
            db,
            user=current_user,
            modality="face",
            failure_code="FACE_SOURCE_NOT_FOUND",
            message="Facial analysis requires an explicit browser-captured image payload.",
            source_type=source_type,
            model_registry=active_model,
        )
    else:
        try:
            _, result = predict_with_active_model(
                db,
                modality="face",
                payload={"image_data_url": request.image_data_url},
            )
            snapshot = create_feature_snapshot(
                db,
                student_id=current_user.id,
                modality="face",
                source_type=source_type,
                source_record_id=None,
                source_timestamp=None,
                feature_schema_version=active_model.feature_schema_version or MODEL_EVIDENCE["face"]["schema"],
                preprocessing_version=active_model.preprocessing_version or MODEL_EVIDENCE["face"]["preprocessing"],
                features_json=result.features,
                metadata_json={
                    "explicit_capture_reference": request.source_reference_id,
                    "raw_image_stored": False,
                    "feature_shape": result.metadata.get("feature_shape"),
                    "face_detection_status": result.metadata.get("face_detection_status"),
                },
            )
            prediction = create_prediction(
                db,
                student_id=current_user.id,
                modality="face",
                status_value="succeeded",
                is_available=True,
                output_type="machine_learning",
                source_type=source_type,
                source_record_id=None,
                source_timestamp=None,
                feature_snapshot=snapshot,
                probability=result.probability,
                confidence=result.confidence,
                label=result.label,
                raw_output_json={"emotion_label": result.label, "class_probabilities": result.probabilities},
                metadata_json={**result.metadata, "class_probabilities": result.probabilities, "raw_image_stored": False},
                model_registry=active_model,
            )
        except (RuntimeModelUnavailable, RuntimeInferenceError, FacePreprocessingError):
            prediction = create_failed_prediction(
                db,
                user=current_user,
                modality="face",
                failure_code="FACE_RUNTIME_FAILED",
                message="The active face runtime model could not safely produce a prediction.",
                source_type=source_type,
                model_registry=active_model,
            )
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_face_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/behavioral/predict", response_model=ModalityPredictionResponse)
def predict_behavioral(
    request: BehavioralPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "behavioral")
    prediction = create_behavioral_prediction(db, current_user)
    trigger_fusion_for_prediction(db, prediction, trigger_source="modality_behavioral_predict", actor=current_user)
    db.refresh(prediction)
    return prediction_to_response(prediction)


@router.post("/behavioral/telemetry", response_model=BehavioralTelemetryResponse)
def record_behavioral_telemetry(
    request: BehavioralTelemetryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student_creator(current_user)
    _require_modality_consent(db, current_user, "behavioral")
    allowed_metadata = {
        key: value
        for key, value in (request.metadata or {}).items()
        if str(key).lower() not in {"text", "raw_text", "keystrokes", "key_events", "mouse_path", "pointer_path", "coordinates"}
    }
    event = BehavioralTelemetryEvent(
        student_id=current_user.id,
        event_type=request.event_type,
        source_page=request.source_page,
        session_id=request.session_id,
        session_duration_seconds=request.session_duration_seconds,
        interaction_count=request.interaction_count,
        response_latency_ms=request.response_latency_ms,
        typing_active_ms=request.typing_active_ms,
        typing_pause_count=request.typing_pause_count,
        typed_character_count=request.typed_character_count,
        metadata_json={
            "privacy": "aggregate_only_no_raw_keystrokes_no_pointer_paths",
            "client_metadata": allowed_metadata,
        },
        consent_policy_version=CURRENT_POLICY_VERSION,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    stored_fields = [
        field
        for field in (
            "session_duration_seconds",
            "interaction_count",
            "response_latency_ms",
            "typing_active_ms",
            "typing_pause_count",
            "typed_character_count",
        )
        if getattr(event, field) is not None
    ]
    return BehavioralTelemetryResponse(
        id=event.id,
        user_id=current_user.id,
        event_type=event.event_type,
        source_page=event.source_page,
        session_id=event.session_id,
        consent_policy_version=event.consent_policy_version,
        created_at=event.created_at,
        stored_fields=stored_fields,
    )


@router.get("/predictions", response_model=ModalityPredictionListResponse)
def list_predictions(
    modality: CanonicalModality | None = None,
    user_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = user_id or current_user.id
    if target_user_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to list these predictions")

    query = db.query(ModalityPrediction).filter(ModalityPrediction.student_id == target_user_id)
    if modality:
        query = query.filter(ModalityPrediction.modality == modality.value)
    predictions = query.order_by(ModalityPrediction.created_at.desc()).limit(limit).all()
    return {
        "user_id": target_user_id,
        "predictions": [prediction_to_response(prediction) for prediction in predictions],
    }


@router.get("/predictions/latest", response_model=ModalityPredictionListResponse)
def latest_predictions(
    user_id: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    target_user_id = user_id or current_user.id
    if target_user_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to list these predictions")

    latest = []
    for modality in CanonicalModality:
        prediction = (
            db.query(ModalityPrediction)
            .filter(
                ModalityPrediction.student_id == target_user_id,
                ModalityPrediction.modality == modality.value,
            )
            .order_by(ModalityPrediction.created_at.desc())
            .first()
        )
        if prediction:
            latest.append(prediction_to_response(prediction))
    return {"user_id": target_user_id, "predictions": latest}


@router.get("/predictions/{prediction_id}", response_model=ModalityPredictionResponse)
def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prediction = db.query(ModalityPrediction).filter(ModalityPrediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    authorize_prediction_read(current_user, prediction)
    return prediction_to_response(prediction)


@router.get("/availability", response_model=ModalityAvailabilityResponse)
def get_availability(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    modalities = availability_contract()
    active_models = {
        modality: get_active_model(db, modality=modality)
        for modality in ("profile", "text", "speech", "face")
    }
    for item in modalities:
        active_model = active_models.get(item["modality"])
        if active_model:
            item["implemented"] = True
            item["runtime_model_active"] = True
            if item["modality"] == "profile":
                item["limitations"] = [SCREENING_LIMITATION]
            elif item["modality"] == "text":
                item["limitations"] = [
                    "SafeTalk chat is separate and is not used as this modality model.",
                    SCREENING_LIMITATION,
                ]
            elif item["modality"] == "speech":
                item["limitations"] = ["Speech emotion is technically active but fusion excluded without an approved risk mapping."]
            elif item["modality"] == "face":
                item["limitations"] = ["Facial emotion is technically active but low reliability and fusion excluded."]
    return {"modalities": modalities}
