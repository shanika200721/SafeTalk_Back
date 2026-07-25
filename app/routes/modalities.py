from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import (
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
from app.ml.runtime.registry import predict_with_active_model
from app.services.model_registry import get_active_model
from app.services.consent import require_active_consent
from app.services.modalities import (
    MODALITY_CONSENT_TYPES,
    MODEL_EVIDENCE,
    SCREENING_LIMITATION,
    availability_contract,
    authorize_prediction_read,
    create_prediction,
    create_dass21_prediction_for_assessment,
    create_failed_prediction,
    create_feature_snapshot,
    create_mood_prediction_for_checkin,
    create_profile_prediction_for_assessment,
    create_unavailable_prediction,
    populate_dass21_metadata,
    prediction_to_response,
    verify_owned_chat_message,
)
from app.utils.assessment_calculator import ProfileRiskCalculator
from app.utils.dass21_calculator import DASS21Calculator


router = APIRouter(prefix="/api/modalities", tags=["Modalities"])


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
    db.commit()
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
    db.commit()
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
    db.commit()
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
    db.commit()
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

    if request.chat_message_id is not None:
        message = verify_owned_chat_message(db, request.chat_message_id, current_user)
        source_record_id = message.id
        source_timestamp = message.created_at
        source_type = "chat_voice_message"
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
    db.commit()
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
    prediction = create_unavailable_prediction(
        db,
        user=current_user,
        modality="face",
        failure_code="MODEL_NOT_ACTIVE",
        message="Backend face inference is not active. Frontend random emotion output is not accepted as evidence.",
        source_type="future_face_reference" if request.source_reference_id else "no_face_source",
    )
    db.commit()
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
    prediction = create_unavailable_prediction(
        db,
        user=current_user,
        modality="behavioral",
        failure_code="MODALITY_NOT_VALIDATED",
        message="Behavioral modality is not validated for runtime prediction.",
        source_type="not_validated",
    )
    db.commit()
    db.refresh(prediction)
    return prediction_to_response(prediction)


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
        for modality in ("profile", "text")
    }
    for item in modalities:
        active_model = active_models.get(item["modality"])
        if active_model:
            item["implemented"] = True
            item["runtime_model_active"] = True
            item["limitations"] = [SCREENING_LIMITATION] if item["modality"] == "profile" else [
                "SafeTalk chat is separate and is not used as this modality model.",
                SCREENING_LIMITATION,
            ]
    return {"modalities": modalities}
