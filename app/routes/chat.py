from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.database import get_db
from app.models.database_models import CounselorAssignment, User, ChatMessage, ModalityPrediction, UserRole
from app.routes.auth import get_current_user
from app.services.consent import get_latest_consent
from app.ml.runtime.base import RuntimeInferenceError, RuntimeModelUnavailable
from app.ml.runtime.registry import predict_with_active_model
from app.ml.runtime.speech import SPEECH_RISK_MAPPING_STATUS, SPEECH_RUNTIME_LIMITATION, SpeechRuntimeLoader
from app.ml.runtime.speech_preprocessor import SpeechPreprocessingError, validate_speech_audio_quality
from app.services.model_registry import get_active_model
from app.services.modalities import (
    MODEL_EVIDENCE,
    create_failed_prediction,
    create_feature_snapshot,
    create_prediction,
    create_unavailable_prediction,
    trigger_fusion_for_prediction,
)
from app.services.counselor_workflow import has_active_assignment, is_counselor_role
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import mimetypes

router = APIRouter(prefix="/api/chat", tags=["Chat"])

UPLOAD_DIR = Path("uploaded_audio")
MAX_AUDIO_BYTES = 5 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a"}
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
}
ALLOWED_MESSAGE_TYPES = {"text", "system", "attachment", "call"}


def _normalize_audio_mime_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()

# ==================== Pydantic Models ====================

class ChatMessageCreate(BaseModel):
    receiver_id: int
    message: str
    message_type: str = "text"

class ChatMessageResponse(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    message: str
    message_type: str
    ai_analysis_requested: bool = False
    ai_analysis_status: str = "not_requested"
    ai_prediction_id: Optional[int] = None
    is_read: bool
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    delivery_status: str = "sent"
    created_at: datetime
    sender_username: str
    
    class Config:
        from_attributes = True

class ChatConversationResponse(BaseModel):
    id: int
    conversation_id: str
    user_id: int
    username: str
    full_name: str
    last_message: Optional[str]
    last_message_time: Optional[datetime]
    unread_count: int
    latest_message_type: Optional[str] = None
    conversation_status: str = "active"
    assigned_counselor: Optional[dict] = None
    student: Optional[dict] = None


class CallRequestCreate(BaseModel):
    receiver_id: int


class CallSignalResponse(BaseModel):
    id: int
    student_id: int
    counselor_id: int
    caller_id: int
    receiver_id: int
    caller_name: str
    receiver_name: str
    student_name: str
    student_email: Optional[str] = None
    counselor_name: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    declined_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class CallWebRTCSignalPayload(BaseModel):
    offer: Optional[Dict[str, Any]] = None
    answer: Optional[Dict[str, Any]] = None
    ice_candidate: Optional[Dict[str, Any]] = None


class CallWebRTCSignalResponse(BaseModel):
    call_id: int
    offer: Optional[Dict[str, Any]] = None
    answer: Optional[Dict[str, Any]] = None
    ice_candidates: List[Dict[str, Any]] = []
    updated_at: Optional[datetime] = None


def _role_value(user: User) -> str:
    return getattr(user.role, "value", user.role)


def _is_student(user: User) -> bool:
    return _role_value(user) == "student"


def _is_admin(user: User) -> bool:
    return _role_value(user) == "admin"


def _conversation_pair_id(first_id: int, second_id: int) -> str:
    left, right = sorted([int(first_id), int(second_id)])
    return f"direct:{left}:{right}"


def _active_counselor_assignment(db: Session, *, student_id: int, counselor_id: int) -> Optional[CounselorAssignment]:
    return (
        db.query(CounselorAssignment)
        .filter(
            CounselorAssignment.student_id == student_id,
            CounselorAssignment.counselor_id == counselor_id,
            CounselorAssignment.active.is_(True),
        )
        .first()
    )


def _student_counselor_pair(first: User, second: User) -> tuple[User, User] | None:
    if _is_student(first) and is_counselor_role(second):
        return first, second
    if _is_student(second) and is_counselor_role(first):
        return second, first
    return None


def _require_chat_user(current_user: User) -> None:
    if _is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators cannot routinely access private counselor chat content",
        )
    if not (_is_student(current_user) or is_counselor_role(current_user)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Direct chat is not available for this account")


def _authorize_direct_chat(db: Session, current_user: User, other_user: User) -> tuple[User, User]:
    _require_chat_user(current_user)
    pair = _student_counselor_pair(current_user, other_user)
    if pair is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Direct chat is limited to assigned student-counselor conversations",
        )
    student, counselor = pair
    if not has_active_assignment(db, counselor.id, student.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Counselor assignment is required for this conversation",
        )
    return student, counselor


def _user_summary(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name or user.username,
        "role": _role_value(user),
    }


def _analysis_consent_record(db: Session, user: User):
    consent = get_latest_consent(db, user.id, "voice_processing")
    if consent and consent.is_granted and consent.withdrawn_at is None:
        return consent
    return None


def _text_analysis_consent_record(db: Session, user: User):
    consent = get_latest_consent(db, user.id, "text_processing")
    if consent and consent.is_granted and consent.withdrawn_at is None:
        return consent
    return None


def _speech_failure_code(status_value: str) -> str:
    return {
        "too_short": "AUDIO_TOO_SHORT",
        "silent": "AUDIO_SILENT",
        "corrupt": "AUDIO_CORRUPT",
        "unsupported": "UNSUPPORTED_AUDIO_CODEC",
        "low_quality": "AUDIO_LOW_QUALITY",
    }.get(status_value, "SPEECH_ANALYSIS_FAILED")


def _persist_speech_analysis(
    db: Session,
    *,
    chat_message: ChatMessage,
    student: User,
    file_path: Path,
    content_type: str,
    consent_record_id: Optional[int],
    conversation_id: str,
):
    active_model = get_active_model(db, modality="speech")
    if not active_model:
        prediction = create_unavailable_prediction(
            db,
            user=student,
            modality="speech",
            failure_code="MODEL_NOT_ACTIVE",
            message="The speech runtime model is not active; voice message delivery was not blocked.",
            source_type="counselor_chat_voice_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
        )
        prediction.metadata_json = {
            **(prediction.metadata_json or {}),
            "source_type": "counselor_chat_voice_message",
            "source_reference": chat_message.id,
            "conversation_reference": conversation_id,
            "analysis_consent_record_id": consent_record_id,
            "analysis_requested": True,
            "student_is_audio_speaker": True,
            "fusion_status": "excluded_model_not_active",
            "fusion_eligible": False,
            "normalized_score": None,
            "risk_mapping_version": None,
            "limitations": [
                "Voice-emotion analysis is supporting evidence only and not a diagnosis.",
                "Speech runtime inference is inactive; no emotion label or contribution was generated.",
            ],
        }
        return prediction

    quality = validate_speech_audio_quality(file_path, content_type=content_type)
    if not quality.accepted:
        return create_prediction(
            db,
            student_id=student.id,
            modality="speech",
            status_value="failed",
            is_available=False,
            output_type="machine_learning",
            source_type="counselor_chat_voice_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
            failure_code=_speech_failure_code(quality.status),
            failure_message_safe="Voice-emotion analysis was unavailable because the uploaded audio did not meet quality requirements.",
            raw_output_json={"quality": quality.__dict__},
            metadata_json={
                "source_type": "counselor_chat_voice_message",
                "source_reference": chat_message.id,
                "conversation_reference": conversation_id,
                "analysis_consent_record_id": consent_record_id,
                "analysis_requested": True,
                "student_is_audio_speaker": True,
                "fusion_status": "excluded_audio_quality",
                "fusion_eligible": False,
                "normalized_score": None,
                "risk_mapping_version": None,
                "limitation": "Poor-quality audio was not converted into a risk score.",
            },
            model_registry=active_model,
            valid_for_hours=None,
            data_quality_status=quality.status,
            data_quality_flags=quality.flags,
        )

    loader = SpeechRuntimeLoader()
    try:
        result = loader.predict(active_model, {"path": str(file_path), "content_type": content_type})
    except (RuntimeModelUnavailable, RuntimeInferenceError, SpeechPreprocessingError) as exc:
        return create_prediction(
            db,
            student_id=student.id,
            modality="speech",
            status_value="failed",
            is_available=False,
            output_type="machine_learning",
            source_type="counselor_chat_voice_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
            failure_code="SPEECH_RUNTIME_FAILED",
            failure_message_safe="Voice-emotion analysis was unavailable; voice message delivery was not blocked.",
            raw_output_json={"error": exc.__class__.__name__},
            metadata_json={
                "source_type": "counselor_chat_voice_message",
                "source_reference": chat_message.id,
                "conversation_reference": conversation_id,
                "analysis_consent_record_id": consent_record_id,
                "analysis_requested": True,
                "student_is_audio_speaker": True,
                "fusion_status": "excluded_runtime_failure",
                "fusion_eligible": False,
                "normalized_score": None,
                "risk_mapping_version": None,
            },
            model_registry=active_model,
            valid_for_hours=None,
            data_quality_status="corrupt",
            data_quality_flags=[exc.__class__.__name__],
        )

    metadata = result.metadata or {}
    snapshot = create_feature_snapshot(
        db,
        student_id=student.id,
        modality="speech",
        source_type="counselor_chat_voice_message",
        source_record_id=chat_message.id,
        source_timestamp=chat_message.created_at,
        feature_schema_version=active_model.feature_schema_version or "1.0.0",
        preprocessing_version=active_model.preprocessing_version or "speech-runtime-v1",
        features_json=result.features,
        data_quality_status=metadata.get("data_quality_status") or "accepted",
        data_quality_flags=metadata.get("data_quality_flags") or [],
        metadata_json={
            "audio_container": Path(file_path).suffix.lower().lstrip("."),
            "original_mime_type": content_type,
            "feature_shape": metadata.get("feature_shape"),
        },
    )


def _persist_student_chat_text_analysis(
    db: Session,
    *,
    chat_message: ChatMessage,
    student: User,
    counselor: User,
    consent_record_id: int,
    conversation_id: str,
):
    existing = (
        db.query(ModalityPrediction)
        .filter(
            ModalityPrediction.modality == "text",
            ModalityPrediction.source_type == "counselor_chat_text_message",
            ModalityPrediction.source_record_id == chat_message.id,
        )
        .order_by(ModalityPrediction.created_at.desc(), ModalityPrediction.id.desc())
        .first()
    )
    if existing:
        return existing

    text = (chat_message.message or "").strip()
    snapshot = create_feature_snapshot(
        db,
        student_id=student.id,
        modality="text",
        source_type="counselor_chat_text_message",
        source_record_id=chat_message.id,
        source_timestamp=chat_message.created_at,
        feature_schema_version=MODEL_EVIDENCE["text"]["schema"],
        preprocessing_version=MODEL_EVIDENCE["text"]["preprocessing"],
        features_json={
            "text_length": len(text),
            "contains_raw_text": False,
            "source_type": "counselor_chat_text_message",
        },
        metadata_json={
            "raw_text_stored_in_feature_snapshot": False,
            "student_authored": True,
            "bot_generated": False,
            "counselor_authored": False,
        },
    )
    active_model = None
    try:
        active_model, result = predict_with_active_model(db, modality="text", payload={"text": text})
        snapshot.feature_schema_version = active_model.feature_schema_version or snapshot.feature_schema_version
        snapshot.preprocessing_version = active_model.preprocessing_version or snapshot.preprocessing_version
        snapshot.features_json = result.features
        return create_prediction(
            db,
            student_id=student.id,
            modality="text",
            status_value="succeeded",
            is_available=True,
            output_type="machine_learning",
            source_type="counselor_chat_text_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
            feature_snapshot=snapshot,
            probability=result.probability,
            confidence=result.confidence,
            label=result.label,
            raw_output_json={"class_probabilities": result.probabilities},
            metadata_json={
                **result.metadata,
                "class_probabilities": result.probabilities,
                "source_type": "counselor_chat_text_message",
                "conversation_reference": conversation_id,
                "counselor_id": counselor.id,
                "analysis_consent_record_id": consent_record_id,
                "student_authored": True,
                "bot_generated": False,
                "counselor_authored": False,
                "raw_text_stored": False,
                "selection_policy": "most_recent_valid_text_prediction",
            },
            model_registry=active_model,
        )
    except RuntimeModelUnavailable:
        prediction = create_unavailable_prediction(
            db,
            user=student,
            modality="text",
            failure_code="MODEL_NOT_ACTIVE",
            message="The trained text modality model is not active in the runtime API.",
            source_type="counselor_chat_text_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
            feature_snapshot=snapshot,
        )
        prediction.metadata_json = {
            **(prediction.metadata_json or {}),
            "source_type": "counselor_chat_text_message",
            "conversation_reference": conversation_id,
            "counselor_id": counselor.id,
            "analysis_consent_record_id": consent_record_id,
            "student_authored": True,
            "bot_generated": False,
            "counselor_authored": False,
            "raw_text_stored": False,
        }
        return prediction
    except RuntimeInferenceError:
        prediction = create_failed_prediction(
            db,
            user=student,
            modality="text",
            failure_code="INFERENCE_FAILED",
            message="The active text runtime model could not safely analyze this counselor-chat message.",
            source_type="counselor_chat_text_message",
            source_record_id=chat_message.id,
            source_timestamp=chat_message.created_at,
            feature_snapshot=snapshot,
            model_registry=active_model,
        )
        prediction.metadata_json = {
            **(prediction.metadata_json or {}),
            "source_type": "counselor_chat_text_message",
            "conversation_reference": conversation_id,
            "counselor_id": counselor.id,
            "analysis_consent_record_id": consent_record_id,
            "student_authored": True,
            "bot_generated": False,
            "counselor_authored": False,
            "raw_text_stored": False,
        }
        return prediction
    return create_prediction(
        db,
        student_id=student.id,
        modality="speech",
        status_value="succeeded",
        is_available=True,
        output_type="machine_learning",
        source_type="counselor_chat_voice_message",
        source_record_id=chat_message.id,
        source_timestamp=chat_message.created_at,
        feature_snapshot=snapshot,
        probability=result.probability,
        confidence=result.confidence,
        label=result.label,
        raw_output_json={
            "emotion_label": result.label,
            "class_probabilities": result.probabilities,
            "confidence": result.confidence,
        },
        metadata_json={
            "source_type": "counselor_chat_voice_message",
            "source_reference": chat_message.id,
            "conversation_reference": conversation_id,
            "analysis_consent_record_id": consent_record_id,
            "analysis_requested": True,
            "student_is_audio_speaker": True,
            "emotion_label": result.label,
            "class_probabilities": result.probabilities,
            "confidence_band": metadata.get("confidence_band"),
            "data_quality_status": metadata.get("data_quality_status") or "accepted",
            "technical_status": "technically_verified_but_fusion_excluded",
            "fusion_status": SPEECH_RISK_MAPPING_STATUS,
            "fusion_eligible": False,
            "normalized_score": None,
            "risk_mapping_version": None,
            "limitation": SPEECH_RUNTIME_LIMITATION,
        },
        model_registry=active_model,
        valid_for_hours=24,
        data_quality_status=metadata.get("data_quality_status") or "accepted",
        data_quality_flags=metadata.get("data_quality_flags") or [],
    )


def _message_response(message: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        message=message.message,
        message_type=message.message_type,
        ai_analysis_requested=message.ai_analysis_requested,
        ai_analysis_status=message.ai_analysis_status,
        ai_prediction_id=message.ai_prediction_id,
        is_read=message.is_read,
        sent_at=message.sent_at or message.created_at,
        delivered_at=message.delivered_at,
        read_at=message.read_at,
        delivery_status=message.delivery_status or ("read" if message.is_read else "sent"),
        created_at=message.created_at,
        sender_username=message.sender.username,
    )


def _call_metadata(
    *,
    status_value: str,
    student: User,
    counselor: User,
    caller: User,
    receiver: User,
    requested_at: datetime,
    **extra,
) -> dict:
    return {
        "call_status": status_value,
        "student_id": student.id,
        "counselor_id": counselor.id,
        "student_name": student.full_name or student.username,
        "student_email": student.email,
        "counselor_name": counselor.full_name or counselor.username,
        "caller_id": caller.id,
        "receiver_id": receiver.id,
        "caller_name": caller.full_name or caller.username,
        "receiver_name": receiver.full_name or receiver.username,
        "requested_at": requested_at.isoformat(),
        **extra,
    }


def _call_response(message: ChatMessage) -> CallSignalResponse:
    metadata = message.metadata_json or {}
    student = message.sender if _is_student(message.sender) else message.receiver
    counselor = message.receiver if is_counselor_role(message.receiver) else message.sender
    return CallSignalResponse(
        id=message.id,
        student_id=metadata.get("student_id") or student.id,
        counselor_id=metadata.get("counselor_id") or counselor.id,
        caller_id=metadata.get("caller_id") or message.sender_id,
        receiver_id=metadata.get("receiver_id") or message.receiver_id,
        caller_name=metadata.get("caller_name") or message.sender.full_name or message.sender.username,
        receiver_name=metadata.get("receiver_name") or message.receiver.full_name or message.receiver.username,
        student_name=metadata.get("student_name") or student.full_name or student.username,
        student_email=metadata.get("student_email") or student.email,
        counselor_name=metadata.get("counselor_name") or counselor.full_name or counselor.username,
        status=metadata.get("call_status") or "ringing",
        created_at=message.created_at,
        updated_at=message.updated_at,
        answered_at=datetime.fromisoformat(metadata["answered_at"]) if metadata.get("answered_at") else None,
        declined_at=datetime.fromisoformat(metadata["declined_at"]) if metadata.get("declined_at") else None,
        cancelled_at=datetime.fromisoformat(metadata["cancelled_at"]) if metadata.get("cancelled_at") else None,
        ended_at=datetime.fromisoformat(metadata["ended_at"]) if metadata.get("ended_at") else None,
    )


def _webrtc_signal_response(message: ChatMessage) -> CallWebRTCSignalResponse:
    metadata = message.metadata_json or {}
    webrtc = metadata.get("webrtc") or {}
    return CallWebRTCSignalResponse(
        call_id=message.id,
        offer=webrtc.get("offer"),
        answer=webrtc.get("answer"),
        ice_candidates=webrtc.get("ice_candidates") or [],
        updated_at=message.updated_at,
    )


def _get_authorized_call(message_id: int, current_user: User, db: Session) -> ChatMessage:
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id, ChatMessage.message_type == "call").first()
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call request not found")
    if not _is_chat_participant(message, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not authorized to access this call")
    other_user = message.receiver if message.sender_id == current_user.id else message.sender
    _authorize_direct_chat(db, current_user, other_user)
    return message

def _is_chat_participant(message: ChatMessage, user: User) -> bool:
    return message.sender_id == user.id or message.receiver_id == user.id

def _safe_audio_path(filename: str) -> Path:
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid filename",
        )
    return UPLOAD_DIR / filename

def _get_authorized_voice_message(
    message_id: int,
    current_user: User,
    db: Session,
) -> ChatMessage:
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message or message.message_type != "voice":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio message not found",
        )
    if not _is_chat_participant(message, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this audio message",
        )
    other_user = message.receiver if message.sender_id == current_user.id else message.sender
    _authorize_direct_chat(db, current_user, other_user)
    return message

def _audio_response(message: ChatMessage) -> FileResponse:
    filename = Path(message.message or "").name
    file_path = _safe_audio_path(filename)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found",
        )

    mime_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        str(file_path),
        media_type=mime_type or "audio/wav",
        filename=filename,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=300",
            "Content-Disposition": "inline",
        },
    )

def _resolve_voice_message_by_filename(filename: str, db: Session) -> ChatMessage:
    safe_name = _safe_audio_path(filename).name
    candidates = [
        safe_name,
        f"uploaded_audio/{safe_name}",
        f"uploaded_audio\\{safe_name}",
    ]
    message = (
        db.query(ChatMessage)
        .filter(ChatMessage.message_type == "voice", ChatMessage.message.in_(candidates))
        .order_by(ChatMessage.created_at.desc())
        .first()
    )
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio message not found",
        )
    return message

# ==================== Chat Endpoints ====================

@router.post("/send", response_model=ChatMessageResponse)
def send_message(
    message_data: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a chat message to an assigned student/counselor participant."""
    
    # Verify receiver exists
    receiver = db.query(User).filter(User.id == message_data.receiver_id).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Receiver not found"
        )
    if message_data.message_type not in ALLOWED_MESSAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported message type")
    if message_data.message_type == "text" and not message_data.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    student, counselor = _authorize_direct_chat(db, current_user, receiver)
    text_analysis_consent = None
    if message_data.message_type == "text" and current_user.id == student.id:
        text_analysis_consent = _text_analysis_consent_record(db, current_user)
    
    # Create message
    chat_message = ChatMessage(
        sender_id=current_user.id,
        receiver_id=message_data.receiver_id,
        message=message_data.message,
        message_type=message_data.message_type,
        ai_analysis_requested=bool(text_analysis_consent),
        ai_analysis_status="pending" if text_analysis_consent else "not_requested",
        sent_at=datetime.utcnow(),
        delivery_status="sent",
        metadata_json={
            "conversation_id": _conversation_pair_id(student.id, counselor.id),
            "student_id": student.id,
            "counselor_id": counselor.id,
            "sender_role": _role_value(current_user),
            "text_analysis_consent_record_id": text_analysis_consent.id if text_analysis_consent else None,
            "delivery_independent_from_inference": True,
        },
    )
    
    db.add(chat_message)
    db.flush()

    if text_analysis_consent:
        prediction = _persist_student_chat_text_analysis(
            db,
            chat_message=chat_message,
            student=student,
            counselor=counselor,
            consent_record_id=text_analysis_consent.id,
            conversation_id=_conversation_pair_id(student.id, counselor.id),
        )
        chat_message.ai_prediction_id = prediction.id
        chat_message.ai_analysis_status = (
            "succeeded"
            if prediction.status == "succeeded"
            else "unavailable"
            if prediction.status == "unavailable"
            else "failed"
        )
        trigger_fusion_for_prediction(db, prediction, trigger_source="counselor_chat_text_analysis", actor=current_user)

    db.commit()
    db.refresh(chat_message)
    
    return _message_response(chat_message)


@router.post("/calls/request", response_model=CallSignalResponse)
def request_call(
    payload: CallRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create an immediate call request between assigned student/counselor participants."""
    receiver = db.query(User).filter(User.id == payload.receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call receiver not found")
    student, counselor = _authorize_direct_chat(db, current_user, receiver)

    existing = (
        db.query(ChatMessage)
        .filter(
            or_(
                and_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == receiver.id),
                and_(ChatMessage.sender_id == receiver.id, ChatMessage.receiver_id == current_user.id),
            ),
            ChatMessage.message_type == "call",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .first()
    )
    if existing and (existing.metadata_json or {}).get("call_status") in {"ringing", "answered"}:
        return _call_response(existing)

    now = datetime.utcnow()
    call_message = ChatMessage(
        sender_id=current_user.id,
        receiver_id=receiver.id,
        message="Immediate call requested",
        message_type="call",
        sent_at=now,
        delivery_status="sent",
        metadata_json=_call_metadata(
            status_value="ringing",
            student=student,
            counselor=counselor,
            caller=current_user,
            receiver=receiver,
            requested_at=now,
        ),
    )
    db.add(call_message)
    db.commit()
    db.refresh(call_message)
    return _call_response(call_message)


@router.get("/calls/incoming", response_model=List[CallSignalResponse])
def incoming_calls(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return ringing call requests for the signed-in chat participant."""
    _require_chat_user(current_user)
    calls = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.receiver_id == current_user.id,
            ChatMessage.message_type == "call",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(20)
        .all()
    )
    return [
        _call_response(call)
        for call in calls
        if (call.metadata_json or {}).get("call_status") == "ringing"
    ]


@router.get("/calls/outgoing", response_model=List[CallSignalResponse])
def outgoing_calls(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return recent call requests created by the signed-in chat participant."""
    _require_chat_user(current_user)
    calls = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.sender_id == current_user.id,
            ChatMessage.message_type == "call",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(10)
        .all()
    )
    return [_call_response(call) for call in calls]


@router.get("/calls/active", response_model=List[CallSignalResponse])
def active_calls(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return active ringing or answered calls for the signed-in chat participant."""
    _require_chat_user(current_user)
    calls = (
        db.query(ChatMessage)
        .filter(
            or_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == current_user.id),
            ChatMessage.message_type == "call",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(20)
        .all()
    )
    return [
        _call_response(call)
        for call in calls
        if (call.metadata_json or {}).get("call_status") in {"ringing", "answered"}
    ]


@router.get("/calls/{call_id}/signal", response_model=CallWebRTCSignalResponse)
def get_call_signal(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return WebRTC signaling data for an answered call."""
    call = _get_authorized_call(call_id, current_user, db)
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "answered":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Call is not ready for audio signaling")
    return _webrtc_signal_response(call)


@router.post("/calls/{call_id}/signal", response_model=CallWebRTCSignalResponse)
def update_call_signal(
    call_id: int,
    payload: CallWebRTCSignalPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Store WebRTC offer, answer, and ICE candidates for browser voice calls."""
    call = _get_authorized_call(call_id, current_user, db)
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "answered":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Call is not ready for audio signaling")

    now = datetime.utcnow()
    webrtc = {
        "offer": None,
        "answer": None,
        "ice_candidates": [],
        **(metadata.get("webrtc") or {}),
    }

    if payload.offer:
        webrtc["offer"] = {
            **payload.offer,
            "from_user_id": current_user.id,
            "created_at": now.isoformat(),
        }
        webrtc["answer"] = None
        webrtc["ice_candidates"] = [
            candidate for candidate in (webrtc.get("ice_candidates") or [])
            if candidate.get("from_user_id") == current_user.id
        ]

    if payload.answer:
        webrtc["answer"] = {
            **payload.answer,
            "from_user_id": current_user.id,
            "created_at": now.isoformat(),
        }

    if payload.ice_candidate:
        candidate = {
            **payload.ice_candidate,
            "from_user_id": current_user.id,
            "created_at": now.isoformat(),
        }
        candidates = (webrtc.get("ice_candidates") or []) + [candidate]
        webrtc["ice_candidates"] = candidates[-100:]

    call.metadata_json = {
        **metadata,
        "webrtc": webrtc,
    }
    db.commit()
    db.refresh(call)
    return _webrtc_signal_response(call)


@router.post("/calls/{call_id}/answer", response_model=CallSignalResponse)
def answer_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    call = _get_authorized_call(call_id, current_user, db)
    if call.receiver_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the call receiver can answer this call")
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "ringing":
        return _call_response(call)
    now = datetime.utcnow()
    call.metadata_json = {
        **metadata,
        "call_status": "answered",
        "answered_at": now.isoformat(),
        "answered_by": current_user.id,
    }
    call.is_read = True
    call.read_at = now
    call.delivery_status = "read"
    db.commit()
    db.refresh(call)
    return _call_response(call)


@router.post("/calls/{call_id}/decline", response_model=CallSignalResponse)
def decline_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    call = _get_authorized_call(call_id, current_user, db)
    if call.receiver_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the call receiver can decline this call")
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "ringing":
        return _call_response(call)
    now = datetime.utcnow()
    call.metadata_json = {
        **metadata,
        "call_status": "declined",
        "declined_at": now.isoformat(),
        "declined_by": current_user.id,
    }
    call.is_read = True
    call.read_at = now
    call.delivery_status = "read"
    db.commit()
    db.refresh(call)
    return _call_response(call)


@router.post("/calls/{call_id}/cancel", response_model=CallSignalResponse)
def cancel_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    call = _get_authorized_call(call_id, current_user, db)
    if call.sender_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the call requester can cancel this call")
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "ringing":
        return _call_response(call)
    now = datetime.utcnow()
    call.metadata_json = {
        **metadata,
        "call_status": "cancelled",
        "cancelled_at": now.isoformat(),
    }
    db.commit()
    db.refresh(call)
    return _call_response(call)


@router.post("/calls/{call_id}/end", response_model=CallSignalResponse)
def end_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    call = _get_authorized_call(call_id, current_user, db)
    metadata = call.metadata_json or {}
    if metadata.get("call_status") != "answered":
        return _call_response(call)
    now = datetime.utcnow()
    call.metadata_json = {
        **metadata,
        "call_status": "ended",
        "ended_at": now.isoformat(),
        "ended_by": current_user.id,
    }
    db.commit()
    db.refresh(call)
    return _call_response(call)

@router.get("/conversations", response_model=List[ChatConversationResponse])
def get_conversations(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get visible direct-chat conversation partners with last message preview."""
    _require_chat_user(current_user)
    
    # Get all users this person has chatted with
    base_query = db.query(
        ChatMessage.sender_id, ChatMessage.receiver_id
    ).filter(
        or_(
            ChatMessage.sender_id == current_user.id,
            ChatMessage.receiver_id == current_user.id
        )
    )
    conversation_partners = base_query.all()
    
    partner_ids = set()
    for sender_id, receiver_id in conversation_partners:
        if sender_id != current_user.id:
            partner_ids.add(sender_id)
        if receiver_id != current_user.id:
            partner_ids.add(receiver_id)
    
    conversations = []
    for partner_id in partner_ids:
        partner = db.query(User).filter(User.id == partner_id).first()
        if not partner:
            continue
        try:
            student, counselor = _authorize_direct_chat(db, current_user, partner)
        except HTTPException:
            continue
        # Get last message
        last_msg = db.query(ChatMessage).filter(
            or_(
                and_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == partner_id),
                and_(ChatMessage.sender_id == partner_id, ChatMessage.receiver_id == current_user.id)
            )
        ).order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).first()
        
        # Count unread messages from this partner
        unread_count = db.query(ChatMessage).filter(
            ChatMessage.sender_id == partner_id,
            ChatMessage.receiver_id == current_user.id,
            ChatMessage.is_read == False
        ).count()
        
        conversations.append(ChatConversationResponse(
            id=partner.id,
            conversation_id=_conversation_pair_id(student.id, counselor.id),
            user_id=partner.id,
            username=partner.username,
            full_name=partner.full_name or partner.username,
            last_message=("[voice message]" if last_msg and last_msg.message_type == "voice" else last_msg.message[:50]) if last_msg else None,
            last_message_time=last_msg.created_at if last_msg else None,
            unread_count=unread_count,
            latest_message_type=last_msg.message_type if last_msg else None,
            conversation_status="active",
            assigned_counselor=_user_summary(counselor),
            student=_user_summary(student) if is_counselor_role(current_user) else None,
        ))
    
    # Sort by last message time (most recent first)
    conversations.sort(key=lambda x: x.last_message_time or datetime.min, reverse=True)
    
    return conversations[:limit]

@router.get("/messages/{user_id}", response_model=List[ChatMessageResponse])
def get_messages(
    user_id: int,
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get paginated messages between current user and an authorized participant."""
    
    # Verify the other user exists
    other_user = db.query(User).filter(User.id == user_id).first()
    if not other_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    _authorize_direct_chat(db, current_user, other_user)
    
    # Get messages (both sent and received)
    messages = db.query(ChatMessage).filter(
        or_(
            and_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == user_id),
            and_(ChatMessage.sender_id == user_id, ChatMessage.receiver_id == current_user.id)
        )
    ).order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).offset(offset).limit(limit).all()
    
    # Mark messages from the other user as read
    db.query(ChatMessage).filter(
        ChatMessage.sender_id == user_id,
        ChatMessage.receiver_id == current_user.id,
        ChatMessage.is_read == False
    ).update({
        ChatMessage.is_read: True,
        ChatMessage.read_at: datetime.utcnow(),
        ChatMessage.delivery_status: "read",
    })
    db.commit()

    for msg in messages:
        if msg.receiver_id == current_user.id and msg.is_read:
            msg.delivery_status = "read"
            msg.read_at = msg.read_at or datetime.utcnow()

    return [_message_response(msg) for msg in messages]

@router.get("/unread-count")
def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get total unread message count"""
    
    _require_chat_user(current_user)
    unread_count = db.query(ChatMessage).filter(
        ChatMessage.receiver_id == current_user.id,
        ChatMessage.is_read == False
    ).count()
    
    return {"unread_count": unread_count}

@router.post("/mark-read/{message_id}")
def mark_message_read(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a message as read"""
    
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # Only receiver can mark as read
    if message.receiver_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only mark your own messages as read"
        )
    
    message.is_read = True
    message.read_at = datetime.utcnow()
    message.delivery_status = "read"
    db.commit()
    
    return {"status": "success", "message_id": message_id}

@router.get("/counselors")
def get_available_counselors(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of available counselors for chat"""
    _require_chat_user(current_user)
    
    if is_counselor_role(current_user):
        return []

    assigned_ids = [
        row[0]
        for row in db.query(CounselorAssignment.counselor_id)
        .filter(CounselorAssignment.student_id == current_user.id, CounselorAssignment.active.is_(True))
        .distinct()
        .all()
    ]
    if not assigned_ids:
        return []

    counselors = db.query(User).filter(
        User.role.in_([UserRole.COUNSELOR, UserRole.PSYCHIATRIST]),
        User.is_active == True,
        User.id.in_(assigned_ids),
    ).all()
    
    return [
        {
            "id": counselor.id,
            "username": counselor.username,
            "full_name": counselor.full_name or counselor.username,
            "role": counselor.role
        }
        for counselor in counselors
    ]

@router.post("/send-voice", response_model=ChatMessageResponse)
def send_voice_message(
    receiver_id: int = Form(...),
    analyze_emotional_tone: bool = Form(False),
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a voice message. Delivery is independent from optional speech analysis."""
    if analyze_emotional_tone and current_user.role.value != "student":
        analyze_emotional_tone = False
    
    # Verify receiver exists
    receiver = db.query(User).filter(User.id == receiver_id).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Receiver not found"
        )
    student, counselor = _authorize_direct_chat(db, current_user, receiver)
    analysis_consent = None
    if analyze_emotional_tone:
        analysis_consent = _analysis_consent_record(db, current_user)
        if analysis_consent is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "VOICE_ANALYSIS_CONSENT_REQUIRED", "message": "Voice message can be sent without analysis, but voice-emotion analysis requires active consent."},
            )
    
    original_name = audio.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_AUDIO_EXTENSION", "message": "Unsupported audio file extension"},
        )

    normalized_content_type = _normalize_audio_mime_type(audio.content_type)
    if normalized_content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_AUDIO_MIME_TYPE", "message": "Unsupported audio file type"},
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"voice_{uuid4().hex}{extension}"
    file_path = _safe_audio_path(filename)

    total_bytes = 0
    try:
        with file_path.open("wb") as buffer:
            while True:
                chunk = audio.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_AUDIO_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={"code": "AUDIO_TOO_LARGE", "message": "Audio file exceeds the maximum allowed size"},
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save audio file",
        )

    if total_bytes == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_AUDIO_FILE", "message": "Audio file is empty"},
        )
    
    # Create message record
    chat_message = ChatMessage(
        sender_id=current_user.id,
        receiver_id=receiver_id,
        message=filename,
        message_type='voice',
        ai_analysis_requested=analyze_emotional_tone,
        ai_analysis_status="pending" if analyze_emotional_tone else "not_requested",
        sent_at=datetime.utcnow(),
        delivery_status="sent",
        metadata_json={
            "student_notice": "When enabled, your voice message may be analyzed for emotional tone and included as supporting screening evidence.",
            "delivery_independent_from_inference": True,
            "conversation_id": _conversation_pair_id(student.id, counselor.id),
            "student_id": student.id,
            "counselor_id": counselor.id,
            "uploader_user_id": current_user.id,
            "uploader_role": _role_value(current_user),
            "original_mime_type": audio.content_type,
            "accepted_mime_type": normalized_content_type,
            "normalized_format": extension.lstrip("."),
            "size_bytes": total_bytes,
            "analysis_requested": analyze_emotional_tone,
            "analysis_consent_record_id": analysis_consent.id if analysis_consent else None,
            "retention_status": "retained_for_private_chat",
        },
    )
    
    db.add(chat_message)
    db.flush()
    if analyze_emotional_tone:
        conversation_id = _conversation_pair_id(student.id, counselor.id)
        prediction = _persist_speech_analysis(
            db,
            chat_message=chat_message,
            student=current_user,
            file_path=file_path,
            content_type=normalized_content_type,
            consent_record_id=analysis_consent.id if analysis_consent else None,
            conversation_id=conversation_id,
        )
        chat_message.ai_prediction_id = prediction.id
        chat_message.ai_analysis_status = (
            "succeeded"
            if prediction.status == "succeeded"
            else "unavailable"
            if prediction.status == "unavailable"
            else "failed"
        )
        if prediction.status == "succeeded" and (prediction.metadata_json or {}).get("fusion_eligible") is True:
            trigger_fusion_for_prediction(db, prediction, trigger_source="counselor_chat_voice_analysis", actor=current_user)
    db.commit()
    db.refresh(chat_message)
    
    return _message_response(chat_message)

@router.get("/messages/{message_id}/audio")
def get_message_audio(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stream a voice message after participant authorization."""
    message = _get_authorized_voice_message(message_id, current_user, db)
    return _audio_response(message)

@router.get("/audio/{filename}")
def get_audio(
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deprecated filename route; still requires participant authorization."""
    message = _resolve_voice_message_by_filename(filename, db)
    if not _is_chat_participant(message, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this audio message",
        )
    other_user = message.receiver if message.sender_id == current_user.id else message.sender
    _authorize_direct_chat(db, current_user, other_user)
    return _audio_response(message)
