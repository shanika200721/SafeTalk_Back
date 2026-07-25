from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.database import get_db
from app.models.database_models import User, UserRole, SafeTalkBotMessage, SafeTalkConversation
from app.routes.auth import get_current_user
from app.ml.counselor_safetalk_bot import CounselorSafeTalkBot
import logging
from app.services.safetalk_safety import (
    RESPONSE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    LIMITATIONS,
    make_context_state,
    make_title,
    render_safety_response,
    route_safetalk_message,
    load_crisis_resources,
)

logger = logging.getLogger(__name__)

# Initialize bot (singleton pattern)
_bot_instance = None

def get_bot() -> CounselorSafeTalkBot:
    """Get or create counselor bot instance with Groq LLM support"""
    global _bot_instance
    if _bot_instance is None:
        logger.warning("[BOT] Initializing SafeTalk Bot with Groq LLM support...")
        _bot_instance = CounselorSafeTalkBot()
        logger.warning(f"[BOT] Bot initialized. LLM Provider: {_bot_instance.llm_provider}")
    return _bot_instance

router = APIRouter(prefix="/api/bot", tags=["SafeTalk Bot"])

# ==================== Pydantic Models ====================

class BotMessageCreate(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class BotMessageResponse(BaseModel):
    id: int
    conversation_id: Optional[int] = None
    user_message: str
    bot_response: str
    intent: str
    route: Optional[str] = None
    severity: Optional[str] = None
    topic_label: Optional[str] = None
    response_policy_version: Optional[str] = None
    response_variant_id: Optional[str] = None
    legacy_response: bool = False
    confidence: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class BotChatResponse(BaseModel):
    """For counselor/admin - full technical details"""
    user_message: str
    main_response: str
    alternative_responses: List[str]
    intent: str
    confidence: float
    crisis_level: int
    empathy_level: int
    is_crisis: bool
    techniques_used: List[str]
    follow_up_questions: List[str]
    suggested_actions: List[str]
    crisis_resources: List[str]
    response_type: str
    timestamp: datetime

class StudentBotResponse(BaseModel):
    """For students - clean response without technical details"""
    conversation_id: int
    message_id: int
    route: str
    severity: str
    response: str
    message: str
    follow_up: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    resource_actions: List[Dict[str, Any]] = Field(default_factory=list)
    safety_check_required: bool
    human_contact_recommended: bool
    alert_created: bool = False
    counselor_contacted: bool = False
    emergency_services_contacted: bool = False
    limitations: List[str] = Field(default_factory=lambda: LIMITATIONS)
    is_crisis: bool
    crisis_resources: List[str] = Field(default_factory=list)
    timestamp: datetime

class BotHistoryResponse(BaseModel):
    messages: List[BotMessageResponse]
    total_count: int


class SafeTalkConversationCreate(BaseModel):
    pass


class SafeTalkConversationResponse(BaseModel):
    id: int
    title: str
    topic_label: str
    status: str
    last_message_preview: Optional[str] = None
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    response_policy_version: Optional[str] = None
    legacy_response: bool = False
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SafeTalkConversationDetail(BaseModel):
    conversation: SafeTalkConversationResponse
    messages: List[BotMessageResponse]


class SafeTalkResourceResponse(BaseModel):
    version: str
    review_status: str
    notes: str
    fallback: Dict[str, Any]
    resources: List[Dict[str, Any]]


def _require_student(user: User) -> None:
    if user.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="SafeTalk bot conversations are available to authenticated students only",
        )


def _owned_conversation(db: Session, user: User, conversation_id: int) -> SafeTalkConversation:
    conversation = (
        db.query(SafeTalkConversation)
        .filter(SafeTalkConversation.id == conversation_id, SafeTalkConversation.user_id == user.id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SafeTalk conversation not found")
    return conversation


def _get_or_create_conversation(db: Session, user: User, conversation_id: Optional[int]) -> SafeTalkConversation:
    if conversation_id is not None:
        conversation = _owned_conversation(db, user, conversation_id)
        if conversation.status == "archived":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Archived SafeTalk conversations cannot receive new messages")
        return conversation

    conversation = (
        db.query(SafeTalkConversation)
        .filter(SafeTalkConversation.user_id == user.id, SafeTalkConversation.status == "active")
        .order_by(SafeTalkConversation.updated_at.desc(), SafeTalkConversation.id.desc())
        .first()
    )
    if conversation:
        return conversation

    conversation = SafeTalkConversation(
        user_id=user.id,
        title="New SafeTalk conversation",
        topic_label="Other",
        context_state={"version": "safetalk-context-v1"},
        safety_policy_version=SAFETY_POLICY_VERSION,
    )
    db.add(conversation)
    db.flush()
    return conversation


def _history_label_for(message: SafeTalkBotMessage) -> str:
    if message.topic_label:
        return message.topic_label
    return route_safetalk_message(message.user_message or "").topic_label


def _message_response(message: SafeTalkBotMessage) -> BotMessageResponse:
    topic_label = _history_label_for(message)
    route = message.route or route_safetalk_message(message.user_message or "").route
    details = message.response_details or {}
    policy_version = (
        getattr(message, "response_policy_version", None)
        or details.get("response_policy_version")
        or details.get("policy_version")
    )
    return BotMessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        user_message=message.user_message,
        bot_response=message.bot_response,
        intent=topic_label,
        route=route,
        severity=message.severity,
        topic_label=topic_label,
        response_policy_version=policy_version,
        response_variant_id=getattr(message, "response_variant_id", None) or details.get("response_variant_id"),
        legacy_response=policy_version not in {None, RESPONSE_POLICY_VERSION},
        confidence=0.0,
        created_at=message.created_at,
    )


def _conversation_response(conversation: SafeTalkConversation) -> SafeTalkConversationResponse:
    messages = list(getattr(conversation, "messages", []) or [])
    latest_message = max(messages, key=lambda item: (item.created_at or datetime.min, item.id or 0), default=None)
    latest_details = latest_message.response_details if latest_message and latest_message.response_details else {}
    latest_policy = (
        getattr(latest_message, "response_policy_version", None)
        if latest_message
        else None
    ) or latest_details.get("response_policy_version") or getattr(conversation, "response_policy_version", None)
    preview = None
    if latest_message:
        preview_source = latest_message.user_message or latest_message.bot_response or ""
        preview = preview_source.strip()
        if len(preview) > 96:
            preview = preview[:93].rstrip() + "..."
    return SafeTalkConversationResponse(
        id=conversation.id,
        title=conversation.title or conversation.topic_label or "SafeTalk conversation",
        topic_label=conversation.topic_label or "Other",
        status=conversation.status,
        last_message_preview=preview,
        last_message_at=latest_message.created_at if latest_message else conversation.updated_at,
        message_count=len(messages),
        response_policy_version=latest_policy or RESPONSE_POLICY_VERSION,
        legacy_response=bool(latest_message and latest_policy not in {None, RESPONSE_POLICY_VERSION}),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        archived_at=conversation.archived_at,
    )

# ==================== Bot Endpoints ====================

@router.post("/safetalk/chat", response_model=StudentBotResponse)
def chat_with_safetalk(
    message_data: BotMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to SafeTalk through the deterministic Phase 4F safety layer."""
    _require_student(current_user)
    
    user_message = message_data.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )

    conversation = _get_or_create_conversation(db, current_user, message_data.conversation_id)
    safety_route = route_safetalk_message(user_message, context_state=conversation.context_state)
    rendered = render_safety_response(safety_route, context_state=conversation.context_state)
    response_text = rendered["message"]
    if rendered.get("follow_up"):
        response_text = f"{response_text}\n\n{rendered['follow_up']}"

    context_state = make_context_state(
        safety_route,
        previous=conversation.context_state,
        now=datetime.utcnow(),
    )
    conversation.context_state = {
        **context_state,
        "conversation_id": conversation.id,
        "previous_user_message": user_message,
        "previous_response_variant_id": rendered.get("variant_id"),
    }
    conversation.safety_policy_version = SAFETY_POLICY_VERSION
    if hasattr(conversation, "response_policy_version"):
        conversation.response_policy_version = RESPONSE_POLICY_VERSION
    conversation.topic_label = safety_route.topic_label
    if not conversation.title or conversation.title == "New SafeTalk conversation":
        conversation.title = make_title(user_message, safety_route.topic_label)
    conversation.updated_at = datetime.utcnow()

    severity_to_level = {"none": 0, "low": 2, "medium": 5, "high": 7, "crisis": 9, "imminent": 10}
    bot_message = SafeTalkBotMessage(
        user_id=current_user.id,
        conversation_id=conversation.id,
        user_message=user_message,
        bot_response=response_text,
        intent=safety_route.topic_label,
        confidence=0.0,
        crisis_level=severity_to_level.get(safety_route.severity, 0),
        route=safety_route.route,
        severity=safety_route.severity,
        topic_label=safety_route.topic_label,
        response_template_version=rendered["template_version"],
        safety_check_required=safety_route.requires_direct_safety_question,
        human_contact_recommended=safety_route.recommend_immediate_support,
        safety_policy_version=SAFETY_POLICY_VERSION,
        response_policy_version=RESPONSE_POLICY_VERSION,
        response_variant_id=rendered.get("variant_id"),
        response_details={
            "policy_version": SAFETY_POLICY_VERSION,
            "response_policy_version": RESPONSE_POLICY_VERSION,
            "route": safety_route.route,
            "severity": safety_route.severity,
            "template_id": safety_route.safe_response_template_id,
            "response_variant_id": rendered.get("variant_id"),
            "internal_reason_code": safety_route.internal_reason_code,
            "classifier_used": safety_route.classifier_used,
            "resource_action_count": len(rendered["resource_actions"]),
            "alert_created": False,
            "counselor_contacted": False,
            "emergency_services_contacted": False,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    
    db.add(bot_message)
    db.commit()
    db.refresh(bot_message)
    
    is_crisis = safety_route.route in {
        "possible_self_harm_or_crisis",
        "explicit_suicidal_intent",
        "imminent_self_harm",
    }
    return StudentBotResponse(
        conversation_id=conversation.id,
        message_id=bot_message.id,
        route=safety_route.route,
        severity=safety_route.severity,
        response=response_text,
        message=response_text,
        follow_up=rendered.get("follow_up"),
        recommended_actions=rendered["recommended_actions"],
        resource_actions=rendered["resource_actions"],
        safety_check_required=safety_route.requires_direct_safety_question,
        human_contact_recommended=safety_route.recommend_immediate_support,
        is_crisis=is_crisis,
        crisis_resources=[item["label"] for item in rendered["resource_actions"]] if is_crisis else [],
        timestamp=bot_message.created_at
    )


@router.post("/safetalk/conversations/{conversation_id}/messages", response_model=StudentBotResponse)
def add_safetalk_conversation_message(
    conversation_id: int,
    message_data: BotMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return chat_with_safetalk(
        BotMessageCreate(message=message_data.message, conversation_id=conversation_id),
        current_user=current_user,
        db=db,
    )


@router.post("/safetalk/conversations", response_model=SafeTalkConversationResponse)
def start_safetalk_conversation(
    _: SafeTalkConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    conversation = SafeTalkConversation(
        user_id=current_user.id,
        title="New SafeTalk conversation",
        topic_label="Other",
        context_state={"version": "safetalk-context-v1", "conversation_id": None},
        safety_policy_version=SAFETY_POLICY_VERSION,
    )
    if hasattr(conversation, "response_policy_version"):
        conversation.response_policy_version = RESPONSE_POLICY_VERSION
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    conversation.context_state = {**(conversation.context_state or {}), "conversation_id": conversation.id}
    db.commit()
    db.refresh(conversation)
    return _conversation_response(conversation)


@router.get("/safetalk/conversations", response_model=List[SafeTalkConversationResponse])
def list_safetalk_conversations(
    current_user: User = Depends(get_current_user),
    include_archived: bool = False,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    query = db.query(SafeTalkConversation).filter(SafeTalkConversation.user_id == current_user.id)
    if not include_archived:
        query = query.filter(SafeTalkConversation.status != "archived")
    conversations = query.order_by(SafeTalkConversation.updated_at.desc(), SafeTalkConversation.id.desc()).limit(limit).all()
    return [_conversation_response(conversation) for conversation in conversations]


@router.get("/safetalk/conversations/{conversation_id}", response_model=SafeTalkConversationDetail)
def get_safetalk_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    limit: int = 100,
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    conversation = _owned_conversation(db, current_user, conversation_id)
    messages = (
        db.query(SafeTalkBotMessage)
        .filter(
            SafeTalkBotMessage.user_id == current_user.id,
            SafeTalkBotMessage.conversation_id == conversation.id,
        )
        .order_by(SafeTalkBotMessage.created_at.asc(), SafeTalkBotMessage.id.asc())
        .limit(limit)
        .all()
    )
    return SafeTalkConversationDetail(
        conversation=_conversation_response(conversation),
        messages=[_message_response(message) for message in messages],
    )


@router.post("/safetalk/conversations/{conversation_id}/archive", response_model=SafeTalkConversationResponse)
def archive_safetalk_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_student(current_user)
    conversation = _owned_conversation(db, current_user, conversation_id)
    conversation.status = "archived"
    conversation.archived_at = datetime.utcnow()
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conversation)
    return _conversation_response(conversation)


@router.patch("/safetalk/conversations/{conversation_id}/archive", response_model=SafeTalkConversationResponse)
def patch_archive_safetalk_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return archive_safetalk_conversation(conversation_id, current_user=current_user, db=db)


@router.get("/safetalk/resources", response_model=SafeTalkResourceResponse)
def get_safetalk_resources(
    current_user: User = Depends(get_current_user),
):
    _require_student(current_user)
    return load_crisis_resources()

@router.get("/safetalk/history", response_model=BotHistoryResponse)
def get_safetalk_history(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get chat history with SafeTalk bot"""
    _require_student(current_user)
    
    messages = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).order_by(SafeTalkBotMessage.created_at.desc()).offset(offset).limit(limit).all()
    
    total_count = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).count()
    
    return BotHistoryResponse(
        messages=[_message_response(msg) for msg in messages],
        total_count=total_count
    )

@router.get("/safetalk/stats")
def get_safetalk_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics about bot interactions"""
    _require_student(current_user)
    
    all_messages = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).all()
    
    # Count intents
    intent_counts = {}
    for msg in all_messages:
        intent = _history_label_for(msg)
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    return {
        "total_conversations": len(all_messages),
        "intent_distribution": intent_counts,
        "average_confidence": 0.0,
        "last_chat": all_messages[0].created_at if all_messages else None
    }

@router.get("/info")
def get_bot_info():
    """Get SafeTalk bot information for the active deterministic safety layer."""
    return {
        "bot_name": "SafeTalk Bot",
        "version": SAFETY_POLICY_VERSION,
        "capabilities": [
            "Deterministic safety routing",
            "Greeting, positive, neutral, coping and information support",
            "Crisis wording with safe human-support recommendations",
        ],
        "limitations": LIMITATIONS,
        "clinical_use_boundary": "non_diagnostic_support_only",
        "alert_behavior": {
            "alerts_created": False,
            "counselor_contacted": False,
            "emergency_services_contacted": False,
        },
    }
