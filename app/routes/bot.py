from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.database import get_db
from app.models.database_models import User, SafeTalkBotMessage
from app.routes.auth import get_current_user
from app.ml.counselor_safetalk_bot import CounselorSafeTalkBot
import numpy as np
import logging

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

class BotMessageResponse(BaseModel):
    id: int
    user_message: str
    bot_response: str
    intent: str
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
    response: str  # Just the bot's message
    is_crisis: bool  # Flag if crisis detected
    crisis_resources: List[str] = []  # Crisis help if needed
    timestamp: datetime

class BotHistoryResponse(BaseModel):
    messages: List[BotMessageResponse]
    total_count: int

# ==================== Bot Endpoints ====================

@router.post("/safetalk/chat", response_model=StudentBotResponse)
def chat_with_safetalk(
    message_data: BotMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to SafeTalk bot and get Groq LLM response (clean student-friendly format)"""
    
    user_message = message_data.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Get bot with Groq support
    bot = get_bot()
    bot_response_full = bot.generate_response(user_message, user_id=current_user.id)
    
    # Determine if crisis: Use keyword detection + only severe crisis_level scores
    # Only flag as crisis for actual suicidal ideation/self-harm (crisis_level >= 9)
    crisis_level = bot_response_full.get("crisis_level", 0)
    is_crisis_from_keywords = bot_response_full.get("is_crisis", False)
    is_crisis = is_crisis_from_keywords or crisis_level >= 9  # Only true crisis, not just sadness/stress
    
    # Map response to database model
    bot_message = SafeTalkBotMessage(
        user_id=current_user.id,
        user_message=user_message,
        bot_response=bot_response_full["main_response"],
        intent=bot_response_full.get("intent", "counselor_response"),
        confidence=bot_response_full.get("confidence", 0.95),
        crisis_level=crisis_level,
        response_details={
            "llm_used": bot_response_full.get("llm_used", False),
            "llm_provider": bot_response_full.get("llm_provider"),
            "techniques": bot_response_full.get("techniques_used", []),
            "timestamp": str(bot_response_full.get("timestamp", datetime.now()))
        }
    )
    
    db.add(bot_message)
    db.commit()
    db.refresh(bot_message)
    
    # Return clean student response (NO technical metadata, intent tags, etc)
    return StudentBotResponse(
        response=bot_response_full["main_response"],
        is_crisis=is_crisis,
        crisis_resources=bot_response_full.get("crisis_resources", []) if is_crisis else [],
        timestamp=bot_message.created_at
    )

@router.get("/safetalk/history", response_model=BotHistoryResponse)
def get_safetalk_history(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get chat history with SafeTalk bot"""
    
    messages = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).order_by(SafeTalkBotMessage.created_at.desc()).offset(offset).limit(limit).all()
    
    total_count = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).count()
    
    return BotHistoryResponse(
        messages=[
            BotMessageResponse(
                id=msg.id,
                user_message=msg.user_message,
                bot_response=msg.bot_response,
                intent=msg.intent,
                confidence=msg.confidence,
                created_at=msg.created_at
            )
            for msg in messages
        ],
        total_count=total_count
    )

@router.get("/safetalk/stats")
def get_safetalk_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics about bot interactions"""
    
    all_messages = db.query(SafeTalkBotMessage).filter(
        SafeTalkBotMessage.user_id == current_user.id
    ).all()
    
    # Count intents
    intent_counts = {}
    for msg in all_messages:
        intent = msg.intent or "unknown"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Calculate average confidence
    avg_confidence = np.mean([msg.confidence for msg in all_messages]) if all_messages else 0
    
    return {
        "total_conversations": len(all_messages),
        "intent_distribution": intent_counts,
        "average_confidence": float(avg_confidence),
        "last_chat": all_messages[0].created_at if all_messages else None
    }

@router.get("/info")
def get_bot_info():
    """Get information about SafeTalk bot - LLM-enhanced version"""
    
    llm_bot = get_bot()
    return llm_bot.get_bot_info()
