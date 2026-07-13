from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from datetime import datetime
from typing import List, Optional
from app.database import get_db
from app.models.database_models import User, ChatMessage
from app.routes.auth import get_current_user
from pydantic import BaseModel
import os
import shutil

router = APIRouter(prefix="/api/chat", tags=["Chat"])

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
    is_read: bool
    created_at: datetime
    sender_username: str
    
    class Config:
        from_attributes = True

class ChatConversationResponse(BaseModel):
    user_id: int
    username: str
    full_name: str
    last_message: Optional[str]
    last_message_time: Optional[datetime]
    unread_count: int

# ==================== Chat Endpoints ====================

@router.post("/send", response_model=ChatMessageResponse)
def send_message(
    message_data: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a chat message to another user"""
    
    # Verify receiver exists
    receiver = db.query(User).filter(User.id == message_data.receiver_id).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Receiver not found"
        )
    
    # Create message
    chat_message = ChatMessage(
        sender_id=current_user.id,
        receiver_id=message_data.receiver_id,
        message=message_data.message,
        message_type=message_data.message_type
    )
    
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    
    return ChatMessageResponse(
        id=chat_message.id,
        sender_id=chat_message.sender_id,
        receiver_id=chat_message.receiver_id,
        message=chat_message.message,
        message_type=chat_message.message_type,
        is_read=chat_message.is_read,
        created_at=chat_message.created_at,
        sender_username=current_user.username
    )

@router.get("/conversations", response_model=List[ChatConversationResponse])
def get_conversations(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all conversation partners with last message preview"""
    
    # Get all users this person has chatted with
    conversation_partners = db.query(
        ChatMessage.sender_id, ChatMessage.receiver_id
    ).filter(
        or_(
            ChatMessage.sender_id == current_user.id,
            ChatMessage.receiver_id == current_user.id
        )
    ).all()
    
    partner_ids = set()
    for sender_id, receiver_id in conversation_partners:
        if sender_id != current_user.id:
            partner_ids.add(sender_id)
        if receiver_id != current_user.id:
            partner_ids.add(receiver_id)
    
    conversations = []
    for partner_id in partner_ids:
        # Get last message
        last_msg = db.query(ChatMessage).filter(
            or_(
                and_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == partner_id),
                and_(ChatMessage.sender_id == partner_id, ChatMessage.receiver_id == current_user.id)
            )
        ).order_by(ChatMessage.created_at.desc()).first()
        
        # Count unread messages from this partner
        unread_count = db.query(ChatMessage).filter(
            ChatMessage.sender_id == partner_id,
            ChatMessage.receiver_id == current_user.id,
            ChatMessage.is_read == False
        ).count()
        
        partner = db.query(User).filter(User.id == partner_id).first()
        if partner:
            conversations.append(ChatConversationResponse(
                user_id=partner.id,
                username=partner.username,
                full_name=partner.full_name or partner.username,
                last_message=last_msg.message[:50] if last_msg else None,
                last_message_time=last_msg.created_at if last_msg else None,
                unread_count=unread_count
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
    """Get all messages between current user and another user"""
    
    # Verify the other user exists
    other_user = db.query(User).filter(User.id == user_id).first()
    if not other_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get messages (both sent and received)
    messages = db.query(ChatMessage).filter(
        or_(
            and_(ChatMessage.sender_id == current_user.id, ChatMessage.receiver_id == user_id),
            and_(ChatMessage.sender_id == user_id, ChatMessage.receiver_id == current_user.id)
        )
    ).order_by(ChatMessage.created_at.desc()).offset(offset).limit(limit).all()
    
    # Mark messages from the other user as read
    db.query(ChatMessage).filter(
        ChatMessage.sender_id == user_id,
        ChatMessage.receiver_id == current_user.id,
        ChatMessage.is_read == False
    ).update({
        ChatMessage.is_read: True,
        ChatMessage.read_at: datetime.utcnow()
    })
    db.commit()
    
    # Reverse to show chronological order
    messages.reverse()
    
    return [
        ChatMessageResponse(
            id=msg.id,
            sender_id=msg.sender_id,
            receiver_id=msg.receiver_id,
            message=msg.message,
            message_type=msg.message_type,
            is_read=msg.is_read,
            created_at=msg.created_at,
            sender_username=msg.sender.username
        )
        for msg in messages
    ]

@router.get("/unread-count")
def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get total unread message count"""
    
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
    db.commit()
    
    return {"status": "success", "message_id": message_id}

@router.get("/counselors")
def get_available_counselors(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of available counselors for chat"""
    from app.models.database_models import UserRole
    
    counselors = db.query(User).filter(
        User.role.in_([UserRole.COUNSELOR, UserRole.PSYCHIATRIST]),
        User.is_active == True,
        User.id != current_user.id
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
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a voice message (audio file) to another user"""
    
    # Verify receiver exists
    receiver = db.query(User).filter(User.id == receiver_id).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Receiver not found"
        )
    
    # Create audio directory if not exists
    audio_dir = "uploaded_audio"
    os.makedirs(audio_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.utcnow().timestamp()
    filename = f"voice_{current_user.id}_{receiver_id}_{timestamp}.wav"
    file_path = os.path.join(audio_dir, filename)
    
    # Save audio file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save audio file: {str(e)}"
        )
    
    # Create message record
    chat_message = ChatMessage(
        sender_id=current_user.id,
        receiver_id=receiver_id,
        message=file_path,  # Store file path in message field
        message_type='voice'
    )
    
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    
    return ChatMessageResponse(
        id=chat_message.id,
        sender_id=chat_message.sender_id,
        receiver_id=chat_message.receiver_id,
        message=chat_message.message,
        message_type=chat_message.message_type,
        is_read=chat_message.is_read,
        created_at=chat_message.created_at,
        sender_username=current_user.username
    )

@router.get("/audio/{filename}")
def get_audio(
    filename: str,
    db: Session = Depends(get_db)
):
    """Stream audio file for voice message - no auth required for HTML5 audio element"""
    import mimetypes
    from fastapi.responses import FileResponse
    
    # Validate filename to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid filename"
        )
    
    file_path = f"uploaded_audio/{filename}"
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found"
        )
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "audio/wav"
    
    # Return file with proper headers for HTML5 audio element
    return FileResponse(
        file_path,
        media_type=mime_type,
        filename=filename,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": "inline"
        }
    )
