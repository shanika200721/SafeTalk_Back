from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes
from app.routes import auth, assessments, checkin, counselor, resources, student, chat, bot
from app.database import engine
from app.models.database_models import Base

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Suicide Prevention Agent API",
    description="AI-powered suicide prevention system for mental health assessment and support",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware - this MUST be added first before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    max_age=86400,
)

# Custom middleware to ensure CORS headers are always present
@app.middleware("http")
async def ensure_cors_headers(request, call_next):
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            }
        )
    
    response = await call_next(request)
    
    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    
    return response

# Include route modules
app.include_router(auth.router)
app.include_router(assessments.router)
app.include_router(checkin.router)
app.include_router(counselor.router)
app.include_router(student.router)
app.include_router(resources.router)
app.include_router(chat.router)
app.include_router(bot.router)

# ==================== Root & Health Endpoints ====================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Suicide Prevention AI Agent API",
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Suicide Prevention API"
    }

@app.get("/api/info")
def api_info():
    """Get API information"""
    return {
        "name": "Suicide Prevention Agent API",
        "version": "1.0.0",
        "description": "AI-powered suicide prevention system for mental health assessment and support",
        "endpoints": {
            "auth": "/api/auth/*",
            "assessments": "/api/assessments/*",
            "daily_checkin": "/api/checkin/*",
            "student": "/api/student/*",
            "counselor": "/api/counselor/*",
            "chat": "/api/chat/*",
            "resources": "/api/resources/*"
        },
        "documentation": "/api/docs"
    }

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    # Log the actual error
    print(f"ERROR: {type(exc).__name__}: {str(exc)}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "exception": str(exc)  # Include error details for debugging
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )