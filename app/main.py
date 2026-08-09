from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from datetime import datetime
import os
from sqlalchemy.exc import SQLAlchemyError

# Import routes
from app.routes import auth, assessments, checkin, counselor, resources, student, chat, bot, consents, modalities, models, fusion, support, admin, wellness, profile_assessment
from app.database import engine
from app.core.config import settings
from app.core.errors import (
    http_exception_handler,
    sqlalchemy_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import configure_logging, get_logger
from app.core.middleware import production_middleware
from app.models.database_models import Base
from app.runtime_health import operational_checks, validate_startup_requirements


configure_logging()
logger = get_logger("app.main")
settings.validate_production_safety()

DEV_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]


def resolve_cors_origins():
    origins = list(settings.CORS_ORIGINS)
    if settings.ENVIRONMENT.lower() in {"development", "test"} and "*" not in origins:
        origins.extend(DEV_CORS_ORIGINS)
    return list(dict.fromkeys(origins))


cors_origins = resolve_cors_origins()

if settings.is_sqlite and settings.ENVIRONMENT.lower() in {"development", "test"}:
    # Temporary compatibility fallback while Alembic migrations are adopted.
    Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Suicide Prevention Agent API",
    description="AI-powered suicide prevention system for mental health assessment and support",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


@app.on_event("startup")
def startup_validation():
    checks = validate_startup_requirements()
    logger.info(
        "startup_validation_completed",
        extra={
            "runtime_result": checks["status"],
            "environment": settings.ENVIRONMENT,
            "ffmpeg_available": checks["checks"]["ffmpeg"]["status"] == "ok",
            "face_detector_available": checks["checks"]["face_detector"].get("available"),
        },
    )

# Add CORS middleware - this MUST be added first before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    max_age=86400,
)

app.middleware("http")(production_middleware)

# Custom middleware to ensure CORS headers are always present
@app.middleware("http")
async def ensure_cors_headers(request, call_next):
    request_origin = request.headers.get("origin")
    if "*" in cors_origins:
        allow_origin = "*"
    elif request_origin in cors_origins:
        allow_origin = request_origin
    else:
        allow_origin = cors_origins[0] if cors_origins else ""

    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            }
        )
    
    response = await call_next(request)
    
    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = allow_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    
    return response

# Include route modules
app.include_router(auth.router)
app.include_router(assessments.router)
app.include_router(checkin.router)
app.include_router(counselor.router)
app.include_router(wellness.router)
app.include_router(student.router)
app.include_router(resources.router)
app.include_router(chat.router)
app.include_router(bot.router)
app.include_router(consents.router)
app.include_router(modalities.router)
app.include_router(models.router)
app.include_router(fusion.router)
app.include_router(support.router)
app.include_router(admin.router)
app.include_router(profile_assessment.router)

# ==================== Root & Health Endpoints ====================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Suicide Prevention AI Agent API",
        "status": "running",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "documentation": "/api/docs",
    }

@app.get("/health")
def health_check():
    """Liveness endpoint with safe operational metadata."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Suicide Prevention API",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/api/health")
def api_health_check():
    return health_check()


@app.get("/ready")
def readiness_check():
    payload = operational_checks()
    if payload["status"] != "ready":
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/ready")
def api_readiness_check():
    return readiness_check()

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
            "resources": "/api/resources/*",
            "modalities": "/api/modalities/*",
            "models": "/api/models/*",
            "fusion": "/api/fusion/*",
        },
        "documentation": "/api/docs"
    }

# ==================== Error Handlers ====================

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=settings.ENVIRONMENT == "development"
    )
