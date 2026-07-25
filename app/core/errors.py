from datetime import datetime
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.logging import get_logger


logger = get_logger("app.errors")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def error_payload(
    *,
    code: str,
    message: str,
    status_code: int,
    request_id: str,
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "status_code": status_code,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }
    if message:
        legacy_key = message.split(".", 1)[0]
        payload["error"][legacy_key] = True
    if details is not None:
        payload["error"]["details"] = details
    return payload


def _safe_detail(detail: Any) -> tuple[str, Any | None]:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or "Request failed"), detail
    return str(detail), None


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message, details = _safe_detail(exc.detail)
    status_code = exc.status_code
    code = details.get("code") if isinstance(details, dict) and details.get("code") else f"HTTP_{status_code}"
    return JSONResponse(
        status_code=status_code,
        headers=exc.headers,
        content=error_payload(
            code=str(code),
            message=message,
            status_code=status_code,
            request_id=_request_id(request),
            details=details,
        ),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=_request_id(request),
            details=exc.errors(),
        ),
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error(
        "database_error",
        exc_info=exc,
        extra={"request_id": _request_id(request), "path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload(
            code="DATABASE_ERROR",
            message="A database operation failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=_request_id(request),
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        exc_info=exc,
        extra={"request_id": _request_id(request), "path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=_request_id(request),
        ),
    )
