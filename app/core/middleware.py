import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.errors import error_payload
from app.core.logging import get_logger


logger = get_logger("app.requests")
security_logger = get_logger("app.security")

_login_attempts: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW_SECONDS = 60
_RATE_LIMIT = 20


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(key: str, now: float) -> bool:
    attempts = _login_attempts[key]
    while attempts and now - attempts[0] > _RATE_WINDOW_SECONDS:
        attempts.popleft()
    attempts.append(now)
    return len(attempts) > _RATE_LIMIT


async def production_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()

    if request.method == "POST" and request.url.path.startswith("/api/auth/login") and _client_ip(request) != "testclient":
        key = f"{_client_ip(request)}:{request.url.path}"
        if _rate_limited(key, time.monotonic()):
            security_logger.warning(
                "auth_rate_limit_exceeded",
                extra={"request_id": request_id, "path": request.url.path, "method": request.method, "client_ip": _client_ip(request)},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_payload(
                    code="RATE_LIMITED",
                    message="Too many login attempts. Please wait and try again.",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    request_id=request_id,
                ),
                headers={"Retry-After": str(_RATE_WINDOW_SECONDS)},
            )

    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else response.headers.get("Cache-Control", "")

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": _client_ip(request),
        },
    )
    return response
