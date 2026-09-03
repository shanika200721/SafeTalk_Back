import json
import logging
import sys
from datetime import datetime
from typing import Any


SECURITY_LEVEL = 35
AUDIT_LEVEL = 25

logging.addLevelName(SECURITY_LEVEL, "SECURITY")
logging.addLevelName(AUDIT_LEVEL, "AUDIT")


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for API, audit, and security events."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "client_ip",
            "endpoint",
            "error_category",
            "modality",
            "model_name",
            "model_version",
            "runtime_result",
            "failure_code",
            "fusion_status",
            "environment",
            "ffmpeg_available",
            "face_detector_available",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_safetalk_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    setattr(root, "_safetalk_configured", True)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
