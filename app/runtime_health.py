from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from app.core.config import settings
from app.database import engine
from app.ml.runtime.face_detector import detector_status
from app.ml.runtime.speech_preprocessor import ffmpeg_version
from app.services.model_registry import APPROVED_RUNTIME_ARTIFACTS, _resolve_repo_path


REQUIRED_ENV_BY_ENVIRONMENT = {
    "development": ["DATABASE_URL", "SECRET_KEY", "MODEL_ROOT", "UPLOAD_ROOT"],
    "test": ["DATABASE_URL", "SECRET_KEY", "MODEL_ROOT", "UPLOAD_ROOT"],
    "staging": ["DATABASE_URL", "SECRET_KEY", "CORS_ORIGINS", "MODEL_ROOT", "UPLOAD_ROOT", "FFMPEG_BINARY"],
    "production": ["DATABASE_URL", "SECRET_KEY", "CORS_ORIGINS", "MODEL_ROOT", "UPLOAD_ROOT", "FFMPEG_BINARY"],
}


def _component(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def check_database() -> dict[str, Any]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = set(inspect(connection).get_table_names())
        return _component("ok", dialect=engine.dialect.name, required_tables_present="model_registry" in tables)
    except Exception as exc:
        return _component("failed", error_category=exc.__class__.__name__)


def check_migrations() -> dict[str, Any]:
    try:
        config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        script = ScriptDirectory.from_config(config)
        heads = sorted(script.get_heads())
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            if "alembic_version" not in tables:
                return _component("failed", head=heads[-1] if heads else None, current=None, compatible=False)
            rows = connection.execute(text("SELECT version_num FROM alembic_version")).fetchall()
        current = sorted(str(row[0]) for row in rows)
        compatible = bool(current) and set(current).issubset(set(heads)) and set(heads).issubset(set(current))
        return _component(
            "ok" if compatible else "failed",
            head=heads[-1] if len(heads) == 1 else heads,
            current=current[0] if len(current) == 1 else current,
            compatible=compatible,
        )
    except Exception as exc:
        return _component("failed", error_category=exc.__class__.__name__, compatible=False)


def check_model_artifacts() -> dict[str, Any]:
    missing = []
    present = 0
    for artifact in APPROVED_RUNTIME_ARTIFACTS:
        path = _resolve_repo_path(artifact)
        if path.exists() and path.is_file():
            present += 1
        else:
            missing.append(Path(artifact).name)
    return _component(
        "ok" if not missing else "failed",
        approved_artifact_count=len(APPROVED_RUNTIME_ARTIFACTS),
        present_count=present,
        missing_count=len(missing),
    )


def check_model_registry() -> dict[str, Any]:
    try:
        with engine.connect() as connection:
            total = connection.execute(text("SELECT COUNT(*) FROM model_registry")).scalar_one()
            active_rows = connection.execute(
                text("SELECT modality, COUNT(*) FROM model_registry WHERE is_active = :active GROUP BY modality"),
                {"active": True},
            ).fetchall()
        active = {str(row[0]): int(row[1]) for row in active_rows}
        return _component("ok", registered_count=int(total), active_modalities=active)
    except Exception as exc:
        return _component("failed", error_category=exc.__class__.__name__)


def check_ffmpeg() -> dict[str, Any]:
    version = ffmpeg_version()
    return _component("ok" if version.get("available") else "failed", available=bool(version.get("available")), version=version.get("version"))


def check_face_detector() -> dict[str, Any]:
    status = detector_status()
    required = bool(settings.FACE_DETECTOR_REQUIRED)
    available = bool(status.get("available"))
    if available:
        state = "ok"
    elif required:
        state = "failed"
    else:
        state = "degraded_optional_dependency"
    return _component(
        state,
        available=available,
        required=required,
        detector=status.get("detector"),
        failure_code=status.get("failure_code"),
        version=status.get("version"),
    )


def check_environment() -> dict[str, Any]:
    environment = settings.ENVIRONMENT.lower()
    required = REQUIRED_ENV_BY_ENVIRONMENT.get(environment, REQUIRED_ENV_BY_ENVIRONMENT["production"])
    missing = []
    for name in required:
        value = getattr(settings, name, None)
        if value is None or value == "" or (isinstance(value, (list, tuple, set, dict)) and not value):
            missing.append(name)
    unsafe = []
    if environment in {"staging", "production"}:
        if settings.SECRET_KEY == "change-me-in-your-local-env":
            unsafe.append("SECRET_KEY_DEFAULT")
        if "*" in settings.CORS_ORIGINS:
            unsafe.append("CORS_WILDCARD")
        if settings.is_sqlite:
            unsafe.append("SQLITE_DATABASE")
    return _component("ok" if not missing and not unsafe else "failed", environment=environment, missing=missing, unsafe=unsafe)


def operational_checks() -> dict[str, Any]:
    checks = {
        "environment": check_environment(),
        "database": check_database(),
        "migrations": check_migrations(),
        "model_artifacts": check_model_artifacts(),
        "model_registry": check_model_registry(),
        "ffmpeg": check_ffmpeg(),
        "face_detector": check_face_detector(),
    }
    ready = all(
        component["status"] == "ok" or component["status"] == "degraded_optional_dependency"
        for component in checks.values()
    )
    if checks["ffmpeg"]["status"] != "ok":
        ready = False
    return {
        "status": "ready" if ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "Suicide Prevention API",
        "checks": checks,
    }


def validate_startup_requirements() -> dict[str, Any]:
    checks = operational_checks()
    environment = settings.ENVIRONMENT.lower()
    if environment in {"staging", "production"} and checks["status"] != "ready":
        failed = [name for name, check in checks["checks"].items() if check["status"] == "failed"]
        raise RuntimeError(f"Startup validation failed for {environment}: {', '.join(failed)}")
    return checks
