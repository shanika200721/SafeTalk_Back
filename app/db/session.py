from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings


def _engine_kwargs(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        return kwargs

    return {
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,
    }


engine = create_engine(settings.DATABASE_URL, **_engine_kwargs(settings.DATABASE_URL))


if settings.is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_db():
    """FastAPI dependency that rolls back failed transactions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
