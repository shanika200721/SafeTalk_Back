from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL - you can use PostgreSQL, MySQL, or SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./test.db"  # Default to SQLite for development
)

# For PostgreSQL (recommended for production)
# DATABASE_URL = "postgresql://user:password@localhost/suicide_prevention_db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
