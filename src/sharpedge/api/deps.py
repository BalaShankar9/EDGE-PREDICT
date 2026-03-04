"""FastAPI dependency injection."""
from sqlalchemy.orm import Session
from sharpedge.db.engine import SessionLocal


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
