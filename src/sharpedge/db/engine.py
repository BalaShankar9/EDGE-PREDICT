from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sharpedge.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    return SessionLocal()
