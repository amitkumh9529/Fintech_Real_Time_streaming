"""SQLAlchemy models + session for the transaction log / predictions database."""
import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///mlruns/local_predictions.db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    transaction_id = Column(Integer, nullable=True)
    amount = Column(Float)
    actual_class = Column(Integer, nullable=True)  # ground truth, if known (replay mode)
    fraud_probability = Column(Float)
    decision = Column(String(16))
    confidence = Column(Float)
    model_version = Column(String(32))
    rule_override_applied = Column(Boolean, default=False)
    latency_ms = Column(Float)


def init_db():
    Base.metadata.create_all(bind=engine)
    columns = {column["name"] for column in inspect(engine).get_columns("prediction_log")}
    if "transaction_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE prediction_log ADD COLUMN transaction_id INTEGER")
            )


def get_session():
    return SessionLocal()
