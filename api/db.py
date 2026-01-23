"""
SQLite database layer using SQLAlchemy.

Local-only, no cloud dependencies.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from paths import DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# Database Models
# ─────────────────────────────────────────────────────────────────────────────

class ImageRecord(Base):
    """Stored image with metadata and analysis results."""
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    file_type = Column(String(32))  # fits, png, jpeg
    file_size = Column(Integer)  # bytes
    width = Column(Integer)
    height = Column(Integer)
    
    # Duplicate detection
    perceptual_hash = Column(String(128), index=True)  # For duplicate detection
    
    # Analysis results
    class_label = Column(String(64))
    class_confidence = Column(Float)
    class_probabilities = Column(JSON)  # dict of class -> prob
    ood_score = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    embedding_id = Column(Integer)  # index in FAISS
    
    # LLM annotation
    llm_description = Column(Text)
    llm_hypothesis = Column(Text)
    llm_follow_up = Column(Text)
    llm_model = Column(String(64))
    
    # Metadata
    source = Column(String(64))  # user_upload, import, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime)
    annotated_at = Column(DateTime)


class AnalysisLog(Base):
    """Log of analysis runs for debugging."""
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, index=True)
    analysis_type = Column(String(32))  # classify, anomaly, embed, annotate
    duration_ms = Column(Integer)
    success = Column(Boolean)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Database Operations
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables."""
    # Ensure data directory exists
    db_path = DATABASE_URL.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Operations
# ─────────────────────────────────────────────────────────────────────────────

def create_image(db: Session, filename: str, filepath: str, **kwargs) -> ImageRecord:
    """Create a new image record."""
    record = ImageRecord(filename=filename, filepath=filepath, **kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_image(db: Session, image_id: int) -> Optional[ImageRecord]:
    """Get image by ID."""
    return db.query(ImageRecord).filter(ImageRecord.id == image_id).first()


def get_images(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    anomaly_only: bool = False,
) -> List[ImageRecord]:
    """Get list of images."""
    query = db.query(ImageRecord)
    if anomaly_only:
        query = query.filter(ImageRecord.is_anomaly == True)
    return query.order_by(ImageRecord.created_at.desc()).offset(skip).limit(limit).all()


def update_image(db: Session, image_id: int, **kwargs) -> Optional[ImageRecord]:
    """Update image record."""
    record = get_image(db, image_id)
    if record:
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)
        db.commit()
        db.refresh(record)
    return record


def delete_image(db: Session, image_id: int) -> bool:
    """Delete image record."""
    record = get_image(db, image_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False


def log_analysis(
    db: Session,
    image_id: int,
    analysis_type: str,
    duration_ms: int,
    success: bool,
    error_message: str = None,
):
    """Log an analysis run."""
    log = AnalysisLog(
        image_id=image_id,
        analysis_type=analysis_type,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message,
    )
    db.add(log)
    db.commit()


def get_stats(db: Session) -> dict:
    """Get database statistics."""
    total = db.query(ImageRecord).count()
    anomalies = db.query(ImageRecord).filter(ImageRecord.is_anomaly == True).count()
    analyzed = db.query(ImageRecord).filter(ImageRecord.class_label != None).count()
    annotated = db.query(ImageRecord).filter(ImageRecord.llm_description != None).count()
    
    return {
        "total_images": total,
        "anomalies": anomalies,
        "analyzed": analyzed,
        "annotated": annotated,
    }


def get_all_hashes(db: Session) -> List[str]:
    """Get all perceptual hashes from database for duplicate detection."""
    records = db.query(ImageRecord.perceptual_hash).filter(
        ImageRecord.perceptual_hash != None
    ).all()
    return [r[0] for r in records]


def find_by_hash(db: Session, hash_value: str) -> Optional[ImageRecord]:
    """Find image by perceptual hash."""
    return db.query(ImageRecord).filter(
        ImageRecord.perceptual_hash == hash_value
    ).first()


def is_duplicate_hash(db: Session, hash_value: str) -> bool:
    """Check if hash already exists in database."""
    return db.query(ImageRecord).filter(
        ImageRecord.perceptual_hash == hash_value
    ).first() is not None
