"""
Pydantic models for API request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Image Models
# ─────────────────────────────────────────────────────────────────────────────

class ImageBase(BaseModel):
    """Base image fields."""
    filename: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ImageCreate(ImageBase):
    """Request to create image record."""
    source: str = "user_upload"


class ImageSummary(ImageBase):
    """Image summary for list views."""
    id: int
    filepath: Optional[str] = None  # Added for cross-reference coordinate extraction
    class_label: Optional[str] = None
    class_confidence: Optional[float] = None
    ood_score: Optional[float] = None
    is_anomaly: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class ImageDetail(ImageSummary):
    """Full image detail with analysis results."""
    filepath: str
    class_probabilities: Optional[Dict[str, float]] = None
    llm_description: Optional[str] = None
    llm_hypothesis: Optional[str] = None
    llm_follow_up: Optional[str] = None
    llm_model: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    annotated_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Models
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """Result of image classification."""
    class_label: str
    confidence: float
    probabilities: Dict[str, float]


class AnomalyResult(BaseModel):
    """Result of anomaly detection."""
    ood_score: float
    is_anomaly: bool
    threshold: float


class SimilarImage(BaseModel):
    """Similar image result."""
    image_id: int
    similarity: float
    filename: str


class SimilarityResult(BaseModel):
    """Result of similarity search."""
    query_id: int
    similar: List[SimilarImage]


class FullAnalysisResult(BaseModel):
    """Result of full analysis pipeline."""
    image_id: int
    classification: ClassificationResult
    anomaly: AnomalyResult
    similar: List[SimilarImage]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation Models
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationResult(BaseModel):
    """LLM annotation result."""
    image_id: int
    description: str
    hypothesis: str
    follow_up: str
    model_used: str


# ─────────────────────────────────────────────────────────────────────────────
# Chat Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Chat message from user."""
    message: str


class ChatResponse(BaseModel):
    """Chat response from agent."""
    reply: str
    tool_calls: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Stats Models
# ─────────────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    """Database statistics."""
    total_images: int
    anomalies: int
    analyzed: int
    annotated: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    ml_model_loaded: bool
    llm_available: bool
