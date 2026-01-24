"""
FastAPI application for AstroLens.

Local-first API for image analysis with ML and LLM.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .db import init_db, get_db, create_image, get_image, get_images, update_image, delete_image, get_stats, log_analysis
from .models import (
    ImageSummary, ImageDetail, ClassificationResult, AnomalyResult,
    SimilarImage, SimilarityResult, FullAnalysisResult, AnnotationResult,
    ChatMessage, ChatResponse, StatsResponse, HealthResponse,
)
from paths import IMAGES_DIR, WEIGHTS_PATH

# Lazy imports to avoid loading heavy modules at startup
_classifier = None
_ood_detector = None
_embeddings = None
_annotator = None
_agent = None

logger = logging.getLogger(__name__)

# Configuration
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "10.0"))
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")


def get_classifier():
    """Lazy-load classifier."""
    global _classifier
    if _classifier is None:
        from inference.classifier import AstroClassifier
        # Check if fine-tuned weights exist (folder with model.safetensors or config.json)
        weights_valid = (
            Path(WEIGHTS_PATH).exists() and 
            Path(WEIGHTS_PATH).is_dir() and
            (Path(WEIGHTS_PATH) / "config.json").exists()
        )
        _classifier = AstroClassifier(weights_path=WEIGHTS_PATH if weights_valid else None)
    return _classifier


def get_ood_detector():
    """Lazy-load OOD detector."""
    global _ood_detector
    if _ood_detector is None:
        from inference.ood import OODDetector
        _ood_detector = OODDetector(threshold=OOD_THRESHOLD)
    return _ood_detector


def get_embeddings():
    """Lazy-load embeddings store."""
    global _embeddings
    if _embeddings is None:
        from inference.embeddings import EmbeddingStore
        _embeddings = EmbeddingStore()
    return _embeddings


def get_annotator():
    """Lazy-load LLM annotator."""
    global _annotator
    if _annotator is None:
        from annotator.chain import ImageAnnotator
        _annotator = ImageAnnotator(provider=LLM_PROVIDER)
    return _annotator


def get_agent():
    """Lazy-load LangChain agent."""
    global _agent
    if _agent is None:
        from agent.agent import AstroLensAgent
        _agent = AstroLensAgent(provider=LLM_PROVIDER)
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Initialize database
    init_db()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("AstroLens API started")
    yield
    logger.info("AstroLens API shutting down")


app = FastAPI(
    title="AstroLens API",
    description="AI-powered image analysis with ML classification and LLM insights",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health & Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ml_loaded = _classifier is not None
    llm_available = LLM_PROVIDER != "none" and (
        os.environ.get("OPENAI_API_KEY") or LLM_PROVIDER == "ollama"
    )
    return HealthResponse(
        status="ok",
        version="0.1.0",
        ml_model_loaded=ml_loaded,
        llm_available=llm_available,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """Get database statistics."""
    return get_stats(db)


# ─────────────────────────────────────────────────────────────────────────────
# Image CRUD
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/images", response_model=List[ImageSummary])
async def list_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10000),  # Increased for batch operations
    anomaly_only: bool = Query(False),
    db: Session = Depends(get_db),
):
    """List all images."""
    records = get_images(db, skip=skip, limit=limit, anomaly_only=anomaly_only)
    return [ImageSummary.model_validate(r) for r in records]


@app.post("/images", response_model=ImageSummary)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a new image."""
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/fits", "application/fits"}
    if file.content_type not in allowed_types and not file.filename.endswith(".fits"):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    filepath = IMAGES_DIR / safe_name
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Get file info
    file_size = filepath.stat().st_size
    file_type = filepath.suffix.lower().lstrip(".")

    # Try to get dimensions
    width, height = None, None
    try:
        if file_type in ("png", "jpg", "jpeg"):
            from PIL import Image
            with Image.open(filepath) as img:
                width, height = img.size
        elif file_type == "fits":
            from astropy.io import fits
            with fits.open(filepath) as hdu:
                if hdu[0].data is not None:
                    height, width = hdu[0].data.shape[:2]
    except Exception:
        pass

    # Create database record
    record = create_image(
        db,
        filename=file.filename,
        filepath=str(filepath),
        file_type=file_type,
        file_size=file_size,
        width=width,
        height=height,
        source="user_upload",
    )

    return ImageSummary.model_validate(record)


@app.get("/images/{image_id}", response_model=ImageDetail)
async def get_image_detail(image_id: int, db: Session = Depends(get_db)):
    """Get image detail."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")
    return ImageDetail.model_validate(record)


@app.get("/images/{image_id}/file")
async def get_image_file(image_id: int, db: Session = Depends(get_db)):
    """Get actual image file."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")
    if not Path(record.filepath).exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(record.filepath)


@app.delete("/images/{image_id}")
async def remove_image(image_id: int, db: Session = Depends(get_db)):
    """Delete an image."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")
    
    # Delete file
    try:
        Path(record.filepath).unlink(missing_ok=True)
    except Exception:
        pass
    
    # Delete record
    delete_image(db, image_id)
    return {"ok": True}


class ImageUpdate(BaseModel):
    """Request body for updating image analysis results."""
    class_label: Optional[str] = None
    class_confidence: Optional[float] = None
    ood_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    perceptual_hash: Optional[str] = None


@app.patch("/images/{image_id}")
async def update_image_record(
    image_id: int,
    update_data: ImageUpdate,
    db: Session = Depends(get_db),
):
    """Update image analysis results (used by discovery loop)."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")
    
    updates = {}
    if update_data.class_label is not None:
        updates["class_label"] = update_data.class_label
    if update_data.class_confidence is not None:
        updates["class_confidence"] = update_data.class_confidence
    if update_data.ood_score is not None:
        updates["ood_score"] = update_data.ood_score
    if update_data.is_anomaly is not None:
        updates["is_anomaly"] = update_data.is_anomaly
    if update_data.perceptual_hash is not None:
        updates["perceptual_hash"] = update_data.perceptual_hash
    if updates:
        updates["analyzed_at"] = datetime.utcnow()
        update_image(db, image_id, **updates)
    
    return {"ok": True, "updated": list(updates.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analysis/classify/{image_id}", response_model=ClassificationResult)
async def classify_image(image_id: int, db: Session = Depends(get_db)):
    """Classify an image using ML model."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")

    start = time.time()
    try:
        classifier = get_classifier()
        result = classifier.classify(record.filepath)
        
        # Update database
        update_image(
            db, image_id,
            class_label=result.class_label,
            class_confidence=result.confidence,
            class_probabilities=result.probabilities,
            analyzed_at=datetime.utcnow(),
        )
        
        log_analysis(db, image_id, "classify", int((time.time() - start) * 1000), True)
        
        return ClassificationResult(
            class_label=result.class_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
        )
    except Exception as e:
        log_analysis(db, image_id, "classify", int((time.time() - start) * 1000), False, str(e))
        raise HTTPException(500, f"Classification failed: {e}")


@app.post("/analysis/anomaly/{image_id}", response_model=AnomalyResult)
async def detect_anomaly(image_id: int, db: Session = Depends(get_db)):
    """Detect if image is an anomaly (out-of-distribution)."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")

    start = time.time()
    try:
        # Need classification first to get logits
        classifier = get_classifier()
        class_result = classifier.classify(record.filepath)
        
        ood = get_ood_detector()
        result = ood.detect(class_result.logits)
        
        # Update database
        update_image(
            db, image_id,
            ood_score=result.ood_score,
            is_anomaly=result.is_anomaly,
        )
        
        log_analysis(db, image_id, "anomaly", int((time.time() - start) * 1000), True)
        
        return AnomalyResult(
            ood_score=result.ood_score,
            is_anomaly=result.is_anomaly,
            threshold=result.threshold,
        )
    except Exception as e:
        log_analysis(db, image_id, "anomaly", int((time.time() - start) * 1000), False, str(e))
        raise HTTPException(500, f"Anomaly detection failed: {e}")


@app.post("/analysis/similar/{image_id}", response_model=SimilarityResult)
async def find_similar(
    image_id: int,
    k: int = Query(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Find similar images using embedding similarity."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")

    try:
        # Check if FAISS has any embeddings
        store = get_embeddings()
        if store.count() == 0:
            # No embeddings yet - return empty with helpful message
            return SimilarityResult(query_id=image_id, similar=[])
        
        # Get embedding for query image
        classifier = get_classifier()
        class_result = classifier.classify(record.filepath)
        
        # Search FAISS
        similar_ids, similarities = store.search(class_result.embedding, k=k + 1)
        
        # Filter out self and get filenames
        results = []
        for idx, sim in zip(similar_ids, similarities):
            if idx != image_id and idx != -1:
                similar_record = get_image(db, idx)
                if similar_record:
                    results.append(SimilarImage(
                        image_id=idx,
                        similarity=sim,
                        filename=similar_record.filename,
                    ))
        
        return SimilarityResult(query_id=image_id, similar=results[:k])
    except Exception as e:
        logger.error(f"Similarity search failed for image {image_id}: {e}")
        raise HTTPException(500, f"Similarity search failed: {e}")


@app.post("/analysis/full/{image_id}", response_model=FullAnalysisResult)
async def full_analysis(image_id: int, db: Session = Depends(get_db)):
    """Run full analysis pipeline: classify, anomaly, embed, similar."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")

    try:
        classifier = get_classifier()
        ood = get_ood_detector()
        store = get_embeddings()
        
        # Classify
        class_result = classifier.classify(record.filepath)
        
        # Anomaly
        anomaly_result = ood.detect(class_result.logits)
        
        # Store embedding and save
        store.add(image_id, class_result.embedding)
        store.save()  # Persist to disk
        
        # Find similar
        similar_ids, similarities = store.search(class_result.embedding, k=6)
        similar = []
        for idx, sim in zip(similar_ids, similarities):
            if idx != image_id:
                sim_record = get_image(db, idx)
                if sim_record:
                    similar.append(SimilarImage(
                        image_id=idx,
                        similarity=sim,
                        filename=sim_record.filename,
                    ))
        
        # Update database
        update_image(
            db, image_id,
            class_label=class_result.class_label,
            class_confidence=class_result.confidence,
            class_probabilities=class_result.probabilities,
            ood_score=anomaly_result.ood_score,
            is_anomaly=anomaly_result.is_anomaly,
            embedding_id=image_id,
            analyzed_at=datetime.utcnow(),
        )
        
        return FullAnalysisResult(
            image_id=image_id,
            classification=ClassificationResult(
                class_label=class_result.class_label,
                confidence=class_result.confidence,
                probabilities=class_result.probabilities,
            ),
            anomaly=AnomalyResult(
                ood_score=anomaly_result.ood_score,
                is_anomaly=anomaly_result.is_anomaly,
                threshold=anomaly_result.threshold,
            ),
            similar=similar[:5],
        )
    except Exception as e:
        raise HTTPException(500, f"Full analysis failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM Annotation
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/annotate/{image_id}", response_model=AnnotationResult)
async def annotate_image(image_id: int, db: Session = Depends(get_db)):
    """Generate LLM annotation for an image."""
    record = get_image(db, image_id)
    if not record:
        raise HTTPException(404, "Image not found")

    start = time.time()
    try:
        annotator = get_annotator()
        result = annotator.annotate(
            image_path=record.filepath,
            class_label=record.class_label or "unknown",
            confidence=record.class_confidence or 0.0,
            ood_score=record.ood_score or 0.0,
        )
        
        # Update database
        update_image(
            db, image_id,
            llm_description=result.description,
            llm_hypothesis=result.hypothesis,
            llm_follow_up=result.follow_up,
            llm_model=result.model_used,
            annotated_at=datetime.utcnow(),
        )
        
        log_analysis(db, image_id, "annotate", int((time.time() - start) * 1000), True)
        
        return AnnotationResult(
            image_id=image_id,
            description=result.description,
            hypothesis=result.hypothesis,
            follow_up=result.follow_up,
            model_used=result.model_used,
        )
    except Exception as e:
        log_analysis(db, image_id, "annotate", int((time.time() - start) * 1000), False, str(e))
        raise HTTPException(500, f"Annotation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Chat Agent
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, db: Session = Depends(get_db)):
    """Chat with the LangChain agent."""
    try:
        agent = get_agent()
        result = agent.chat(message.message, db=db)
        return ChatResponse(
            reply=result.get("output", ""),
            tool_calls=result.get("tool_calls", []),
        )
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Candidates (Anomalies)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/candidates", response_model=List[ImageSummary])
async def list_candidates(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List anomaly candidates (is_anomaly=True), sorted by OOD score."""
    records = get_images(db, skip=skip, limit=limit, anomaly_only=True)
    return [ImageSummary.model_validate(r) for r in records]
