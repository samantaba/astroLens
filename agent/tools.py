"""
LangChain tools for the AstroLens agent.

Each tool provides a specific capability that the agent can invoke.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.tools import tool
from paths import WEIGHTS_PATH

logger = logging.getLogger(__name__)

# Global reference to database session (set by agent)
_db = None


def set_db(db):
    """Set the database session for tools to use."""
    global _db
    _db = db


@tool
def list_images(limit: int = 10, anomaly_only: bool = False, unanalyzed_only: bool = False) -> str:
    """
    List images in the database.
    
    Args:
        limit: Maximum number of images to return
        anomaly_only: If True, only return flagged anomalies
        unanalyzed_only: If True, only return images without analysis
    
    Returns:
        Formatted list of images with IDs and basic info
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_images, ImageRecord
    
    if unanalyzed_only:
        # Query specifically for unanalyzed images
        records = _db.query(ImageRecord).filter(ImageRecord.class_label == None).limit(limit).all()
    else:
        records = get_images(_db, limit=limit, anomaly_only=anomaly_only)
    
    if not records:
        if unanalyzed_only:
            return "No unanalyzed images found - all images have been processed!"
        return "No images found."
    
    lines = []
    for r in records:
        status = "⚠️ ANOMALY" if r.is_anomaly else ""
        label = r.class_label or "[unanalyzed]"
        lines.append(f"• ID {r.id}: {r.filename} - {label} {status}")
    
    return "\n".join(lines)


@tool
def get_image_info(image_id: int) -> str:
    """
    Get detailed information about a specific image.
    
    Args:
        image_id: The ID of the image
    
    Returns:
        Detailed information including classification and annotations
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_image
    
    record = get_image(_db, image_id)
    if not record:
        return f"Image {image_id} not found."
    
    lines = [
        f"**Image {record.id}:** {record.filename}",
        f"- Type: {record.file_type}",
        f"- Size: {record.width}x{record.height}" if record.width else "- Size: unknown",
    ]
    
    if record.class_label:
        lines.append(f"- Classification: {record.class_label} ({record.class_confidence:.1%})")
    if record.ood_score is not None:
        lines.append(f"- Anomaly Score: {record.ood_score:.2f} ({'ANOMALY' if record.is_anomaly else 'normal'})")
    if record.llm_description:
        lines.append(f"- Description: {record.llm_description}")
    
    return "\n".join(lines)


@tool
def analyze_image(image_id: int) -> str:
    """
    Run full ML analysis on an image (classification + anomaly detection).
    
    Args:
        image_id: The ID of the image to analyze
    
    Returns:
        Analysis results including classification and anomaly status
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_image, update_image
    from inference.classifier import AstroClassifier
    from inference.ood import OODDetector
    from datetime import datetime
    import os
    
    record = get_image(_db, image_id)
    if not record:
        return f"Image {image_id} not found."
    
    try:
        # Lazy load classifier with fine-tuned weights
        weights_path = WEIGHTS_PATH
        weights_valid = os.path.isdir(weights_path) and os.path.exists(os.path.join(weights_path, "config.json"))
        classifier = AstroClassifier(weights_path=weights_path if weights_valid else None)
        ood = OODDetector(threshold=float(os.environ.get("OOD_THRESHOLD", "10.0")))
        
        # Classify
        result = classifier.classify(record.filepath)
        anomaly = ood.detect(result.logits)
        
        # Update database
        update_image(
            _db, image_id,
            class_label=result.class_label,
            class_confidence=result.confidence,
            class_probabilities=result.probabilities,
            ood_score=anomaly.ood_score,
            is_anomaly=anomaly.is_anomaly,
            analyzed_at=datetime.utcnow(),
        )
        
        status = "⚠️ FLAGGED AS ANOMALY" if anomaly.is_anomaly else "Normal"
        return (
            f"Analysis complete for image {image_id}:\n"
            f"- Classification: {result.class_label} ({result.confidence:.1%})\n"
            f"- Anomaly Score: {anomaly.ood_score:.2f}\n"
            f"- Status: {status}"
        )
    except Exception as e:
        return f"Analysis failed: {e}"


@tool
def annotate_image(image_id: int) -> str:
    """
    Generate LLM annotation for an image (description and hypothesis).
    
    Args:
        image_id: The ID of the image to annotate
    
    Returns:
        LLM-generated description and hypothesis
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_image, update_image
    from annotator.chain import ImageAnnotator
    from datetime import datetime
    import os
    
    record = get_image(_db, image_id)
    if not record:
        return f"Image {image_id} not found."
    
    try:
        provider = os.environ.get("LLM_PROVIDER", "openai")
        annotator = ImageAnnotator(provider=provider)
        
        result = annotator.annotate(
            image_path=record.filepath,
            class_label=record.class_label or "unknown",
            confidence=record.class_confidence or 0.0,
            ood_score=record.ood_score or 0.0,
        )
        
        # Update database
        update_image(
            _db, image_id,
            llm_description=result.description,
            llm_hypothesis=result.hypothesis,
            llm_follow_up=result.follow_up,
            llm_model=result.model_used,
            annotated_at=datetime.utcnow(),
        )
        
        return (
            f"Annotation for image {image_id}:\n"
            f"**Description:** {result.description}\n"
            f"**Hypothesis:** {result.hypothesis}\n"
            f"**Follow-up:** {result.follow_up}"
        )
    except Exception as e:
        return f"Annotation failed: {e}"


@tool
def find_similar_images(image_id: int, count: int = 5) -> str:
    """
    Find images similar to the given image.
    
    Args:
        image_id: The ID of the query image
        count: Number of similar images to find
    
    Returns:
        List of similar images with similarity scores
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_image
    from inference.classifier import AstroClassifier
    from inference.embeddings import EmbeddingStore
    import os
    
    record = get_image(_db, image_id)
    if not record:
        return f"Image {image_id} not found."
    
    try:
        weights_path = WEIGHTS_PATH
        weights_valid = os.path.isdir(weights_path) and os.path.exists(os.path.join(weights_path, "config.json"))
        classifier = AstroClassifier(weights_path=weights_path if weights_valid else None)
        store = EmbeddingStore()
        
        # Get embedding
        result = classifier.classify(record.filepath)
        
        # Search
        ids, sims = store.search(result.embedding, k=count + 1)
        
        lines = [f"Images similar to {record.filename}:"]
        for img_id, sim in zip(ids, sims):
            if img_id != image_id:
                sim_record = get_image(_db, img_id)
                if sim_record:
                    lines.append(f"• ID {img_id}: {sim_record.filename} (similarity: {sim:.2%})")
        
        return "\n".join(lines) if len(lines) > 1 else "No similar images found."
    except Exception as e:
        return f"Similarity search failed: {e}"


@tool
def get_statistics() -> str:
    """
    Get statistics about the image collection.
    
    Returns:
        Summary statistics including counts and anomalies
    """
    if _db is None:
        return "Database not available"
    
    from api.db import get_stats
    
    stats = get_stats(_db)
    
    return (
        f"**Collection Statistics:**\n"
        f"- Total images: {stats['total_images']}\n"
        f"- Analyzed: {stats['analyzed']}\n"
        f"- Anomalies: {stats['anomalies']}\n"
        f"- Annotated: {stats['annotated']}"
    )


# Export all tools
ALL_TOOLS = [
    list_images,
    get_image_info,
    analyze_image,
    annotate_image,
    find_similar_images,
    get_statistics,
]

