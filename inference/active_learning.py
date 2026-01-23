"""
Active Learning & Multi-Source Diversity Module

Provides intelligent source selection and uncertainty-based sample flagging.

Active Learning:
- Flags images where model is uncertain (confidence 40-60%)
- These samples are most valuable for human labeling
- Adds them to a review queue for manual annotation

Multi-Source Diversity:
- Tracks OOD scores per source
- Prioritizes sources that yield higher OOD (more interesting) images
- Dynamically adjusts download ratios
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class SourceStats:
    """Statistics for a single data source."""
    name: str
    total_downloaded: int = 0
    total_analyzed: int = 0
    anomalies_found: int = 0
    near_misses: int = 0
    avg_ood_score: float = 0.0
    max_ood_score: float = 0.0
    avg_confidence: float = 0.0
    diversity_score: float = 1.0  # Higher = more interesting, should download more
    last_updated: str = ""
    
    def update(self, ood_score: float, confidence: float, is_anomaly: bool, is_near_miss: bool):
        """Update stats with a new analyzed image."""
        self.total_analyzed += 1
        
        # Running average
        n = self.total_analyzed
        self.avg_ood_score = ((n - 1) * self.avg_ood_score + ood_score) / n
        self.avg_confidence = ((n - 1) * self.avg_confidence + confidence) / n
        
        if ood_score > self.max_ood_score:
            self.max_ood_score = ood_score
        
        if is_anomaly:
            self.anomalies_found += 1
        
        if is_near_miss:
            self.near_misses += 1
        
        # Diversity score: higher OOD + more anomalies = more interesting
        self.diversity_score = (
            self.avg_ood_score * 0.3 +
            (self.anomalies_found / max(self.total_analyzed, 1)) * 5.0 +
            (self.near_misses / max(self.total_analyzed, 1)) * 2.0 +
            1.0  # Base score
        )
        
        self.last_updated = datetime.now().isoformat()


@dataclass
class UncertainSample:
    """A sample the model is uncertain about - candidate for human review."""
    image_id: int
    image_path: str
    class_label: str
    confidence: float
    ood_score: float
    uncertainty_type: str  # "low_confidence", "borderline_ood", "class_ambiguity"
    top_classes: List[Tuple[str, float]] = field(default_factory=list)
    flagged_at: str = ""
    reviewed: bool = False
    human_label: Optional[str] = None


class SourceDiversityManager:
    """
    Manages multi-source diversity for optimal anomaly discovery.
    
    Tracks which sources yield more interesting results and
    dynamically adjusts download ratios.
    """
    
    def __init__(self, state_file: Path = None):
        self.sources: Dict[str, SourceStats] = {}
        self.state_file = state_file
        
        # Default sources
        default_sources = ["sdss", "ztf", "apod", "eso"]
        for name in default_sources:
            self.sources[name] = SourceStats(name=name)
        
        if state_file and state_file.exists():
            self._load_state()
    
    def _load_state(self):
        """Load saved state."""
        try:
            with open(self.state_file) as f:
                data = json.load(f)
                for name, stats in data.items():
                    self.sources[name] = SourceStats(**stats)
        except Exception as e:
            logger.warning(f"Failed to load source stats: {e}")
    
    def save_state(self):
        """Save current state."""
        if self.state_file:
            with open(self.state_file, "w") as f:
                # Convert numpy types to Python types for JSON serialization
                data = convert_numpy_types({name: asdict(s) for name, s in self.sources.items()})
                json.dump(data, f, indent=2)
    
    def record_download(self, source: str, count: int):
        """Record downloaded images from a source."""
        if source not in self.sources:
            self.sources[source] = SourceStats(name=source)
        self.sources[source].total_downloaded += count
    
    def record_analysis(
        self, 
        source: str, 
        ood_score: float, 
        confidence: float,
        is_anomaly: bool,
        is_near_miss: bool
    ):
        """Record analysis result for a source."""
        if source not in self.sources:
            self.sources[source] = SourceStats(name=source)
        
        self.sources[source].update(ood_score, confidence, is_anomaly, is_near_miss)
    
    def get_download_ratios(self, total_images: int = 30) -> Dict[str, int]:
        """
        Get optimal download counts per source based on diversity scores.
        
        Sources with higher diversity scores get more downloads.
        """
        if not self.sources:
            # Equal distribution
            return {s: total_images // 3 for s in ["sdss", "ztf", "apod"]}
        
        # Calculate ratios based on diversity scores
        total_score = sum(s.diversity_score for s in self.sources.values())
        
        if total_score == 0:
            total_score = len(self.sources)
        
        ratios = {}
        remaining = total_images
        
        for name, stats in sorted(self.sources.items(), key=lambda x: x[1].diversity_score, reverse=True):
            if remaining <= 0:
                break
            
            # Allocate proportionally, minimum 5 images
            share = max(5, int((stats.diversity_score / total_score) * total_images))
            share = min(share, remaining)
            ratios[name] = share
            remaining -= share
        
        # Distribute any remaining
        if remaining > 0:
            for name in ratios:
                ratios[name] += remaining // len(ratios)
                break
        
        return ratios
    
    def get_summary(self) -> str:
        """Get human-readable summary of source performance."""
        lines = ["📊 Source Diversity Scores:"]
        
        for name, stats in sorted(self.sources.items(), key=lambda x: x[1].diversity_score, reverse=True):
            lines.append(
                f"  {name}: score={stats.diversity_score:.2f}, "
                f"anomalies={stats.anomalies_found}, near_misses={stats.near_misses}, "
                f"avg_ood={stats.avg_ood_score:.2f}"
            )
        
        return "\n".join(lines)


class ActiveLearningManager:
    """
    Manages active learning - flagging uncertain samples for human review.
    
    Identifies images where the model is uncertain:
    - Low confidence predictions (40-60%)
    - Borderline OOD scores (near threshold)
    - Ambiguous class predictions (top 2 classes are close)
    """
    
    def __init__(
        self, 
        review_queue_file: Path = None,
        confidence_low: float = 0.4,
        confidence_high: float = 0.6,
        ood_margin: float = 0.3,  # Fraction of threshold for "borderline"
    ):
        self.review_queue: List[UncertainSample] = []
        self.queue_file = review_queue_file
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.ood_margin = ood_margin
        
        if review_queue_file and review_queue_file.exists():
            self._load_queue()
    
    def _load_queue(self):
        """Load saved review queue."""
        try:
            with open(self.queue_file) as f:
                data = json.load(f)
                self.review_queue = [UncertainSample(**s) for s in data]
        except Exception as e:
            logger.warning(f"Failed to load review queue: {e}")
    
    def save_queue(self):
        """Save review queue."""
        if self.queue_file:
            with open(self.queue_file, "w") as f:
                # Convert numpy types to Python types for JSON serialization
                data = convert_numpy_types([asdict(s) for s in self.review_queue])
                json.dump(data, f, indent=2)
    
    def check_uncertainty(
        self,
        image_id: int,
        image_path: str,
        class_label: str,
        confidence: float,
        ood_score: float,
        ood_threshold: float,
        probabilities: Dict[str, float] = None,
    ) -> Optional[UncertainSample]:
        """
        Check if an image should be flagged for human review.
        
        Returns UncertainSample if uncertain, None otherwise.
        """
        uncertainty_type = None
        top_classes = []
        
        # Check 1: Low confidence prediction
        if self.confidence_low <= confidence <= self.confidence_high:
            uncertainty_type = "low_confidence"
        
        # Check 2: Borderline OOD score
        threshold_margin = ood_threshold * self.ood_margin
        if abs(ood_score - ood_threshold) < threshold_margin:
            uncertainty_type = uncertainty_type or "borderline_ood"
        
        # Check 3: Ambiguous class predictions (top 2 are close)
        if probabilities:
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_classes = sorted_probs[:3]
            
            if len(sorted_probs) >= 2:
                diff = sorted_probs[0][1] - sorted_probs[1][1]
                if diff < 0.1:  # Top 2 classes within 10% of each other
                    uncertainty_type = uncertainty_type or "class_ambiguity"
        
        if uncertainty_type:
            sample = UncertainSample(
                image_id=image_id,
                image_path=image_path,
                class_label=class_label,
                confidence=confidence,
                ood_score=ood_score,
                uncertainty_type=uncertainty_type,
                top_classes=top_classes,
                flagged_at=datetime.now().isoformat(),
            )
            self.review_queue.append(sample)
            return sample
        
        return None
    
    def get_pending_reviews(self, limit: int = 10) -> List[UncertainSample]:
        """Get samples pending human review."""
        return [s for s in self.review_queue if not s.reviewed][:limit]
    
    def submit_review(self, image_id: int, human_label: str):
        """Record human review for a sample."""
        for sample in self.review_queue:
            if sample.image_id == image_id:
                sample.reviewed = True
                sample.human_label = human_label
                self.save_queue()
                return True
        return False
    
    def get_reviewed_for_training(self) -> List[UncertainSample]:
        """Get reviewed samples that can be added to training data."""
        return [s for s in self.review_queue if s.reviewed and s.human_label]
    
    def get_stats(self) -> Dict:
        """Get active learning statistics."""
        pending = sum(1 for s in self.review_queue if not s.reviewed)
        reviewed = sum(1 for s in self.review_queue if s.reviewed)
        
        by_type = {}
        for s in self.review_queue:
            by_type[s.uncertainty_type] = by_type.get(s.uncertainty_type, 0) + 1
        
        return {
            "total_flagged": len(self.review_queue),
            "pending_review": pending,
            "reviewed": reviewed,
            "by_uncertainty_type": by_type,
        }

