"""
YOLO Transient Detector Integration

Integrates the trained YOLO model with the Discovery pipeline to:
1. Confirm OOD-flagged anomalies as real transients
2. Localize where in the image the transient is
3. Reduce false positives from the ViT+OOD pipeline

Usage:
    from inference.yolo_detector import YOLOTransientDetector
    
    detector = YOLOTransientDetector()
    result = detector.detect(image_path)
    if result.is_transient:
        print(f"Transient at {result.box} with {result.confidence:.2%} confidence")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model path
MODEL_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "models" / "transient_detector" / "weights"
DEFAULT_MODEL = MODEL_DIR / "best.pt"


@dataclass
class DetectionResult:
    """Result from YOLO transient detection."""
    image_path: str
    is_transient: bool
    confidence: float  # 0-1
    box: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2 (normalized)
    box_pixels: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2 (pixels)
    class_name: str = "transient"
    all_detections: List[dict] = None  # All boxes found
    
    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"


class YOLOTransientDetector:
    """
    YOLO-based transient detector.
    
    Uses the trained YOLOv8 model to detect transient events in astronomical images.
    Designed to work as a second-stage filter after ViT+OOD detection.
    """
    
    def __init__(self, model_path: Optional[Path] = None, confidence_threshold: float = 0.25):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model weights. Uses default if not provided.
            confidence_threshold: Minimum confidence for detection (0-1).
        """
        self.model_path = model_path or DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        if not self.model_path.exists():
            logger.warning(f"YOLO model not found at {self.model_path}")
            logger.info("Run the transient pipeline to train the model first")
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info(f"YOLO model loaded from {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def is_available(self) -> bool:
        """Check if the YOLO model is loaded and ready."""
        return self.model is not None
    
    def detect(self, image_path: str, save_annotated: bool = False) -> DetectionResult:
        """
        Detect transients in an image.
        
        Args:
            image_path: Path to the image file.
            save_annotated: Whether to save an annotated copy with boxes drawn.
        
        Returns:
            DetectionResult with detection information.
        """
        if not self.is_available():
            return DetectionResult(
                image_path=str(image_path),
                is_transient=False,
                confidence=0.0,
                all_detections=[]
            )
        
        try:
            # Run inference
            results = self.model(
                str(image_path),
                verbose=False,
                conf=self.confidence_threshold,
                save=save_annotated
            )
            
            # Parse results
            boxes = results[0].boxes
            
            if boxes is None or len(boxes) == 0:
                return DetectionResult(
                    image_path=str(image_path),
                    is_transient=False,
                    confidence=0.0,
                    all_detections=[]
                )
            
            # Get best detection
            all_detections = []
            best_conf = 0.0
            best_box = None
            best_box_pixels = None
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()  # x1, y1, x2, y2 in pixels
                xywhn = boxes.xywhn[i].tolist() if hasattr(boxes, 'xywhn') else None
                
                detection = {
                    "confidence": conf,
                    "box_pixels": tuple(int(x) for x in xyxy),
                    "class": "transient"
                }
                all_detections.append(detection)
                
                if conf > best_conf:
                    best_conf = conf
                    best_box_pixels = tuple(int(x) for x in xyxy)
                    # Normalize box
                    img_w = results[0].orig_shape[1]
                    img_h = results[0].orig_shape[0]
                    best_box = (
                        xyxy[0] / img_w,
                        xyxy[1] / img_h,
                        xyxy[2] / img_w,
                        xyxy[3] / img_h
                    )
            
            return DetectionResult(
                image_path=str(image_path),
                is_transient=True,
                confidence=best_conf,
                box=best_box,
                box_pixels=best_box_pixels,
                all_detections=all_detections
            )
            
        except Exception as e:
            logger.error(f"YOLO detection failed for {image_path}: {e}")
            return DetectionResult(
                image_path=str(image_path),
                is_transient=False,
                confidence=0.0,
                all_detections=[]
            )
    
    def detect_batch(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> List[DetectionResult]:
        """
        Detect transients in multiple images.
        
        Args:
            image_paths: List of image paths.
            progress_callback: Optional callback(current, total) for progress.
        
        Returns:
            List of DetectionResult objects.
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            result = self.detect(path)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def confirm_ood_candidates(
        self,
        ood_candidates: List[dict],
        min_confidence: float = 0.5
    ) -> Tuple[List[dict], List[dict]]:
        """
        Filter OOD candidates using YOLO.
        
        Takes candidates flagged by OOD detection and runs YOLO to confirm
        which are likely real transients vs artifacts.
        
        Args:
            ood_candidates: List of dicts with 'image_path' key.
            min_confidence: Minimum YOLO confidence to confirm as transient.
        
        Returns:
            Tuple of (confirmed_transients, rejected_artifacts)
        """
        confirmed = []
        rejected = []
        
        for candidate in ood_candidates:
            image_path = candidate.get("image_path") or candidate.get("filepath")
            if not image_path:
                rejected.append(candidate)
                continue
            
            result = self.detect(image_path)
            
            if result.is_transient and result.confidence >= min_confidence:
                candidate["yolo_confirmed"] = True
                candidate["yolo_confidence"] = result.confidence
                candidate["yolo_box"] = result.box_pixels
                confirmed.append(candidate)
            else:
                candidate["yolo_confirmed"] = False
                candidate["yolo_confidence"] = result.confidence
                rejected.append(candidate)
        
        logger.info(f"YOLO confirmed {len(confirmed)}/{len(ood_candidates)} OOD candidates")
        return confirmed, rejected


# Convenience function
def detect_transient(image_path: str) -> DetectionResult:
    """Quick function to detect transients in an image."""
    detector = YOLOTransientDetector()
    return detector.detect(image_path)
