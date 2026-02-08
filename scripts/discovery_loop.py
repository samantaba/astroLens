#!/usr/bin/env python3
"""
AstroLens Autonomous Discovery Loop

Runs continuously in the background to:
1. Download images from multiple astronomical sources
2. Analyze each image for anomalies using OOD detection  
3. Track near-misses and promising candidates
4. Periodically fine-tune the model on accumulated data
5. Adaptively adjust detection thresholds

This is designed to run until an anomaly is discovered or manually stopped.

Usage:
    # Run with default settings
    python scripts/discovery_loop.py
    
    # Run aggressively (more downloads, lower threshold)
    python scripts/discovery_loop.py --aggressive
    
    # Run in background
    nohup python scripts/discovery_loop.py > discovery.log 2>&1 &

Environment:
    OPENAI_API_KEY - Optional, for LLM-assisted analysis
    NASA_API_KEY   - Optional, for more APOD downloads
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import (
    DOWNLOADS_DIR, 
    WEIGHTS_DIR, 
    DATASETS_DIR,
    DATA_DIR,
    WEIGHTS_PATH,
)

from scripts.discovery_logger import get_discovery_logger

# Initialize structured logger
structured_logger = get_discovery_logger()

# API configuration
API_BASE = "http://localhost:8000"

# Configure logging
LOG_FILE = DATA_DIR / "discovery_loop.log"
CANDIDATES_FILE = DATA_DIR / "anomaly_candidates.json"
STATE_FILE = DATA_DIR / "discovery_state.json"
SOURCE_STATS_FILE = DATA_DIR / "source_stats.json"
REVIEW_QUEUE_FILE = DATA_DIR / "review_queue.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    """Track a single training run's metrics."""
    run_number: int = 0
    dataset: str = ""
    started_at: str = ""
    duration_minutes: float = 0.0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    improvement_pct: float = 0.0
    loss_before: float = 0.0
    loss_after: float = 0.0
    epochs_completed: int = 0
    success: bool = False


@dataclass
class DiscoveryStats:
    """Track discovery progress across cycles."""
    started_at: str = ""
    cycles_completed: int = 0
    total_downloaded: int = 0
    total_analyzed: int = 0
    duplicates_skipped: int = 0
    anomalies_found: int = 0
    near_misses: int = 0  # High OOD but below threshold
    uncertain_flagged: int = 0  # Flagged for active learning
    finetune_runs: int = 0
    current_threshold: float = 3.0
    highest_ood_score: float = 0.0
    highest_ood_image: str = ""
    last_cycle_at: str = ""
    anomaly_ids: List[int] = field(default_factory=list)
    # Model improvement tracking
    labeled_anomalies_downloaded: int = 0
    model_accuracy: float = 0.0  # Current model accuracy
    initial_accuracy: float = 0.0  # Accuracy at start
    total_improvement_pct: float = 0.0  # Total improvement since start
    training_history: List[dict] = field(default_factory=list)  # List of TrainingRun as dicts


@dataclass 
class AnomalyCandidate:
    """A promising image that might be an anomaly."""
    image_id: int
    image_path: str
    ood_score: float
    threshold_at_detection: float
    classification: str
    confidence: float
    source: str
    detected_at: str
    is_confirmed: bool = False


class DiscoveryLoop:
    """
    Autonomous discovery agent that hunts for anomalies.
    
    Strategy:
    1. Download in small batches from diverse sources
    2. Analyze immediately to catch anomalies fast
    3. Track "near-misses" (high OOD but not quite anomalous)
    4. Gradually lower threshold if no anomalies found
    5. Trigger fine-tuning to improve model periodically
    6. Run until stopped or anomaly confirmed
    
    Modes:
    - Normal: 1-min cycles, 30 images, fine-tune every 15 cycles
    - Aggressive: 30-sec cycles, 50 images, fine-tune every 10 cycles
    - Turbo: No wait, 100 images, fine-tune every 5 cycles (fastest)
    """

    def __init__(
        self,
        cycle_interval_minutes: float = 1.0,  # Changed: 1 minute default (was 5)
        images_per_cycle: int = 30,
        finetune_every_n_cycles: int = 15,  # Changed: more frequent (was 20)
        adaptive_threshold: bool = True,
        initial_threshold: float = 2.5,  # Changed: more sensitive (was 3.0)
        min_threshold: float = 0.5,  # Changed: can go lower (was 1.0)
        threshold_decay: float = 0.95,
        aggressive: bool = False,
        turbo: bool = False,
    ):
        self.cycle_interval = int(cycle_interval_minutes * 60)  # seconds
        self.images_per_cycle = images_per_cycle
        self.finetune_every_n = finetune_every_n_cycles
        self.adaptive_threshold = adaptive_threshold
        self.min_threshold = min_threshold
        self.threshold_decay = threshold_decay
        
        # Aggressive mode: faster cycles, more images, lower threshold
        if aggressive:
            self.cycle_interval = 30  # 30 seconds
            self.images_per_cycle = 50
            self.finetune_every_n = 10
            initial_threshold = 2.0
        
        # Turbo mode: no wait, maximum throughput
        if turbo:
            self.cycle_interval = 5  # Minimal wait (just to not hammer APIs)
            self.images_per_cycle = 100
            self.finetune_every_n = 5
            initial_threshold = 1.5
            self.threshold_decay = 0.98  # Slower decay since we're already low
        
        # Load or create state
        self.stats = self._load_state()
        if self.stats.current_threshold == 0:
            self.stats.current_threshold = initial_threshold
        
        self.candidates: List[AnomalyCandidate] = self._load_candidates()
        self.running = True
        self.api_available = False
        
        # Initialize components lazily
        self._classifier = None
        self._ood_detector = None
        self._duplicate_detector = None
        self._source_manager = None
        self._active_learning = None
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_state(self) -> DiscoveryStats:
        """Load saved state or create new."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                    return DiscoveryStats(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return DiscoveryStats(started_at=datetime.now().isoformat())

    def _save_state(self):
        """Persist state to disk."""
        with open(STATE_FILE, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            data = self._convert_numpy_types(asdict(self.stats))
            json.dump(data, f, indent=2)
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    def _load_candidates(self) -> List[AnomalyCandidate]:
        """Load saved candidates."""
        if CANDIDATES_FILE.exists():
            try:
                with open(CANDIDATES_FILE) as f:
                    data = json.load(f)
                    return [AnomalyCandidate(**c) for c in data]
            except Exception:
                pass
        return []

    def _save_candidates(self):
        """Persist candidates to disk."""
        with open(CANDIDATES_FILE, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            data = self._convert_numpy_types([asdict(c) for c in self.candidates])
            json.dump(data, f, indent=2)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("\n🛑 Shutdown signal received. Saving state...")
        self.running = False
        self._save_state()
        self._save_candidates()
        self._print_summary()
        sys.exit(0)

    @property
    def classifier(self):
        """Lazy-load classifier."""
        if self._classifier is None:
            from inference.classifier import AstroClassifier
            weights = WEIGHTS_PATH if Path(WEIGHTS_PATH).exists() else None
            self._classifier = AstroClassifier(weights_path=weights)
            logger.info(f"Loaded classifier with {self._classifier.num_classes} classes")
            logger.info(f"🖥️  Inference device: {self._classifier.device.upper()}")
            if self._classifier.device == "mps":
                logger.info("   ✓ Apple Metal GPU acceleration enabled (4x faster)")
            elif self._classifier.device == "cuda":
                logger.info("   ✓ NVIDIA CUDA GPU acceleration enabled (4x faster)")
            else:
                logger.info("   ⚠ Running on CPU (slower inference)")
        return self._classifier

    @property
    def ood_detector(self):
        """Lazy-load OOD detector with current threshold."""
        # Use aggressive mode when threshold is low (breakthrough hunting)
        aggressive = self.stats.current_threshold < 1.0
        
        if self._ood_detector is None:
            from inference.ood import OODDetector
            self._ood_detector = OODDetector(
                threshold=self.stats.current_threshold,
                aggressive_mode=aggressive,
            )
            if aggressive:
                logger.info("🎯 OOD detector using AGGRESSIVE mode (threshold < 1.0)")
        else:
            # Update threshold if changed
            self._ood_detector.threshold = self.stats.current_threshold
            # Reinitialize if aggressive mode changed
            if aggressive and self._ood_detector.voting_threshold != 1:
                from inference.ood import OODDetector
                self._ood_detector = OODDetector(
                    threshold=self.stats.current_threshold,
                    aggressive_mode=True,
                )
        return self._ood_detector

    def calibrate_ood_detector(self, sample_size: int = 200):
        """
        Calibrate OOD detector using in-distribution samples from database.
        
        This is CRITICAL for proper anomaly detection. Without calibration,
        the threshold is meaningless and almost nothing will be detected.
        """
        logger.info("🔧 Calibrating OOD detector on in-distribution data...")
        
        try:
            import numpy as np
            
            # Get sample of normal images from database
            response = httpx.get(
                f"{API_BASE}/images",
                params={"limit": sample_size, "anomaly_only": False},
            )
            response.raise_for_status()
            images = response.json()
            
            if len(images) < 20:
                logger.warning(f"  ⚠ Only {len(images)} images available, need more for calibration")
                return False
            
            # Run classifier on each to get logits/embeddings
            embeddings = []
            logits_list = []
            labels = []
            
            class_to_idx = {c: i for i, c in enumerate(self.classifier.classes)}
            
            for img in images[:sample_size]:
                try:
                    filepath = img.get("filepath")
                    if not filepath or not Path(filepath).exists():
                        continue
                    
                    result = self.classifier.classify(filepath)
                    embeddings.append(result.embedding)
                    logits_list.append(result.logits)
                    
                    # Use predicted class as pseudo-label
                    label_idx = class_to_idx.get(result.class_label, 0)
                    labels.append(label_idx)
                    
                except Exception as e:
                    continue
            
            if len(embeddings) < 20:
                logger.warning(f"  ⚠ Only {len(embeddings)} valid samples, need more")
                return False
            
            # Convert to numpy arrays
            embeddings = np.array(embeddings)
            logits_array = np.array(logits_list)
            labels = np.array(labels)
            
            # Calibrate the detector
            self.ood_detector.calibrate(
                embeddings=embeddings,
                logits=logits_array,
                labels=labels,
                target_fpr=0.05,  # 5% false positive rate
            )
            
            # Update threshold based on calibration
            old_threshold = self.stats.current_threshold
            self.stats.current_threshold = self.ood_detector.energy_threshold
            
            logger.info(f"  ✓ Calibrated on {len(embeddings)} samples")
            logger.info(f"  📊 New thresholds:")
            logger.info(f"     MSP: {self.ood_detector.msp_threshold:.3f}")
            logger.info(f"     Energy: {self.ood_detector.energy_threshold:.3f}")
            logger.info(f"     Mahalanobis: {self.ood_detector.mahal_threshold:.3f}")
            logger.info(f"  📉 Threshold: {old_threshold:.3f} → {self.stats.current_threshold:.3f}")
            
            self._save_state()
            return True
            
        except Exception as e:
            logger.warning(f"  ⚠ Calibration failed: {e}")
            return False

    @property
    def duplicate_detector(self):
        """Lazy-load duplicate detector."""
        if self._duplicate_detector is None:
            from inference.duplicates import DuplicateDetector
            self._duplicate_detector = DuplicateDetector(
                hash_size=16,
                similarity_threshold=0.92,
            )
            # Load existing hashes from database
            self._load_known_hashes()
        return self._duplicate_detector

    @property
    def source_manager(self):
        """Lazy-load source diversity manager."""
        if self._source_manager is None:
            from inference.active_learning import SourceDiversityManager
            self._source_manager = SourceDiversityManager(state_file=SOURCE_STATS_FILE)
        return self._source_manager

    @property
    def active_learning(self):
        """Lazy-load active learning manager."""
        if self._active_learning is None:
            from inference.active_learning import ActiveLearningManager
            self._active_learning = ActiveLearningManager(
                review_queue_file=REVIEW_QUEUE_FILE,
                confidence_low=0.4,
                confidence_high=0.6,
            )
        return self._active_learning

    def _load_known_hashes(self):
        """Load known hashes from database."""
        try:
            import httpx
            # Use API's max limit of 2000
            resp = httpx.get("http://localhost:8000/images?limit=2000", timeout=30)
            if resp.status_code == 200:
                images = resp.json()
                logger.info(f"Database has {len(images)} existing images")
        except Exception as e:
            logger.debug(f"Could not load hashes from API: {e}")

    def check_api(self) -> bool:
        """Check if AstroLens API is running."""
        try:
            import httpx
            resp = httpx.get("http://localhost:8000/health", timeout=5)
            self.api_available = resp.status_code == 200
        except Exception:
            self.api_available = False
        return self.api_available

    def download_images(self) -> List[Tuple[Path, str]]:
        """
        Download images from multiple sources with diversity-based allocation.
        
        Includes Galaxy Zoo anomalies for better anomaly detection training.
        
        Returns list of (file_path, source_name) tuples.
        """
        from scripts.nightly_ingest import (
            download_sdss_galaxies,
            download_nasa_apod,
            download_ztf_alerts,
            download_galaxy_zoo_anomalies,
            download_real_supernovae,
            download_gravitational_lenses,
            download_galaxy_mergers,
            download_peculiar_galaxies,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_dir = DOWNLOADS_DIR / f"discovery_{timestamp}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = []  # List of (path, source)
        
        # Get optimized download ratios based on source performance
        ratios = self.source_manager.get_download_ratios(self.images_per_cycle)
        logger.info(f"  📊 Source allocation: {ratios}")
        
        # All download functions - COMPREHENSIVE anomaly coverage
        download_fns = {
            "sdss": lambda n: download_sdss_galaxies(n, cycle_dir / "sdss"),
            "ztf": lambda n: download_ztf_alerts(n, cycle_dir / "ztf"),
            "apod": lambda n: download_nasa_apod(min(n, 7), cycle_dir / "apod"),
            "galaxy_zoo_anomalies": lambda n: download_galaxy_zoo_anomalies(n, cycle_dir / "gz_anomalies"),
        }
        
        for source_name, count in ratios.items():
            if source_name not in download_fns:
                continue
            
            try:
                files = download_fns[source_name](count)
                # Track source with each file
                for f in files:
                    all_files.append((f, source_name))
                
                self.source_manager.record_download(source_name, len(files))
                logger.info(f"  📥 {source_name}: {len(files)} images")
            except Exception as e:
                logger.warning(f"  ⚠ {source_name} download failed: {e}")
        
        # REAL ANOMALY SOURCES - Download every cycle for breakthrough detection
        # These are ACTUAL anomalies from verified catalogs, not random positions
        anomaly_sources = [
            ("supernovae", lambda: download_real_supernovae(5, cycle_dir / "supernovae")),
            ("gravitational_lenses", lambda: download_gravitational_lenses(3, cycle_dir / "lenses")),
            ("galaxy_mergers", lambda: download_galaxy_mergers(5, cycle_dir / "mergers")),
            ("peculiar_galaxies", lambda: download_peculiar_galaxies(3, cycle_dir / "peculiar")),
            ("galaxy_zoo_anomalies", lambda: download_galaxy_zoo_anomalies(5, cycle_dir / "gz_anomalies")),
        ]
        
        logger.info("  🎯 Downloading REAL anomalies from catalogs...")
        anomaly_count = 0
        for source_name, download_fn in anomaly_sources:
            try:
                files = download_fn()
                for f in files:
                    all_files.append((f, source_name))
                anomaly_count += len(files)
                if files:
                    logger.info(f"    ✓ {source_name}: {len(files)} verified anomalies")
            except Exception as e:
                logger.debug(f"    ⚠ {source_name}: {e}")
        
        if anomaly_count > 0:
            logger.info(f"  🎯 Total verified anomalies this cycle: {anomaly_count}")
        
        # Count all labeled anomalies in this batch
        anomaly_source_names = {"supernovae", "gravitational_lenses", "galaxy_mergers", 
                               "peculiar_galaxies", "galaxy_zoo_anomalies"}
        labeled_count = sum(1 for _, source in all_files if source in anomaly_source_names)
        self.stats.labeled_anomalies_downloaded += labeled_count
        self.stats.total_downloaded += len(all_files)
        return all_files

    def check_duplicate(self, image_path: Path) -> bool:
        """Check if image is a duplicate."""
        result = self.duplicate_detector.check_duplicate(str(image_path))
        if result.is_duplicate:
            logger.debug(f"Duplicate: {image_path.name} (sim: {result.similarity:.2%})")
            self.stats.duplicates_skipped += 1
            return True
        # Register the new hash
        self.duplicate_detector.add_hash(result.hash_value, str(image_path))
        return False

    def upload_to_api(self, image_path: Path, analysis: Dict) -> Optional[int]:
        """
        Upload image to API and store analysis results.
        
        Returns image_id if successful, None otherwise.
        """
        if not self.api_available:
            return None
        
        try:
            import httpx
            
            # Upload image
            with open(image_path, "rb") as f:
                resp = httpx.post(
                    "http://localhost:8000/images",
                    files={"file": (image_path.name, f)},
                    timeout=30,
                )
            
            if resp.status_code != 200:
                logger.warning(f"  ⚠ POST failed for {image_path.name}: {resp.status_code}")
                return None
            
            image_data = resp.json()
            image_id = image_data.get("id")
            
            if not image_id:
                logger.warning(f"  ⚠ No image_id returned for {image_path.name}")
                return None
            
            # Ensure all values are Python native types (not numpy) for JSON serialization
            class_label = str(analysis.get("class_label", "Unknown"))
            confidence = self._to_python_native(analysis.get("confidence", 0.0))
            ood_score = self._to_python_native(analysis.get("ood_score", 0.0))
            is_anomaly = bool(analysis.get("is_anomaly", False))
            
            # Update with analysis results - THIS IS CRITICAL FOR IMAGES TO SHOW AS ANALYZED
            update_data = {
                "class_label": class_label,
                "class_confidence": float(confidence) if confidence is not None else 0.0,
                "ood_score": float(ood_score) if ood_score is not None else 0.0,
                "is_anomaly": is_anomaly,
            }
            
            update_resp = httpx.patch(
                f"http://localhost:8000/images/{image_id}",
                json=update_data,
                timeout=10,
            )
            
            if update_resp.status_code != 200:
                logger.warning(f"  ⚠ PATCH failed for {image_path.name} (id={image_id}): {update_resp.status_code} - {update_resp.text[:100]}")
                # Still return ID since image was uploaded
                return image_id
            
            logger.debug(f"  ✓ Uploaded and analyzed: {image_path.name} -> id={image_id}")
            return image_id
            
        except Exception as e:
            logger.warning(f"  ⚠ Upload failed for {image_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _to_python_native(self, obj):
        """Recursively convert numpy/tensor types to Python native types."""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: self._to_python_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_python_native(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar or tensor
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # Try to convert to float/int if it looks like a number
            try:
                if isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
            except (TypeError, AttributeError):
                pass
            return obj

    def analyze_image(
        self, 
        image_path: Path, 
        source: str = "unknown",
        skip_duplicate_check: bool = False
    ) -> Dict:
        """
        Analyze a single image for anomalies using ensemble OOD detection.
        
        Returns dict with classification info and anomaly status.
        Returns None if duplicate or analysis fails.
        """
        try:
            # Check for duplicate first
            if not skip_duplicate_check and self.check_duplicate(image_path):
                return None  # Skip duplicates
            
            # Run classification (returns embedding for Mahalanobis)
            result = self.classifier.classify(str(image_path))
            
            # Run ensemble OOD detection with embedding
            ood_result = self.ood_detector.detect(result.logits, result.embedding)
            
            # Check both OOD score AND if class itself is an anomaly type
            is_anomaly_class = self.classifier.is_anomaly_class(result.class_label)
            is_anomaly = ood_result.is_anomaly or is_anomaly_class
            
            # Determine if near-miss (high OOD but below threshold)
            near_miss_threshold = self.stats.current_threshold * 0.8
            is_near_miss = (not is_anomaly and ood_result.ood_score > near_miss_threshold)
            
            # Convert ALL numpy types to Python natives for JSON serialization
            # Use explicit conversion to avoid float32 serialization errors
            ood_score_native = self._to_python_native(ood_result.ood_score)
            confidence_native = self._to_python_native(result.confidence)
            threshold_native = self._to_python_native(ood_result.threshold)
            votes_native = self._to_python_native(ood_result.votes)
            
            analysis = {
                "path": str(image_path),
                "source": source,
                "class_label": str(result.class_label),
                "confidence": float(confidence_native) if confidence_native is not None else 0.0,
                "ood_score": float(ood_score_native) if ood_score_native is not None else 0.0,
                "ood_votes": int(votes_native) if votes_native is not None else 0,
                "method_scores": self._to_python_native(ood_result.method_scores),
                "is_anomaly": bool(is_anomaly),
                "is_anomaly_class": bool(is_anomaly_class),
                "is_near_miss": bool(is_near_miss),
                "threshold": float(threshold_native) if threshold_native is not None else 3.0,
                "is_duplicate": False,
            }
            
            # Record for source diversity
            self.source_manager.record_analysis(
                source=source,
                ood_score=ood_result.ood_score,
                confidence=result.confidence,
                is_anomaly=is_anomaly,
                is_near_miss=is_near_miss,
            )
            
            # Upload to API so it appears in gallery
            image_id = self.upload_to_api(image_path, analysis)
            if image_id:
                analysis["image_id"] = image_id
                
                # Check for active learning (uncertain samples)
                # Convert probabilities to Python floats for JSON serialization
                probs = self._to_python_native(result.probabilities) if result.probabilities else None
                uncertain = self.active_learning.check_uncertainty(
                    image_id=image_id,
                    image_path=str(image_path),
                    class_label=str(result.class_label),
                    confidence=float(confidence_native) if confidence_native else 0.0,
                    ood_score=float(ood_score_native) if ood_score_native else 0.0,
                    ood_threshold=float(threshold_native) if threshold_native else 3.0,
                    probabilities=probs,
                )
                
                if uncertain:
                    self.stats.uncertain_flagged += 1
                    logger.info(f"  🔍 Flagged for review: {image_path.name} ({uncertain.uncertainty_type})")
            
            return analysis
            
        except Exception as e:
            # Log at WARNING level so we can see failures!
            logger.warning(f"  ⚠ Analysis failed for {image_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def process_cycle(self):
        """Run one discovery cycle with multi-source diversity and active learning."""
        cycle_start = datetime.now()
        cycle_num = self.stats.cycles_completed + 1
        
        # Log cycle start for timing
        structured_logger.start_cycle(cycle_num, self.stats.current_threshold)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CYCLE {cycle_num}")
        logger.info(f"   Threshold: {self.stats.current_threshold:.2f}")
        logger.info(f"{'='*60}")
        
        # Download with diversity-based allocation
        files_with_sources = self.download_images()
        if not files_with_sources:
            logger.warning("No images downloaded this cycle")
            return
        
        # Analyze
        logger.info(f"\n🔬 Analyzing {len(files_with_sources)} images...")
        
        anomalies_this_cycle = 0
        near_misses_this_cycle = 0
        analyzed_this_cycle = 0
        skipped_this_cycle = 0
        
        for filepath, source in files_with_sources:
            try:
                result = self.analyze_image(filepath, source=source)
                if result is None:
                    skipped_this_cycle += 1
                    continue
                
                analyzed_this_cycle += 1
                
                self.stats.total_analyzed += 1
                
                # Track highest OOD score ever seen (ensure Python native types)
                ood_score = float(result["ood_score"]) if result["ood_score"] is not None else 0.0
                if ood_score > self.stats.highest_ood_score:
                    self.stats.highest_ood_score = float(ood_score)
                    self.stats.highest_ood_image = str(result["path"])
                    logger.info(f"  🔥 New highest OOD: {ood_score:.3f} - {filepath.name}")
                    logger.info(f"     Votes: {result.get('ood_votes', 0)}/3 methods agree")
                
                # Check for anomaly
                if result["is_anomaly"]:
                    anomalies_this_cycle += 1
                    self.stats.anomalies_found += 1
                    
                    logger.info(f"\n  ⚡⚡⚡ ANOMALY DETECTED! ⚡⚡⚡")
                    logger.info(f"      File: {filepath.name}")
                    logger.info(f"      Source: {source}")
                    logger.info(f"      Class: {result['class_label']} ({result['confidence']:.1%})")
                    logger.info(f"      OOD Score: {result['ood_score']:.3f} (votes: {result.get('ood_votes', 0)})")
                    
                    # Save candidate (ensure all values are Python native types)
                    candidate = AnomalyCandidate(
                        image_id=int(result.get("image_id", len(self.candidates))),
                        image_path=str(filepath),
                        ood_score=float(result["ood_score"]) if result["ood_score"] is not None else 0.0,
                        threshold_at_detection=float(result["threshold"]) if result["threshold"] is not None else 3.0,
                        classification=str(result["class_label"]),
                        confidence=float(result["confidence"]) if result["confidence"] is not None else 0.0,
                        source=str(source),
                        detected_at=datetime.now().isoformat(),
                    )
                    self.candidates.append(candidate)
                    self._save_candidates()
                    
                    # Desktop notification
                    self._send_notification(
                        "🔭 AstroLens: Anomaly Detected!",
                        f"{result['class_label']} from {source} - OOD: {result['ood_score']:.2f}"
                    )
                    
                    # Log to structured logger (ensure native types)
                    structured_logger.log_anomaly(
                        cycle=int(self.stats.cycles_completed + 1),
                        image_path=str(filepath),
                        source=str(source),
                        class_label=str(result['class_label']),
                        confidence=float(result['confidence']) if result['confidence'] is not None else 0.0,
                        ood_score=float(result['ood_score']) if result['ood_score'] is not None else 0.0,
                        ood_votes=int(result.get('ood_votes', 0)),
                    )
                
                # Track near-misses
                elif result.get("is_near_miss", False):
                    near_misses_this_cycle += 1
                    self.stats.near_misses += 1
                    logger.info(f"  📍 Near-miss: {filepath.name} (OOD: {result['ood_score']:.3f}) from {source}")
                    
            except Exception as e:
                logger.warning(f"  ⚠ Error processing {filepath.name}: {e}")
                structured_logger.log_error(
                    cycle=self.stats.cycles_completed + 1,
                    error_type="image_processing",
                    message=f"{filepath.name}: {str(e)}",
                    recoverable=True,
                )
                continue
        
        # Save source stats
        self.source_manager.save_state()
        self.active_learning.save_queue()
        
        # Cycle summary with timing
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        images_per_second = analyzed_this_cycle / cycle_duration if cycle_duration > 0 else 0
        
        logger.info(f"\n📊 Cycle Summary:")
        logger.info(f"   Downloaded: {len(files_with_sources)}")
        logger.info(f"   Analyzed: {analyzed_this_cycle}")
        logger.info(f"   Skipped: {skipped_this_cycle} (duplicates or errors)")
        logger.info(f"   Anomalies: {anomalies_this_cycle}")
        logger.info(f"   Near-misses: {near_misses_this_cycle}")
        logger.info(f"   Uncertain (for review): {self.stats.uncertain_flagged} total")
        logger.info(f"   ⏱ Duration: {cycle_duration:.1f}s ({images_per_second:.2f} img/sec)")
        
        # Show source diversity summary occasionally
        if self.stats.cycles_completed % 5 == 0:
            logger.info(f"\n{self.source_manager.get_summary()}")
        
        self.stats.cycles_completed += 1
        self.stats.last_cycle_at = datetime.now().isoformat()
        
        # Log to structured logger
        source_counts = {}
        for _, source in files_with_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Ensure all values are Python native types for JSON serialization
        structured_logger.end_cycle(
            cycle_number=int(self.stats.cycles_completed),
            images_downloaded=int(len(files_with_sources)),
            images_analyzed=int(analyzed_this_cycle),
            duplicates_skipped=int(skipped_this_cycle),
            anomalies_found=int(anomalies_this_cycle),
            near_misses=int(near_misses_this_cycle),
            flagged_for_review=int(self.stats.uncertain_flagged),
            highest_ood=float(self.stats.highest_ood_score) if self.stats.highest_ood_score else 0.0,
            threshold=float(self.stats.current_threshold) if self.stats.current_threshold else 3.0,
            sources=source_counts,
        )
        
        # Adaptive threshold adjustment
        if self.adaptive_threshold and anomalies_this_cycle == 0:
            self._adjust_threshold()
        
        # Check if fine-tuning is due
        if self.stats.cycles_completed % self.finetune_every_n == 0:
            self._trigger_finetune()
        
        # Save state
        self._save_state()

    def _adjust_threshold(self):
        """Gradually lower threshold if no anomalies found."""
        old_threshold = float(self.stats.current_threshold)
        new_threshold = float(max(
            self.min_threshold,
            old_threshold * self.threshold_decay
        ))
        
        if new_threshold < old_threshold:
            self.stats.current_threshold = float(new_threshold)
            logger.info(f"  📉 Threshold adjusted: {old_threshold:.3f} → {new_threshold:.3f}")

    def _enrich_anomaly_dataset(self):
        """
        Copy downloaded anomalies to the training dataset for continuous improvement.
        
        This ensures the model keeps learning from new anomaly examples
        downloaded from real catalogs (supernovae, lenses, mergers, etc.)
        """
        import shutil
        
        anomaly_dataset = DATASETS_DIR / "anomalies" / "train"
        
        # Source directories and their target classes
        source_mappings = [
            (DOWNLOADS_DIR / "real_supernovae", "supernova"),
            (DOWNLOADS_DIR / "gravitational_lenses", "gravitational_lens"),
            (DOWNLOADS_DIR / "galaxy_mergers", "merger"),
            (DOWNLOADS_DIR / "peculiar_galaxies", "unusual_morphology"),
            (DOWNLOADS_DIR / "galaxy_zoo_anomalies", "unusual_morphology"),
        ]
        
        total_copied = 0
        
        for source_dir, target_class in source_mappings:
            if not source_dir.exists():
                continue
            
            target_dir = anomaly_dataset / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in source_dir.glob("*.jpg"):
                dest = target_dir / img_file.name
                if not dest.exists():
                    try:
                        shutil.copy(img_file, dest)
                        total_copied += 1
                    except Exception:
                        pass
        
        if total_copied > 0:
            logger.info(f"  📥 Enriched anomaly dataset: +{total_copied} new images")
        
        # Log dataset statistics
        total_anomaly_images = 0
        for class_dir in anomaly_dataset.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg")))
                total_anomaly_images += count
        
        if total_anomaly_images > 0:
            logger.info(f"  📊 Anomaly dataset now has {total_anomaly_images} images")

    def _trigger_finetune(self):
        """
        Trigger model fine-tuning on combined dataset.
        
        Uses both galaxy10 (normal) and anomalies (unusual) datasets
        to teach the model what anomalies look like.
        
        IMPORTANT: Fine-tuning runs WITHOUT timeout to allow completion.
        Progress is streamed to the log in real-time.
        """
        logger.info("\n🎓 Triggering fine-tuning cycle...")
        logger.info("   This may take 30-60+ minutes on CPU. Progress will be shown below.")
        finetune_start = datetime.now()
        
        # FIRST: Enrich anomaly dataset with newly downloaded images
        self._enrich_anomaly_dataset()
        
        # Dataset rotation strategy:
        # - Prioritize anomalies for breakthrough detection (2 out of 4 cycles)
        # - Include galaxy10 for normal class calibration
        # - Include galaxy_zoo for diverse morphologies
        #
        # Pattern: anomalies -> galaxy10 -> anomalies -> galaxy_zoo
        # This ensures anomaly training happens 50% of the time
        datasets = ["anomalies", "galaxy10", "anomalies", "galaxy_zoo"]
        dataset_idx = self.stats.finetune_runs % len(datasets)
        dataset = datasets[dataset_idx]
        
        logger.info(f"  📚 Training rotation: cycle {self.stats.finetune_runs + 1}, dataset: {dataset}")
        
        # Configure training based on mode
        # NO TIMEOUT - Training runs until completion (resume from checkpoint if needed)
        if hasattr(self, 'turbo') and self.turbo:
            epochs = 3  # Reasonable training in turbo mode
            max_training_minutes = None  # No timeout - runs to completion
            logger.info(f"  Training on: {dataset} (turbo: {epochs} epochs, no timeout)")
        elif hasattr(self, 'aggressive') and self.aggressive:
            epochs = 4  # More training in aggressive mode
            max_training_minutes = None  # No timeout - runs to completion
            logger.info(f"  Training on: {dataset} (aggressive: {epochs} epochs, no timeout)")
        else:
            epochs = 5  # Full training in normal mode
            max_training_minutes = None  # No timeout - training runs to completion
            logger.info(f"  Training on: {dataset} (normal: {epochs} epochs, no timeout)")
        
        try:
            # Run fine-tuning with RESUME support
            # This allows training to continue from checkpoints if timed out previously
            
            # Set environment to disable tqdm (progress bars don't work in subprocess)
            env = os.environ.copy()
            env["ASTROLENS_SUBPROCESS"] = "1"
            
            process = subprocess.Popen(
                [
                    sys.executable, 
                    "finetuning/pipeline.py",
                    "--dataset", dataset,
                    "--epochs", str(epochs),
                    "--skip_download",
                    "--verbose",  # Enable verbose output
                    "--resume",   # Resume from checkpoint if available
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            
            # Stream output in real-time - NO TIMEOUT (training runs to completion)
            epoch_count = 0
            last_progress_log = datetime.now()
            step_count = 0
            
            # Track accuracy values for improvement calculation
            first_accuracy = None
            last_accuracy = None
            first_loss = None
            last_loss = None
            
            import re
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Read output line
                try:
                    line = process.stdout.readline()
                except Exception:
                    break
                    
                if not line:
                    continue
                    
                line = line.strip()
                if not line:
                    continue
                
                # Skip tqdm progress bar lines (contain \r or |)
                if '\r' in line or ('|' in line and '%' in line):
                    continue
                
                # Extract accuracy and loss from eval lines
                # Format: {'eval_loss': 0.5119, 'eval_accuracy': 0.8201, ...}
                if "'eval_accuracy'" in line:
                    acc_match = re.search(r"'eval_accuracy':\s*([\d.]+)", line)
                    loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
                    if acc_match:
                        acc_value = float(acc_match.group(1))
                        if first_accuracy is None:
                            first_accuracy = acc_value
                        last_accuracy = acc_value
                    if loss_match:
                        loss_value = float(loss_match.group(1))
                        if first_loss is None:
                            first_loss = loss_value
                        last_loss = loss_value
                
                # Log training metrics (loss, accuracy, step info)
                if "'loss'" in line or "'eval_loss'" in line or "'accuracy'" in line or "Step " in line:
                    logger.info(f"  📈 {line}")
                    step_count += 1
                elif any(kw in line.lower() for kw in ['epoch', 'training', 'evaluating', 'complete', 'error', 'warning', 'saved', 'saving']):
                    logger.info(f"  [TRAIN] {line}")
                    
                    # Track epoch progress
                    if 'epoch' in line.lower() and ('/' in line or 'eval' in line.lower()):
                        epoch_count += 1
                        logger.info(f"  📊 Completed epoch {epoch_count}/{epochs}")
                
                # Log periodic progress every 2 minutes
                now = datetime.now()
                if (now - last_progress_log).total_seconds() > 120:
                    elapsed = (now - finetune_start).total_seconds() / 60
                    logger.info(f"  ⏱ Training in progress: {elapsed:.1f} minutes elapsed, {step_count} steps logged")
                    last_progress_log = now
            
            process.wait()
            
            duration_minutes = (datetime.now() - finetune_start).total_seconds() / 60
            
            if process.returncode == 0:
                self.stats.finetune_runs += 1
                logger.info(f"  ✓ Fine-tuning complete (run #{self.stats.finetune_runs})")
                logger.info(f"  ⏱ Total duration: {duration_minutes:.1f} minutes")
                
                # Calculate and log improvement
                improvement_pct = 0.0
                if first_accuracy is not None and last_accuracy is not None:
                    if first_accuracy > 0:
                        improvement_pct = ((last_accuracy - first_accuracy) / first_accuracy) * 100
                    
                    logger.info(f"  📊 Accuracy: {first_accuracy:.1%} → {last_accuracy:.1%}")
                    if improvement_pct > 0:
                        logger.info(f"  📈 Improvement: +{improvement_pct:.1f}%")
                    elif improvement_pct < 0:
                        logger.info(f"  📉 Change: {improvement_pct:.1f}%")
                    
                    # Update model accuracy in stats
                    # IMPORTANT: Only use Galaxy10 for primary accuracy tracking
                    # (anomalies/galaxy_zoo have much lower baseline accuracy)
                    is_primary_dataset = dataset in ("galaxy10", "Galaxy10")
                    
                    if is_primary_dataset:
                        if self.stats.initial_accuracy == 0 or self.stats.initial_accuracy < 0.5:
                            # Set initial from Galaxy10 (should be ~80%+)
                            self.stats.initial_accuracy = first_accuracy
                        self.stats.model_accuracy = last_accuracy
                        
                        # Calculate total improvement since start (Galaxy10 only)
                        if self.stats.initial_accuracy > 0.5:
                            self.stats.total_improvement_pct = (
                                (last_accuracy - self.stats.initial_accuracy) / self.stats.initial_accuracy * 100
                            )
                    else:
                        # For anomaly/galaxy_zoo runs, log but don't update primary metrics
                        logger.info(f"  ℹ️  {dataset} accuracy: {last_accuracy:.1%} (not primary metric)")
                
                # Record training run in history
                training_run = {
                    "run_number": self.stats.finetune_runs,
                    "dataset": dataset,
                    "started_at": finetune_start.isoformat(),
                    "duration_minutes": round(duration_minutes, 1),
                    "accuracy_before": round(first_accuracy or 0, 4),
                    "accuracy_after": round(last_accuracy or 0, 4),
                    "improvement_pct": round(improvement_pct, 2),
                    "loss_before": round(first_loss or 0, 4),
                    "loss_after": round(last_loss or 0, 4),
                    "epochs_completed": epoch_count,
                    "success": True,
                }
                self.stats.training_history.append(training_run)
                
                # Log to structured logger
                structured_logger.log_finetune(
                    run_number=self.stats.finetune_runs,
                    dataset=dataset,
                    duration_minutes=duration_minutes,
                    success=True,
                )
                
                # Reload classifier with new weights
                self._classifier = None  # Force reload
                logger.info("  🔄 Classifier will reload with new weights on next analysis")
                
                # Save state to persist training history
                self._save_state()
            else:
                logger.warning(f"  ⚠ Fine-tuning failed (exit code {process.returncode})")
                structured_logger.log_finetune(
                    run_number=self.stats.finetune_runs + 1,
                    dataset=dataset,
                    duration_minutes=duration_minutes,
                    success=False,
                    error=f"Exit code {process.returncode}",
                )
                
        except Exception as e:
            logger.warning(f"  ⚠ Fine-tuning error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            structured_logger.log_error(
                cycle=self.stats.cycles_completed,
                error_type="finetune_error",
                message=str(e),
                recoverable=True,
            )

    def _send_notification(self, title: str, message: str):
        """Send desktop notification (without sound)."""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":
                # No sound - silent notification
                script = f'display notification "{message}" with title "{title}"'
                subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
            elif system == "Linux":
                subprocess.run(["notify-send", title, message], check=False, capture_output=True)
        except Exception:
            pass

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "="*60)
        print("🔭 ASTROLENS DISCOVERY LOOP SUMMARY")
        print("="*60)
        print(f"Started:        {self.stats.started_at}")
        print(f"Cycles:         {self.stats.cycles_completed}")
        print(f"Downloaded:     {self.stats.total_downloaded} images")
        print(f"Duplicates:     {self.stats.duplicates_skipped} skipped")
        print(f"Analyzed:       {self.stats.total_analyzed} images")
        print(f"Anomalies:      {self.stats.anomalies_found}")
        print(f"Near-misses:    {self.stats.near_misses}")
        print(f"Uncertain:      {self.stats.uncertain_flagged} flagged for review")
        print(f"Fine-tune runs: {self.stats.finetune_runs}")
        print(f"Final threshold: {self.stats.current_threshold:.3f}")
        print(f"Highest OOD:    {self.stats.highest_ood_score:.3f}")
        
        # Show model training progress
        print(f"\n🎓 Model Training Progress:")
        print(f"   Labeled anomalies: {self.stats.labeled_anomalies_downloaded}")
        if self.stats.model_accuracy > 0:
            print(f"   Current accuracy:  {self.stats.model_accuracy:.1%}")
        if self.stats.initial_accuracy > 0 and self.stats.total_improvement_pct != 0:
            sign = "+" if self.stats.total_improvement_pct > 0 else ""
            print(f"   Total improvement: {sign}{self.stats.total_improvement_pct:.1f}%")
        if self.stats.training_history:
            print(f"   Training runs:     {len(self.stats.training_history)}")
        
        # Show source diversity
        print(f"\n{self.source_manager.get_summary()}")
        
        # Show active learning stats
        al_stats = self.active_learning.get_stats()
        print(f"\n🔍 Active Learning:")
        print(f"   Total flagged: {al_stats['total_flagged']}")
        print(f"   Pending review: {al_stats['pending_review']}")
        
        if self.stats.highest_ood_image:
            print(f"Best candidate: {self.stats.highest_ood_image}")
        
        if self.candidates:
            print(f"\n📁 {len(self.candidates)} candidates saved to:")
            print(f"   {CANDIDATES_FILE}")
        
        print("="*60)

    def run(self):
        """Main loop - runs until stopped."""
        logger.info("="*60)
        logger.info("🚀 ASTROLENS AUTONOMOUS DISCOVERY LOOP")
        logger.info("="*60)
        logger.info(f"Cycle interval: {self.cycle_interval // 60} minutes")
        logger.info(f"Images/cycle: {self.images_per_cycle}")
        logger.info(f"Fine-tune every: {self.finetune_every_n} cycles")
        logger.info(f"Initial threshold: {self.stats.current_threshold}")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*60)
        
        # Initialize classifier early to log device info
        logger.info("\n🔧 Initializing ML models...")
        _ = self.classifier  # Force load to get device info
        
        # Log start to structured logger with device info
        structured_logger.log_start({
            "cycle_interval_seconds": int(self.cycle_interval),
            "images_per_cycle": int(self.images_per_cycle),
            "finetune_every_n": int(self.finetune_every_n),
            "initial_threshold": float(self.stats.current_threshold),
            "inference_device": str(self._classifier.device) if self._classifier else "unknown",
            "num_classes": int(self._classifier.num_classes) if self._classifier else 0,
            "aggressive": hasattr(self, 'aggressive') and self.aggressive,
            "turbo": hasattr(self, 'turbo') and self.turbo,
        })
        
        # Check API status
        if self.check_api():
            logger.info("✓ AstroLens API is running")
        else:
            logger.warning("⚠ API not running - will analyze locally only")
        
        # Check if OOD detector needs calibration
        # If threshold >> highest_ood_score and we have data, calibration is probably needed
        threshold_miscalibrated = (
            self.stats.highest_ood_score > 0 and
            self.stats.current_threshold > self.stats.highest_ood_score * 3
        )
        
        if threshold_miscalibrated:
            logger.info("\n⚠️  OOD threshold appears miscalibrated:")
            logger.info(f"   Threshold: {self.stats.current_threshold:.3f}")
            logger.info(f"   Highest OOD seen: {self.stats.highest_ood_score:.3f}")
            logger.info("   Attempting auto-calibration...")
            self.calibrate_ood_detector()
        
        logger.info("\n📊 Sources enabled: SDSS, ZTF, NASA APOD, Galaxy Zoo Anomalies")
        logger.info("   Galaxy Zoo anomalies provide labeled unusual galaxies for training")
        
        while self.running:
            try:
                self.process_cycle()
                
                # Check if we found anomalies
                if self.stats.anomalies_found > 0:
                    logger.info("\n🎉 ANOMALY FOUND! Review candidates in the app.")
                    # Keep running to find more, but celebrate
                
                # Wait for next cycle
                if self.running:
                    logger.info(f"\n⏳ Next cycle in {self.cycle_interval // 60} minutes...")
                    time.sleep(self.cycle_interval)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                import traceback
                tb = traceback.format_exc()
                structured_logger.log_error(
                    cycle=self.stats.cycles_completed + 1,
                    error_type="cycle_error",
                    message=f"{str(e)}\n{tb[:300]}",
                    recoverable=True,
                )
                time.sleep(60)  # Wait a minute before retry
        
        # Log stop event
        structured_logger.log_stop(self._convert_numpy_types(asdict(self.stats)))
        self._print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run autonomous anomaly discovery loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/discovery_loop.py                    # Normal mode (1-min cycles)
    python scripts/discovery_loop.py --aggressive       # Fast hunting (30-sec cycles)
    python scripts/discovery_loop.py --turbo            # Maximum speed (no wait)
    python scripts/discovery_loop.py --interval 0.5     # 30-second cycles
    python scripts/discovery_loop.py --threshold 1.5    # Lower threshold (more sensitive)
        """
    )
    
    parser.add_argument(
        "--aggressive", "-a",
        action="store_true",
        help="Aggressive mode: 30-sec cycles, 50 images, fine-tune every 10 cycles",
    )
    parser.add_argument(
        "--turbo", "-t",
        action="store_true",
        help="Turbo mode: 5-sec cycles, 100 images, maximum throughput",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Minutes between cycles (default: 1.0, use 0 for no wait)",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=30,
        help="Images per cycle (default: 30)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Initial OOD threshold (default: 3.0, lower = more sensitive)",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive threshold lowering",
    )
    parser.add_argument(
        "--finetune-every",
        type=int,
        default=20,
        help="Fine-tune every N cycles (default: 20)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset discovery state and start fresh",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Force OOD detector calibration before running",
    )
    parser.add_argument(
        "--crossref",
        action="store_true",
        help="Cross-reference all anomalies against catalogs (SIMBAD, NED) and exit",
    )
    parser.add_argument(
        "--crossref-limit",
        type=int,
        default=100,
        help="Maximum anomalies to cross-reference (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Reset state if requested
    if args.reset:
        for f in [STATE_FILE, CANDIDATES_FILE]:
            if f.exists():
                f.unlink()
        logger.info("State reset - starting fresh")
    
    # Cross-reference mode: query catalogs and exit
    if args.crossref:
        logger.info("="*60)
        logger.info("🔍 CATALOG CROSS-REFERENCE MODE")
        logger.info("="*60)
        
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        # Get anomalies from API
        try:
            response = httpx.get(
                f"{API_BASE}/candidates",
                params={"limit": args.crossref_limit},
            )
            response.raise_for_status()
            anomalies = response.json()
            
            if not anomalies:
                logger.info("No anomaly candidates found in database")
                return
            
            logger.info(f"Found {len(anomalies)} anomaly candidates")
            logger.info(f"Querying SIMBAD and NED for each (this may take a while)...")
            
            def progress_callback(current, total, result):
                status = "✓ KNOWN" if result.is_known else "★ UNKNOWN"
                logger.info(f"  [{current}/{total}] {status}: {Path(result.image_path).name}")
                if result.primary_match:
                    logger.info(f"       → {result.primary_match.object_name} ({result.primary_match.object_type})")
            
            stats = xref.cross_reference_all(
                [{"id": a["id"], "filepath": a["filepath"]} for a in anomalies],
                progress_callback=progress_callback,
            )
            
            logger.info("\n" + "="*60)
            logger.info("📊 CROSS-REFERENCE RESULTS")
            logger.info("="*60)
            logger.info(f"Total checked:    {stats['total']}")
            logger.info(f"Known objects:    {stats['known']} (already in catalogs)")
            logger.info(f"UNKNOWN objects:  {stats['unknown']} ← potential discoveries!")
            logger.info(f"With publications:{stats['published']}")
            logger.info(f"Errors:           {stats['errors']}")
            logger.info(f"Skipped (cached): {stats['skipped']}")
            logger.info("="*60)
            
            summary = xref.get_summary()
            logger.info(f"\n💾 Results saved to: {ARTIFACTS_DIR / 'data' / 'cross_reference_results.json'}")
            
        except Exception as e:
            logger.error(f"Cross-reference failed: {e}")
        
        return
    
    loop = DiscoveryLoop(
        cycle_interval_minutes=args.interval,
        images_per_cycle=args.images,
        initial_threshold=args.threshold,
        adaptive_threshold=not args.no_adaptive,
        finetune_every_n_cycles=args.finetune_every,
        aggressive=args.aggressive,
        turbo=args.turbo,
    )
    
    # Force calibration if requested
    if args.calibrate:
        logger.info("\n🔧 Forcing OOD detector calibration...")
        loop.calibrate_ood_detector()
    
    loop.run()


if __name__ == "__main__":
    main()

