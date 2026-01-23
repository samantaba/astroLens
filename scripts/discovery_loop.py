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
        return self._classifier

    @property
    def ood_detector(self):
        """Lazy-load OOD detector with current threshold."""
        if self._ood_detector is None:
            from inference.ood import OODDetector
            self._ood_detector = OODDetector(threshold=self.stats.current_threshold)
        else:
            # Update threshold if changed
            self._ood_detector.threshold = self.stats.current_threshold
        return self._ood_detector

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
        
        Returns list of (file_path, source_name) tuples.
        """
        from scripts.nightly_ingest import (
            download_sdss_galaxies,
            download_nasa_apod,
            download_ztf_alerts,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_dir = DOWNLOADS_DIR / f"discovery_{timestamp}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = []  # List of (path, source)
        
        # Get optimized download ratios based on source performance
        ratios = self.source_manager.get_download_ratios(self.images_per_cycle)
        logger.info(f"  📊 Source allocation: {ratios}")
        
        # Download functions
        download_fns = {
            "sdss": lambda n: download_sdss_galaxies(n, cycle_dir / "sdss"),
            "ztf": lambda n: download_ztf_alerts(n, cycle_dir / "ztf"),
            "apod": lambda n: download_nasa_apod(min(n, 7), cycle_dir / "apod"),
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
            
            # Update with analysis results - THIS IS CRITICAL FOR IMAGES TO SHOW AS ANALYZED
            update_resp = httpx.patch(
                f"http://localhost:8000/images/{image_id}",
                json={
                    "class_label": analysis["class_label"],
                    "class_confidence": float(analysis["confidence"]),
                    "ood_score": float(analysis["ood_score"]),
                    "is_anomaly": bool(analysis["is_anomaly"]),
                },
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
            return None

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
            
            # Convert numpy types to Python natives for JSON serialization
            analysis = {
                "path": str(image_path),
                "source": source,
                "class_label": result.class_label,
                "confidence": float(result.confidence),
                "ood_score": float(ood_result.ood_score),
                "ood_votes": int(ood_result.votes),
                "method_scores": {k: float(v) for k, v in ood_result.method_scores.items()},
                "is_anomaly": bool(is_anomaly),
                "is_anomaly_class": bool(is_anomaly_class),
                "is_near_miss": bool(is_near_miss),
                "threshold": float(ood_result.threshold),
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
                probs = {k: float(v) for k, v in result.probabilities.items()} if result.probabilities else None
                uncertain = self.active_learning.check_uncertainty(
                    image_id=image_id,
                    image_path=str(image_path),
                    class_label=result.class_label,
                    confidence=float(result.confidence),
                    ood_score=float(ood_result.ood_score),
                    ood_threshold=float(ood_result.threshold),
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
                
                # Track highest OOD score ever seen
                if result["ood_score"] > self.stats.highest_ood_score:
                    self.stats.highest_ood_score = result["ood_score"]
                    self.stats.highest_ood_image = result["path"]
                    logger.info(f"  🔥 New highest OOD: {result['ood_score']:.3f} - {filepath.name}")
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
                    
                    # Save candidate
                    candidate = AnomalyCandidate(
                        image_id=result.get("image_id", len(self.candidates)),
                        image_path=str(filepath),
                        ood_score=result["ood_score"],
                        threshold_at_detection=result["threshold"],
                        classification=result["class_label"],
                        confidence=result["confidence"],
                        source=source,
                        detected_at=datetime.now().isoformat(),
                    )
                    self.candidates.append(candidate)
                    self._save_candidates()
                    
                    # Desktop notification
                    self._send_notification(
                        "🔭 AstroLens: Anomaly Detected!",
                        f"{result['class_label']} from {source} - OOD: {result['ood_score']:.2f}"
                    )
                    
                    # Log to structured logger
                    structured_logger.log_anomaly(
                        cycle=self.stats.cycles_completed + 1,
                        image_path=str(filepath),
                        source=source,
                        class_label=result['class_label'],
                        confidence=result['confidence'],
                        ood_score=result['ood_score'],
                        ood_votes=result.get('ood_votes', 0),
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
        
        # Cycle summary
        logger.info(f"\n📊 Cycle Summary:")
        logger.info(f"   Downloaded: {len(files_with_sources)}")
        logger.info(f"   Analyzed: {analyzed_this_cycle}")
        logger.info(f"   Skipped: {skipped_this_cycle} (duplicates or errors)")
        logger.info(f"   Anomalies: {anomalies_this_cycle}")
        logger.info(f"   Near-misses: {near_misses_this_cycle}")
        logger.info(f"   Uncertain (for review): {self.stats.uncertain_flagged} total")
        
        # Show source diversity summary occasionally
        if self.stats.cycles_completed % 5 == 0:
            logger.info(f"\n{self.source_manager.get_summary()}")
        
        self.stats.cycles_completed += 1
        self.stats.last_cycle_at = datetime.now().isoformat()
        
        # Log to structured logger
        source_counts = {}
        for _, source in files_with_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        structured_logger.end_cycle(
            cycle_number=self.stats.cycles_completed,
            images_downloaded=len(files_with_sources),
            images_analyzed=len(files_with_sources) - self.stats.duplicates_skipped,
            duplicates_skipped=self.stats.duplicates_skipped,
            anomalies_found=anomalies_this_cycle,
            near_misses=near_misses_this_cycle,
            flagged_for_review=self.stats.uncertain_flagged,
            highest_ood=self.stats.highest_ood_score,
            threshold=self.stats.current_threshold,
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
        old_threshold = self.stats.current_threshold
        new_threshold = max(
            self.min_threshold,
            old_threshold * self.threshold_decay
        )
        
        if new_threshold < old_threshold:
            self.stats.current_threshold = new_threshold
            logger.info(f"  📉 Threshold adjusted: {old_threshold:.3f} → {new_threshold:.3f}")

    def _trigger_finetune(self):
        """
        Trigger model fine-tuning on combined dataset.
        
        Uses both galaxy10 (normal) and anomalies (unusual) datasets
        to teach the model what anomalies look like.
        """
        logger.info("\n🎓 Triggering fine-tuning cycle...")
        finetune_start = datetime.now()
        
        # Alternate between datasets to build diverse knowledge
        datasets = ["galaxy10", "anomalies", "galaxy_zoo"]
        dataset_idx = self.stats.finetune_runs % len(datasets)
        dataset = datasets[dataset_idx]
        
        logger.info(f"  Training on: {dataset}")
        
        try:
            # Run the fine-tuning pipeline with longer timeout
            # Fine-tuning can take 45-60 minutes on CPU
            result = subprocess.run(
                [
                    sys.executable, 
                    "finetuning/pipeline.py",
                    "--dataset", dataset,
                    "--epochs", "2",  # Reduced for faster iterations
                    "--skip_download",
                ],
                capture_output=True,
                text=True,
                timeout=3600,  # 60 min max (increased from 30)
            )
            
            duration_minutes = (datetime.now() - finetune_start).total_seconds() / 60
            
            if result.returncode == 0:
                self.stats.finetune_runs += 1
                logger.info(f"  ✓ Fine-tuning complete (run #{self.stats.finetune_runs})")
                
                # Log to structured logger
                structured_logger.log_finetune(
                    run_number=self.stats.finetune_runs,
                    dataset=dataset,
                    duration_minutes=duration_minutes,
                    success=True,
                )
                
                # Reload classifier with new weights
                self._classifier = None  # Force reload
            else:
                error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                logger.warning(f"  ⚠ Fine-tuning failed: {error_msg}")
                structured_logger.log_finetune(
                    run_number=self.stats.finetune_runs + 1,
                    dataset=dataset,
                    duration_minutes=duration_minutes,
                    success=False,
                    error=error_msg,
                )
                
        except subprocess.TimeoutExpired:
            logger.warning("  ⚠ Fine-tuning timed out")
            structured_logger.log_finetune(
                run_number=self.stats.finetune_runs + 1,
                dataset=dataset,
                duration_minutes=30.0,
                success=False,
                error="Timeout after 30 minutes",
            )
        except Exception as e:
            logger.warning(f"  ⚠ Fine-tuning error: {e}")
            structured_logger.log_error(
                cycle=self.stats.cycles_completed,
                error_type="finetune_error",
                message=str(e),
                recoverable=True,
            )

    def _send_notification(self, title: str, message: str):
        """Send desktop notification."""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":
                script = f'display notification "{message}" with title "{title}" sound name "Glass"'
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
        
        # Log start to structured logger
        structured_logger.log_start({
            "cycle_interval_seconds": self.cycle_interval,
            "images_per_cycle": self.images_per_cycle,
            "finetune_every_n": self.finetune_every_n,
            "initial_threshold": self.stats.current_threshold,
            "aggressive": hasattr(self, 'aggressive') and self.aggressive,
            "turbo": hasattr(self, 'turbo') and self.turbo,
        })
        
        # Check API status
        if self.check_api():
            logger.info("✓ AstroLens API is running")
        else:
            logger.warning("⚠ API not running - will analyze locally only")
        
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
    
    args = parser.parse_args()
    
    # Reset state if requested
    if args.reset:
        for f in [STATE_FILE, CANDIDATES_FILE]:
            if f.exists():
                f.unlink()
        logger.info("State reset - starting fresh")
    
    loop = DiscoveryLoop(
        cycle_interval_minutes=args.interval,
        images_per_cycle=args.images,
        initial_threshold=args.threshold,
        adaptive_threshold=not args.no_adaptive,
        finetune_every_n_cycles=args.finetune_every,
        aggressive=args.aggressive,
        turbo=args.turbo,
    )
    
    loop.run()


if __name__ == "__main__":
    main()

