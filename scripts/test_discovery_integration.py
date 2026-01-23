#!/usr/bin/env python3
"""
AstroLens Discovery Loop Integration Test

Tests the complete discovery loop workflow including:
- API connectivity
- Image downloading from sources
- Analysis pipeline (classification, OOD detection)
- Database updates
- Fine-tuning trigger
- Structured logging

Usage:
    # Quick test (2 mini-cycles)
    python scripts/test_discovery_integration.py --quick
    
    # Full test (5 cycles with fine-tuning)
    python scripts/test_discovery_integration.py --full
    
    # Continuous test (runs for specified minutes)
    python scripts/test_discovery_integration.py --duration 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR

# Test results directory
TEST_RESULTS_DIR = DATA_DIR / "test_results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Optional[Dict[str, Any]] = None


class DiscoveryIntegrationTest:
    """Integration test suite for discovery loop."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def log(self, message: str):
        """Log with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def record(self, name: str, passed: bool, duration_ms: float, message: str = "", details: Dict = None):
        """Record test result."""
        result = TestResult(name, passed, duration_ms, message, details)
        self.results.append(result)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        self.log(f"  {status}: {name} ({duration_ms:.0f}ms) {message}")
        
    def test_api_connectivity(self) -> bool:
        """Test 1: API is running and responsive."""
        start = time.time()
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
                r = client.get("/health")
                passed = r.status_code == 200
                self.record("API Connectivity", passed, (time.time() - start) * 1000,
                           f"Status: {r.status_code}")
                return passed
        except Exception as e:
            self.record("API Connectivity", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_image_upload(self) -> Optional[int]:
        """Test 2: Upload a test image."""
        start = time.time()
        try:
            import httpx
            from PIL import Image
            import io
            
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color=(100, 100, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            
            with httpx.Client(base_url="http://localhost:8000", timeout=30) as client:
                files = {"file": ("test_image.jpg", buffer, "image/jpeg")}
                r = client.post("/images", files=files)
                
                if r.status_code == 200:
                    image_id = r.json().get("id")
                    self.record("Image Upload", True, (time.time() - start) * 1000,
                               f"Image ID: {image_id}")
                    return image_id
                else:
                    self.record("Image Upload", False, (time.time() - start) * 1000,
                               f"Status: {r.status_code}")
                    return None
        except Exception as e:
            self.record("Image Upload", False, (time.time() - start) * 1000, str(e))
            return None
    
    def test_analysis_pipeline(self, image_id: int) -> bool:
        """Test 3: Run full analysis on an image."""
        start = time.time()
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:8000", timeout=60) as client:
                r = client.post(f"/analysis/full/{image_id}")
                
                if r.status_code == 200:
                    data = r.json()
                    self.record("Analysis Pipeline", True, (time.time() - start) * 1000,
                               details={
                                   "class_label": data.get("classification", {}).get("class_label"),
                                   "confidence": data.get("classification", {}).get("confidence"),
                                   "ood_score": data.get("anomaly", {}).get("ood_score"),
                               })
                    return True
                else:
                    self.record("Analysis Pipeline", False, (time.time() - start) * 1000,
                               f"Status: {r.status_code}")
                    return False
        except Exception as e:
            self.record("Analysis Pipeline", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_patch_update(self, image_id: int) -> bool:
        """Test 4: PATCH update works (critical for discovery loop)."""
        start = time.time()
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
                r = client.patch(
                    f"/images/{image_id}",
                    json={
                        "class_label": "test_class",
                        "class_confidence": 0.95,
                        "ood_score": 1.5,
                        "is_anomaly": False,
                    }
                )
                
                passed = r.status_code == 200
                self.record("PATCH Update", passed, (time.time() - start) * 1000,
                           f"Status: {r.status_code}, Response: {r.json() if passed else r.text[:100]}")
                return passed
        except Exception as e:
            self.record("PATCH Update", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_image_analyzed_status(self, image_id: int) -> bool:
        """Test 5: Verify image shows as analyzed in gallery."""
        start = time.time()
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
                r = client.get(f"/images/{image_id}")
                
                if r.status_code == 200:
                    data = r.json()
                    has_class = data.get("class_label") is not None
                    self.record("Image Analyzed Status", has_class, (time.time() - start) * 1000,
                               f"class_label: {data.get('class_label')}")
                    return has_class
                else:
                    self.record("Image Analyzed Status", False, (time.time() - start) * 1000,
                               f"Status: {r.status_code}")
                    return False
        except Exception as e:
            self.record("Image Analyzed Status", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_classifier_loading(self) -> bool:
        """Test 6: Classifier loads and works."""
        start = time.time()
        try:
            from inference.classifier import AstroClassifier
            from PIL import Image
            import io
            
            # Create test image
            img = Image.new('RGB', (224, 224), color=(100, 100, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # Save temporarily
            test_path = DATA_DIR / "test_classifier_image.jpg"
            img.save(test_path)
            
            classifier = AstroClassifier()
            result = classifier.classify(str(test_path))
            
            # Cleanup
            test_path.unlink(missing_ok=True)
            
            self.record("Classifier Loading", True, (time.time() - start) * 1000,
                       f"Class: {result.class_label}, Confidence: {result.confidence:.2%}")
            return True
        except Exception as e:
            self.record("Classifier Loading", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_ood_detection(self) -> bool:
        """Test 7: OOD detection works."""
        start = time.time()
        try:
            from inference.ood import OODDetector
            import numpy as np
            
            detector = OODDetector(use_ensemble=True)
            logits = np.array([2.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
            embedding = np.random.randn(768)
            
            result = detector.detect(logits, embedding)
            
            self.record("OOD Detection", True, (time.time() - start) * 1000,
                       f"Score: {result.ood_score:.3f}, Votes: {result.votes}")
            return True
        except Exception as e:
            self.record("OOD Detection", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_structured_logging(self) -> bool:
        """Test 8: Structured logger works."""
        start = time.time()
        try:
            from scripts.discovery_logger import get_discovery_logger
            
            logger = get_discovery_logger()
            logger.start_cycle(999, 2.0)
            logger.end_cycle(
                cycle_number=999,
                images_downloaded=10,
                images_analyzed=8,
                duplicates_skipped=2,
                anomalies_found=0,
                near_misses=1,
                flagged_for_review=3,
                highest_ood=1.5,
                threshold=2.0,
                sources={"sdss": 5, "ztf": 5},
            )
            
            summary = logger.get_summary()
            self.record("Structured Logging", True, (time.time() - start) * 1000,
                       f"Log file: {summary.get('log_file')}")
            return True
        except Exception as e:
            self.record("Structured Logging", False, (time.time() - start) * 1000, str(e))
            return False
    
    def test_discovery_mini_cycle(self) -> bool:
        """Test 9: Run a mini discovery cycle."""
        start = time.time()
        try:
            from scripts.discovery_loop import DiscoveryLoop
            
            # Create a very small test loop
            loop = DiscoveryLoop(
                cycle_interval_minutes=0,
                images_per_cycle=5,  # Very small
                finetune_every_n_cycles=100,  # Won't trigger
                initial_threshold=2.0,
            )
            
            # Run just download (no analysis to keep fast)
            files = loop.download_images()
            
            passed = len(files) > 0
            self.record("Mini Discovery Cycle", passed, (time.time() - start) * 1000,
                       f"Downloaded: {len(files)} images")
            return passed
        except Exception as e:
            self.record("Mini Discovery Cycle", False, (time.time() - start) * 1000, str(e))
            return False
    
    def cleanup_test_image(self, image_id: int):
        """Clean up test image."""
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
                client.delete(f"/images/{image_id}")
        except Exception:
            pass
    
    def run_quick(self) -> bool:
        """Run quick test suite."""
        self.log("\n" + "="*60)
        self.log("🧪 ASTROLENS QUICK INTEGRATION TEST")
        self.log("="*60)
        
        # Test 1: API
        if not self.test_api_connectivity():
            self.log("\n⚠️  API not running. Start it with: uvicorn api.main:app")
            return False
        
        # Test 2: Upload
        image_id = self.test_image_upload()
        
        # Test 3: Analysis
        if image_id:
            self.test_analysis_pipeline(image_id)
        
        # Test 4: PATCH (critical fix)
        if image_id:
            self.test_patch_update(image_id)
        
        # Test 5: Verify analyzed
        if image_id:
            self.test_image_analyzed_status(image_id)
            self.cleanup_test_image(image_id)
        
        # Test 6: Classifier
        self.test_classifier_loading()
        
        # Test 7: OOD
        self.test_ood_detection()
        
        # Test 8: Logging
        self.test_structured_logging()
        
        return self.print_summary()
    
    def run_full(self) -> bool:
        """Run full test suite including mini cycle."""
        self.log("\n" + "="*60)
        self.log("🧪 ASTROLENS FULL INTEGRATION TEST")
        self.log("="*60)
        
        # Run quick tests first
        if not self.run_quick():
            return False
        
        # Additional full tests
        self.log("\n--- Extended Tests ---")
        self.test_discovery_mini_cycle()
        
        return self.print_summary()
    
    def run_continuous(self, duration_minutes: float) -> bool:
        """Run continuous test for specified duration."""
        self.log("\n" + "="*60)
        self.log(f"🧪 ASTROLENS CONTINUOUS TEST ({duration_minutes} minutes)")
        self.log("="*60)
        
        from scripts.discovery_loop import DiscoveryLoop
        
        # Create loop with fast settings
        loop = DiscoveryLoop(
            cycle_interval_minutes=0,
            images_per_cycle=20,
            finetune_every_n_cycles=3,  # Trigger fine-tuning
            initial_threshold=2.0,
        )
        
        end_time = datetime.now().timestamp() + (duration_minutes * 60)
        cycles = 0
        
        self.log("\nStarting continuous test loop...")
        
        try:
            while datetime.now().timestamp() < end_time:
                cycle_start = time.time()
                
                try:
                    loop.process_cycle()
                    cycles += 1
                    
                    self.record(f"Cycle {cycles}", True, (time.time() - cycle_start) * 1000,
                               f"Threshold: {loop.stats.current_threshold:.3f}")
                except Exception as e:
                    self.record(f"Cycle {cycles}", False, (time.time() - cycle_start) * 1000, str(e))
                
                # Short pause between cycles
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.log("\nTest interrupted by user")
        
        self.log(f"\nCompleted {cycles} cycles")
        return self.print_summary()
    
    def print_summary(self) -> bool:
        """Print test summary and save results."""
        self.log("\n" + "="*60)
        self.log("📊 TEST SUMMARY")
        self.log("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        self.log(f"  Passed: {passed}/{total}")
        self.log(f"  Failed: {failed}/{total}")
        self.log(f"  Duration: {(datetime.now() - self.start_time).total_seconds():.1f}s")
        
        if failed > 0:
            self.log("\n  Failed tests:")
            for r in self.results:
                if not r.passed:
                    self.log(f"    - {r.name}: {r.message}")
        
        # Save results
        result_file = TEST_RESULTS_DIR / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            json.dump({
                "timestamp": self.start_time.isoformat(),
                "passed": passed,
                "failed": failed,
                "total": total,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration_ms": r.duration_ms,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in self.results
                ],
            }, f, indent=2)
        
        self.log(f"\n  Results saved: {result_file}")
        self.log("="*60)
        
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="AstroLens Integration Test")
    parser.add_argument("--quick", action="store_true", help="Quick test (API, upload, analysis)")
    parser.add_argument("--full", action="store_true", help="Full test including mini cycle")
    parser.add_argument("--duration", type=float, default=0, help="Continuous test duration in minutes")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    tester = DiscoveryIntegrationTest(verbose=not args.quiet)
    
    if args.duration > 0:
        success = tester.run_continuous(args.duration)
    elif args.full:
        success = tester.run_full()
    else:
        success = tester.run_quick()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

