#!/usr/bin/env python3
"""
AstroLens Debug and Analysis Tool

Comprehensive debugging and analysis for:
- Catalog cross-reference validation
- OOD detection calibration
- Model performance analysis
- System health checks

Usage:
    python tests/debug_and_analyze.py --all           # Run all diagnostics
    python tests/debug_and_analyze.py --crossref      # Test cross-reference
    python tests/debug_and_analyze.py --ood           # Analyze OOD detection
    python tests/debug_and_analyze.py --model         # Check model performance
    python tests/debug_and_analyze.py --health        # System health check
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT.parent / "astrolens_artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


class CrossRefDiagnostics:
    """Diagnostics for catalog cross-reference."""
    
    def __init__(self):
        self.results_file = DATA_DIR / "cross_reference_results.json"
        self.log_file = DATA_DIR / "cross_reference_log.json"
    
    def run(self) -> Dict:
        """Run cross-reference diagnostics."""
        print_header("CATALOG CROSS-REFERENCE DIAGNOSTICS")
        
        results = {"issues": [], "recommendations": []}
        
        # Check results file
        if not self.results_file.exists():
            print_warning("No cross-reference results file found")
            results["issues"].append("No cross-reference results")
            results["recommendations"].append("Run cross-reference from Verify tab")
            return results
        
        with open(self.results_file, "r") as f:
            data = json.load(f)
        
        total = data.get("total_results", 0)
        items = data.get("results", [])
        
        print_info(f"Total results: {total}")
        
        if total == 0:
            print_warning("No cross-reference results yet")
            results["recommendations"].append("Run cross-reference on anomaly candidates")
            return results
        
        # Analyze results
        known = sum(1 for r in items if r.get("is_known"))
        unknown = sum(1 for r in items if not r.get("is_known"))
        errors = sum(1 for r in items if r.get("status") == "error")
        verified = sum(1 for r in items if r.get("human_verified"))
        
        print(f"  Known objects:    {known} ({known/total*100:.1f}%)")
        print(f"  Unknown objects:  {unknown} ({unknown/total*100:.1f}%)")
        print(f"  Errors:          {errors}")
        print(f"  Human verified:   {verified}")
        
        # Check for issues
        if unknown == total and total > 10:
            print_error("All results are 'unknown' - likely a problem!")
            results["issues"].append("All results unknown")
            
            # Analyze why
            print("\n  Investigating...")
            
            # Check search radius
            radii = [r.get("query_radius_arcsec", 0) for r in items[:10]]
            avg_radius = sum(radii) / len(radii) if radii else 0
            
            if avg_radius < 30:
                print_warning(f"  Search radius too small: {avg_radius}″")
                results["issues"].append(f"Search radius {avg_radius}″ is too small")
                results["recommendations"].append("Increase search radius to 60-120 arcseconds")
            
            # Check coordinates
            sample = items[0] if items else {}
            ra = sample.get("query_ra", 0)
            dec = sample.get("query_dec", 0)
            print_info(f"  Sample coordinates: RA={ra}, Dec={dec}")
            
            if ra == 0 and dec == 0:
                print_error("  Coordinates are 0,0 - extraction failed!")
                results["issues"].append("Coordinate extraction failed")
                results["recommendations"].append("Check filename format includes ra_dec pattern")
        
        elif known == 0 and unknown > 0:
            print_warning("No known objects found yet")
            results["recommendations"].append("Try larger search radius or check coordinates")
        
        elif known > unknown:
            print_success(f"Most objects ({known}/{total}) matched to catalogs")
        
        # Check for duplicate queries
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                log_data = json.load(f)
            
            queries = log_data.get("queries", [])
            print(f"\n  Total catalog queries: {len(queries)}")
            
            # Check success rate
            successes = sum(1 for q in queries if q.get("success"))
            if queries:
                success_rate = successes / len(queries) * 100
                print(f"  Query success rate: {success_rate:.1f}%")
                
                if success_rate < 90:
                    print_warning("  Low query success rate")
                    results["issues"].append(f"Query success rate only {success_rate:.1f}%")
        
        # Test live query
        print("\n  Testing live catalog query...")
        self._test_live_query(results)
        
        return results
    
    def _test_live_query(self, results: Dict):
        """Test a live catalog query."""
        try:
            from catalog.cross_reference import CatalogCrossReference
            
            # Test with known object: Virgo cluster region
            xref = CatalogCrossReference(search_radius_arcsec=120)
            
            print("  Querying SIMBAD for Virgo cluster region...")
            start = time.time()
            matches = xref.query_simbad(187.5, 12.5)  # Near M87
            duration = time.time() - start
            
            if matches:
                print_success(f"  SIMBAD: Found {len(matches)} objects in {duration:.1f}s")
                print(f"    First match: {matches[0].object_name} ({matches[0].object_type})")
            else:
                print_warning(f"  SIMBAD: No matches found (query took {duration:.1f}s)")
                results["issues"].append("SIMBAD query returned no results for known region")
                results["recommendations"].append("Check network connectivity and SIMBAD service status")
            
            print("  Querying VizieR/SDSS...")
            start = time.time()
            vizier_matches = xref.query_vizier(180.0, 30.0)
            duration = time.time() - start
            
            if vizier_matches:
                print_success(f"  VizieR: Found {len(vizier_matches)} objects in {duration:.1f}s")
            else:
                print_info(f"  VizieR: No matches (may be outside SDSS coverage)")
            
        except Exception as e:
            print_error(f"  Query test failed: {e}")
            results["issues"].append(f"Live query failed: {e}")


class OODDiagnostics:
    """Diagnostics for OOD detection."""
    
    def __init__(self):
        self.state_file = DATA_DIR / "discovery_state.json"
    
    def run(self) -> Dict:
        """Run OOD detection diagnostics."""
        print_header("OOD DETECTION DIAGNOSTICS")
        
        results = {"issues": [], "recommendations": []}
        
        # Load state
        if not self.state_file.exists():
            print_warning("No discovery state file found")
            results["recommendations"].append("Run discovery loop to generate state")
            return results
        
        with open(self.state_file, "r") as f:
            state = json.load(f)
        
        threshold = state.get("current_threshold", 0)
        highest_ood = state.get("highest_ood_score", 0)
        anomalies = state.get("anomalies_found", 0)
        
        print(f"  Current threshold:   {threshold:.3f}")
        print(f"  Highest OOD score:   {highest_ood:.3f}")
        print(f"  Anomalies found:     {anomalies}")
        
        # Check for miscalibration
        if highest_ood > 0 and threshold > highest_ood * 2:
            print_error("Threshold is much higher than highest OOD score!")
            print_info(f"  Ratio: threshold/highest_ood = {threshold/highest_ood:.1f}x")
            results["issues"].append("OOD threshold miscalibrated")
            results["recommendations"].append("Run OOD calibration from Verify tab")
            results["recommendations"].append(f"Or manually set threshold to ~{highest_ood * 1.2:.2f}")
        elif highest_ood == 0:
            print_warning("No OOD scores recorded yet")
            results["recommendations"].append("Run discovery to analyze images")
        else:
            print_success("Threshold appears properly calibrated")
        
        # Analyze training history
        training_history = state.get("training_history", [])
        if training_history:
            print(f"\n  Training runs: {len(training_history)}")
            
            # Check for improvement
            galaxy10_runs = [r for r in training_history if r.get("dataset") == "galaxy10"]
            anomaly_runs = [r for r in training_history if r.get("dataset") == "anomalies"]
            
            if galaxy10_runs:
                last_g10 = galaxy10_runs[-1]
                print(f"  Galaxy10 accuracy: {last_g10.get('accuracy_after', 0):.1%}")
            
            if anomaly_runs:
                last_anom = anomaly_runs[-1]
                if last_anom.get("epochs_completed", 0) == 0:
                    print_warning("  Anomaly training completing 0 epochs (early exit)")
                    results["issues"].append("Anomaly training not learning")
                    results["recommendations"].append("Review anomaly dataset quality")
        
        # Test OOD detector
        print("\n  Testing OOD detector...")
        self._test_ood_detector(results)
        
        return results
    
    def _test_ood_detector(self, results: Dict):
        """Test the OOD detector."""
        try:
            from inference.ood import OODDetector
            import numpy as np
            
            detector = OODDetector()
            
            # Test with confident prediction
            confident_logits = np.array([10.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01])
            result_confident = detector.detect(confident_logits)
            
            print(f"  Confident prediction: OOD={result_confident.ood_score:.3f}, votes={result_confident.votes}")
            
            # Test with uncertain prediction
            uncertain_logits = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1])
            result_uncertain = detector.detect(uncertain_logits)
            
            print(f"  Uncertain prediction: OOD={result_uncertain.ood_score:.3f}, votes={result_uncertain.votes}")
            
            if result_uncertain.ood_score > result_confident.ood_score:
                print_success("  OOD detector correctly ranks uncertain higher")
            else:
                print_warning("  OOD detector may not be working correctly")
                results["issues"].append("OOD detector ranking incorrect")
            
        except Exception as e:
            print_error(f"  OOD test failed: {e}")
            results["issues"].append(f"OOD detector error: {e}")


class ModelDiagnostics:
    """Diagnostics for model performance."""
    
    def __init__(self):
        self.weights_dir = ARTIFACTS_DIR / "weights"
        self.state_file = DATA_DIR / "discovery_state.json"
    
    def run(self) -> Dict:
        """Run model diagnostics."""
        print_header("MODEL PERFORMANCE DIAGNOSTICS")
        
        results = {"issues": [], "recommendations": []}
        
        # Check weights
        print("  Checking model weights...")
        
        weights_path = self.weights_dir / "vit_astrolens"
        if not weights_path.exists():
            print_warning("No fine-tuned weights found")
            results["recommendations"].append("Run fine-tuning to improve model")
        else:
            config_file = weights_path / "config.json"
            if config_file.exists():
                print_success(f"  Found fine-tuned model at {weights_path}")
                
                # Check evaluation report
                eval_file = weights_path / "evaluation_report.json"
                if eval_file.exists():
                    with open(eval_file, "r") as f:
                        eval_data = json.load(f)
                    
                    accuracy = eval_data.get("accuracy", 0)
                    print(f"  Evaluation accuracy: {accuracy:.1%}")
                    
                    if accuracy < 0.7:
                        print_warning("  Accuracy below 70%")
                        results["issues"].append(f"Model accuracy only {accuracy:.1%}")
                        results["recommendations"].append("Consider more training epochs or data")
        
        # Check training history
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
        # Get Galaxy10 metrics from training history (the primary metrics)
        training_history = state.get("training_history", [])
        galaxy10_runs = [r for r in training_history if r.get("dataset") == "galaxy10"]
        
        if galaxy10_runs:
            first_g10 = galaxy10_runs[0]
            last_g10 = galaxy10_runs[-1]
            initial = first_g10.get("accuracy_before", 0)
            accuracy = last_g10.get("accuracy_after", 0)
            improvement = ((accuracy - initial) / initial * 100) if initial > 0 else 0
            
            print(f"\n  Galaxy10 Model Performance:")
            print(f"    Initial accuracy:  {initial:.1%}")
            print(f"    Current accuracy:  {accuracy:.1%}")
            print(f"    Total improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                print_success(f"  Model improved by {improvement:.1f}%")
            elif accuracy < 0.7:
                print_warning("  Model accuracy below 70%")
                results["issues"].append(f"Galaxy10 accuracy only {accuracy:.1%}")
        else:
            print_info("  No Galaxy10 training runs yet")
        
        # Show anomaly training status
        anomaly_runs = [r for r in training_history if r.get("dataset") == "anomalies"]
        if anomaly_runs:
            last_anom = anomaly_runs[-1]
            anom_acc = last_anom.get("accuracy_after", 0)
            epochs = last_anom.get("epochs_completed", 0)
            print(f"\n  Anomaly Training Status:")
            print(f"    Accuracy: {anom_acc:.1%} (different scale)")
            print(f"    Last epochs: {epochs}")
            if epochs == 0 and len(anomaly_runs) > 1:
                print_warning("  Anomaly training keeps exiting early")
                results["issues"].append("Anomaly training not learning")
        
        # Test model inference
        print("\n  Testing model inference...")
        self._test_inference(results)
        
        return results
    
    def _test_inference(self, results: Dict):
        """Test model inference."""
        try:
            # Check if we have sample images
            images_dir = DATA_DIR / "images"
            if not images_dir.exists():
                print_info("  No images directory found")
                return
            
            sample_images = list(images_dir.glob("*.jpg"))[:3]
            if not sample_images:
                print_info("  No sample images found")
                return
            
            from inference.classifier import AstroClassifier
            
            weights_path = self.weights_dir / "vit_astrolens"
            weights = str(weights_path) if weights_path.exists() else None
            
            classifier = AstroClassifier(weights_path=weights)
            print(f"  Loaded classifier with {classifier.num_classes} classes")
            print(f"  Device: {classifier.device}")
            
            # Test inference speed
            times = []
            for img_path in sample_images:
                try:
                    start = time.time()
                    result = classifier.classify(str(img_path))
                    times.append(time.time() - start)
                    print(f"    {img_path.name}: {result.class_label} ({result.confidence:.1%})")
                except Exception as e:
                    print_error(f"    {img_path.name}: {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                print(f"\n  Average inference time: {avg_time*1000:.0f}ms")
                
                if avg_time > 1.0:
                    print_warning("  Inference is slow (>1s per image)")
                    results["recommendations"].append("Consider GPU acceleration")
            
        except Exception as e:
            print_error(f"  Inference test failed: {e}")
            results["issues"].append(f"Inference error: {e}")


class HealthCheck:
    """System health checks."""
    
    def run(self) -> Dict:
        """Run system health checks."""
        print_header("SYSTEM HEALTH CHECK")
        
        results = {"issues": [], "recommendations": []}
        
        # Check API
        print("  Checking API...")
        self._check_api(results)
        
        # Check disk space
        print("\n  Checking disk space...")
        self._check_disk(results)
        
        # Check dependencies
        print("\n  Checking dependencies...")
        self._check_dependencies(results)
        
        return results
    
    def _check_api(self, results: Dict):
        """Check API status."""
        try:
            import httpx
            
            response = httpx.get("http://localhost:8000/health", timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"  API running: {data.get('status')}")
                print(f"    ML model loaded: {data.get('ml_model_loaded')}")
                print(f"    LLM available: {data.get('llm_available')}")
            else:
                print_error(f"  API returned {response.status_code}")
                results["issues"].append("API not responding correctly")
                
        except Exception as e:
            print_warning(f"  API not running: {e}")
            results["recommendations"].append("Start API with: python -m api.main")
    
    def _check_disk(self, results: Dict):
        """Check disk space."""
        import shutil
        
        total, used, free = shutil.disk_usage(ARTIFACTS_DIR)
        
        free_gb = free / (1024 ** 3)
        used_gb = used / (1024 ** 3)
        
        print(f"  Artifacts directory: {ARTIFACTS_DIR}")
        print(f"  Free space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print_warning("  Low disk space!")
            results["issues"].append(f"Only {free_gb:.1f} GB free")
            results["recommendations"].append("Clean up old downloads/images")
        else:
            print_success(f"  Disk space OK ({free_gb:.1f} GB free)")
        
        # Check artifacts size
        if ARTIFACTS_DIR.exists():
            size = sum(f.stat().st_size for f in ARTIFACTS_DIR.rglob("*") if f.is_file())
            size_gb = size / (1024 ** 3)
            print(f"  Artifacts size: {size_gb:.2f} GB")
    
    def _check_dependencies(self, results: Dict):
        """Check Python dependencies."""
        required = ["torch", "transformers", "fastapi", "httpx", "numpy", "PIL"]
        
        for pkg in required:
            try:
                if pkg == "PIL":
                    import PIL
                else:
                    __import__(pkg)
                print_success(f"  {pkg} installed")
            except ImportError:
                print_error(f"  {pkg} not installed")
                results["issues"].append(f"Missing dependency: {pkg}")
                results["recommendations"].append(f"Install with: pip install {pkg}")


def main():
    parser = argparse.ArgumentParser(description="AstroLens Debug and Analysis")
    
    parser.add_argument("--all", "-a", action="store_true", help="Run all diagnostics")
    parser.add_argument("--crossref", "-c", action="store_true", help="Cross-reference diagnostics")
    parser.add_argument("--ood", "-o", action="store_true", help="OOD detection diagnostics")
    parser.add_argument("--model", "-m", action="store_true", help="Model diagnostics")
    parser.add_argument("--health", action="store_true", dest="health_check", help="Health check")
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not any([args.all, args.crossref, args.ood, args.model, args.health_check]):
        args.all = True
    
    all_results = {}
    
    if args.all or args.health_check:
        all_results["health"] = HealthCheck().run()
    
    if args.all or args.crossref:
        all_results["crossref"] = CrossRefDiagnostics().run()
    
    if args.all or args.ood:
        all_results["ood"] = OODDiagnostics().run()
    
    if args.all or args.model:
        all_results["model"] = ModelDiagnostics().run()
    
    # Summary
    print_header("SUMMARY")
    
    all_issues = []
    all_recommendations = []
    
    for category, results in all_results.items():
        all_issues.extend(results.get("issues", []))
        all_recommendations.extend(results.get("recommendations", []))
    
    if all_issues:
        print(f"{Colors.RED}Issues Found ({len(all_issues)}):{Colors.END}")
        for issue in all_issues:
            print(f"  • {issue}")
    else:
        print_success("No issues found!")
    
    if all_recommendations:
        print(f"\n{Colors.YELLOW}Recommendations:{Colors.END}")
        for rec in all_recommendations:
            print(f"  → {rec}")
    
    print()


if __name__ == "__main__":
    main()
