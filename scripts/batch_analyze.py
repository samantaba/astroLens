#!/usr/bin/env python3
"""
Batch Analyze - Run OOD detection on all images in database.

This script:
1. Gets all images from the database
2. Runs OOD detection on each
3. Marks anomalies (is_anomaly=True) for high OOD scores
4. Does NOT download new images or fine-tune

Usage:
    python scripts/batch_analyze.py
    python scripts/batch_analyze.py --threshold 0.7
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from PIL import Image

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ood import OODDetector
from paths import DATA_DIR

API_BASE = "http://localhost:8000"
LOG_FILE = DATA_DIR / "batch_analyze.log"


def log(msg: str):
    """Log to console and file."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_all_images(limit: int = 5000) -> list:
    """Get all images from database."""
    try:
        response = httpx.get(
            f"{API_BASE}/images",
            params={"limit": limit},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log(f"Error fetching images: {e}")
        return []


def update_image(image_id: int, ood_score: float, is_anomaly: bool) -> bool:
    """Update image with OOD results."""
    try:
        response = httpx.patch(
            f"{API_BASE}/images/{image_id}",
            json={
                "ood_score": ood_score,
                "is_anomaly": is_anomaly,
            },
            timeout=10.0,
        )
        return response.status_code == 200
    except Exception as e:
        return False


def run_batch_analysis(threshold: float = 0.7, limit: int = 5000):
    """Run OOD detection on all images."""
    log("=" * 60)
    log("BATCH ANALYSIS - OOD Detection")
    log("=" * 60)
    log(f"Threshold: {threshold}")
    log(f"Limit: {limit}")
    log("")
    
    # Load OOD detector
    log("Loading OOD detector...")
    try:
        detector = OODDetector()
        log(f"✓ OOD detector loaded")
    except Exception as e:
        log(f"✗ Failed to load OOD detector: {e}")
        return
    
    # Get images
    log("Fetching images from database...")
    images = get_all_images(limit)
    log(f"Found {len(images)} images")
    
    if not images:
        log("No images to analyze")
        return
    
    # Analyze each image
    stats = {
        "total": len(images),
        "analyzed": 0,
        "anomalies": 0,
        "errors": 0,
        "skipped": 0,
    }
    
    log("")
    log("Analyzing images...")
    
    for i, img in enumerate(images):
        image_id = img.get("id")
        filepath = img.get("filepath", "")
        filename = img.get("filename", "")
        
        # Check file exists
        if not filepath or not Path(filepath).exists():
            stats["skipped"] += 1
            continue
        
        try:
            # Run OOD detection
            result = detector.predict(filepath)
            ood_score = float(result.get("ood_score", 0))
            is_anomaly = ood_score >= threshold
            
            # Update database
            if update_image(image_id, ood_score, is_anomaly):
                stats["analyzed"] += 1
                if is_anomaly:
                    stats["anomalies"] += 1
                    log(f"  ✓ [{i+1}/{len(images)}] ANOMALY: {filename} (score={ood_score:.3f})")
            else:
                stats["errors"] += 1
                
        except Exception as e:
            stats["errors"] += 1
            log(f"  ✗ Error analyzing {filename}: {e}")
        
        # Progress every 50 images
        if (i + 1) % 50 == 0:
            log(f"  Progress: {i+1}/{len(images)} ({stats['anomalies']} anomalies so far)")
    
    # Summary
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Total images: {stats['total']}")
    log(f"  Analyzed: {stats['analyzed']}")
    log(f"  Anomalies found: {stats['anomalies']}")
    log(f"  Skipped (no file): {stats['skipped']}")
    log(f"  Errors: {stats['errors']}")
    log(f"  Anomaly rate: {stats['anomalies']/max(1, stats['analyzed'])*100:.1f}%")
    log("")
    
    # Save stats
    stats_file = DATA_DIR / "batch_analyze_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            **stats,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    log(f"Stats saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch OOD Analysis")
    parser.add_argument("--threshold", "-t", type=float, default=0.7,
                        help="OOD score threshold for anomaly (default: 0.7)")
    parser.add_argument("--limit", "-l", type=int, default=5000,
                        help="Max images to analyze (default: 5000)")
    args = parser.parse_args()
    
    run_batch_analysis(args.threshold, args.limit)


if __name__ == "__main__":
    main()
