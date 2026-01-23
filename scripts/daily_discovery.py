#!/usr/bin/env python3
"""
AstroLens Daily Discovery Script

Automates the discovery workflow:
1. Download fresh images from astronomical sources
2. Upload to AstroLens for analysis
3. Report any anomalies found
4. (Optional) Trigger model retraining

Run daily via cron or manually:
    python scripts/daily_discovery.py

Options:
    --sources sdss,apod        Sources to download from
    --count 50                 Images per source
    --analyze                  Analyze after upload
    --retrain                  Trigger model retraining if anomalies found
    --quiet                    Less output
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DOWNLOADS_DIR
from scripts.nightly_ingest import (
    download_sdss_galaxies,
    download_nasa_apod,
    download_ztf_alerts,
    upload_to_astrolens,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_discovery(
    sources: list,
    count_per_source: int,
    analyze: bool = True,
    retrain_threshold: int = 5,
    quiet: bool = False,
):
    """
    Run the daily discovery workflow.
    
    Args:
        sources: List of sources to download from
        count_per_source: Number of images per source
        analyze: Whether to analyze after upload
        retrain_threshold: Number of anomalies to trigger retraining
        quiet: Less output
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads_dir = DOWNLOADS_DIR / timestamp
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    if not quiet:
        print("=" * 60)
        print(f"🔭 AstroLens Daily Discovery")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    # Track results
    results = {
        "timestamp": timestamp,
        "sources": {},
        "total_downloaded": 0,
        "total_uploaded": 0,
        "total_analyzed": 0,
        "anomalies_found": 0,
        "anomaly_ids": [],
    }
    
    # Download from each source
    all_files = []
    
    source_funcs = {
        "sdss": lambda: download_sdss_galaxies(count_per_source, downloads_dir / "sdss"),
        "apod": lambda: download_nasa_apod(min(count_per_source, 30), downloads_dir / "apod"),
        "ztf": lambda: download_ztf_alerts(count_per_source, downloads_dir / "ztf"),
    }
    
    for source in sources:
        source = source.strip().lower()
        if source not in source_funcs:
            logger.warning(f"Unknown source: {source}")
            continue
        
        if not quiet:
            print(f"\n📥 Downloading from {source.upper()}...")
        
        try:
            files = source_funcs[source]()
            results["sources"][source] = len(files)
            results["total_downloaded"] += len(files)
            all_files.extend(files)
            
            if not quiet:
                print(f"   ✓ {len(files)} images")
                
        except Exception as e:
            logger.error(f"Failed to download from {source}: {e}")
            results["sources"][source] = 0
    
    if not all_files:
        logger.warning("No images downloaded!")
        return results
    
    # Upload and analyze
    if not quiet:
        print(f"\n📤 Uploading {len(all_files)} images...")
    
    try:
        upload_results = upload_to_astrolens(
            all_files,
            analyze=analyze,
            alert=True,
        )
        
        results["total_uploaded"] = upload_results.get("uploaded", 0)
        results["total_analyzed"] = upload_results.get("analyzed", 0)
        results["anomalies_found"] = upload_results.get("anomalies", 0)
        
        if not quiet:
            print(f"   ✓ Uploaded: {results['total_uploaded']}")
            print(f"   ✓ Analyzed: {results['total_analyzed']}")
            print(f"   ⚡ Anomalies: {results['anomalies_found']}")
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.info("Make sure the API is running: uvicorn api.main:app --port 8000")
    
    # Summary
    if not quiet:
        print("\n" + "=" * 60)
        print("📊 Discovery Summary")
        print("=" * 60)
        print(f"   Downloaded: {results['total_downloaded']} images")
        print(f"   Analyzed: {results['total_analyzed']} images")
        print(f"   Anomalies: {results['anomalies_found']} found")
        
        if results["anomalies_found"] > 0:
            print("\n   🔥 Check the AstroLens app to review anomalies!")
        
        if results["anomalies_found"] >= retrain_threshold:
            print(f"\n   💡 Tip: You have {results['anomalies_found']} anomalies.")
            print("      Consider reviewing and adding to training data.")
    
    # Save results
    results_file = downloads_dir / "discovery_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run daily discovery workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="sdss",
        help="Comma-separated sources: sdss,apod,ztf (default: sdss)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Images per source (default: 50)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip analysis (just download and upload)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less output",
    )
    
    args = parser.parse_args()
    
    sources = [s.strip() for s in args.sources.split(",")]
    
    run_discovery(
        sources=sources,
        count_per_source=args.count,
        analyze=not args.no_analyze,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

