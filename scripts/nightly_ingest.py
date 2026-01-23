#!/usr/bin/env python3
"""
Nightly Image Ingestion Script

Downloads astronomical images from public sources and processes them through AstroLens.

Sources:
- SDSS (Sloan Digital Sky Survey) random galaxies
- ZTF (Zwicky Transient Facility) transients via ALeRCE + ESO cutouts
- NASA APOD (Astronomy Picture of the Day)
- ESO DSS (European Southern Observatory Digitized Sky Survey)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from paths import DOWNLOADS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = os.environ.get("ASTROLENS_API", "http://localhost:8000")

# SSL context for downloads (bypass cert issues on macOS)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_sdss_galaxies(count: int = 10, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download galaxy images from SDSS SkyServer using random positions.
    
    Args:
        count: Number of images to download
        output_dir: Output directory
        on_progress: Optional callback(current, total) called after each download
    """
    logger.info(f"Downloading {count} galaxies from SDSS...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "sdss"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    attempts = 0
    max_attempts = count * 3
    
    while len(downloaded) < count and attempts < max_attempts:
        attempts += 1
        
        # Random position in SDSS footprint (North Galactic Cap)
        ra = random.uniform(120, 240)
        dec = random.uniform(5, 55)
        
        img_url = (
            f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.4&width=256&height=256"
        )
        
        try:
            img_req = urllib.request.Request(img_url)
            img_req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(img_req, context=ssl_context, timeout=15) as img_response:
                img_data = img_response.read()
            
            if len(img_data) < 3000:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"sdss_{idx:04d}_ra{ra:.1f}_dec{dec:.1f}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if idx % 10 == 0 or idx <= 5:
                logger.info(f"  ✓ {idx}/{count}")
            
            # Emit progress callback
            if on_progress:
                on_progress(idx, count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)}/{count} SDSS images")
    return downloaded


def download_nasa_apod(days: int = 7, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download recent NASA Astronomy Picture of the Day images.
    
    Args:
        days: Number of days to download
        output_dir: Output directory
        on_progress: Optional callback(current, total) called after each download
    """
    api_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
    logger.info(f"Downloading last {days} days of NASA APOD...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "apod"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}&date={date}"
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as response:
                data = json.loads(response.read())
            
            if data.get("media_type") != "image":
                # Emit progress even for skipped items
                if on_progress:
                    on_progress(i + 1, days)
                continue
            
            img_url = data.get("hdurl") or data.get("url")
            if not img_url:
                if on_progress:
                    on_progress(i + 1, days)
                continue
            
            img_req = urllib.request.Request(img_url)
            with urllib.request.urlopen(img_req, context=ssl_context, timeout=30) as img_response:
                img_data = img_response.read()
            
            ext = img_url.split(".")[-1][:4]
            filename = output_dir / f"apod_{date}.{ext}"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            logger.info(f"  ✓ {filename.name}")
            
            if on_progress:
                on_progress(i + 1, days)
            
        except Exception as e:
            logger.debug(f"  APOD {date}: {e}")
            if on_progress:
                on_progress(i + 1, days)
    
    logger.info(f"  ✅ Complete: {len(downloaded)} APOD images")
    return downloaded


def download_ztf_alerts(count: int = 20, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download transient sky images using SDSS cutouts for known transient locations.
    
    Args:
        count: Number of images to download
        output_dir: Output directory
        on_progress: Optional callback(current, total) called after each download
    """
    logger.info(f"Downloading {count} transient region images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "ztf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    transient_coords = []
    
    # Try to get real transient coordinates from ALeRCE
    classes = ["SNIa", "SNII", "SNIbc", "SLSN"]
    
    for class_name in classes:
        if len(transient_coords) >= count * 2:
            break
        
        try:
            query_url = f"https://api.alerce.online/ztf/v1/objects?classifier=lc_classifier&class_name={class_name}&page_size=50"
            
            req = urllib.request.Request(query_url)
            req.add_header("User-Agent", "AstroLens/1.0")
            req.add_header("Accept", "application/json")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                data = json.loads(response.read())
            
            for obj in data.get("items", []):
                ra = obj.get("meanra")
                dec = obj.get("meandec")
                oid = obj.get("oid", "unknown")
                
                if ra and dec:
                    transient_coords.append((ra, dec, oid, class_name))
                    
        except Exception:
            continue
    
    logger.info(f"  Found {len(transient_coords)} transient coordinates")
    
    # Download images for coordinates within SDSS footprint, or use random SDSS positions
    attempts = 0
    max_attempts = count * 3
    
    while len(downloaded) < count and attempts < max_attempts:
        attempts += 1
        
        if transient_coords and attempts <= len(transient_coords):
            # Use real transient coordinate
            ra, dec, oid, class_name = transient_coords[attempts - 1]
            
            # Check if in SDSS footprint
            if not (120 <= ra <= 240 and 5 <= dec <= 55):
                # Outside SDSS, use random position in footprint instead
                ra = random.uniform(120, 240)
                dec = random.uniform(5, 55)
                oid = f"region_{attempts}"
        else:
            # Fall back to random SDSS position
            ra = random.uniform(120, 240)
            dec = random.uniform(5, 55)
            oid = f"region_{attempts}"
            class_name = "transient_region"
        
        # Download SDSS cutout
        img_url = (
            f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.4&width=256&height=256"
        )
        
        try:
            img_req = urllib.request.Request(img_url)
            img_req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(img_req, context=ssl_context, timeout=10) as img_resp:
                img_data = img_resp.read()
            
            if len(img_data) < 3000:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"transient_{idx:04d}_{oid}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if idx % 5 == 0 or idx <= 3:
                logger.info(f"  ✓ {idx}/{count}")
            
            # Emit progress callback
            if on_progress:
                on_progress(idx, count)
                
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} transient images")
    return downloaded


def download_eso_dss(count: int = 20, output_dir: Path = None) -> List[Path]:
    """
    Download random sky survey images from ESO DSS.
    
    Good for general sky backgrounds and testing.
    """
    logger.info(f"Downloading {count} ESO DSS images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "eso"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    attempts = 0
    max_attempts = count * 2
    
    while len(downloaded) < count and attempts < max_attempts:
        attempts += 1
        
        # Random sky position
        ra = random.uniform(0, 360)
        dec = random.uniform(-60, 60)
        
        eso_url = f"https://archive.eso.org/dss/dss/image?ra={ra}&dec={dec}&x=10&y=10&mime-type=image/jpeg"
        
        try:
            req = urllib.request.Request(eso_url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                img_data = resp.read()
            
            if len(img_data) < 500:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"eso_{idx:04d}_ra{ra:.1f}_dec{dec:.1f}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if idx % 10 == 0 or idx <= 3:
                logger.info(f"  ✓ {idx}/{count}")
                
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} ESO images")
    return downloaded


def check_api_running() -> bool:
    """Check if the AstroLens API is running."""
    try:
        req = urllib.request.Request(f"{API_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def upload_to_astrolens(
    files: List[Path],
    analyze: bool = True,
    alert: bool = False,
) -> dict:
    """Upload images to AstroLens API."""
    
    if not check_api_running():
        logger.warning("AstroLens API not running - skipping upload")
        return {"uploaded": 0, "analyzed": 0, "anomalies": 0}
    
    results = {"uploaded": 0, "analyzed": 0, "anomalies": 0}
    
    with httpx.Client(base_url=API_URL, timeout=60.0) as client:
        for filepath in files:
            try:
                with open(filepath, "rb") as f:
                    response = client.post(
                        "/images",
                        files={"file": (filepath.name, f)},
                    )
                
                if response.status_code == 200:
                    results["uploaded"] += 1
                    image_data = response.json()
                    image_id = image_data.get("id")
                    
                    if analyze and image_id:
                        try:
                            analysis = client.post(f"/analysis/full/{image_id}")
                            if analysis.status_code == 200:
                                results["analyzed"] += 1
                                data = analysis.json()
                                if data.get("anomaly", {}).get("is_anomaly"):
                                    results["anomalies"] += 1
                                    if alert:
                                        send_alert(
                                            "Anomaly Detected",
                                            f"{filepath.name}: {data.get('classification', {}).get('class_label', 'Unknown')}"
                                        )
                        except Exception:
                            pass
                            
            except Exception as e:
                logger.debug(f"Upload failed {filepath.name}: {e}")
    
    logger.info(f"  Uploaded: {results['uploaded']}, Analyzed: {results['analyzed']}, Anomalies: {results['anomalies']}")
    return results


def send_alert(title: str, message: str):
    """Send desktop notification."""
    import platform
    system = platform.system()
    
    try:
        if system == "Darwin":
            import subprocess
            script = f'display notification "{message}" with title "{title}" sound name "Glass"'
            subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        elif system == "Linux":
            import subprocess
            subprocess.run(["notify-send", title, message], check=False, capture_output=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Download astronomical images")
    parser.add_argument("--source", choices=["sdss", "ztf", "apod", "eso", "all"], default="all")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--upload", action="store_true", help="Upload to AstroLens")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--alert", action="store_true", help="Send anomaly alerts")
    
    args = parser.parse_args()
    
    all_downloads = []
    
    if args.source in ["sdss", "all"]:
        all_downloads.extend(download_sdss_galaxies(args.count))
    
    if args.source in ["ztf", "all"]:
        all_downloads.extend(download_ztf_alerts(args.count))
    
    if args.source in ["apod", "all"]:
        all_downloads.extend(download_nasa_apod(min(args.count, 30)))
    
    if args.source == "eso":
        all_downloads.extend(download_eso_dss(args.count))
    
    logger.info(f"Total downloaded: {len(all_downloads)} images")
    
    if args.upload and all_downloads:
        upload_to_astrolens(all_downloads, analyze=args.analyze, alert=args.alert)


if __name__ == "__main__":
    main()
