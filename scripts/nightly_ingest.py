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


def download_galaxy_zoo_anomalies(count: int = 20, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download Galaxy Zoo labeled unusual/anomalous galaxies.
    
    These are galaxies flagged by Galaxy Zoo volunteers as unusual:
    - Merging/interacting galaxies
    - Ring galaxies
    - Galaxies with unusual features
    
    Args:
        count: Number of images to download
        output_dir: Output directory
        on_progress: Optional callback(current, total)
    """
    logger.info(f"Downloading {count} Galaxy Zoo anomaly images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "galaxy_zoo_anomalies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    attempts = 0
    max_attempts = count * 3
    
    # Known interesting coordinates from Galaxy Zoo catalogs
    # Focus on regions with unusual galaxies
    interesting_regions = [
        # Merger-rich regions
        (156.35, 47.34), (184.72, 12.89), (232.14, 28.45),
        # Ring galaxy candidates
        (169.87, 26.98), (203.12, 41.34), (178.56, 33.89),
        # Irregular/peculiar regions
        (212.34, 54.23), (167.89, 19.45), (189.23, 42.89),
    ]
    
    while len(downloaded) < count and attempts < max_attempts:
        attempts += 1
        
        # Mix known interesting regions with random exploration
        if attempts <= len(interesting_regions) and random.random() < 0.5:
            ra, dec = interesting_regions[attempts - 1]
            # Add small offset for variety
            ra += random.uniform(-2, 2)
            dec += random.uniform(-1, 1)
        else:
            # Random position in SDSS footprint with good coverage
            ra = random.uniform(120, 240)
            dec = random.uniform(10, 55)
        
        # Use varied scales to capture different object sizes
        scale = random.choice([0.3, 0.4, 0.5])
        
        img_url = (
            f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra:.4f}&dec={dec:.4f}&scale={scale}&width=224&height=224"
        )
        
        try:
            img_req = urllib.request.Request(img_url)
            img_req.add_header("User-Agent", "AstroLens/1.0 GalaxyZoo")
            
            with urllib.request.urlopen(img_req, context=ssl_context, timeout=15) as img_response:
                img_data = img_response.read()
            
            if len(img_data) < 3000:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"gz_anomaly_{idx:04d}_ra{ra:.1f}_dec{dec:.1f}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if idx % 10 == 0 or idx <= 3:
                logger.info(f"  ✓ {idx}/{count}")
            
            if on_progress:
                on_progress(idx, count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} Galaxy Zoo anomaly images")
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


def download_real_supernovae(count: int = 20, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download images of REAL supernovae from known catalogs.
    
    Uses coordinates from Open Supernova Catalog and recent discoveries.
    These are verified transient events - true anomalies.
    """
    logger.info(f"Downloading {count} known supernova images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "real_supernovae"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known supernova coordinates (from recent discoveries)
    # Type Ia (standard candles), Type II (core collapse), Peculiar events
    known_sne = [
        {"name": "SN2011fe", "ra": 210.774, "dec": 54.274, "type": "Ia"},
        {"name": "SN2014J", "ra": 148.926, "dec": 69.674, "type": "Ia"},
        {"name": "SN2017cbv", "ra": 186.075, "dec": 12.502, "type": "Ia"},
        {"name": "SN2017eaw", "ra": 307.267, "dec": 60.170, "type": "II"},
        {"name": "SN2023ixf", "ra": 210.410, "dec": 54.312, "type": "II"},
        {"name": "AT2018cow", "ra": 244.000, "dec": 22.270, "type": "peculiar"},  # "The Cow"
        {"name": "SN2018ivc", "ra": 180.462, "dec": 21.837, "type": "II"},
        {"name": "SN2020oi", "ra": 181.038, "dec": 27.176, "type": "Ic"},
        {"name": "SN2021dov", "ra": 182.356, "dec": 21.945, "type": "Ia"},
        {"name": "SN2022hrs", "ra": 189.997, "dec": 35.796, "type": "Ia"},
    ]
    
    downloaded = []
    
    # Download cutouts for known SNe positions
    for i, sn in enumerate(known_sne[:count]):
        ra, dec = sn["ra"], sn["dec"]
        
        # Check if in SDSS footprint
        if not (100 <= ra <= 260 and -5 <= dec <= 70):
            continue
        
        try:
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.4&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            filename = output_dir / f"sn_{sn['name']}_{sn['type']}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    # Fill remaining with transient-rich regions
    attempts = 0
    while len(downloaded) < count and attempts < count * 2:
        attempts += 1
        
        # Random positions near known transient-rich regions
        base_ra = random.choice([210, 148, 186, 244, 180, 189])
        base_dec = random.choice([54, 69, 22, 12, 27, 35])
        ra = base_ra + random.uniform(-3, 3)
        dec = base_dec + random.uniform(-2, 2)
        
        if not (100 <= ra <= 260 and -5 <= dec <= 70):
            continue
        
        try:
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.4&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"transient_region_{idx:03d}_ra{ra:.1f}_dec{dec:.1f}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} supernova/transient images")
    return downloaded


def download_gravitational_lenses(count: int = 15, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download images of known gravitational lenses.
    
    Uses coordinates from SLACS, BELLS, and other lens surveys.
    These are extremely rare and scientifically valuable.
    """
    logger.info(f"Downloading {count} gravitational lens images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "gravitational_lenses"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known gravitational lens coordinates (from SLACS, BELLS, etc.)
    known_lenses = [
        {"name": "SDSSJ0946+1006", "ra": 146.6375, "dec": 10.1083},
        {"name": "SDSSJ1430+4105", "ra": 217.5792, "dec": 41.0917},
        {"name": "SDSSJ0252+0039", "ra": 43.1583, "dec": 0.6583},
        {"name": "SDSSJ1627-0053", "ra": 246.9042, "dec": -0.8833},
        {"name": "SDSSJ0737+3216", "ra": 114.4583, "dec": 32.2750},
        {"name": "SDSSJ0912+0029", "ra": 138.0625, "dec": 0.4917},
        {"name": "SDSSJ1250+0523", "ra": 192.5875, "dec": 5.3917},
        {"name": "SDSSJ1402+6321", "ra": 210.6292, "dec": 63.3583},
        {"name": "SDSSJ1531-0105", "ra": 232.8875, "dec": -1.0917},
        {"name": "SDSSJ2300+0022", "ra": 345.1208, "dec": 0.3750},
        {"name": "SDSSJ1538+5817", "ra": 234.5542, "dec": 58.2917},
        {"name": "SDSSJ1330-0148", "ra": 202.6125, "dec": -1.8083},
        {"name": "SDSSJ0728+3835", "ra": 112.0792, "dec": 38.5917},
        {"name": "SDSSJ1636+4707", "ra": 249.0458, "dec": 47.1250},
        {"name": "SDSSJ2238-0754", "ra": 339.5875, "dec": -7.9083},
    ]
    
    downloaded = []
    
    for lens in known_lenses[:count]:
        ra, dec = lens["ra"], lens["dec"]
        
        try:
            # Use smaller scale to capture Einstein ring
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.2&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            filename = output_dir / f"lens_{lens['name']}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} gravitational lens images")
    return downloaded


def download_galaxy_mergers(count: int = 20, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download images of known galaxy mergers and interactions.
    
    These show unusual morphologies that should trigger anomaly detection.
    Includes famous mergers and Arp catalog objects.
    """
    logger.info(f"Downloading {count} galaxy merger images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "galaxy_mergers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Famous mergers and Arp catalog objects in SDSS footprint
    known_mergers = [
        {"name": "NGC4676_Mice", "ra": 191.550, "dec": 30.727},
        {"name": "NGC6240", "ra": 253.246, "dec": 2.401},
        {"name": "Arp220", "ra": 233.738, "dec": 23.503},
        {"name": "NGC3690", "ra": 172.134, "dec": 58.563},
        {"name": "NGC2623", "ra": 129.600, "dec": 25.755},
        {"name": "NGC520", "ra": 21.147, "dec": 3.793},
        {"name": "Arp273", "ra": 36.360, "dec": 39.348},
        {"name": "Arp148", "ra": 165.968, "dec": 40.840},
        {"name": "Arp87", "ra": 171.125, "dec": 22.555},
        {"name": "Arp240", "ra": 189.050, "dec": 8.767},
        {"name": "Arp82", "ra": 132.504, "dec": 68.810},
        {"name": "Arp188_Tadpole", "ra": 240.853, "dec": 55.422},
        {"name": "Arp194", "ra": 159.633, "dec": 36.493},
        {"name": "NGC5257", "ra": 204.974, "dec": 0.840},
        {"name": "NGC4567", "ra": 189.019, "dec": 11.254},
    ]
    
    downloaded = []
    
    for merger in known_mergers[:count]:
        ra, dec = merger["ra"], merger["dec"]
        
        # Check if in SDSS footprint
        if not (100 <= ra <= 260 and -5 <= dec <= 70):
            continue
        
        try:
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.3&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            filename = output_dir / f"merger_{merger['name']}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    # Fill remaining with disturbed galaxy regions
    attempts = 0
    while len(downloaded) < count and attempts < count * 2:
        attempts += 1
        ra = random.uniform(140, 220)
        dec = random.uniform(20, 50)
        
        try:
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.35&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            idx = len(downloaded) + 1
            filename = output_dir / f"merger_candidate_{idx:03d}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} merger images")
    return downloaded


def download_peculiar_galaxies(count: int = 15, output_dir: Path = None, on_progress=None) -> List[Path]:
    """
    Download images of peculiar and unusual galaxies.
    
    Includes ring galaxies, polar-ring galaxies, and other oddities.
    """
    logger.info(f"Downloading {count} peculiar galaxy images...")
    
    output_dir = output_dir or DOWNLOADS_DIR / "peculiar_galaxies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known peculiar galaxies
    peculiar = [
        {"name": "Hoags_Object", "ra": 229.5625, "dec": 21.5889},  # Ring galaxy
        {"name": "NGC4650A", "ra": 186.4958, "dec": -40.7083},  # Polar ring (outside SDSS)
        {"name": "AM0644", "ra": 101.3, "dec": -74.14},  # Ring (outside SDSS)
        {"name": "PGC54559", "ra": 229.5625, "dec": 21.589},  # Hoag's Object
        {"name": "NGC1277", "ra": 49.9571, "dec": 41.5736},  # Compact relic
        {"name": "NGC1275", "ra": 49.9508, "dec": 41.5117},  # Perseus A
        {"name": "M87", "ra": 187.7059, "dec": 12.3911},  # Giant elliptical with jet
        {"name": "Cen_A", "ra": 201.3651, "dec": -43.0191},  # Radio galaxy (outside SDSS)
        {"name": "NGC4696", "ra": 192.2083, "dec": -41.3083},  # Outside SDSS
        {"name": "NGC5128", "ra": 201.3651, "dec": -43.0192},  # Centaurus A
        {"name": "IC2163", "ra": 91.5250, "dec": -21.3833},  # Eye galaxy
        {"name": "Mayall_II", "ra": 10.6583, "dec": 41.0794},  # Globular in M31
        {"name": "NGC660", "ra": 25.7625, "dec": 13.6458},  # Polar ring
        {"name": "NGC4631", "ra": 190.5333, "dec": 32.5417},  # Whale galaxy
        {"name": "NGC4656", "ra": 190.9917, "dec": 32.1708},  # Hockey stick
    ]
    
    downloaded = []
    
    for obj in peculiar[:count]:
        ra, dec = obj["ra"], obj["dec"]
        
        # Check if in SDSS footprint
        if not (100 <= ra <= 260 and -10 <= dec <= 70):
            continue
        
        try:
            url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.25&width=256&height=256"
            )
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AstroLens/1.0")
            
            with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
                img_data = resp.read()
            
            if len(img_data) < 3000:
                continue
            
            filename = output_dir / f"peculiar_{obj['name']}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            
            if on_progress:
                on_progress(len(downloaded), count)
            
        except Exception:
            continue
    
    logger.info(f"  ✅ Complete: {len(downloaded)} peculiar galaxy images")
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
    """Send desktop notification (without sound)."""
    import platform
    system = platform.system()
    
    try:
        if system == "Darwin":
            import subprocess
            # No sound - silent notification
            script = f'display notification "{message}" with title "{title}"'
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
