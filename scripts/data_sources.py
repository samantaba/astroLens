#!/usr/bin/env python3
"""
Data Sources for AstroLens

Downloads galaxy images from multiple astronomical archives:
1. DECaLS/DESI Legacy Survey - Deep galaxy images
2. Pan-STARRS - Wide-field coverage
3. SDSS - Sloan Digital Sky Survey
4. Hubble Legacy Archive - High-resolution
5. Galaxy Zoo - Citizen science classifications

Usage:
    python scripts/data_sources.py --test          # Test all sources
    python scripts/data_sources.py --download 100  # Download 100 images
    python scripts/data_sources.py --source decals # Specific source
"""

import argparse
import io
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output directory
DOWNLOADS_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "data" / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GalaxyImage:
    """Downloaded galaxy image metadata."""
    source: str
    ra: float
    dec: float
    filename: str
    filepath: str
    size_kb: float
    download_time: float


class DataSource:
    """Base class for data sources."""
    
    name: str = "base"
    base_url: str = ""
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0, follow_redirects=True)
    
    def test_connection(self) -> bool:
        """Test if source is accessible."""
        raise NotImplementedError
    
    def download_image(self, ra: float, dec: float, size_arcmin: float = 1.0) -> Optional[GalaxyImage]:
        """Download image at given coordinates."""
        raise NotImplementedError
    
    def get_random_coordinates(self) -> Tuple[float, float]:
        """Get random sky coordinates in source's footprint."""
        # Default: random in SDSS footprint (North)
        ra = random.uniform(120, 240)
        dec = random.uniform(0, 60)
        return ra, dec
    
    def close(self):
        self.client.close()


class DECaLSSource(DataSource):
    """
    DECaLS/DESI Legacy Survey
    
    Deep imaging, good for faint galaxies.
    Covers ~14,000 sq deg in g, r, z bands.
    """
    
    name = "decals"
    base_url = "https://www.legacysurvey.org/viewer"
    
    def test_connection(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/")
            return r.status_code == 200
        except:
            return False
    
    def download_image(self, ra: float, dec: float, size_arcmin: float = 1.0) -> Optional[GalaxyImage]:
        """Download DECaLS cutout image."""
        start = time.time()
        
        # Convert arcmin to pixels (0.262 arcsec/pixel)
        pixscale = 0.262
        size_pix = int(size_arcmin * 60 / pixscale)
        size_pix = min(max(size_pix, 64), 512)  # Clamp
        
        url = (
            f"{self.base_url}/jpeg-cutout"
            f"?ra={ra}&dec={dec}&size={size_pix}&layer=ls-dr10&pixscale={pixscale}"
        )
        
        try:
            r = self.client.get(url)
            if r.status_code != 200:
                return None
            
            # Save image
            filename = f"decals_ra{ra:.3f}_dec{dec:.3f}.jpg"
            filepath = DOWNLOADS_DIR / filename
            
            img = Image.open(io.BytesIO(r.content))
            img.save(filepath, "JPEG", quality=95)
            
            return GalaxyImage(
                source=self.name,
                ra=ra,
                dec=dec,
                filename=filename,
                filepath=str(filepath),
                size_kb=len(r.content) / 1024,
                download_time=time.time() - start,
            )
        except Exception as e:
            logger.error(f"DECaLS download failed: {e}")
            return None
    
    def get_random_coordinates(self) -> Tuple[float, float]:
        # DECaLS covers dec < +32 (mostly south)
        ra = random.uniform(0, 360)
        dec = random.uniform(-20, 32)
        return ra, dec


class PanSTARRSSource(DataSource):
    """
    Pan-STARRS (PS1)
    
    Covers 3/4 of the sky (dec > -30).
    Good depth, 5 bands (grizy).
    """
    
    name = "panstarrs"
    base_url = "https://ps1images.stsci.edu/cgi-bin/ps1cutouts"
    
    def test_connection(self) -> bool:
        try:
            r = self.client.get("https://ps1images.stsci.edu/")
            return r.status_code == 200
        except:
            return False
    
    def download_image(self, ra: float, dec: float, size_arcmin: float = 1.0) -> Optional[GalaxyImage]:
        """Download Pan-STARRS cutout."""
        start = time.time()
        
        # PS1 uses size in arcsec
        size_arcsec = size_arcmin * 60
        
        # First get the image URL
        params = {
            "ra": ra,
            "dec": dec,
            "size": int(size_arcsec),
            "filter": "color",
            "output_size": 256,
            "format": "jpg",
        }
        
        try:
            # Use PS1 image server directly
            url = (
                f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
                f"?ra={ra}&dec={dec}&filters=gri"
            )
            r = self.client.get(url)
            
            if r.status_code != 200 or "filename" not in r.text.lower():
                # Fallback: try direct color cutout
                url = (
                    f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
                    f"?ra={ra}&dec={dec}&size={int(size_arcsec)}&format=jpg&red=y&green=i&blue=g"
                )
                r = self.client.get(url)
                
                if r.status_code != 200 or len(r.content) < 1000:
                    return None
            
            # Save
            filename = f"ps1_ra{ra:.3f}_dec{dec:.3f}.jpg"
            filepath = DOWNLOADS_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(r.content)
            
            return GalaxyImage(
                source=self.name,
                ra=ra,
                dec=dec,
                filename=filename,
                filepath=str(filepath),
                size_kb=len(r.content) / 1024,
                download_time=time.time() - start,
            )
        except Exception as e:
            logger.error(f"Pan-STARRS download failed: {e}")
            return None
    
    def get_random_coordinates(self) -> Tuple[float, float]:
        # PS1 covers dec > -30
        ra = random.uniform(0, 360)
        dec = random.uniform(-30, 80)
        return ra, dec


class SDSSSource(DataSource):
    """
    SDSS (Sloan Digital Sky Survey)
    
    The classic galaxy survey.
    Covers ~14,500 sq deg in North.
    """
    
    name = "sdss"
    base_url = "https://skyserver.sdss.org/dr18/SkyServerWS"
    
    def test_connection(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/")
            return r.status_code in [200, 404]  # 404 is OK, server is up
        except:
            return False
    
    def download_image(self, ra: float, dec: float, size_arcmin: float = 1.0) -> Optional[GalaxyImage]:
        """Download SDSS cutout."""
        start = time.time()
        
        # SDSS ImgCutout service
        scale = 0.4  # arcsec/pixel
        width = int(size_arcmin * 60 / scale)
        height = width
        width = min(max(width, 64), 512)
        height = width
        
        url = (
            f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
        )
        
        try:
            r = self.client.get(url)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            
            filename = f"sdss_ra{ra:.3f}_dec{dec:.3f}.jpg"
            filepath = DOWNLOADS_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(r.content)
            
            return GalaxyImage(
                source=self.name,
                ra=ra,
                dec=dec,
                filename=filename,
                filepath=str(filepath),
                size_kb=len(r.content) / 1024,
                download_time=time.time() - start,
            )
        except Exception as e:
            logger.error(f"SDSS download failed: {e}")
            return None
    
    def get_random_coordinates(self) -> Tuple[float, float]:
        # SDSS footprint (approximate)
        ra = random.uniform(110, 260)
        dec = random.uniform(-5, 70)
        return ra, dec


class HubbleSource(DataSource):
    """
    Hubble Legacy Archive
    
    High-resolution space-based imaging.
    Limited sky coverage but excellent quality.
    """
    
    name = "hubble"
    base_url = "https://hla.stsci.edu"
    
    def test_connection(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/")
            return r.status_code == 200
        except:
            return False
    
    def download_image(self, ra: float, dec: float, size_arcmin: float = 0.5) -> Optional[GalaxyImage]:
        """Download Hubble cutout (if available)."""
        start = time.time()
        
        # HLA cutout service
        url = (
            f"{self.base_url}/cgi-bin/fitscut.cgi"
            f"?ra={ra}&dec={dec}&size={size_arcmin*60}&format=jpeg"
        )
        
        try:
            r = self.client.get(url)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            
            filename = f"hubble_ra{ra:.3f}_dec{dec:.3f}.jpg"
            filepath = DOWNLOADS_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(r.content)
            
            return GalaxyImage(
                source=self.name,
                ra=ra,
                dec=dec,
                filename=filename,
                filepath=str(filepath),
                size_kb=len(r.content) / 1024,
                download_time=time.time() - start,
            )
        except Exception as e:
            logger.error(f"Hubble download failed: {e}")
            return None


# Known galaxy coordinates for testing
KNOWN_GALAXIES = [
    # Name, RA, Dec
    ("M31 Andromeda", 10.68, 41.27),
    ("M33 Triangulum", 23.46, 30.66),
    ("M51 Whirlpool", 202.47, 47.20),
    ("M81 Bode's", 148.89, 69.07),
    ("M82 Cigar", 148.97, 69.68),
    ("M101 Pinwheel", 210.80, 54.35),
    ("NGC 1300", 49.92, -19.41),
    ("NGC 4038 Antennae", 180.47, -18.87),
    ("NGC 5128 Centaurus A", 201.37, -43.02),
    ("IC 1101", 225.08, 5.74),
]


def test_all_sources() -> dict:
    """Test all data sources."""
    results = {}
    
    sources = [
        DECaLSSource(),
        PanSTARRSSource(),
        SDSSSource(),
        # HubbleSource(),  # Often slow/limited
    ]
    
    for source in sources:
        logger.info(f"Testing {source.name}...")
        
        # Test connection
        connected = source.test_connection()
        
        # Test download
        downloaded = None
        if connected:
            ra, dec = source.get_random_coordinates()
            downloaded = source.download_image(ra, dec)
        
        results[source.name] = {
            "connected": connected,
            "downloaded": downloaded is not None,
            "filepath": downloaded.filepath if downloaded else None,
            "size_kb": downloaded.size_kb if downloaded else 0,
            "time_s": downloaded.download_time if downloaded else 0,
        }
        
        status = "✓" if results[source.name]["downloaded"] else "✗"
        logger.info(f"  {status} {source.name}: connected={connected}, downloaded={results[source.name]['downloaded']}")
        
        source.close()
    
    return results


def download_batch(n: int = 10, source_name: Optional[str] = None) -> List[GalaxyImage]:
    """Download batch of galaxy images."""
    
    if source_name:
        sources = {
            "decals": DECaLSSource,
            "panstarrs": PanSTARRSSource,
            "sdss": SDSSSource,
            "hubble": HubbleSource,
        }
        if source_name not in sources:
            logger.error(f"Unknown source: {source_name}")
            return []
        source_classes = [sources[source_name]]
    else:
        source_classes = [DECaLSSource, SDSSSource, PanSTARRSSource]
    
    downloaded = []
    
    for SourceClass in source_classes:
        source = SourceClass()
        per_source = n // len(source_classes)
        
        logger.info(f"Downloading {per_source} images from {source.name}...")
        
        for i in range(per_source):
            ra, dec = source.get_random_coordinates()
            img = source.download_image(ra, dec)
            
            if img:
                downloaded.append(img)
                logger.info(f"  [{i+1}/{per_source}] {img.filename} ({img.size_kb:.1f} KB)")
            else:
                logger.warning(f"  [{i+1}/{per_source}] Failed at ({ra:.2f}, {dec:.2f})")
            
            time.sleep(0.5)  # Rate limiting
        
        source.close()
    
    return downloaded


def download_known_galaxies() -> List[GalaxyImage]:
    """Download images of known galaxies for testing."""
    downloaded = []
    
    sources = [SDSSSource(), DECaLSSource()]
    
    for name, ra, dec in KNOWN_GALAXIES[:5]:  # First 5
        logger.info(f"Downloading {name} ({ra}, {dec})...")
        
        for source in sources:
            img = source.download_image(ra, dec, size_arcmin=2.0)
            if img:
                downloaded.append(img)
                logger.info(f"  ✓ {source.name}: {img.filename}")
                break
        else:
            logger.warning(f"  ✗ Could not download {name}")
        
        time.sleep(0.5)
    
    for source in sources:
        source.close()
    
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="AstroLens Data Sources")
    parser.add_argument("--test", action="store_true", help="Test all sources")
    parser.add_argument("--download", type=int, metavar="N", help="Download N images")
    parser.add_argument("--source", type=str, help="Specific source (decals, sdss, panstarrs)")
    parser.add_argument("--known", action="store_true", help="Download known galaxies")
    args = parser.parse_args()
    
    if args.test:
        print("\n" + "="*60)
        print("TESTING DATA SOURCES")
        print("="*60 + "\n")
        
        results = test_all_sources()
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for name, r in results.items():
            status = "✓" if r["downloaded"] else "✗"
            print(f"{status} {name}: ", end="")
            if r["downloaded"]:
                print(f"{r['size_kb']:.1f} KB in {r['time_s']:.2f}s")
            else:
                print("FAILED" if r["connected"] else "NOT CONNECTED")
    
    elif args.download:
        images = download_batch(args.download, args.source)
        print(f"\nDownloaded {len(images)} images to {DOWNLOADS_DIR}")
    
    elif args.known:
        images = download_known_galaxies()
        print(f"\nDownloaded {len(images)} known galaxies")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
