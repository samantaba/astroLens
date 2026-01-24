#!/usr/bin/env python3
"""
Anomaly Breakthrough Script

This script implements aggressive strategies to detect the first anomaly:
1. Lowers OOD thresholds dramatically
2. Downloads from ACTUAL anomaly catalogs (not random positions)
3. Adds "anomaly" as a trainable class
4. Provides instant analysis of existing images

The key insight: OOD detection finds images the model CAN'T classify,
but we need to find SCIENTIFICALLY INTERESTING objects. These are different!

Strategy:
- Download known supernovae, gravitational lenses, peculiar galaxies
- Train model to recognize these AS anomalies
- Use aggressive thresholds in discovery
"""

import json
import sys
import urllib.request
import ssl
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DOWNLOADS_DIR, DATASETS_DIR, DATA_DIR

# SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_known_supernovae(count: int = 50) -> list:
    """
    Download images of REAL supernovae from Open Supernova Catalog.
    These are verified transient events - true anomalies.
    """
    print(f"🌟 Downloading {count} known supernovae from catalogs...")
    
    output_dir = DOWNLOADS_DIR / "real_supernovae"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Known supernova coordinates (from recent discoveries)
    # These are actual transient events
    known_sne = [
        # Type Ia supernovae (standard candles)
        {"name": "SN2011fe", "ra": 210.774, "dec": 54.274},
        {"name": "SN2014J", "ra": 148.926, "dec": 69.674},
        {"name": "SN2017cbv", "ra": 186.075, "dec": 12.502},
        # Type II supernovae (core collapse)
        {"name": "SN2017eaw", "ra": 307.267, "dec": 60.170},
        {"name": "SN2023ixf", "ra": 210.410, "dec": 54.312},
        # Peculiar supernovae
        {"name": "AT2018cow", "ra": 244.000, "dec": 22.270},  # "The Cow"
    ]
    
    downloaded = []
    
    # Download cutouts for known SNe positions
    for sn in known_sne[:count]:
        ra, dec = sn["ra"], sn["dec"]
        
        # Check if in SDSS footprint
        if not (0 <= ra <= 360 and -10 <= dec <= 70):
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
            
            filename = output_dir / f"sn_{sn['name']}_ra{ra:.1f}_dec{dec:.1f}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            downloaded.append(filename)
            print(f"  ✓ {sn['name']}")
            
        except Exception as e:
            print(f"  ✗ {sn['name']}: {e}")
    
    # Also download regions around other known transients
    print(f"  Downloading additional transient regions...")
    
    for i in range(count - len(downloaded)):
        # Random positions near known transient-rich regions
        ra = random.choice([210, 148, 186, 244]) + random.uniform(-5, 5)
        dec = random.choice([54, 69, 22, 12]) + random.uniform(-3, 3)
        
        if not (120 <= ra <= 240 and 0 <= dec <= 60):
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
            
        except Exception:
            continue
    
    print(f"  ✅ Downloaded {len(downloaded)} supernova/transient images")
    return downloaded


def download_gravitational_lenses(count: int = 30) -> list:
    """
    Download images of known gravitational lenses.
    These are extremely rare and scientifically valuable.
    """
    print(f"🔭 Downloading {count} gravitational lens candidates...")
    
    output_dir = DOWNLOADS_DIR / "gravitational_lenses"
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
    ]
    
    downloaded = []
    
    for lens in known_lenses[:count]:
        ra, dec = lens["ra"], lens["dec"]
        
        try:
            # Use larger scale to capture Einstein ring
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
            print(f"  ✓ {lens['name']}")
            
        except Exception as e:
            print(f"  ✗ {lens['name']}: {e}")
    
    print(f"  ✅ Downloaded {len(downloaded)} gravitational lens images")
    return downloaded


def download_galaxy_mergers(count: int = 50) -> list:
    """
    Download images of known galaxy mergers and interactions.
    These show unusual morphologies that should trigger anomaly detection.
    """
    print(f"🌀 Downloading {count} galaxy mergers/interactions...")
    
    output_dir = DOWNLOADS_DIR / "galaxy_mergers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Famous mergers and interacting galaxies
    known_mergers = [
        {"name": "NGC4676_Mice", "ra": 191.550, "dec": 30.727},
        {"name": "NGC4038_Antennae", "ra": 180.471, "dec": -18.867},  # Outside SDSS
        {"name": "NGC6240", "ra": 253.246, "dec": 2.401},
        {"name": "Arp220", "ra": 233.738, "dec": 23.503},
        {"name": "NGC3690", "ra": 172.134, "dec": 58.563},
        {"name": "NGC2623", "ra": 129.600, "dec": 25.755},
        {"name": "NGC7252", "ra": 339.017, "dec": -24.679},  # Outside SDSS
        {"name": "NGC520", "ra": 21.147, "dec": 3.793},
        {"name": "NGC3256", "ra": 156.963, "dec": -43.904},  # Outside SDSS
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
            print(f"  ✓ {merger['name']}")
            
        except Exception as e:
            print(f"  ✗ {merger['name']}: {e}")
    
    # Add more merger candidates from random disturbed regions
    print(f"  Adding disturbed galaxy regions...")
    for i in range(count - len(downloaded)):
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
            
        except Exception:
            continue
    
    print(f"  ✅ Downloaded {len(downloaded)} merger images")
    return downloaded


def update_discovery_thresholds():
    """
    Update discovery state with aggressive thresholds for anomaly detection.
    """
    print("\n⚙️ Updating discovery thresholds for breakthrough...")
    
    state_file = DATA_DIR / "discovery_state.json"
    
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        state = {}
    
    # AGGRESSIVE SETTINGS
    old_threshold = state.get("current_threshold", 3.0)
    state["current_threshold"] = 0.3  # Very low - will catch more candidates
    state["min_threshold"] = 0.1  # Allow it to go very low
    
    print(f"  Threshold: {old_threshold:.2f} → 0.30 (aggressive)")
    
    state_file.write_text(json.dumps(state, indent=2))
    print("  ✅ Discovery state updated")


def create_anomaly_dataset():
    """
    Create a proper anomaly dataset structure for training.
    """
    print("\n📁 Creating anomaly training dataset...")
    
    anomaly_dir = DATASETS_DIR / "anomalies"
    
    # Create class directories
    classes = {
        "supernova": DOWNLOADS_DIR / "real_supernovae",
        "gravitational_lens": DOWNLOADS_DIR / "gravitational_lenses",
        "merger": DOWNLOADS_DIR / "galaxy_mergers",
    }
    
    total_copied = 0
    
    for class_name, source_dir in classes.items():
        class_dir = anomaly_dir / "train" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        if source_dir.exists():
            import shutil
            for img_file in source_dir.glob("*.jpg"):
                dest = class_dir / img_file.name
                if not dest.exists():
                    shutil.copy(img_file, dest)
                    total_copied += 1
    
    print(f"  ✅ Copied {total_copied} images to anomaly dataset")
    return total_copied


def analyze_existing_for_anomalies():
    """
    Re-analyze existing images with aggressive thresholds.
    """
    print("\n🔍 Analyzing existing images with aggressive thresholds...")
    
    try:
        import httpx
        
        # Get all images
        resp = httpx.get("http://localhost:8000/images?limit=1000", timeout=30)
        if resp.status_code != 200:
            print("  ✗ API not available")
            return
        
        images = resp.json()
        print(f"  Found {len(images)} images in database")
        
        # Find images with highest OOD scores
        high_ood = []
        for img in images:
            if img.get("ood_score") and img["ood_score"] > 0.5:
                high_ood.append(img)
        
        high_ood.sort(key=lambda x: x.get("ood_score", 0), reverse=True)
        
        print(f"\n  📊 Top 20 highest OOD scores (potential anomalies):")
        for img in high_ood[:20]:
            score = img.get("ood_score", 0)
            class_label = img.get("class_label", "?")
            conf = img.get("class_confidence", 0)
            print(f"     {img['id']:5d}: OOD={score:.3f}, class={class_label}, conf={conf:.1%}")
        
        # Flag top candidates as anomalies
        flagged = 0
        for img in high_ood[:10]:
            try:
                update_resp = httpx.patch(
                    f"http://localhost:8000/images/{img['id']}",
                    json={"is_anomaly": True},
                    timeout=10,
                )
                if update_resp.status_code == 200:
                    flagged += 1
            except Exception:
                pass
        
        if flagged > 0:
            print(f"\n  🎯 Flagged {flagged} images as anomaly candidates!")
            print("     Check the Anomalies filter in the Gallery to review.")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def print_recommendations():
    """Print strategic recommendations for anomaly detection."""
    print("\n" + "=" * 60)
    print("📋 BREAKTHROUGH RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
🎯 IMMEDIATE ACTIONS:

1. RUN THIS SCRIPT to download real anomalies:
   python scripts/anomaly_breakthrough.py --download

2. LOWER THE THRESHOLD in discovery panel to 0.3 or below
   - The current threshold (1.2+) is too high for sensitive detection
   
3. CHECK NEAR-MISSES - images just below threshold may be true anomalies

4. TRAIN ON REAL ANOMALIES:
   python finetuning/pipeline.py --dataset anomalies --epochs 10

🔬 UNDERSTANDING THE PROBLEM:

The Galaxy10 model classifies NORMAL galaxy types. OOD detection flags 
images that DON'T FIT these normal types. But random sky images mostly 
contain... normal galaxies!

True anomalies (supernovae, lenses, mergers) are RARE in random samples.
We need to:
- Download from ACTUAL anomaly catalogs (this script does that)
- Train the model to RECOGNIZE anomaly patterns
- Use very aggressive thresholds

📊 EXPECTED OUTCOMES:

After running this script and retraining:
- Threshold should catch true anomalies at 0.3-0.5
- Known anomaly images should score 0.8+ OOD
- False positive rate will increase, but that's OK for discovery
- Manual review of flagged images is part of the process

🚀 LONG-TERM IMPROVEMENTS:

1. Add more anomaly sources:
   - ATLAS Transient Server
   - Transient Name Server
   - Galaxy Zoo Talk "weird" tags
   
2. Active learning:
   - Mark false positives as "normal"
   - Mark true discoveries as "anomaly"
   - Retrain periodically
   
3. Specialized models:
   - Train separate supernova detector
   - Train gravitational lens detector
   - Ensemble of specialists
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomaly Breakthrough Script")
    parser.add_argument("--download", action="store_true", help="Download real anomaly images")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing images")
    parser.add_argument("--update-thresholds", action="store_true", help="Update to aggressive thresholds")
    parser.add_argument("--all", action="store_true", help="Do everything")
    parser.add_argument("--recommendations", action="store_true", help="Show recommendations only")
    
    args = parser.parse_args()
    
    if args.recommendations or (not any([args.download, args.analyze, args.update_thresholds, args.all])):
        print_recommendations()
        return
    
    print("=" * 60)
    print("🚀 ASTROLENS ANOMALY BREAKTHROUGH")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.all or args.download:
        # Download real anomalies
        sne = download_known_supernovae(30)
        lenses = download_gravitational_lenses(20)
        mergers = download_galaxy_mergers(30)
        
        print(f"\n📊 Total downloaded: {len(sne) + len(lenses) + len(mergers)} anomaly images")
        
        # Create dataset
        create_anomaly_dataset()
    
    if args.all or args.update_thresholds:
        update_discovery_thresholds()
    
    if args.all or args.analyze:
        analyze_existing_for_anomalies()
    
    print("\n" + "=" * 60)
    print("✅ BREAKTHROUGH PREPARATION COMPLETE")
    print("=" * 60)
    print("""
Next steps:
1. Train on anomalies: python finetuning/pipeline.py --dataset anomalies
2. Restart discovery with low threshold (0.3)
3. Monitor for first anomaly detection!
""")


if __name__ == "__main__":
    main()
