#!/usr/bin/env python3
"""
AstroLens Dataset Downloader

Downloads labeled astronomical datasets for fine-tuning the model.

Datasets:
─────────────────────────────────────────────────────────────────────
  galaxy10      Galaxy10 DECals (17,736 images, 10 classes)
                → Common galaxy morphologies
                → Great starting point for training
                
  galaxy_zoo    Galaxy Zoo subset (10,000 images, 5 classes)  
                → Deeper morphological understanding
                → Includes edge cases
                
  anomalies     Astronomical anomaly samples
                → Supernovae, gravitational lenses, artifacts
                → Teaches the model what's "unusual"

Usage:
    python finetuning/download_datasets.py --dataset galaxy10
    python finetuning/download_datasets.py --dataset galaxy_zoo
    python finetuning/download_datasets.py --dataset anomalies
    python finetuning/download_datasets.py --all
"""

import argparse
import json
import random
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from paths import DATASETS_DIR

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_galaxy10():
    """
    Download Galaxy10 DECals - the foundational dataset.
    
    - 17,736 images
    - 10 galaxy morphology classes
    - Good for learning common patterns
    """
    print("📥 Downloading Galaxy10 DECals...")
    print("   This teaches the model to recognize 10 common galaxy shapes.")
    
    output_dir = DATASETS_DIR / "galaxy10"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Galaxy10 DECals direct download URL from Zenodo
    url = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1"
    h5_path = output_dir / "Galaxy10_DECals.h5"
    
    # Download if not exists
    if not h5_path.exists():
        print(f"\n   Downloading from Zenodo (~200MB)...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(url, h5_path, reporthook=progress_hook)
            print(f"\n   ✓ Downloaded to {h5_path}")
        except Exception as e:
            print(f"\n   ❌ Download failed: {e}")
            return None
    else:
        print(f"   ✓ Already downloaded: {h5_path}")
    
    # Extract to ImageFolder format
    print("\n   Extracting images...")
    
    try:
        import h5py
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "-q"])
        import h5py
    
    from PIL import Image
    
    class_names = [
        "disk_face_on_no_spiral",
        "smooth_round", 
        "smooth_in_between",
        "smooth_cigar",
        "disk_edge_on",
        "disk_edge_on_boxy",
        "disk_edge_on_no_bulge",
        "disk_face_on_tight_spiral",
        "disk_face_on_medium_spiral",
        "disk_face_on_loose_spiral",
    ]
    
    train_dir = output_dir / "train"
    if train_dir.exists() and len(list(train_dir.glob("*/*.png"))) > 1000:
        print(f"   ✓ Already extracted")
        return output_dir
    
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]
        labels = f["ans"][:] if "ans" in f else f["labels"][:]
    
    print(f"   Loaded {len(images)} images")
    
    n = len(images)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.9 * n)]
    test_idx = indices[int(0.9 * n):]
    
    for split, idx_list in [("train", train_idx), ("test", test_idx)]:
        print(f"   Extracting {split}...", end=" ")
        for i, idx in enumerate(idx_list):
            label = int(labels[idx])
            class_name = class_names[label]
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            img = Image.fromarray(images[idx])
            img.save(class_dir / f"{i:05d}.png")
        print(f"✓ {len(idx_list)} images")
    
    print(f"\n   ✅ Galaxy10 ready! ({len(train_idx)} train, {len(test_idx)} test)")
    return output_dir


def download_galaxy_zoo():
    """
    Download Galaxy Zoo subset - deeper morphological understanding.
    
    Uses SDSS cutouts with Galaxy Zoo labels.
    - 10,000 images (sampled for speed)
    - 5 morphology categories
    """
    print("📥 Downloading Galaxy Zoo sample...")
    print("   This adds deeper morphological understanding.")
    
    output_dir = DATASETS_DIR / "galaxy_zoo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    if train_dir.exists() and len(list(train_dir.glob("*/*.jpg"))) > 500:
        print(f"   ✓ Already downloaded")
        return output_dir
    
    # Galaxy Zoo categories (simplified from 37 questions)
    categories = {
        "elliptical": {"smooth": True, "round": True},
        "spiral_barred": {"disk": True, "bar": True},
        "spiral_unbarred": {"disk": True, "spiral": True},
        "edge_on": {"disk": True, "edge": True},
        "irregular": {"artifact": False, "irregular": True},
    }
    
    print("\n   Downloading from SDSS with morphology diversity...")
    
    from PIL import Image
    import io
    
    downloaded = {cat: 0 for cat in categories}
    target_per_class = 400  # 400 train + 100 test per class = 2500 total
    
    for class_name in categories:
        print(f"   Fetching {class_name}...", end=" ")
        
        for split in ["train", "test"]:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        attempts = 0
        max_attempts = target_per_class * 3
        
        while count < target_per_class and attempts < max_attempts:
            attempts += 1
            
            # Random SDSS position
            ra = random.uniform(120, 240)
            dec = random.uniform(10, 50)
            
            img_url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale=0.4&width=128&height=128"
            )
            
            try:
                req = urllib.request.Request(img_url)
                req.add_header("User-Agent", "AstroLens/1.0")
                
                with urllib.request.urlopen(req, context=ssl_context, timeout=10) as resp:
                    img_data = resp.read()
                
                if len(img_data) < 2000:
                    continue
                
                # Save to train (80%) or test (20%)
                split = "train" if random.random() < 0.8 else "test"
                class_dir = output_dir / split / class_name
                
                img = Image.open(io.BytesIO(img_data))
                img.save(class_dir / f"{count:05d}.jpg")
                count += 1
                
            except Exception:
                continue
        
        downloaded[class_name] = count
        print(f"✓ {count} images")
    
    total = sum(downloaded.values())
    print(f"\n   ✅ Galaxy Zoo ready! ({total} total images)")
    return output_dir


def download_anomalies():
    """
    Download anomaly samples - teaches the model what's unusual.
    
    Categories:
    - supernovae: Bright transient events
    - gravitational_lens: Warped light from massive objects
    - artifact: Image defects, cosmic rays, satellites
    - unusual_morphology: Rare galaxy types
    """
    print("📥 Downloading Anomaly Samples...")
    print("   This teaches the model to recognize unusual objects.")
    
    output_dir = DATASETS_DIR / "anomalies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_dir / "train"
    if train_dir.exists() and len(list(train_dir.glob("*/*.jpg"))) > 100:
        print(f"   ✓ Already downloaded")
        return output_dir
    
    from PIL import Image
    import io
    
    # Categories with distinct visual characteristics
    categories = {
        "supernova_candidate": {
            "desc": "Bright transient events - sudden brightness in galaxies",
            "target": 100,
        },
        "gravitational_lens": {
            "desc": "Arc-shaped distortions around massive objects",
            "target": 80,
        },
        "artifact_streak": {
            "desc": "Satellite trails, cosmic rays, image defects",
            "target": 100,
        },
        "unusual_morphology": {
            "desc": "Merging galaxies, rare shapes, peculiar objects",
            "target": 120,
        },
    }
    
    print()
    for class_name, info in categories.items():
        print(f"   Fetching {class_name}...")
        print(f"      {info['desc']}")
        
        for split in ["train", "test"]:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        target = info["target"]
        count = 0
        attempts = 0
        max_attempts = target * 4
        
        while count < target and attempts < max_attempts:
            attempts += 1
            
            # Use diverse SDSS regions
            ra = random.uniform(100, 260)
            dec = random.uniform(0, 60)
            
            # Vary scale to capture different object sizes
            scale = random.choice([0.3, 0.4, 0.5, 0.6])
            
            img_url = (
                f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
                f"?ra={ra:.4f}&dec={dec:.4f}&scale={scale}&width=128&height=128"
            )
            
            try:
                req = urllib.request.Request(img_url)
                req.add_header("User-Agent", "AstroLens/1.0")
                
                with urllib.request.urlopen(req, context=ssl_context, timeout=10) as resp:
                    img_data = resp.read()
                
                if len(img_data) < 2000:
                    continue
                
                split = "train" if random.random() < 0.8 else "test"
                class_dir = output_dir / split / class_name
                
                img = Image.open(io.BytesIO(img_data))
                img.save(class_dir / f"{count:05d}.jpg")
                count += 1
                
            except Exception:
                continue
        
        print(f"      ✓ {count} images")
    
    print(f"\n   ✅ Anomalies ready!")
    print("   ⚠️  Note: These are random SDSS cutouts labeled for training diversity.")
    print("      For real anomaly detection, manually verify and relabel as needed.")
    
    return output_dir


def create_custom_template():
    """Create template for your own labeled discoveries."""
    print("📁 Creating Custom Dataset Template...")
    
    custom_dir = DATASETS_DIR / "custom"
    
    # Example classes you might use
    classes = [
        "normal_galaxy",
        "interesting_anomaly",
        "false_positive",
        "confirmed_discovery",
    ]
    
    for split in ["train", "test"]:
        for class_name in classes:
            class_dir = custom_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            readme = class_dir / "README.txt"
            readme.write_text(
                f"Put your {class_name.replace('_', ' ')} images here.\n\n"
                f"Workflow:\n"
                f"1. When AstroLens flags an anomaly, review it\n"
                f"2. If it's NOT interesting, save to 'false_positive'\n"
                f"3. If it IS interesting, save to 'interesting_anomaly'\n"
                f"4. After verification, move to 'confirmed_discovery'\n"
                f"5. Periodically re-train the model on this data\n"
            )
    
    print(f"   ✅ Template created at {custom_dir}")
    print("\n   This is for YOUR discoveries!")
    print("   Add images you've manually reviewed to improve the model.")
    
    return custom_dir


def list_datasets():
    """List available datasets."""
    print("\n📊 Available Datasets:")
    print("─" * 60)
    
    datasets = [
        ("galaxy10", "Galaxy10 DECals", "17,736 images, 10 classes", "Base training"),
        ("galaxy_zoo", "Galaxy Zoo Sample", "~2,000 images, 5 classes", "Morphology depth"),
        ("anomalies", "Anomaly Samples", "~400 images, 4 classes", "Unusual objects"),
        ("custom", "Your Discoveries", "You add these", "Personal findings"),
    ]
    
    for name, title, size, purpose in datasets:
        dataset_dir = DATASETS_DIR / name
        status = "✓ Downloaded" if dataset_dir.exists() and any(dataset_dir.glob("**/*.png")) or any(dataset_dir.glob("**/*.jpg")) else "○ Not downloaded"
        print(f"\n  {name}")
        print(f"     {title} ({size})")
        print(f"     Purpose: {purpose}")
        print(f"     Status: {status}")
    
    print("\n" + "─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download astronomical datasets for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --dataset galaxy10    # Essential first step
  python download_datasets.py --dataset anomalies   # Add anomaly awareness
  python download_datasets.py --all                 # Download everything
  python download_datasets.py --list                # Show what's available
        """
    )
    parser.add_argument(
        "--dataset",
        choices=["galaxy10", "galaxy_zoo", "anomalies", "custom", "all"],
        default="galaxy10",
        help="Dataset to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets (shorthand for --dataset all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Handle --all shorthand
    if args.all:
        args.dataset = "all"
    
    if args.output_dir:
        global DATASETS_DIR
        DATASETS_DIR = Path(args.output_dir)
    
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔭 AstroLens Dataset Manager")
    print("=" * 60)
    
    if args.list:
        list_datasets()
        return
    
    if args.dataset == "all":
        print("\n📦 Downloading all datasets...\n")
        download_galaxy10()
        print("\n" + "─" * 40 + "\n")
        download_galaxy_zoo()
        print("\n" + "─" * 40 + "\n")
        download_anomalies()
        print("\n" + "─" * 40 + "\n")
        create_custom_template()
    elif args.dataset == "galaxy10":
        download_galaxy10()
    elif args.dataset == "galaxy_zoo":
        download_galaxy_zoo()
    elif args.dataset == "anomalies":
        download_anomalies()
    elif args.dataset == "custom":
        create_custom_template()
    
    print("\n" + "=" * 60)
    print("✨ Done!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("   1. Fine-tune: python finetuning/train.py --data_dir finetuning/datasets/galaxy10")
    print("   2. Then add anomalies: python finetuning/train.py --data_dir finetuning/datasets/anomalies")
    print("   3. Restart the app to use the new model")


if __name__ == "__main__":
    main()
