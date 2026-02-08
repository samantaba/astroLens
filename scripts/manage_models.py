#!/usr/bin/env python3
"""
Model Management Script for AstroLens

Handles uploading and downloading trained model weights.
Supports Git LFS and GitHub Releases as storage backends.

Usage:
    python scripts/manage_models.py --setup-lfs     # Setup Git LFS tracking
    python scripts/manage_models.py --copy-model     # Copy YOLO model to repo for LFS
    python scripts/manage_models.py --check          # Check model availability
    
For GitHub Release uploads (alternative to LFS):
    python scripts/manage_models.py --create-release v1.0.0
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT.parent / "astrolens_artifacts"

# Model paths
MODELS = {
    "yolo_transient": {
        "source": ARTIFACTS_DIR / "transient_data" / "models" / "transient_detector" / "weights" / "best.pt",
        "dest": PROJECT_ROOT / "models" / "yolo_transient_best.pt",
        "description": "YOLOv8 Transient Detector (trained on TNS+ZTF data)",
        "size_mb": 6,
    },
}

# Git LFS patterns
LFS_PATTERNS = [
    "models/*.pt",
    "models/*.pth",
    "models/*.onnx",
    "models/*.safetensors",
    "models/*.bin",
]


def setup_lfs():
    """Setup Git LFS for model files."""
    logger.info("Setting up Git LFS...")
    
    # Check if git-lfs is installed
    result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Git LFS is not installed. Install with:")
        logger.error("  brew install git-lfs  (macOS)")
        logger.error("  apt install git-lfs   (Ubuntu/Debian)")
        logger.error("  choco install git-lfs (Windows)")
        return False
    
    logger.info(f"  Git LFS version: {result.stdout.strip()}")
    
    # Initialize LFS
    subprocess.run(["git", "lfs", "install"], cwd=str(PROJECT_ROOT), check=True)
    
    # Track model patterns
    for pattern in LFS_PATTERNS:
        subprocess.run(
            ["git", "lfs", "track", pattern],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        logger.info(f"  Tracking: {pattern}")
    
    # Create models directory
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create .gitattributes if it was updated
    gitattr = PROJECT_ROOT / ".gitattributes"
    if gitattr.exists():
        logger.info(f"  .gitattributes updated")
    
    logger.info("Git LFS setup complete!")
    return True


def copy_model():
    """Copy trained model to repository for Git LFS tracking."""
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    for name, info in MODELS.items():
        source = info["source"]
        dest = info["dest"]
        
        if not source.exists():
            logger.warning(f"  {name}: Source model not found at {source}")
            continue
        
        size_mb = source.stat().st_size / (1024 * 1024)
        logger.info(f"  {name}: Copying ({size_mb:.1f} MB)")
        
        shutil.copy2(str(source), str(dest))
        logger.info(f"  {name}: Copied to {dest}")
    
    logger.info("\nModels ready for commit. Run:")
    logger.info("  git add models/")
    logger.info("  git commit -m 'Add trained YOLO model via LFS'")
    logger.info("  git push")


def check_models():
    """Check availability of all models."""
    logger.info("Checking model availability:")
    
    for name, info in MODELS.items():
        source = info["source"]
        dest = info["dest"]
        
        source_exists = source.exists()
        dest_exists = dest.exists()
        
        if source_exists:
            size = source.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {name} (artifact): {source} ({size:.1f} MB)")
        else:
            logger.info(f"  ✗ {name} (artifact): Not found")
        
        if dest_exists:
            size = dest.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {name} (repo): {dest} ({size:.1f} MB)")
        else:
            logger.info(f"  ○ {name} (repo): Not yet copied")
    
    # Check Git LFS status
    result = subprocess.run(
        ["git", "lfs", "ls-files"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        logger.info(f"\nGit LFS tracked files:")
        for line in result.stdout.strip().split("\n"):
            logger.info(f"  {line}")
    else:
        logger.info("\nNo files tracked by Git LFS yet")


def create_release(tag: str):
    """Create a GitHub release with model files attached."""
    logger.info(f"Creating GitHub release: {tag}")
    
    # Check gh CLI
    result = subprocess.run(["gh", "version"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("GitHub CLI (gh) not installed. Install with:")
        logger.error("  brew install gh  (macOS)")
        return False
    
    # Collect model files
    files = []
    for name, info in MODELS.items():
        source = info["source"]
        if source.exists():
            files.append(str(source))
            logger.info(f"  Including: {source.name}")
    
    if not files:
        logger.error("No model files found to include in release")
        return False
    
    # Create release
    files_args = []
    for f in files:
        files_args.extend(["--attach", f])
    
    cmd = [
        "gh", "release", "create", tag,
        "--title", f"AstroLens {tag}",
        "--notes", f"AstroLens {tag} release with trained models.\n\nModels included:\n" + 
                   "\n".join(f"- {info['description']}" for info in MODELS.values()),
    ] + files_args
    
    logger.info(f"  Command: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"  ✓ Release created: {result.stdout.strip()}")
        return True
    else:
        logger.error(f"  ✗ Failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AstroLens Model Management")
    parser.add_argument("--setup-lfs", action="store_true", help="Setup Git LFS")
    parser.add_argument("--copy-model", action="store_true", help="Copy models to repo")
    parser.add_argument("--check", action="store_true", help="Check model availability")
    parser.add_argument("--create-release", type=str, metavar="TAG", help="Create GitHub release")
    args = parser.parse_args()
    
    if args.setup_lfs:
        setup_lfs()
    elif args.copy_model:
        copy_model()
    elif args.check:
        check_models()
    elif args.create_release:
        create_release(args.create_release)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
