"""
Centralized path configuration for AstroLens.

All large files (datasets, weights, data) are stored OUTSIDE the git repo
in a sibling folder called 'astrolens_artifacts' to avoid bloating the repo.

Directory Structure:
    projects/
    ├── astroLens/              <- This git repo (code only)
    │   ├── api/
    │   ├── inference/
    │   └── ...
    └── astrolens_artifacts/    <- Large files (gitignored)
        ├── data/               <- SQLite DB, FAISS index, user images
        ├── datasets/           <- Fine-tuning datasets
        ├── downloads/          <- Nightly ingested images
        └── weights/            <- Model weights and checkpoints

Environment Variables (override defaults):
    ASTROLENS_ARTIFACTS_DIR  - Base path for all artifacts
    DATABASE_URL             - SQLite database URL
    FAISS_INDEX_PATH         - FAISS vector index file
    IMAGES_DIR               - User-uploaded images
    WEIGHTS_PATH             - Fine-tuned model weights
    DATASETS_DIR             - Training datasets
    DOWNLOADS_DIR            - Downloaded images from sources
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────────────────────────────────────

# Project root (where this file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Artifacts directory (sibling to project, contains all large files)
# Default: ../astrolens_artifacts relative to project root
ARTIFACTS_DIR = Path(
    os.environ.get(
        "ASTROLENS_ARTIFACTS_DIR",
        PROJECT_ROOT.parent / "astrolens_artifacts"
    )
).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Data Paths (runtime data - database, images, vectors)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", ARTIFACTS_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database
DATABASE_PATH = DATA_DIR / "astrolens.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# FAISS vector index
FAISS_INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", DATA_DIR / "faiss.index"))

# User-uploaded images
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", DATA_DIR / "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Sources configuration
SOURCES_CONFIG_PATH = DATA_DIR / "sources_config.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model Weights
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", ARTIFACTS_DIR / "weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Default fine-tuned model path
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", str(WEIGHTS_DIR / "vit_astrolens"))


# ─────────────────────────────────────────────────────────────────────────────
# Datasets (for fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", ARTIFACTS_DIR / "datasets"))
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Downloads (nightly ingest, daily discovery)
# ─────────────────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", ARTIFACTS_DIR / "downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [DATA_DIR, IMAGES_DIR, WEIGHTS_DIR, DATASETS_DIR, DOWNLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_dataset_path(dataset_name: str) -> Path:
    """Get path to a specific dataset."""
    return DATASETS_DIR / dataset_name


def get_checkpoint_path(model_name: str = "vit_astrolens") -> Path:
    """Get path to model checkpoint directory."""
    return WEIGHTS_DIR / model_name


# Print paths on import (for debugging)
if __name__ == "__main__":
    print("AstroLens Paths Configuration")
    print("=" * 50)
    print(f"PROJECT_ROOT:       {PROJECT_ROOT}")
    print(f"ARTIFACTS_DIR:      {ARTIFACTS_DIR}")
    print(f"DATA_DIR:           {DATA_DIR}")
    print(f"DATABASE_URL:       {DATABASE_URL}")
    print(f"FAISS_INDEX_PATH:   {FAISS_INDEX_PATH}")
    print(f"IMAGES_DIR:         {IMAGES_DIR}")
    print(f"WEIGHTS_PATH:       {WEIGHTS_PATH}")
    print(f"DATASETS_DIR:       {DATASETS_DIR}")
    print(f"DOWNLOADS_DIR:      {DOWNLOADS_DIR}")

