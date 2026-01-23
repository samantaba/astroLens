#!/usr/bin/env python3
"""
Download pre-trained model weights.

Usage: python scripts/download_weights.py
"""

import os
from pathlib import Path

# Weights directory
WEIGHTS_DIR = Path("weights")
WEIGHTS_FILE = WEIGHTS_DIR / "vit_astrolens.pt"


def main():
    """Download or create placeholder weights."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if WEIGHTS_FILE.exists():
        print(f"✓ Weights already exist at {WEIGHTS_FILE}")
        return
    
    print("Downloading pre-trained ViT weights...")
    
    # For now, we'll use the pre-trained ImageNet weights from timm
    # In production, you'd download fine-tuned weights from your server
    
    try:
        import torch
        import timm
        
        # Load pre-trained ViT
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=11)
        
        # Save state dict
        torch.save(model.state_dict(), WEIGHTS_FILE)
        print(f"✓ Saved weights to {WEIGHTS_FILE}")
        print(f"  File size: {WEIGHTS_FILE.stat().st_size / 1e6:.1f} MB")
        
    except ImportError as e:
        print(f"⚠ Could not download weights: {e}")
        print("  Install requirements first: pip install torch timm")
        
        # Create placeholder
        WEIGHTS_FILE.touch()
        print(f"  Created placeholder at {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()

