"""
Image Duplicate Detection using Perceptual Hashing.

Detects near-duplicate images to prevent redundant processing and storage.
Uses multiple hash algorithms for robust matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DuplicateResult:
    """Result of duplicate check."""
    is_duplicate: bool
    hash_value: str
    matching_hash: Optional[str] = None
    similarity: float = 0.0


class DuplicateDetector:
    """
    Perceptual hash-based duplicate detector.
    
    Uses average hash (aHash) for fast comparison.
    Can detect visually similar images even with minor differences.
    """

    def __init__(
        self,
        hash_size: int = 16,
        similarity_threshold: float = 0.90,
    ):
        """
        Args:
            hash_size: Size of hash (larger = more precise, slower)
            similarity_threshold: 0-1, higher = stricter matching
        """
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.known_hashes: Set[str] = set()
        self._hash_to_path: dict = {}

    def compute_hash(self, image_path: str) -> str:
        """
        Compute perceptual hash for an image.
        
        Uses difference hash (dHash) - robust to scaling and minor edits.
        """
        try:
            img = Image.open(image_path)
            
            # Convert to grayscale and resize
            img = img.convert('L')
            img = img.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)
            
            # Compute difference hash
            pixels = np.array(img)
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Convert to hex string
            hash_bits = diff.flatten()
            hash_int = int(''.join(['1' if b else '0' for b in hash_bits]), 2)
            hash_hex = format(hash_int, f'0{self.hash_size * self.hash_size // 4}x')
            
            return hash_hex
            
        except Exception as e:
            logger.warning(f"Failed to hash {image_path}: {e}")
            # Return a unique hash based on file path to avoid false positives
            return f"error_{hash(image_path)}"

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2)) * 4  # Max distance
        
        # Convert hex to binary and count differences
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except ValueError:
            return self.hash_size * self.hash_size  # Max distance on error

    def similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity score (0-1) between two hashes."""
        distance = self.hamming_distance(hash1, hash2)
        max_distance = self.hash_size * self.hash_size
        return 1.0 - (distance / max_distance)

    def check_duplicate(self, image_path: str) -> DuplicateResult:
        """
        Check if an image is a duplicate of any known image.
        
        Args:
            image_path: Path to image to check
            
        Returns:
            DuplicateResult with duplicate status and hash
        """
        new_hash = self.compute_hash(image_path)
        
        # Check exact match first (fast)
        if new_hash in self.known_hashes:
            return DuplicateResult(
                is_duplicate=True,
                hash_value=new_hash,
                matching_hash=new_hash,
                similarity=1.0,
            )
        
        # Check similar hashes
        for known_hash in self.known_hashes:
            sim = self.similarity(new_hash, known_hash)
            if sim >= self.similarity_threshold:
                return DuplicateResult(
                    is_duplicate=True,
                    hash_value=new_hash,
                    matching_hash=known_hash,
                    similarity=sim,
                )
        
        return DuplicateResult(
            is_duplicate=False,
            hash_value=new_hash,
        )

    def add_hash(self, hash_value: str, image_path: str = None):
        """Add a hash to the known set."""
        self.known_hashes.add(hash_value)
        if image_path:
            self._hash_to_path[hash_value] = image_path

    def register_image(self, image_path: str) -> DuplicateResult:
        """
        Check for duplicate and register if new.
        
        Returns:
            DuplicateResult - if not duplicate, hash is added to known set
        """
        result = self.check_duplicate(image_path)
        
        if not result.is_duplicate:
            self.add_hash(result.hash_value, image_path)
        
        return result

    def load_from_database(self, hashes: List[str]):
        """Load existing hashes from database."""
        self.known_hashes = set(hashes)
        logger.info(f"Loaded {len(self.known_hashes)} known hashes")

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "known_hashes": len(self.known_hashes),
            "hash_size": self.hash_size,
            "threshold": self.similarity_threshold,
        }


# Global instance for reuse
_detector: Optional[DuplicateDetector] = None


def get_detector() -> DuplicateDetector:
    """Get or create global duplicate detector."""
    global _detector
    if _detector is None:
        _detector = DuplicateDetector()
    return _detector


def is_duplicate(image_path: str) -> bool:
    """Quick check if image is duplicate."""
    return get_detector().check_duplicate(image_path).is_duplicate


def register_and_check(image_path: str) -> Tuple[bool, str]:
    """
    Check if duplicate and register if new.
    
    Returns:
        Tuple of (is_duplicate, hash_value)
    """
    result = get_detector().register_image(image_path)
    return result.is_duplicate, result.hash_value

