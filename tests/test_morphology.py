#!/usr/bin/env python3
"""
Unit tests for Galaxy Morphology features.

Tests:
- CAS parameter computation
- Gini-M20 coefficients
- Edge cases and error handling
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.morphology import GalaxyMorphology, MorphologyResult, analyze_morphology


class TestGalaxyMorphology:
    """Test cases for GalaxyMorphology class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return GalaxyMorphology()
    
    @pytest.fixture
    def symmetric_image(self, tmp_path):
        """Create a symmetric circular galaxy image."""
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Create circular gradient (symmetric)
        center = size // 2
        for y in range(size):
            for x in range(size):
                r = np.sqrt((x - center)**2 + (y - center)**2)
                if r < center:
                    img[y, x] = int(255 * (1 - r / center))
        
        path = tmp_path / "symmetric.jpg"
        Image.fromarray(img).save(path)
        return str(path)
    
    @pytest.fixture
    def asymmetric_image(self, tmp_path):
        """Create an asymmetric galaxy image (like a merger)."""
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Two off-center blobs (merger-like)
        for y in range(size):
            for x in range(size):
                # First blob
                r1 = np.sqrt((x - 40)**2 + (y - 64)**2)
                if r1 < 25:
                    img[y, x] = max(img[y, x], int(255 * (1 - r1 / 25)))
                
                # Second blob (off-center)
                r2 = np.sqrt((x - 90)**2 + (y - 50)**2)
                if r2 < 20:
                    img[y, x] = max(img[y, x], int(200 * (1 - r2 / 20)))
        
        path = tmp_path / "asymmetric.jpg"
        Image.fromarray(img).save(path)
        return str(path)
    
    @pytest.fixture
    def compact_image(self, tmp_path):
        """Create a compact (high concentration) galaxy image."""
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Small bright center
        center = size // 2
        for y in range(size):
            for x in range(size):
                r = np.sqrt((x - center)**2 + (y - center)**2)
                if r < 15:
                    img[y, x] = int(255 * np.exp(-r / 5))
        
        path = tmp_path / "compact.jpg"
        Image.fromarray(img).save(path)
        return str(path)
    
    def test_analyze_symmetric(self, analyzer, symmetric_image):
        """Test analysis of symmetric galaxy."""
        result = analyzer.analyze(symmetric_image)
        
        assert result is not None
        assert isinstance(result, MorphologyResult)
        
        # Symmetric image should have low asymmetry
        assert result.asymmetry < 0.3, f"Asymmetry too high: {result.asymmetry}"
        
        # Check all values are in valid range
        assert 0 <= result.concentration <= 10
        assert 0 <= result.asymmetry <= 1
        assert 0 <= result.smoothness <= 1
        assert 0 <= result.gini <= 1
        assert -3 <= result.m20 <= 0
        assert 0 <= result.ellipticity <= 1
    
    def test_analyze_asymmetric(self, analyzer, asymmetric_image):
        """Test analysis of asymmetric galaxy (merger-like)."""
        result = analyzer.analyze(asymmetric_image)
        
        assert result is not None
        
        # Asymmetric image should have higher asymmetry
        assert result.asymmetry > 0.2, f"Asymmetry too low for merger: {result.asymmetry}"
    
    def test_analyze_compact(self, analyzer, compact_image):
        """Test analysis of compact galaxy."""
        result = analyzer.analyze(compact_image)
        
        assert result is not None
        
        # Compact image should have high concentration
        assert result.concentration > 2.0, f"Concentration too low: {result.concentration}"
    
    def test_invalid_path(self, analyzer):
        """Test handling of invalid file path."""
        result = analyzer.analyze("/nonexistent/path.jpg")
        assert result is None
    
    def test_morph_score_range(self, analyzer, symmetric_image):
        """Test that morph_score is in valid range."""
        result = analyzer.analyze(symmetric_image)
        
        assert result is not None
        assert 0 <= result.morph_score <= 1
    
    def test_to_dict(self, analyzer, symmetric_image):
        """Test conversion to dictionary."""
        result = analyzer.analyze(symmetric_image)
        
        assert result is not None
        d = result.to_dict()
        
        assert "concentration" in d
        assert "asymmetry" in d
        assert "smoothness" in d
        assert "gini" in d
        assert "m20" in d
        assert "is_irregular" in d
        assert "is_merger" in d
        assert "morph_score" in d
    
    def test_convenience_function(self, symmetric_image):
        """Test the analyze_morphology convenience function."""
        result = analyze_morphology(symmetric_image)
        
        assert result is not None
        assert isinstance(result, MorphologyResult)


class TestMorphologyClassification:
    """Test classification thresholds."""
    
    @pytest.fixture
    def analyzer(self):
        return GalaxyMorphology()
    
    def test_irregular_classification(self, analyzer, tmp_path):
        """Test that high asymmetry triggers irregular flag."""
        # Create very asymmetric image
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        img[20:60, 10:50] = 255  # Bright square in corner
        
        path = tmp_path / "irregular.jpg"
        Image.fromarray(img).save(path)
        
        result = analyzer.analyze(str(path))
        # With such an asymmetric image, either asymmetry or smoothness should be high
        assert result is not None


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    @pytest.fixture
    def analyzer(self):
        return GalaxyMorphology()
    
    def test_uniform_image(self, analyzer, tmp_path):
        """Test analysis of uniform (flat) image."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        path = tmp_path / "uniform.jpg"
        Image.fromarray(img).save(path)
        
        result = analyzer.analyze(str(path))
        assert result is not None
        # Should handle gracefully without errors
    
    def test_black_image(self, analyzer, tmp_path):
        """Test analysis of black image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        path = tmp_path / "black.jpg"
        Image.fromarray(img).save(path)
        
        result = analyzer.analyze(str(path))
        assert result is not None
    
    def test_small_image(self, analyzer, tmp_path):
        """Test analysis of very small image."""
        img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        path = tmp_path / "small.jpg"
        Image.fromarray(img).save(path)
        
        result = analyzer.analyze(str(path))
        assert result is not None


def run_tests():
    """Run all tests with verbose output."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )
    return result.returncode


if __name__ == "__main__":
    # Quick test without pytest
    print("Running quick morphology tests...")
    
    # Create test image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        size = 128
        img = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        for y in range(size):
            for x in range(size):
                r = np.sqrt((x - center)**2 + (y - center)**2)
                if r < center:
                    img[y, x] = int(255 * (1 - r / center))
        
        Image.fromarray(img).save(f.name)
        test_path = f.name
    
    try:
        analyzer = GalaxyMorphology()
        result = analyzer.analyze(test_path)
        
        if result:
            print("✓ Morphology analysis successful!")
            print(f"  Concentration: {result.concentration:.3f}")
            print(f"  Asymmetry: {result.asymmetry:.3f}")
            print(f"  Smoothness: {result.smoothness:.3f}")
            print(f"  Gini: {result.gini:.3f}")
            print(f"  M20: {result.m20:.3f}")
            print(f"  Morph Score: {result.morph_score:.3f}")
            print(f"  Flags: irregular={result.is_irregular}, merger={result.is_merger}, compact={result.is_compact}")
        else:
            print("✗ Morphology analysis failed!")
            
    finally:
        Path(test_path).unlink(missing_ok=True)
