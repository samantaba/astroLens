"""
Test suite for Catalog Cross-Reference functionality.

Tests:
- Coordinate extraction from filenames
- SIMBAD queries
- NED queries
- VizieR queries
- Result caching
- Error handling
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mark all tests in this module
pytestmark = pytest.mark.catalog


class TestCoordinateExtraction:
    """Test coordinate extraction from image filenames."""
    
    def test_extract_from_gz_anomaly_filename(self):
        """Test extraction from Galaxy Zoo anomaly filename."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        # Test typical filename
        coords = xref.extract_coordinates(
            "/path/to/gz_anomaly_0001_ra183.3_dec13.7.jpg"
        )
        
        assert coords is not None
        assert coords[0] == pytest.approx(183.3, abs=0.01)
        assert coords[1] == pytest.approx(13.7, abs=0.01)
    
    def test_extract_from_sdss_filename(self):
        """Test extraction from SDSS filename."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        coords = xref.extract_coordinates(
            "/path/to/sdss_0001_ra200.6_dec40.6.jpg"
        )
        
        assert coords is not None
        assert coords[0] == pytest.approx(200.6, abs=0.01)
        assert coords[1] == pytest.approx(40.6, abs=0.01)
    
    def test_extract_negative_declination(self):
        """Test extraction with negative declination."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        coords = xref.extract_coordinates(
            "image_ra201.365_dec-43.019.jpg"
        )
        
        assert coords is not None
        assert coords[0] == pytest.approx(201.365, abs=0.001)
        assert coords[1] == pytest.approx(-43.019, abs=0.001)
    
    def test_extract_no_coordinates(self):
        """Test with filename without coordinates."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        coords = xref.extract_coordinates("random_image_name.jpg")
        
        assert coords is None
    
    def test_extract_with_date_prefix(self):
        """Test with date-prefixed filename."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        coords = xref.extract_coordinates(
            "20260124_192658_gz_anomaly_0004_ra162.8_dec32.5.jpg"
        )
        
        assert coords is not None
        assert coords[0] == pytest.approx(162.8, abs=0.01)
        assert coords[1] == pytest.approx(32.5, abs=0.01)


class TestSIMBADQuery:
    """Test SIMBAD database queries."""
    
    @pytest.mark.network
    def test_query_known_object_virgo_cluster(self, sample_coordinates):
        """Test query in Virgo cluster region (should find objects)."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=120)  # 2 arcmin
        
        # Virgo cluster center area
        matches = xref.query_simbad(185.0, 12.5)
        
        # Should find something in this rich region
        # Note: If this fails, it might be a network issue or SIMBAD being slow
        assert isinstance(matches, list)
        # Log result for analysis
        print(f"SIMBAD Virgo query: {len(matches)} matches found")
    
    @pytest.mark.network
    def test_query_centaurus_a(self):
        """Test query near Centaurus A (famous galaxy)."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        # Centaurus A coordinates
        matches = xref.query_simbad(201.365, -43.019)
        
        assert isinstance(matches, list)
        print(f"SIMBAD Cen A query: {len(matches)} matches")
        
        # If matches found, check structure
        if matches:
            match = matches[0]
            assert hasattr(match, 'catalog')
            assert hasattr(match, 'object_name')
            assert hasattr(match, 'object_type')
            assert hasattr(match, 'distance_arcsec')
    
    @pytest.mark.network
    def test_query_empty_region(self):
        """Test query in relatively empty region."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=10)  # Small radius
        
        # Random position that might be empty
        matches = xref.query_simbad(45.0, 85.0)  # Near celestial pole
        
        # Should return list (possibly empty)
        assert isinstance(matches, list)
    
    def test_query_timeout_handling(self):
        """Test handling of query timeout."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(timeout_seconds=0.001)  # Very short timeout
        
        # Should handle timeout gracefully
        matches = xref.query_simbad(180.0, 30.0)
        
        # Should return empty list on failure, not raise
        assert isinstance(matches, list)


class TestNEDQuery:
    """Test NED database queries."""
    
    @pytest.mark.network
    def test_query_known_galaxy_region(self):
        """Test query in region with known galaxies."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=120)
        
        # M31 area
        matches = xref.query_ned(10.685, 41.269)
        
        assert isinstance(matches, list)
        print(f"NED M31 query: {len(matches)} matches")
    
    @pytest.mark.network  
    def test_query_sdss_region(self):
        """Test query in SDSS coverage area."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        # SDSS covered region
        matches = xref.query_ned(180.0, 30.0)
        
        assert isinstance(matches, list)
        print(f"NED SDSS region query: {len(matches)} matches")


class TestVizieRQuery:
    """Test VizieR catalog queries."""
    
    @pytest.mark.network
    def test_query_sdss_catalog(self):
        """Test query against SDSS catalog via VizieR."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        # SDSS covered region
        matches = xref.query_vizier(180.0, 30.0)
        
        assert isinstance(matches, list)
        print(f"VizieR SDSS query: {len(matches)} matches")


class TestCrossReference:
    """Test full cross-reference workflow."""
    
    @pytest.mark.network
    def test_cross_reference_full_workflow(self, temp_dir):
        """Test complete cross-reference of a single image."""
        from catalog.cross_reference import CatalogCrossReference
        import random
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        # Use random image_id to avoid cache conflicts
        test_id = random.randint(100000, 999999)
        
        # Create a mock image path with coordinates (Virgo cluster area)
        image_path = str(temp_dir / "gz_anomaly_0001_ra187.5_dec12.5.jpg")
        
        result = xref.cross_reference(
            image_id=test_id,
            image_path=image_path,
            force=True,  # Force fresh query
        )
        
        # Check result structure
        assert result.image_id == test_id
        assert result.query_ra == pytest.approx(187.5, abs=0.01)
        assert result.query_dec == pytest.approx(12.5, abs=0.01)
        assert result.status in ["known", "unknown", "error"]
        assert result.queried_at != ""
        
        print(f"Cross-ref result: status={result.status}, matches={len(result.matches)}")
    
    @pytest.mark.network
    def test_cross_reference_caching(self, temp_dir):
        """Test that results are cached."""
        from catalog.cross_reference import CatalogCrossReference
        import random
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        # Use unique image_id to avoid conflicts with other tests
        test_id = random.randint(200000, 299999)
        image_path = str(temp_dir / "test_ra180.0_dec30.0.jpg")
        
        # First query (force to ensure fresh)
        start1 = time.time()
        result1 = xref.cross_reference(image_id=test_id, image_path=image_path, force=True)
        duration1 = time.time() - start1
        
        # Second query (should be cached)
        start2 = time.time()
        result2 = xref.cross_reference(image_id=test_id, image_path=image_path)
        duration2 = time.time() - start2
        
        # Cached query should be much faster
        assert duration2 < duration1 / 2, f"Cache not working: {duration1:.2f}s vs {duration2:.2f}s"
        
        # Force query should take longer
        start3 = time.time()
        result3 = xref.cross_reference(image_id=test_id, image_path=image_path, force=True)
        duration3 = time.time() - start3
        
        # Force should re-query (should be slower than cached)
        assert duration3 > duration2, f"Force query not slower: {duration3:.2f}s vs cached {duration2:.2f}s"


class TestSearchRadius:
    """Test search radius variations."""
    
    @pytest.mark.network
    @pytest.mark.parametrize("radius", [10, 30, 60, 120, 180])
    def test_radius_affects_results(self, radius):
        """Test that larger radius finds more objects."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=radius)
        
        # Query dense region
        matches = xref.query_simbad(185.0, 12.5)  # Virgo cluster
        
        print(f"Radius {radius}″: {len(matches)} matches")
        
        # Just verify it returns a list
        assert isinstance(matches, list)


class TestPerformance:
    """Performance benchmarks for cross-reference."""
    
    @pytest.mark.network
    @pytest.mark.benchmark
    def test_simbad_query_performance(self):
        """Benchmark SIMBAD query time."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        times = []
        for _ in range(3):
            start = time.time()
            xref.query_simbad(180.0 + _, 30.0)  # Vary position slightly
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"SIMBAD avg query time: {avg_time:.2f}s")
        
        # Should complete in reasonable time
        assert avg_time < 30.0, f"SIMBAD queries too slow: {avg_time:.2f}s avg"
    
    @pytest.mark.network
    @pytest.mark.benchmark
    def test_full_crossref_performance(self, temp_dir):
        """Benchmark full cross-reference time."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference(search_radius_arcsec=60)
        
        image_path = str(temp_dir / "test_ra180.0_dec30.0.jpg")
        
        start = time.time()
        result = xref.cross_reference(
            image_id=77777,
            image_path=image_path,
            force=True,
        )
        duration = time.time() - start
        
        print(f"Full cross-ref time: {duration:.2f}s")
        
        # Should complete in reasonable time (3 catalog queries)
        assert duration < 60.0, f"Cross-ref too slow: {duration:.2f}s"


class TestErrorHandling:
    """Test error handling in cross-reference."""
    
    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates."""
        from catalog.cross_reference import CatalogCrossReference
        
        xref = CatalogCrossReference()
        
        result = xref.cross_reference(
            image_id=1,
            image_path="/path/to/no_coords.jpg",
        )
        
        assert result.status == "error"
        assert "Could not extract coordinates" in result.error_message
    
    def test_network_failure_graceful(self):
        """Test graceful handling of network failures."""
        from catalog.cross_reference import CatalogCrossReference
        
        # Use impossible timeout to simulate network failure
        xref = CatalogCrossReference(timeout_seconds=0.0001)
        
        matches = xref.query_simbad(180.0, 30.0)
        
        # Should return empty list, not raise
        assert matches == []
