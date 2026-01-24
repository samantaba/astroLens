"""
Test suite for AstroLens API endpoints.

Tests:
- Health checks
- Image endpoints
- Analysis endpoints
- Cross-reference endpoints
- Candidates endpoint
"""

import time
from pathlib import Path

import pytest

# Mark all tests in this module
pytestmark = pytest.mark.api


class TestHealthEndpoint:
    """Test API health endpoint."""
    
    def test_health_check(self, api_base):
        """Test health endpoint returns OK."""
        import httpx
        
        try:
            response = httpx.get(f"{api_base}/health", timeout=5.0)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "version" in data
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    def test_health_response_time(self, api_base):
        """Test health endpoint responds quickly."""
        import httpx
        
        try:
            start = time.time()
            response = httpx.get(f"{api_base}/health", timeout=5.0)
            duration = time.time() - start
            
            assert response.status_code == 200
            assert duration < 1.0, f"Health check too slow: {duration:.2f}s"
            
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestStatsEndpoint:
    """Test stats endpoint."""
    
    def test_get_stats(self, api_base):
        """Test stats endpoint returns statistics."""
        import httpx
        
        try:
            response = httpx.get(f"{api_base}/stats", timeout=10.0)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check required fields
            assert "total_images" in data
            assert "analyzed_images" in data
            assert "anomalies" in data
            
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestImagesEndpoint:
    """Test images endpoint."""
    
    def test_list_images(self, api_base):
        """Test listing images."""
        import httpx
        
        try:
            response = httpx.get(
                f"{api_base}/images",
                params={"limit": 10},
                timeout=10.0,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    def test_list_images_pagination(self, api_base):
        """Test images pagination."""
        import httpx
        
        try:
            # Get first page
            response1 = httpx.get(
                f"{api_base}/images",
                params={"limit": 5, "skip": 0},
                timeout=10.0,
            )
            
            # Get second page
            response2 = httpx.get(
                f"{api_base}/images",
                params={"limit": 5, "skip": 5},
                timeout=10.0,
            )
            
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            data1 = response1.json()
            data2 = response2.json()
            
            # If we have enough images, pages should be different
            if len(data1) == 5 and len(data2) > 0:
                # First item of page 2 should not be in page 1
                ids1 = [img.get("id") for img in data1]
                ids2 = [img.get("id") for img in data2]
                assert not set(ids1) & set(ids2), "Pagination returned overlapping results"
            
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestCandidatesEndpoint:
    """Test candidates (anomalies) endpoint."""
    
    def test_list_candidates(self, api_base):
        """Test listing anomaly candidates."""
        import httpx
        
        try:
            response = httpx.get(
                f"{api_base}/candidates",
                params={"limit": 20},
                timeout=10.0,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            
            print(f"Found {len(data)} anomaly candidates")
            
            # If we have candidates, check structure
            if data:
                candidate = data[0]
                assert "id" in candidate
                assert "filepath" in candidate
                
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestCrossRefEndpoint:
    """Test cross-reference endpoint."""
    
    def test_crossref_summary(self, api_base):
        """Test cross-reference summary endpoint."""
        import httpx
        
        try:
            response = httpx.get(
                f"{api_base}/crossref/summary",
                timeout=10.0,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check required fields
            assert "total_checked" in data
            assert "known_objects" in data
            assert "unknown_objects" in data
            
            print(f"CrossRef summary: {data}")
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    def test_crossref_single_image(self, api_base):
        """Test cross-reference single image endpoint."""
        import httpx
        
        try:
            # First get a candidate
            candidates_resp = httpx.get(
                f"{api_base}/candidates",
                params={"limit": 1},
                timeout=10.0,
            )
            
            if candidates_resp.status_code != 200:
                pytest.skip("Could not get candidates")
            
            candidates = candidates_resp.json()
            if not candidates:
                pytest.skip("No candidates to test")
            
            image_id = candidates[0]["id"]
            
            # Cross-reference it
            response = httpx.post(
                f"{api_base}/crossref/{image_id}",
                json={"force": False},
                timeout=60.0,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "image_id" in data
            assert "is_known" in data
            assert "status" in data
            
            print(f"CrossRef result: {data}")
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    def test_crossref_verify(self, api_base):
        """Test cross-reference verification endpoint."""
        import httpx
        
        try:
            # First get a candidate
            candidates_resp = httpx.get(
                f"{api_base}/candidates",
                params={"limit": 1},
                timeout=10.0,
            )
            
            if candidates_resp.status_code != 200:
                pytest.skip("Could not get candidates")
            
            candidates = candidates_resp.json()
            if not candidates:
                pytest.skip("No candidates to test")
            
            image_id = candidates[0]["id"]
            
            # Verify it
            response = httpx.post(
                f"{api_base}/crossref/{image_id}/verify",
                json={"label": "uncertain", "verified_by": "test"},
                timeout=10.0,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("ok") == True
            
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestAPIPerformance:
    """Performance tests for API."""
    
    @pytest.mark.benchmark
    def test_images_list_performance(self, api_base):
        """Benchmark image listing performance."""
        import httpx
        
        try:
            times = []
            for _ in range(5):
                start = time.time()
                response = httpx.get(
                    f"{api_base}/images",
                    params={"limit": 50},
                    timeout=30.0,
                )
                times.append(time.time() - start)
                assert response.status_code == 200
            
            avg_time = sum(times) / len(times)
            print(f"Images list avg time: {avg_time:.3f}s")
            
            assert avg_time < 2.0, f"Images list too slow: {avg_time:.2f}s avg"
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    @pytest.mark.benchmark
    def test_candidates_list_performance(self, api_base):
        """Benchmark candidates listing performance."""
        import httpx
        
        try:
            times = []
            for _ in range(5):
                start = time.time()
                response = httpx.get(
                    f"{api_base}/candidates",
                    params={"limit": 100},
                    timeout=30.0,
                )
                times.append(time.time() - start)
                assert response.status_code == 200
            
            avg_time = sum(times) / len(times)
            print(f"Candidates list avg time: {avg_time:.3f}s")
            
            assert avg_time < 2.0, f"Candidates list too slow: {avg_time:.2f}s avg"
            
        except httpx.ConnectError:
            pytest.skip("API not running")


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_404_on_missing_image(self, api_base):
        """Test 404 response for missing image."""
        import httpx
        
        try:
            response = httpx.get(
                f"{api_base}/images/99999999",
                timeout=10.0,
            )
            
            assert response.status_code == 404
            
        except httpx.ConnectError:
            pytest.skip("API not running")
    
    def test_invalid_parameters(self, api_base):
        """Test handling of invalid parameters."""
        import httpx
        
        try:
            response = httpx.get(
                f"{api_base}/images",
                params={"limit": -1},  # Invalid
                timeout=10.0,
            )
            
            # Should return 422 (validation error) or handle gracefully
            assert response.status_code in [200, 422]
            
        except httpx.ConnectError:
            pytest.skip("API not running")
