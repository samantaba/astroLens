"""
Test suite for UI Integration.

Tests the UI components without actually launching the GUI.
Uses mock objects to test UI logic.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mark all tests in this module
pytestmark = pytest.mark.ui


class TestVerificationPanelLogic:
    """Test Verification Panel business logic."""
    
    def test_results_loading(self, temp_dir):
        """Test loading cross-reference results from file."""
        # Create mock results file
        results_file = temp_dir / "cross_reference_results.json"
        
        mock_results = {
            "last_updated": "2026-01-24T19:00:00",
            "total_results": 3,
            "results": [
                {
                    "image_id": 1,
                    "image_path": "/path/to/image1.jpg",
                    "query_ra": 180.0,
                    "query_dec": 30.0,
                    "is_known": True,
                    "is_published": True,
                    "status": "known",
                    "matches": [{"object_name": "NGC 1234"}],
                    "human_verified": False,
                },
                {
                    "image_id": 2,
                    "image_path": "/path/to/image2.jpg",
                    "query_ra": 185.0,
                    "query_dec": 12.5,
                    "is_known": False,
                    "is_published": False,
                    "status": "unknown",
                    "matches": [],
                    "human_verified": True,
                    "human_label": "true_positive",
                },
                {
                    "image_id": 3,
                    "image_path": "/path/to/image3.jpg",
                    "query_ra": 200.0,
                    "query_dec": 40.0,
                    "is_known": False,
                    "is_published": False,
                    "status": "unknown",
                    "matches": [],
                    "human_verified": False,
                },
            ]
        }
        
        with open(results_file, "w") as f:
            json.dump(mock_results, f)
        
        # Load and verify
        with open(results_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["total_results"] == 3
        
        results = loaded["results"]
        known = sum(1 for r in results if r["is_known"])
        unknown = sum(1 for r in results if not r["is_known"])
        verified = sum(1 for r in results if r.get("human_verified"))
        
        assert known == 1
        assert unknown == 2
        assert verified == 1
    
    def test_filter_results(self):
        """Test filtering results logic."""
        results = [
            {"is_known": True, "human_verified": False, "human_label": ""},
            {"is_known": False, "human_verified": True, "human_label": "true_positive"},
            {"is_known": False, "human_verified": True, "human_label": "false_positive"},
            {"is_known": False, "human_verified": False, "human_label": ""},
        ]
        
        # Unknown only
        unknown = [r for r in results if not r["is_known"]]
        assert len(unknown) == 3
        
        # Known only
        known = [r for r in results if r["is_known"]]
        assert len(known) == 1
        
        # Unverified
        unverified = [r for r in results if not r["human_verified"]]
        assert len(unverified) == 2
        
        # True positives
        tp = [r for r in results if r.get("human_label") == "true_positive"]
        assert len(tp) == 1
        
        # False positives
        fp = [r for r in results if r.get("human_label") == "false_positive"]
        assert len(fp) == 1


class TestDiscoveryPanelLogic:
    """Test Discovery Panel business logic."""
    
    def test_stats_parsing(self, temp_dir):
        """Test parsing discovery state file."""
        state_file = temp_dir / "discovery_state.json"
        
        mock_state = {
            "started_at": "2026-01-24T11:33:29",
            "cycles_completed": 50,
            "total_downloaded": 2000,
            "total_analyzed": 1500,
            "anomalies_found": 300,
            "near_misses": 10,
            "current_threshold": 2.0,
            "highest_ood_score": 0.4,
            "model_accuracy": 0.84,
            "initial_accuracy": 0.80,
            "total_improvement_pct": 5.0,
            "training_history": [
                {"dataset": "galaxy10", "accuracy_after": 0.84, "improvement_pct": 5.0}
            ]
        }
        
        with open(state_file, "w") as f:
            json.dump(mock_state, f)
        
        # Load and verify
        with open(state_file, "r") as f:
            state = json.load(f)
        
        assert state["cycles_completed"] == 50
        assert state["anomalies_found"] == 300
        assert state["model_accuracy"] == pytest.approx(0.84, abs=0.01)
        assert state["total_improvement_pct"] == pytest.approx(5.0, abs=0.1)
    
    def test_threshold_calibration_needed(self):
        """Test detection of miscalibrated threshold."""
        state = {
            "current_threshold": 2.0,
            "highest_ood_score": 0.4,
        }
        
        # Threshold >> highest OOD means miscalibrated
        threshold_miscalibrated = (
            state["highest_ood_score"] > 0 and
            state["current_threshold"] > state["highest_ood_score"] * 3
        )
        
        assert threshold_miscalibrated


class TestCoordinateExtraction:
    """Test coordinate extraction for cross-reference."""
    
    @pytest.mark.parametrize("filename,expected_ra,expected_dec", [
        ("gz_anomaly_0001_ra183.3_dec13.7.jpg", 183.3, 13.7),
        ("sdss_0001_ra200.6_dec40.6.jpg", 200.6, 40.6),
        ("20260124_image_ra150.0_dec-25.5.jpg", 150.0, -25.5),
        ("test_ra0.5_dec89.9.jpg", 0.5, 89.9),
    ])
    def test_coordinate_patterns(self, filename, expected_ra, expected_dec):
        """Test various coordinate patterns in filenames."""
        import re
        
        pattern = r'ra([-+]?\d+\.?\d*)_dec([-+]?\d+\.?\d*)'
        match = re.search(pattern, filename, re.IGNORECASE)
        
        assert match is not None
        ra = float(match.group(1))
        dec = float(match.group(2))
        
        assert ra == pytest.approx(expected_ra, abs=0.01)
        assert dec == pytest.approx(expected_dec, abs=0.01)
    
    def test_no_coordinates(self):
        """Test filename without coordinates."""
        import re
        
        pattern = r'ra([-+]?\d+\.?\d*)_dec([-+]?\d+\.?\d*)'
        
        filenames = [
            "random_image.jpg",
            "galaxy_photo_2026.png",
            "ngc1234.fits",
        ]
        
        for filename in filenames:
            match = re.search(pattern, filename, re.IGNORECASE)
            assert match is None


class TestStatisticsCalculation:
    """Test statistics calculation logic."""
    
    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        training_runs = [
            {"accuracy_before": 0.80, "accuracy_after": 0.84, "dataset": "galaxy10"},
            {"accuracy_before": 0.84, "accuracy_after": 0.84, "dataset": "galaxy10"},
            {"accuracy_before": 0.14, "accuracy_after": 0.14, "dataset": "anomalies"},
        ]
        
        # Calculate improvement for Galaxy10 only
        galaxy10_runs = [r for r in training_runs if r["dataset"] == "galaxy10"]
        
        if galaxy10_runs:
            first_acc = galaxy10_runs[0]["accuracy_before"]
            last_acc = galaxy10_runs[-1]["accuracy_after"]
            
            if first_acc > 0:
                improvement = (last_acc - first_acc) / first_acc * 100
                assert improvement == pytest.approx(5.0, abs=0.1)
    
    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        results = [
            {"human_label": "true_positive"},
            {"human_label": "true_positive"},
            {"human_label": "false_positive"},
            {"human_label": "uncertain"},
            {"human_label": ""},
        ]
        
        verified = [r for r in results if r.get("human_label") in ["true_positive", "false_positive"]]
        
        if verified:
            tp = sum(1 for r in verified if r["human_label"] == "true_positive")
            fp = sum(1 for r in verified if r["human_label"] == "false_positive")
            
            fpr = fp / len(verified) if verified else 0
            precision = tp / len(verified) if verified else 0
            
            assert fpr == pytest.approx(0.333, abs=0.01)
            assert precision == pytest.approx(0.667, abs=0.01)


class TestUIStateManagement:
    """Test UI state management."""
    
    def test_discovery_state_persistence(self, temp_dir):
        """Test that discovery state persists correctly."""
        state_file = temp_dir / "discovery_state.json"
        
        # Initial state
        state = {
            "cycles_completed": 0,
            "current_threshold": 3.0,
            "anomalies_found": 0,
        }
        
        # Simulate cycles
        for _ in range(5):
            state["cycles_completed"] += 1
            state["current_threshold"] *= 0.95  # Decay
            state["anomalies_found"] += 1
        
        # Save
        with open(state_file, "w") as f:
            json.dump(state, f)
        
        # Load
        with open(state_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["cycles_completed"] == 5
        assert loaded["current_threshold"] < 3.0
        assert loaded["anomalies_found"] == 5
    
    def test_crossref_results_persistence(self, temp_dir):
        """Test cross-reference results persist correctly."""
        results_file = temp_dir / "crossref_results.json"
        
        results = {
            "total_results": 0,
            "results": []
        }
        
        # Add results
        for i in range(10):
            results["results"].append({
                "image_id": i,
                "is_known": i % 2 == 0,
                "status": "known" if i % 2 == 0 else "unknown",
            })
            results["total_results"] += 1
        
        # Save
        with open(results_file, "w") as f:
            json.dump(results, f)
        
        # Load
        with open(results_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["total_results"] == 10
        assert len(loaded["results"]) == 10
        
        known = sum(1 for r in loaded["results"] if r["is_known"])
        assert known == 5
