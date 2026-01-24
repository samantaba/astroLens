"""
Pytest configuration and fixtures for AstroLens tests.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test artifacts directory
TEST_ARTIFACTS_DIR = Path(__file__).parent / "test_artifacts"
TEST_ARTIFACTS_DIR.mkdir(exist_ok=True)

# Test results file
TEST_RESULTS_FILE = TEST_ARTIFACTS_DIR / "test_results.json"


class TestResultsCollector:
    """Collect and persist test results for analysis."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.session_start = datetime.now()
        self.load_history()
    
    def load_history(self):
        """Load previous test results."""
        if TEST_RESULTS_FILE.exists():
            try:
                with open(TEST_RESULTS_FILE, "r") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
            except:
                self.history = []
        else:
            self.history = []
    
    def add_result(self, test_name: str, passed: bool, duration: float, 
                   details: Dict = None, category: str = "general"):
        """Add a test result."""
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "duration_seconds": duration,
            "category": category,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
    
    def save(self):
        """Save results to file."""
        session_summary = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "results": self.results,
        }
        
        self.history.append(session_summary)
        
        # Keep last 50 sessions
        self.history = self.history[-50:]
        
        with open(TEST_RESULTS_FILE, "w") as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "history": self.history,
            }, f, indent=2)
    
    def get_regression_data(self) -> Dict:
        """Get data for regression analysis."""
        if not self.history:
            return {}
        
        # Compare current to previous sessions
        test_trends = {}
        for session in self.history:
            for result in session.get("results", []):
                name = result["test_name"]
                if name not in test_trends:
                    test_trends[name] = []
                test_trends[name].append({
                    "passed": result["passed"],
                    "duration": result["duration_seconds"],
                    "timestamp": result["timestamp"],
                })
        
        return test_trends


# Global collector instance
_collector = None


def get_collector() -> TestResultsCollector:
    global _collector
    if _collector is None:
        _collector = TestResultsCollector()
    return _collector


@pytest.fixture(scope="session")
def test_collector():
    """Provide test results collector."""
    collector = get_collector()
    yield collector
    collector.save()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_coordinates():
    """Sample astronomical coordinates for testing."""
    return [
        {"ra": 150.0, "dec": 2.0, "name": "SDSS region"},  # SDSS DR12 coverage
        {"ra": 180.0, "dec": 30.0, "name": "Spring sky"},
        {"ra": 210.0, "dec": 45.0, "name": "Bootes region"},
        {"ra": 185.0, "dec": 12.5, "name": "Virgo cluster"},  # Known galaxy cluster
        {"ra": 201.365, "dec": -43.019, "name": "Centaurus A"},  # Famous galaxy
    ]


@pytest.fixture
def api_base():
    """API base URL."""
    return os.environ.get("ASTROLENS_API_URL", "http://localhost:8000")


@pytest.fixture
def artifacts_dir():
    """Path to astrolens artifacts."""
    return Path(__file__).parent.parent.parent / "astrolens_artifacts"


# Pytest hooks for result collection
def pytest_runtest_makereport(item, call):
    """Collect test results."""
    if call.when == "call":
        collector = get_collector()
        
        # Get category from markers
        category = "general"
        for marker in item.iter_markers():
            if marker.name in ["ui", "api", "catalog", "model", "integration"]:
                category = marker.name
                break
        
        collector.add_result(
            test_name=item.name,
            passed=call.excinfo is None,
            duration=call.duration,
            category=category,
            details={
                "module": item.module.__name__ if item.module else "",
                "nodeid": item.nodeid,
            }
        )


def pytest_sessionfinish(session, exitstatus):
    """Save results at end of session."""
    collector = get_collector()
    collector.save()
