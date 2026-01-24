#!/usr/bin/env python3
"""
AstroLens Comprehensive Test Runner

Runs all tests, collects results, performs regression analysis,
and generates reports.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --quick            # Quick tests only
    python tests/run_tests.py --category api     # Run API tests only
    python tests/run_tests.py --benchmark        # Run benchmarks
    python tests/run_tests.py --regression       # Regression analysis
    python tests/run_tests.py --report           # Generate full report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test artifacts
TESTS_DIR = PROJECT_ROOT / "tests"
ARTIFACTS_DIR = TESTS_DIR / "test_artifacts"
RESULTS_FILE = ARTIFACTS_DIR / "test_results.json"
REPORT_FILE = ARTIFACTS_DIR / "test_report.md"


class TestRunner:
    """Comprehensive test runner with analysis."""
    
    def __init__(self):
        self.results: Dict = {}
        self.start_time: Optional[datetime] = None
        ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    def run_pytest(
        self,
        markers: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict:
        """Run pytest with specified options."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(TESTS_DIR),
            "-v" if verbose else "-q",
            "--tb=short",
            "-x" if not markers else "",  # Stop on first failure for quick runs
        ]
        
        # Add markers
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])
        
        # Add extra args
        if extra_args:
            cmd.extend(extra_args)
        
        # Filter out empty strings
        cmd = [c for c in cmd if c]
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
        self.start_time = datetime.now()
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "markers": markers,
        }
    
    def run_quick_tests(self) -> Dict:
        """Run quick tests (no network, no benchmarks)."""
        print("\n" + "=" * 60)
        print("QUICK TESTS (no network, no benchmarks)")
        print("=" * 60 + "\n")
        
        return self.run_pytest(
            extra_args=[
                "-m", "not network and not benchmark and not integration",
                "--timeout=30",
            ]
        )
    
    def run_category_tests(self, category: str) -> Dict:
        """Run tests for a specific category."""
        print("\n" + "=" * 60)
        print(f"CATEGORY: {category.upper()}")
        print("=" * 60 + "\n")
        
        return self.run_pytest(markers=[category])
    
    def run_all_tests(self) -> Dict:
        """Run all tests including network tests."""
        print("\n" + "=" * 60)
        print("ALL TESTS")
        print("=" * 60 + "\n")
        
        return self.run_pytest()
    
    def run_benchmarks(self) -> Dict:
        """Run performance benchmarks."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 60 + "\n")
        
        return self.run_pytest(markers=["benchmark"])
    
    def load_results(self) -> Dict:
        """Load test results from file."""
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
        return {"history": []}
    
    def analyze_regression(self) -> Dict:
        """Analyze test results for regressions."""
        data = self.load_results()
        history = data.get("history", [])
        
        if len(history) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 test runs for regression analysis"}
        
        # Compare last two sessions
        current = history[-1]
        previous = history[-2]
        
        analysis = {
            "current_session": {
                "timestamp": current.get("session_start", ""),
                "total": current.get("total_tests", 0),
                "passed": current.get("passed", 0),
                "failed": current.get("failed", 0),
            },
            "previous_session": {
                "timestamp": previous.get("session_start", ""),
                "total": previous.get("total_tests", 0),
                "passed": previous.get("passed", 0),
                "failed": previous.get("failed", 0),
            },
            "regressions": [],
            "improvements": [],
            "new_tests": [],
            "removed_tests": [],
        }
        
        # Build test maps
        current_tests = {r["test_name"]: r for r in current.get("results", [])}
        previous_tests = {r["test_name"]: r for r in previous.get("results", [])}
        
        # Find regressions (was passing, now failing)
        for name, result in current_tests.items():
            if name in previous_tests:
                prev = previous_tests[name]
                if prev["passed"] and not result["passed"]:
                    analysis["regressions"].append({
                        "test": name,
                        "previous_duration": prev["duration_seconds"],
                        "current_duration": result["duration_seconds"],
                    })
                elif not prev["passed"] and result["passed"]:
                    analysis["improvements"].append({
                        "test": name,
                        "previous_duration": prev["duration_seconds"],
                        "current_duration": result["duration_seconds"],
                    })
            else:
                analysis["new_tests"].append(name)
        
        # Find removed tests
        for name in previous_tests:
            if name not in current_tests:
                analysis["removed_tests"].append(name)
        
        # Determine status
        if analysis["regressions"]:
            analysis["status"] = "regressions_found"
        elif analysis["improvements"]:
            analysis["status"] = "improvements_found"
        else:
            analysis["status"] = "stable"
        
        return analysis
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        data = self.load_results()
        regression = self.analyze_regression()
        
        report = []
        report.append("# AstroLens Test Report")
        report.append(f"\n**Generated:** {datetime.now().isoformat()}")
        report.append("")
        
        # Current Status
        report.append("## Current Status")
        if data.get("history"):
            latest = data["history"][-1]
            total = latest.get("total_tests", 0)
            passed = latest.get("passed", 0)
            failed = latest.get("failed", 0)
            
            pass_rate = passed / total * 100 if total > 0 else 0
            
            report.append(f"- **Total Tests:** {total}")
            report.append(f"- **Passed:** {passed} ({pass_rate:.1f}%)")
            report.append(f"- **Failed:** {failed}")
            report.append(f"- **Last Run:** {latest.get('session_start', 'Unknown')}")
        else:
            report.append("No test results available.")
        
        # Regression Analysis
        report.append("\n## Regression Analysis")
        if regression.get("status") == "insufficient_data":
            report.append(regression.get("message", ""))
        else:
            status = regression.get("status", "unknown")
            if status == "regressions_found":
                report.append("⚠️ **Regressions Detected!**\n")
                for reg in regression.get("regressions", []):
                    report.append(f"- `{reg['test']}`")
            elif status == "improvements_found":
                report.append("✅ **Improvements Found!**\n")
                for imp in regression.get("improvements", []):
                    report.append(f"- `{imp['test']}`")
            else:
                report.append("✅ **All tests stable.**")
            
            if regression.get("new_tests"):
                report.append("\n### New Tests")
                for t in regression["new_tests"]:
                    report.append(f"- `{t}`")
        
        # Test Categories
        report.append("\n## Test Categories")
        if data.get("history"):
            latest = data["history"][-1]
            categories = {}
            for result in latest.get("results", []):
                cat = result.get("category", "general")
                if cat not in categories:
                    categories[cat] = {"passed": 0, "failed": 0}
                if result["passed"]:
                    categories[cat]["passed"] += 1
                else:
                    categories[cat]["failed"] += 1
            
            report.append("\n| Category | Passed | Failed | Pass Rate |")
            report.append("|----------|--------|--------|-----------|")
            for cat, stats in sorted(categories.items()):
                total = stats["passed"] + stats["failed"]
                rate = stats["passed"] / total * 100 if total > 0 else 0
                report.append(f"| {cat} | {stats['passed']} | {stats['failed']} | {rate:.1f}% |")
        
        # Performance Trends
        report.append("\n## Performance Trends")
        if len(data.get("history", [])) >= 2:
            # Compare durations
            current = data["history"][-1]
            previous = data["history"][-2]
            
            current_tests = {r["test_name"]: r for r in current.get("results", [])}
            previous_tests = {r["test_name"]: r for r in previous.get("results", [])}
            
            slowdowns = []
            speedups = []
            
            for name, result in current_tests.items():
                if name in previous_tests:
                    prev = previous_tests[name]
                    if prev["duration_seconds"] > 0:
                        change = (result["duration_seconds"] - prev["duration_seconds"]) / prev["duration_seconds"] * 100
                        if change > 50:  # 50% slower
                            slowdowns.append((name, change))
                        elif change < -50:  # 50% faster
                            speedups.append((name, change))
            
            if slowdowns:
                report.append("\n### Slowdowns (>50% slower)")
                for name, change in sorted(slowdowns, key=lambda x: -x[1])[:5]:
                    report.append(f"- `{name}`: +{change:.1f}%")
            
            if speedups:
                report.append("\n### Speedups (>50% faster)")
                for name, change in sorted(speedups, key=lambda x: x[1])[:5]:
                    report.append(f"- `{name}`: {change:.1f}%")
        else:
            report.append("Need more test runs for trend analysis.")
        
        # Recommendations
        report.append("\n## Recommendations")
        recommendations = []
        
        if regression.get("regressions"):
            recommendations.append("- 🔴 Fix regressions before merging")
        
        if data.get("history"):
            latest = data["history"][-1]
            failed = latest.get("failed", 0)
            if failed > 0:
                recommendations.append(f"- ⚠️ {failed} failing tests need attention")
        
        if not recommendations:
            recommendations.append("- ✅ All tests passing!")
        
        report.extend(recommendations)
        
        return "\n".join(report)
    
    def save_report(self):
        """Save report to file."""
        report = self.generate_report()
        with open(REPORT_FILE, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="AstroLens Test Runner")
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick tests only (no network, no benchmarks)",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["api", "catalog", "model", "ui", "integration"],
        help="Run tests for specific category",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmarks",
    )
    parser.add_argument(
        "--regression", "-r",
        action="store_true",
        help="Show regression analysis",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full test report",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all tests including network tests",
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.regression:
        analysis = runner.analyze_regression()
        print("\n" + "=" * 60)
        print("REGRESSION ANALYSIS")
        print("=" * 60)
        print(json.dumps(analysis, indent=2))
        return
    
    if args.report:
        runner.save_report()
        print("\n" + runner.generate_report())
        return
    
    # Run tests based on options
    if args.quick:
        result = runner.run_quick_tests()
    elif args.category:
        result = runner.run_category_tests(args.category)
    elif args.benchmark:
        result = runner.run_benchmarks()
    elif args.all:
        result = runner.run_all_tests()
    else:
        # Default: run quick tests
        result = runner.run_quick_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RUN COMPLETE")
    print("=" * 60)
    print(f"Exit code: {result['exit_code']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    
    # Generate report
    runner.save_report()
    
    # Exit with test exit code
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
