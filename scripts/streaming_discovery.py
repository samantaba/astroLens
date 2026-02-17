#!/usr/bin/env python3
"""
AstroLens Streaming Discovery Engine (v1.1.0)

Multi-day autonomous discovery with self-correcting intelligence,
daily reporting, and publishing-ready summaries.

This wraps the core DiscoveryLoop with:
- Daily HTML report generation (charts, candidate rankings)
- Self-correcting strategy (thresholds, source rebalancing)
- Trend analysis and improvement tracking
- Final publishing summary after N days

Usage:
    python scripts/streaming_discovery.py --days 7
    python scripts/streaming_discovery.py --days 3 --aggressive
    python scripts/streaming_discovery.py --report-only  # Generate report from existing data

Reports are saved to: astrolens_artifacts/streaming_reports/
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR, ARTIFACTS_DIR, DOWNLOADS_DIR

# Streaming-specific paths
STREAMING_DIR = ARTIFACTS_DIR / "streaming_reports"
STREAMING_DIR.mkdir(parents=True, exist_ok=True)

STREAMING_STATE_FILE = DATA_DIR / "streaming_state.json"
DAILY_REPORTS_DIR = STREAMING_DIR / "daily"
DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = DATA_DIR / "streaming_discovery.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DailySnapshot:
    """Snapshot of metrics at end of each day."""
    day: int = 0
    date: str = ""
    cycles_completed: int = 0
    images_downloaded: int = 0
    images_analyzed: int = 0
    anomalies_found: int = 0
    near_misses: int = 0
    duplicates_skipped: int = 0
    highest_ood_score: float = 0.0
    threshold_start: float = 0.0
    threshold_end: float = 0.0
    finetune_runs: int = 0
    model_accuracy: float = 0.0
    top_candidates: List[dict] = field(default_factory=list)
    source_effectiveness: Dict[str, dict] = field(default_factory=dict)
    # Self-correction actions taken
    corrections_applied: List[str] = field(default_factory=list)
    # Rate metrics
    anomaly_rate: float = 0.0  # anomalies per 100 images
    images_per_hour: float = 0.0
    # Health & errors
    errors_today: int = 0
    health_issues: List[str] = field(default_factory=list)
    # YOLO transient detection
    yolo_available: bool = False
    yolo_images_scanned: int = 0
    yolo_confirmations: int = 0
    yolo_retrain_runs: int = 0


@dataclass
class StreamingState:
    """Persistent state for multi-day streaming."""
    started_at: str = ""
    target_days: int = 7
    current_day: int = 0
    total_runtime_hours: float = 0.0
    daily_snapshots: List[dict] = field(default_factory=list)
    # Cumulative
    total_images: int = 0
    total_anomalies: int = 0
    total_near_misses: int = 0
    total_corrections: int = 0
    # Strategy
    current_strategy: str = "normal"  # normal, aggressive, turbo
    strategy_history: List[dict] = field(default_factory=list)
    # Best overall
    best_candidates: List[dict] = field(default_factory=list)
    # Completion
    completed: bool = False
    completed_at: str = ""
    # Health monitoring
    total_errors: int = 0
    consecutive_errors: int = 0
    error_log: List[dict] = field(default_factory=list)  # last 50 errors
    api_restarts: int = 0
    last_health_check: str = ""
    # YOLO transient detection (v1.1.0 PRIMARY feature)
    yolo_available: bool = False
    yolo_confirmations: int = 0
    yolo_images_scanned: int = 0         # how many transient-source images YOLO ran on
    yolo_detections: List[dict] = field(default_factory=list)  # top YOLO detections
    yolo_retrain_runs: int = 0
    yolo_false_positive_rate: float = 0.0


class StreamingDiscovery:
    """
    Multi-day streaming discovery orchestrator.

    Wraps the DiscoveryLoop with intelligence layers:
    1. Daily assessment and report generation
    2. Self-correcting strategy (threshold, source, mode adjustments)
    3. Trend analysis across days
    4. Publishing-ready final summary
    """

    def __init__(
        self,
        target_days: int = 7,
        aggressive: bool = False,
        turbo: bool = False,
        daily_report_hour: int = 0,  # Hour (0-23) to generate daily report
    ):
        self.target_days = target_days
        self.aggressive = aggressive
        self.turbo = turbo
        self.daily_report_hour = daily_report_hour
        self.running = True

        # Load or create streaming state
        self.state = self._load_state()
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()

        # Always update target_days and strategy from CLI args
        # (fixes "Day X of 0" when state was saved by --report-only)
        if target_days > 0:
            self.state.target_days = target_days
        self.state.current_strategy = (
            "turbo" if turbo else "aggressive" if aggressive else
            self.state.current_strategy or "normal"
        )

        # Track daily metrics for delta calculation
        self._day_start_metrics = {}
        self._last_report_date = None
        self._discovery_loop = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        atexit.register(self._save_state)

    # ── State persistence ────────────────────────────────────────────────

    def _load_state(self) -> StreamingState:
        if STREAMING_STATE_FILE.exists():
            try:
                with open(STREAMING_STATE_FILE) as f:
                    data = json.load(f)
                    return StreamingState(**data)
            except Exception as e:
                logger.warning(f"Failed to load streaming state: {e}")
        return StreamingState()

    def _save_state(self):
        try:
            with open(STREAMING_STATE_FILE, "w") as f:
                json.dump(self._to_native(asdict(self.state)), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save streaming state: {e}")

    def _to_native(self, obj):
        """Convert numpy/non-JSON types to Python native types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_native(item) for item in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    # ── Handle shutdown ──────────────────────────────────────────────────

    def _handle_shutdown(self, signum=None, frame=None):
        logger.info("\nShutdown signal received. Generating final report...")
        self.running = False
        self._take_daily_snapshot()
        self._generate_daily_report()
        self._generate_final_summary()
        self._save_state()
        self._print_star_reminder()

    # ── Discovery loop integration ───────────────────────────────────────

    def _get_loop(self):
        """Get or create the discovery loop instance."""
        if self._discovery_loop is None:
            from scripts.discovery_loop import DiscoveryLoop
            self._discovery_loop = DiscoveryLoop(
                aggressive=self.aggressive,
                turbo=self.turbo,
            )
        return self._discovery_loop

    def _capture_loop_metrics(self) -> dict:
        """Read current metrics from discovery loop state."""
        state_file = DATA_DIR / "discovery_state.json"
        candidates_file = DATA_DIR / "anomaly_candidates.json"

        metrics = {
            "cycles": 0,
            "downloaded": 0,
            "analyzed": 0,
            "anomalies": 0,
            "near_misses": 0,
            "duplicates": 0,
            "threshold": 3.0,
            "highest_ood": 0.0,
            "finetune_runs": 0,
            "model_accuracy": 0.0,
            "candidates": [],
            # YOLO-specific
            "yolo_scanned": 0,
            "yolo_confirmed": 0,
            "yolo_retrain_runs": 0,
        }

        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                metrics["cycles"] = data.get("cycles_completed", 0)
                metrics["downloaded"] = data.get("total_downloaded", 0)
                metrics["analyzed"] = data.get("total_analyzed", 0)
                metrics["anomalies"] = data.get("anomalies_found", 0)
                metrics["near_misses"] = data.get("near_misses", 0)
                metrics["duplicates"] = data.get("duplicates_skipped", 0)
                metrics["threshold"] = data.get("current_threshold", 3.0)
                metrics["highest_ood"] = data.get("highest_ood_score", 0.0)
                metrics["finetune_runs"] = data.get("finetune_runs", 0)
                metrics["model_accuracy"] = data.get("model_accuracy", 0.0)
                metrics["yolo_retrain_runs"] = data.get("yolo_retrain_runs", 0)
            except Exception:
                pass

        if candidates_file.exists():
            try:
                with open(candidates_file) as f:
                    metrics["candidates"] = json.load(f)
            except Exception:
                pass

        # Count YOLO-specific metrics from candidates
        for c in metrics["candidates"]:
            if c.get("yolo_ran"):
                metrics["yolo_scanned"] += 1
            if c.get("yolo_confirmed"):
                metrics["yolo_confirmed"] += 1

        return metrics

    def _capture_source_stats(self) -> dict:
        """Read source effectiveness stats."""
        source_file = DATA_DIR / "source_stats.json"
        if source_file.exists():
            try:
                with open(source_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ── Daily snapshot & self-correction ─────────────────────────────────

    def _take_daily_snapshot(self):
        """Capture end-of-day metrics and calculate deltas."""
        metrics = self._capture_loop_metrics()
        source_stats = self._capture_source_stats()

        day_num = self.state.current_day + 1

        # Calculate deltas from day start
        start = self._day_start_metrics
        delta_analyzed = metrics["analyzed"] - start.get("analyzed", 0)
        delta_anomalies = metrics["anomalies"] - start.get("anomalies", 0)

        anomaly_rate = (
            (delta_anomalies / delta_analyzed * 100)
            if delta_analyzed > 0
            else 0.0
        )

        # Estimate images per hour (based on ~24h day)
        hours_elapsed = max(1, (datetime.now() - datetime.fromisoformat(
            self.state.started_at
        )).total_seconds() / 3600)
        images_per_hour = metrics["analyzed"] / hours_elapsed

        # Get top 10 candidates sorted by OOD score
        sorted_candidates = sorted(
            metrics["candidates"],
            key=lambda c: c.get("ood_score", 0),
            reverse=True,
        )[:10]

        # Run self-correction analysis
        corrections = self._self_correct(metrics, source_stats, day_num)

        # Health check for snapshot
        health_issues = self._health_check()

        # Count YOLO confirmations from candidates
        yolo_confirmed_count = sum(
            1 for c in metrics["candidates"]
            if c.get("yolo_confirmed")
        )

        snapshot = DailySnapshot(
            day=day_num,
            date=datetime.now().strftime("%Y-%m-%d"),
            cycles_completed=metrics["cycles"] - start.get("cycles", 0),
            images_downloaded=metrics["downloaded"] - start.get("downloaded", 0),
            images_analyzed=delta_analyzed,
            anomalies_found=delta_anomalies,
            near_misses=metrics["near_misses"] - start.get("near_misses", 0),
            duplicates_skipped=metrics["duplicates"] - start.get("duplicates", 0),
            highest_ood_score=metrics["highest_ood"],
            threshold_start=start.get("threshold", 3.0),
            threshold_end=metrics["threshold"],
            finetune_runs=metrics["finetune_runs"],
            model_accuracy=metrics["model_accuracy"],
            top_candidates=sorted_candidates,
            source_effectiveness=source_stats,
            corrections_applied=corrections,
            anomaly_rate=anomaly_rate,
            images_per_hour=images_per_hour,
            errors_today=self.state.total_errors,
            health_issues=health_issues,
            yolo_available=self.state.yolo_available,
            yolo_images_scanned=metrics.get("yolo_scanned", 0),
            yolo_confirmations=yolo_confirmed_count,
            yolo_retrain_runs=metrics.get("yolo_retrain_runs", 0),
        )

        self.state.daily_snapshots.append(asdict(snapshot))
        self.state.current_day = day_num
        self.state.total_images = metrics["analyzed"]
        self.state.total_anomalies = metrics["anomalies"]
        self.state.total_near_misses = metrics["near_misses"]
        self.state.total_corrections += len(corrections)

        # Update YOLO cumulative stats
        self.state.yolo_images_scanned = metrics.get("yolo_scanned", 0)
        self.state.yolo_confirmations = yolo_confirmed_count
        self.state.yolo_retrain_runs = metrics.get("yolo_retrain_runs", 0)

        # Store top YOLO detections separately for publishing
        yolo_detections = [
            c for c in metrics["candidates"] if c.get("yolo_confirmed")
        ]
        yolo_detections.sort(
            key=lambda c: c.get("yolo_confidence", 0), reverse=True
        )
        self.state.yolo_detections = yolo_detections[:30]

        # Update best candidates
        self.state.best_candidates = sorted_candidates[:20]

        self._save_state()
        return snapshot

    def _self_correct(
        self, metrics: dict, source_stats: dict, day_num: int
    ) -> List[str]:
        """
        Self-correcting intelligence layer.

        Analyzes daily performance and makes strategic adjustments:
        1. Threshold correction: too many/few anomalies
        2. Source rebalancing: deprioritize low-yield sources
        3. Strategy escalation: switch to aggressive/turbo if needed
        4. Calibration trigger: recalibrate OOD if scores are clustered
        """
        corrections = []
        loop = self._get_loop()

        # 1. Threshold correction
        delta_analyzed = metrics["analyzed"] - self._day_start_metrics.get("analyzed", 0)
        delta_anomalies = metrics["anomalies"] - self._day_start_metrics.get("anomalies", 0)

        if delta_analyzed > 50:
            anomaly_pct = delta_anomalies / delta_analyzed * 100

            # Healthy anomaly rate for real astronomical data: 1-5%
            # Below 0.5%: too strict. Above 10%: false positive flood.
            THRESHOLD_FLOOR = 1.5  # Never go below this
            THRESHOLD_CEIL = 5.0

            if anomaly_pct > 10:
                # Too many anomalies -> threshold too low, tighten
                old = metrics["threshold"]
                new = min(old * 1.4, THRESHOLD_CEIL)
                loop.stats.current_threshold = new
                loop._save_state()
                corrections.append(
                    f"Threshold raised {old:.2f} -> {new:.2f} "
                    f"(anomaly rate {anomaly_pct:.1f}% too high, target <5%)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

            elif anomaly_pct < 0.3 and delta_analyzed > 300:
                # Too few anomalies -> threshold too high, loosen slightly
                old = metrics["threshold"]
                new = max(old * 0.9, THRESHOLD_FLOOR)
                if new < old:
                    loop.stats.current_threshold = new
                    loop._save_state()
                    corrections.append(
                        f"Threshold lowered {old:.2f} -> {new:.2f} "
                        f"(anomaly rate {anomaly_pct:.1f}% too low, floor={THRESHOLD_FLOOR})"
                    )
                    logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

        # 2. Strategy escalation
        if day_num >= 2 and self.state.total_anomalies == 0:
            if self.state.current_strategy == "normal":
                self.state.current_strategy = "aggressive"
                self.aggressive = True
                corrections.append(
                    "Escalated to AGGRESSIVE mode (0 anomalies after 2 days)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

                # Record strategy change
                self.state.strategy_history.append({
                    "day": day_num,
                    "from": "normal",
                    "to": "aggressive",
                    "reason": "Zero anomalies after 2 days",
                })

            elif day_num >= 3 and self.state.current_strategy == "aggressive":
                self.state.current_strategy = "turbo"
                self.turbo = True
                corrections.append(
                    "Escalated to TURBO mode (0 anomalies after 3 days)"
                )
                logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

                self.state.strategy_history.append({
                    "day": day_num,
                    "from": "aggressive",
                    "to": "turbo",
                    "reason": "Zero anomalies after 3 days",
                })

        # 3. OOD recalibration trigger
        if day_num % 2 == 0 and delta_analyzed > 100:
            try:
                success = loop.calibrate_ood_detector()
                if success:
                    corrections.append(
                        f"Recalibrated OOD detector (day {day_num} routine)"
                    )
                    logger.info(f"  [SELF-CORRECT] {corrections[-1]}")
            except Exception as e:
                logger.warning(f"  Calibration failed: {e}")

        # 4. Source rebalancing based on effectiveness
        if source_stats:
            anomaly_sources = {}
            for source_name, stats in source_stats.items():
                if isinstance(stats, dict):
                    total = stats.get("total_analyzed", 0)
                    anomalies = stats.get("anomalies", 0)
                    if total > 20:
                        anomaly_sources[source_name] = anomalies / total

            if anomaly_sources:
                best = max(anomaly_sources, key=anomaly_sources.get)
                worst = min(anomaly_sources, key=anomaly_sources.get)
                if (
                    anomaly_sources[best] > anomaly_sources[worst] * 3
                    and anomaly_sources[worst] < 0.01
                ):
                    corrections.append(
                        f"Source rebalance: prioritize '{best}' "
                        f"({anomaly_sources[best]:.1%} anomaly rate) "
                        f"over '{worst}' ({anomaly_sources[worst]:.1%})"
                    )
                    logger.info(f"  [SELF-CORRECT] {corrections[-1]}")

        return corrections

    # ── Health monitoring ────────────────────────────────────────────────

    def _health_check(self) -> List[str]:
        """
        Run health checks and return list of issues found.
        Attempts auto-recovery where possible.
        """
        issues = []

        # 1. Check API is alive
        try:
            import httpx
            resp = httpx.get("http://localhost:8000/health", timeout=5)
            if resp.status_code != 200:
                issues.append(f"API returned status {resp.status_code}")
        except Exception:
            issues.append("API not responding on port 8000")

        # 2. Check disk space (artifacts dir)
        try:
            import shutil
            usage = shutil.disk_usage(str(ARTIFACTS_DIR))
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 2:
                issues.append(f"LOW DISK SPACE: {free_gb:.1f}GB free")
            elif free_gb < 5:
                issues.append(f"Disk space warning: {free_gb:.1f}GB free")
        except Exception:
            pass

        # 3. Check discovery loop state file freshness
        state_file = DATA_DIR / "discovery_state.json"
        if state_file.exists():
            age_minutes = (
                datetime.now().timestamp() - state_file.stat().st_mtime
            ) / 60
            if age_minutes > 10:
                issues.append(
                    f"Discovery state stale ({age_minutes:.0f}min old)"
                )

        # 4. Check YOLO availability
        try:
            from inference.yolo_detector import YOLOTransientDetector
            det = YOLOTransientDetector()
            self.state.yolo_available = det.is_available()
        except Exception:
            self.state.yolo_available = False

        # 5. Check error rate
        if self.state.consecutive_errors > 5:
            issues.append(
                f"High error rate: {self.state.consecutive_errors} "
                f"consecutive errors"
            )

        self.state.last_health_check = datetime.now().isoformat()
        return issues

    def _track_error(self, error_msg: str, recoverable: bool = True):
        """Track an error for reporting."""
        self.state.total_errors += 1
        self.state.consecutive_errors += 1

        entry = {
            "time": datetime.now().isoformat(),
            "message": str(error_msg)[:200],
            "recoverable": recoverable,
        }
        self.state.error_log.append(entry)
        # Keep only last 50 errors
        if len(self.state.error_log) > 50:
            self.state.error_log = self.state.error_log[-50:]

    def _clear_error_streak(self):
        """Reset consecutive error counter after successful cycle."""
        self.state.consecutive_errors = 0

    def _send_alert(self, title: str, message: str):
        """Log alert (desktop notifications disabled to avoid disruption)."""
        logger.info(f"  [ALERT] {title}: {message}")

    # ── Report generation ────────────────────────────────────────────────

    def _generate_daily_report(self):
        """Generate an HTML daily report with charts."""
        from scripts.streaming_report import generate_daily_report

        try:
            report_path = generate_daily_report(
                streaming_state=asdict(self.state),
                output_dir=DAILY_REPORTS_DIR,
            )
            logger.info(f"  Daily report saved: {report_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate daily report: {e}")

    def _generate_final_summary(self):
        """Generate the final publishing-ready summary."""
        from scripts.streaming_report import generate_final_summary

        try:
            summary_path = generate_final_summary(
                streaming_state=asdict(self.state),
                output_dir=STREAMING_DIR,
            )
            logger.info(f"\n  Final summary saved: {summary_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate final summary: {e}")

    # ── Star reminder ────────────────────────────────────────────────────

    def _print_star_reminder(self):
        """Print a friendly star reminder at the end of a run."""
        print("\n" + "=" * 60)
        print("  If AstroLens helped your research, please star the repo:")
        print("  https://github.com/samantaba/astroLens")
        print("  It takes 2 seconds and helps others discover the tool.")
        print("=" * 60)

    # ── Main run loop ────────────────────────────────────────────────────

    def run(self):
        """Run multi-day streaming discovery."""
        logger.info("=" * 60)
        logger.info("  ASTROLENS STREAMING DISCOVERY ENGINE v1.1.0")
        logger.info("=" * 60)
        logger.info(f"  Target duration: {self.target_days} days")
        logger.info(f"  Strategy: {self.state.current_strategy}")
        logger.info(f"  Reports: {DAILY_REPORTS_DIR}")
        logger.info(f"  State: {STREAMING_STATE_FILE}")
        logger.info(f"  Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)

        loop = self._get_loop()

        # Capture starting metrics for delta tracking
        self._day_start_metrics = self._capture_loop_metrics()
        self._last_report_date = datetime.now().date()

        start_time = datetime.now()
        target_end = start_time + timedelta(days=self.target_days)

        logger.info(f"\n  Started: {start_time.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Target end: {target_end.strftime('%Y-%m-%d %H:%M')}")
        logger.info("")

        # Check API
        if loop.check_api():
            logger.info("  API is running")
        else:
            logger.warning("  API not running -- will analyze locally only")

        # Initialize models
        logger.info("\n  Initializing ML models...")
        _ = loop.classifier

        # Run initial health check
        logger.info("\n  Running initial health check...")
        initial_issues = self._health_check()
        if initial_issues:
            for issue in initial_issues:
                logger.warning(f"  [HEALTH] {issue}")
        else:
            logger.info("  All health checks passed")

        # Check YOLO availability
        if self.state.yolo_available:
            logger.info("  YOLO transient detector: ACTIVE (second-stage confirmation)")
        else:
            logger.info("  YOLO transient detector: not available (ViT+OOD only)")

        # Main streaming loop
        cycle_count = 0
        last_health_check = datetime.now()
        health_check_interval = 300  # Every 5 minutes

        while self.running and datetime.now() < target_end:
            try:
                # Run one discovery cycle
                loop.process_cycle()
                cycle_count += 1
                self._clear_error_streak()  # Successful cycle

                now = datetime.now()

                # Periodic health check (every 5 minutes)
                if (now - last_health_check).total_seconds() > health_check_interval:
                    issues = self._health_check()
                    if issues:
                        for issue in issues:
                            logger.warning(f"  [HEALTH] {issue}")
                        # Alert on critical issues
                        critical = [i for i in issues if "LOW DISK" in i or "consecutive errors" in i]
                        if critical:
                            self._send_alert(
                                "AstroLens Alert",
                                "; ".join(critical),
                            )
                    last_health_check = now

                # Check if we should generate a daily report
                if (
                    now.date() != self._last_report_date
                    and now.hour >= self.daily_report_hour
                ):
                    logger.info("\n" + "=" * 60)
                    logger.info(f"  DAILY ASSESSMENT - Day {self.state.current_day + 1}")
                    logger.info("=" * 60)

                    # Run full health check for daily report
                    day_issues = self._health_check()

                    snapshot = self._take_daily_snapshot()
                    self._generate_daily_report()

                    # Reset daily start metrics
                    self._day_start_metrics = self._capture_loop_metrics()
                    self._last_report_date = now.date()

                    logger.info(
                        f"  Day {snapshot.day} complete: "
                        f"{snapshot.images_analyzed} analyzed, "
                        f"{snapshot.anomalies_found} anomalies, "
                        f"rate={snapshot.anomaly_rate:.2f}%"
                    )
                    if snapshot.corrections_applied:
                        logger.info(
                            f"  Self-corrections: {len(snapshot.corrections_applied)}"
                        )
                        for c in snapshot.corrections_applied:
                            logger.info(f"    - {c}")
                    if day_issues:
                        logger.warning(f"  Health issues: {len(day_issues)}")
                        for issue in day_issues:
                            logger.warning(f"    - {issue}")

                    # Send daily summary alert
                    self._send_alert(
                        f"AstroLens Day {snapshot.day}",
                        f"{snapshot.images_analyzed} images, "
                        f"{snapshot.anomalies_found} anomalies, "
                        f"rate={snapshot.anomaly_rate:.2f}%",
                    )

                # Save streaming state every 5 cycles for live web UI
                if cycle_count % 5 == 0:
                    # Update cumulative stats from discovery state
                    live_metrics = self._capture_loop_metrics()
                    self.state.total_images = live_metrics["analyzed"]
                    self.state.total_anomalies = live_metrics["anomalies"]
                    self.state.total_near_misses = live_metrics.get("near_misses", 0)
                    self.state.yolo_images_scanned = live_metrics.get("yolo_scanned", 0)
                    self.state.yolo_confirmations = live_metrics.get("yolo_confirmed", 0)
                    self.state.total_runtime_hours = (
                        (now - datetime.fromisoformat(self.state.started_at))
                        .total_seconds() / 3600
                    )
                    self._save_state()

                # Progress update every 10 cycles
                if cycle_count % 10 == 0:
                    elapsed = (now - start_time).total_seconds() / 3600
                    remaining = (target_end - now).total_seconds() / 3600
                    metrics = self._capture_loop_metrics()
                    yolo_str = (
                        f"YOLO: {metrics['yolo_confirmed']}/{metrics['yolo_scanned']} confirmed"
                        if metrics['yolo_scanned'] > 0
                        else "YOLO: waiting for transient sources"
                    )
                    logger.info(
                        f"\n  [PROGRESS] {elapsed:.1f}h elapsed, "
                        f"{remaining:.1f}h remaining | "
                        f"{metrics['analyzed']} analyzed, "
                        f"{metrics['anomalies']} anomalies, "
                        f"threshold={metrics['threshold']:.3f} | "
                        f"{yolo_str} | "
                        f"errors={self.state.total_errors}"
                    )

                # Wait between cycles
                if self.running:
                    time.sleep(loop.cycle_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._track_error(str(e))
                logger.error(f"Streaming cycle error: {e}")
                import traceback
                logger.debug(traceback.format_exc())

                # Alert if too many consecutive errors
                if self.state.consecutive_errors >= 5:
                    self._send_alert(
                        "AstroLens ERROR",
                        f"{self.state.consecutive_errors} consecutive errors. "
                        f"Latest: {str(e)[:80]}",
                    )
                    logger.error(
                        f"  {self.state.consecutive_errors} consecutive errors!"
                        f" Consider restarting if this persists."
                    )

                time.sleep(60)

        # Final wrap-up
        self.state.completed = datetime.now() >= target_end
        self.state.completed_at = datetime.now().isoformat()
        self.state.total_runtime_hours = (
            (datetime.now() - datetime.fromisoformat(self.state.started_at))
            .total_seconds()
            / 3600
        )

        # Take final snapshot and generate reports
        self._take_daily_snapshot()
        self._generate_daily_report()
        self._generate_final_summary()
        self._save_state()

        # Print summary
        self._print_final_summary()
        self._print_star_reminder()

    def _print_final_summary(self):
        """Print final summary to console."""
        print("\n" + "=" * 60)
        print("  STREAMING DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"  Duration: {self.state.total_runtime_hours:.1f} hours")
        print(f"  Days: {self.state.current_day}")
        print(f"  Total images analyzed: {self.state.total_images}")
        print(f"  Total anomalies found: {self.state.total_anomalies}")
        print(f"  Total near-misses: {self.state.total_near_misses}")
        print(f"  Self-corrections applied: {self.state.total_corrections}")
        print(f"  Strategy: {self.state.current_strategy}")

        # YOLO results (primary publishable output)
        print(f"\n  --- YOLO TRANSIENT DETECTION ---")
        print(f"  YOLO available: {'YES' if self.state.yolo_available else 'NO'}")
        print(f"  Transient images scanned: {self.state.yolo_images_scanned}")
        print(f"  YOLO confirmed detections: {self.state.yolo_confirmations}")
        print(f"  YOLO retrain runs: {self.state.yolo_retrain_runs}")
        if self.state.yolo_images_scanned > 0:
            rate = self.state.yolo_confirmations / self.state.yolo_images_scanned * 100
            print(f"  Detection rate: {rate:.2f}%")

        if self.state.yolo_detections:
            print(f"\n  Top YOLO detections (publishable):")
            for i, c in enumerate(self.state.yolo_detections[:5], 1):
                print(
                    f"    {i}. YOLO conf={c.get('yolo_confidence', 0):.1%} "
                    f"| OOD={c.get('ood_score', 0):.3f} "
                    f"| {c.get('classification', '?')} "
                    f"| source={c.get('source', '?')}"
                )

        if self.state.best_candidates:
            print(f"\n  Top OOD anomaly candidates:")
            for i, c in enumerate(self.state.best_candidates[:5], 1):
                yolo = " [YOLO]" if c.get("yolo_confirmed") else ""
                print(
                    f"    {i}. OOD={c.get('ood_score', 0):.3f} "
                    f"| {c.get('classification', '?')} "
                    f"| {c.get('source', '?')}{yolo}"
                )

        print(f"\n  Reports: {STREAMING_DIR}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AstroLens Streaming Discovery Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/streaming_discovery.py --days 7
    python scripts/streaming_discovery.py --days 3 --aggressive
    python scripts/streaming_discovery.py --days 1 --turbo
    python scripts/streaming_discovery.py --report-only
        """,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to run (default: 7)",
    )
    parser.add_argument(
        "--aggressive",
        "-a",
        action="store_true",
        help="Start in aggressive mode",
    )
    parser.add_argument(
        "--turbo",
        "-t",
        action="store_true",
        help="Start in turbo mode (maximum throughput)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing data without running discovery",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset streaming state and start fresh",
    )

    args = parser.parse_args()

    if args.reset:
        # Reset streaming state
        if STREAMING_STATE_FILE.exists():
            STREAMING_STATE_FILE.unlink()
            logger.info("Streaming state reset")

        # Reset discovery state (threshold, counters) but keep downloads
        discovery_state = DATA_DIR / "discovery_state.json"
        candidates_file = DATA_DIR / "anomaly_candidates.json"

        if discovery_state.exists():
            try:
                with open(discovery_state) as f:
                    ds = json.load(f)
                # Reset counters and threshold, but keep training history
                ds["current_threshold"] = 3.0
                ds["anomalies_found"] = 0
                ds["near_misses"] = 0
                ds["uncertain_flagged"] = 0
                ds["highest_ood_score"] = 0.0
                ds["highest_ood_image"] = ""
                ds["anomaly_ids"] = []
                with open(discovery_state, "w") as f:
                    json.dump(ds, f, indent=2)
                logger.info("Discovery state reset (threshold=3.0, anomalies cleared)")
            except Exception as e:
                logger.warning(f"Failed to reset discovery state: {e}")

        if candidates_file.exists():
            # Archive old candidates instead of deleting
            archive = DATA_DIR / f"anomaly_candidates_archive_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            import shutil
            shutil.copy2(candidates_file, archive)
            with open(candidates_file, "w") as f:
                json.dump([], f)
            logger.info(f"Candidates archived to {archive.name}, cleared for fresh start")

    if args.report_only:
        engine = StreamingDiscovery(target_days=0)
        engine._take_daily_snapshot()
        engine._generate_daily_report()
        engine._generate_final_summary()
        engine._save_state()
        logger.info("Reports generated from existing data.")
        return

    engine = StreamingDiscovery(
        target_days=args.days,
        aggressive=args.aggressive,
        turbo=args.turbo,
    )
    engine.run()


if __name__ == "__main__":
    main()
