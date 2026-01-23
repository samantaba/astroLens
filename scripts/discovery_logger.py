"""
Discovery Loop Structured Logger

Creates compact, informative log files with date-based rotation.
Each day gets a new log file to prevent large file accumulation.

Log Format:
- Cycle summaries (not every image)
- Fine-tuning events
- Errors and warnings
- Anomaly detections
- Performance metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import DATA_DIR

# Create logs directory
LOGS_DIR = DATA_DIR / "discovery_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CycleEvent:
    """A single cycle summary."""
    cycle_number: int
    timestamp: str
    duration_seconds: float
    images_downloaded: int
    images_analyzed: int
    duplicates_skipped: int
    anomalies_found: int
    near_misses: int
    flagged_for_review: int
    highest_ood: float
    threshold: float
    sources: Dict[str, int]  # source -> count


@dataclass  
class FineTuneEvent:
    """Fine-tuning run summary."""
    timestamp: str
    run_number: int
    dataset: str
    duration_minutes: float
    success: bool
    error: Optional[str] = None


@dataclass
class AnomalyEvent:
    """Anomaly detection event."""
    timestamp: str
    cycle: int
    image_path: str
    source: str
    class_label: str
    confidence: float
    ood_score: float
    ood_votes: int


@dataclass
class ErrorEvent:
    """Error or warning event."""
    timestamp: str
    cycle: int
    error_type: str
    message: str
    recoverable: bool


class DiscoveryLogger:
    """
    Structured logger for discovery loop.
    
    Creates daily log files with JSON entries for easy parsing.
    Provides summary methods for quick status checks.
    """
    
    def __init__(self, max_entries_per_file: int = 1000):
        self.max_entries = max_entries_per_file
        self._current_date: Optional[str] = None
        self._log_file: Optional[Path] = None
        self._entries: List[Dict] = []
        self._cycle_start: Optional[datetime] = None
        
    def _get_log_file(self) -> Path:
        """Get current log file, rotating daily."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self._current_date != today:
            # Rotate to new file
            self._current_date = today
            self._log_file = LOGS_DIR / f"discovery_{today}.jsonl"
            self._entries = []
            
            # Load existing entries if file exists
            if self._log_file.exists():
                try:
                    with open(self._log_file, "r") as f:
                        for line in f:
                            if line.strip():
                                self._entries.append(json.loads(line))
                except Exception:
                    pass
        
        return self._log_file
    
    def _write_entry(self, entry_type: str, data: Dict[str, Any]):
        """Write a log entry."""
        log_file = self._get_log_file()
        
        entry = {
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        
        self._entries.append(entry)
        
        # Append to file
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Trim if too many entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
    
    def start_cycle(self, cycle_number: int, threshold: float):
        """Mark cycle start for timing."""
        self._cycle_start = datetime.now()
        self._current_cycle = cycle_number
        self._current_threshold = threshold
    
    def end_cycle(
        self,
        cycle_number: int,
        images_downloaded: int,
        images_analyzed: int,
        duplicates_skipped: int,
        anomalies_found: int,
        near_misses: int,
        flagged_for_review: int,
        highest_ood: float,
        threshold: float,
        sources: Dict[str, int],
    ):
        """Log cycle completion."""
        duration = 0.0
        if self._cycle_start:
            duration = (datetime.now() - self._cycle_start).total_seconds()
        
        event = CycleEvent(
            cycle_number=cycle_number,
            timestamp=datetime.now().isoformat(),
            duration_seconds=round(duration, 1),
            images_downloaded=images_downloaded,
            images_analyzed=images_analyzed,
            duplicates_skipped=duplicates_skipped,
            anomalies_found=anomalies_found,
            near_misses=near_misses,
            flagged_for_review=flagged_for_review,
            highest_ood=round(highest_ood, 4),
            threshold=round(threshold, 4),
            sources=sources,
        )
        
        self._write_entry("cycle", asdict(event))
    
    def log_finetune(
        self,
        run_number: int,
        dataset: str,
        duration_minutes: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log fine-tuning event."""
        event = FineTuneEvent(
            timestamp=datetime.now().isoformat(),
            run_number=run_number,
            dataset=dataset,
            duration_minutes=round(duration_minutes, 2),
            success=success,
            error=error,
        )
        
        self._write_entry("finetune", asdict(event))
    
    def log_anomaly(
        self,
        cycle: int,
        image_path: str,
        source: str,
        class_label: str,
        confidence: float,
        ood_score: float,
        ood_votes: int,
    ):
        """Log anomaly detection."""
        event = AnomalyEvent(
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            image_path=image_path,
            source=source,
            class_label=class_label,
            confidence=round(confidence, 4),
            ood_score=round(ood_score, 4),
            ood_votes=ood_votes,
        )
        
        self._write_entry("anomaly", asdict(event))
    
    def log_error(
        self,
        cycle: int,
        error_type: str,
        message: str,
        recoverable: bool = True,
    ):
        """Log error or warning."""
        event = ErrorEvent(
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            error_type=error_type,
            message=message[:500],  # Truncate long messages
            recoverable=recoverable,
        )
        
        self._write_entry("error", asdict(event))
    
    def log_start(self, config: Dict[str, Any]):
        """Log discovery loop start."""
        self._write_entry("start", {
            "config": config,
            "pid": __import__("os").getpid(),
        })
    
    def log_stop(self, stats: Dict[str, Any]):
        """Log discovery loop stop."""
        self._write_entry("stop", {"final_stats": stats})
    
    def get_summary(self, last_n_cycles: int = 10) -> Dict[str, Any]:
        """Get summary of recent activity."""
        _ = self._get_log_file()  # Ensure loaded
        
        cycles = [e for e in self._entries if e.get("type") == "cycle"][-last_n_cycles:]
        finetunes = [e for e in self._entries if e.get("type") == "finetune"]
        anomalies = [e for e in self._entries if e.get("type") == "anomaly"]
        errors = [e for e in self._entries if e.get("type") == "error"]
        
        if cycles:
            avg_duration = sum(c.get("duration_seconds", 0) for c in cycles) / len(cycles)
            total_analyzed = sum(c.get("images_analyzed", 0) for c in cycles)
            total_anomalies = sum(c.get("anomalies_found", 0) for c in cycles)
        else:
            avg_duration = 0
            total_analyzed = 0
            total_anomalies = 0
        
        return {
            "log_file": str(self._log_file),
            "cycles_logged": len(cycles),
            "avg_cycle_duration_seconds": round(avg_duration, 1),
            "total_analyzed_recent": total_analyzed,
            "total_anomalies_recent": total_anomalies,
            "finetune_runs": len(finetunes),
            "anomalies_detected": len(anomalies),
            "errors_logged": len(errors),
            "last_cycle": cycles[-1] if cycles else None,
        }
    
    @staticmethod
    def list_log_files() -> List[Path]:
        """List all log files, newest first."""
        return sorted(LOGS_DIR.glob("discovery_*.jsonl"), reverse=True)
    
    @staticmethod
    def read_log_file(log_path: Path) -> List[Dict]:
        """Read entries from a log file."""
        entries = []
        if log_path.exists():
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return entries


# Global logger instance
_logger: Optional[DiscoveryLogger] = None


def get_discovery_logger() -> DiscoveryLogger:
    """Get or create the discovery logger."""
    global _logger
    if _logger is None:
        _logger = DiscoveryLogger()
    return _logger

