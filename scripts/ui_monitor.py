#!/usr/bin/env python3
"""
UI State Monitor

Monitors the pipeline state and compares with actual data to verify
the UI is displaying correct information.

Usage:
    python scripts/ui_monitor.py --continuous
    python scripts/ui_monitor.py --check
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data"
STATE_FILE = DATA_DIR / "pipeline_state.json"
LOG_FILE = DATA_DIR / "logs" / f"ui_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(message: str, level: str = "INFO"):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_actual_counts() -> dict:
    """Get actual data counts from disk."""
    downloads = DATA_DIR / "downloads"
    training = DATA_DIR / "training"
    yolo = DATA_DIR / "annotations" / "yolo"
    models = DATA_DIR / "models" / "transient_detector"
    
    return {
        "tns_downloads": len(list((downloads / "tns").glob("*.jpg"))) if (downloads / "tns").exists() else 0,
        "ztf_downloads": len(list((downloads / "ztf").glob("*.jpg"))) if (downloads / "ztf").exists() else 0,
        "train_images": len(list((training / "train" / "transient").glob("*.jpg"))) if (training / "train" / "transient").exists() else 0,
        "val_images": len(list((training / "val" / "transient").glob("*.jpg"))) if (training / "val" / "transient").exists() else 0,
        "yolo_annotations": len(list((yolo / "labels" / "train").glob("*.txt"))) if (yolo / "labels" / "train").exists() else 0,
        "model_exists": (models / "weights" / "best.pt").exists(),
    }


def get_state_counts() -> dict:
    """Get counts from state file."""
    if not STATE_FILE.exists():
        return {}
    
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        
        counts = {
            "is_complete": state.get("is_complete", False),
            "is_running": state.get("is_running", False),
            "total_progress": state.get("total_progress", 0),
            "phases": {}
        }
        
        for phase in state.get("phases", []):
            phase_data = {
                "status": phase.get("status", "unknown"),
                "progress": phase.get("progress", 0),
                "subtasks": {}
            }
            for st in phase.get("subtasks", []):
                phase_data["subtasks"][st["id"]] = {
                    "status": st.get("status", "unknown"),
                    "current": st.get("current_count", 0),
                    "target": st.get("target_count", 0),
                }
            counts["phases"][phase["id"]] = phase_data
        
        return counts
    except Exception as e:
        log(f"Error reading state: {e}", "ERROR")
        return {}


def check_consistency() -> tuple:
    """Check if state matches actual data."""
    actual = get_actual_counts()
    state = get_state_counts()
    
    issues = []
    
    # Check TNS count
    if "phase1_data_collection" in state.get("phases", {}):
        subtasks = state["phases"]["phase1_data_collection"].get("subtasks", {})
        if "download_tns" in subtasks:
            state_tns = subtasks["download_tns"]["current"]
            if state_tns != actual["tns_downloads"]:
                issues.append(f"TNS mismatch: state={state_tns}, actual={actual['tns_downloads']}")
        
        if "download_ztf" in subtasks:
            state_ztf = subtasks["download_ztf"]["current"]
            if state_ztf != actual["ztf_downloads"]:
                issues.append(f"ZTF mismatch: state={state_ztf}, actual={actual['ztf_downloads']}")
    
    # Check model status
    if "phase2_yolo_training" in state.get("phases", {}):
        phase2 = state["phases"]["phase2_yolo_training"]
        if phase2["status"] == "completed" and not actual["model_exists"]:
            issues.append("Phase 2 marked complete but model doesn't exist")
        if actual["model_exists"] and phase2["status"] != "completed":
            issues.append("Model exists but Phase 2 not marked complete")
    
    return issues, actual, state


def display_status():
    """Display current status."""
    issues, actual, state = check_consistency()
    
    log("=" * 60)
    log("UI STATE MONITOR")
    log("=" * 60)
    log("")
    
    log("=== Actual Data ===")
    log(f"  TNS downloads: {actual['tns_downloads']}")
    log(f"  ZTF downloads: {actual['ztf_downloads']}")
    log(f"  Training images: {actual['train_images']}")
    log(f"  Validation images: {actual['val_images']}")
    log(f"  YOLO annotations: {actual['yolo_annotations']}")
    log(f"  Model exists: {actual['model_exists']}")
    log("")
    
    log("=== State File ===")
    if state:
        log(f"  Complete: {state.get('is_complete', False)}")
        log(f"  Running: {state.get('is_running', False)}")
        log(f"  Progress: {state.get('total_progress', 0)}%")
        log("")
        
        for phase_id, phase in state.get("phases", {}).items():
            log(f"  {phase_id}: {phase['status']} ({phase['progress']}%)")
            for st_id, st in phase.get("subtasks", {}).items():
                log(f"    - {st_id}: {st['current']}/{st['target']} ({st['status']})")
    else:
        log("  No state file found")
    
    log("")
    log("=== Consistency Check ===")
    if issues:
        for issue in issues:
            log(f"  ⚠ {issue}", "WARN")
    else:
        log("  ✓ State matches actual data")
    
    log("")
    log(f"Log saved to: {LOG_FILE}")
    
    return len(issues) == 0


def monitor_continuous(interval: int = 5):
    """Monitor continuously."""
    log("Starting continuous monitoring...")
    log(f"Interval: {interval} seconds")
    log("Press Ctrl+C to stop\n")
    
    try:
        while True:
            display_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        log("\nMonitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="UI State Monitor")
    parser.add_argument("--continuous", "-c", action="store_true", help="Monitor continuously")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Check interval in seconds")
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuous(args.interval)
    else:
        success = display_status()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
