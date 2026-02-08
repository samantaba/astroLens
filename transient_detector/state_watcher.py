#!/usr/bin/env python3
"""
State Watcher - keeps pipeline_state.json updated with training progress.

This script runs in the background and periodically updates the state file
by reading the training progress from results.csv.

Usage:
    python -m transient_detector.state_watcher
"""

import json
import time
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "pipeline_state.json"
RESULTS_FILE = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "models" / "transient_detector" / "results.csv"

def get_current_epoch():
    """Get current training epoch from results.csv."""
    if not RESULTS_FILE.exists():
        return 0
    try:
        lines = RESULTS_FILE.read_text().strip().split('\n')
        if len(lines) > 1:
            return int(lines[-1].split(',')[0])
    except:
        pass
    return 0

def update_state():
    """Update the pipeline state file."""
    if not STATE_FILE.exists():
        return False
    
    current_epoch = get_current_epoch()
    
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        
        # Calculate total progress (Phase 1=33%, Phase 2=33%, Phase 3=33%)
        phase2_progress = (current_epoch / 100) * 33
        total_progress = 33 + phase2_progress
        
        is_complete = current_epoch >= 100
        state["is_running"] = not is_complete
        state["total_progress"] = min(total_progress, 66) if not is_complete else 66
        
        for phase in state["phases"]:
            if phase["id"] == "phase2_yolo_training":
                phase["progress"] = current_epoch
                if is_complete:
                    phase["status"] = "completed"
                for st in phase["subtasks"]:
                    if st["id"] == "train_yolo":
                        st["current_count"] = current_epoch
                        st["progress"] = float(current_epoch)
                        if is_complete:
                            st["status"] = "completed"
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        
        return current_epoch
    except Exception as e:
        print(f"Error: {e}")
        return -1

def main():
    print("State Watcher Started")
    print(f"Watching: {RESULTS_FILE}")
    print(f"Updating: {STATE_FILE}")
    print("Press Ctrl+C to stop\n")
    
    last_epoch = -1
    
    while True:
        try:
            epoch = update_state()
            if epoch != last_epoch and epoch >= 0:
                print(f"Epoch {epoch}/100")
                last_epoch = epoch
            
            if epoch >= 100:
                print("\nTraining complete!")
                break
            
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nWatcher stopped")
            break

if __name__ == "__main__":
    main()
