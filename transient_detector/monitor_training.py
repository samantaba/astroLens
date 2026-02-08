#!/usr/bin/env python3
"""
Monitor YOLO training progress and update state file.

This script runs in the background and periodically checks the training
output to update the pipeline state file for UI display.

Usage:
    python -m transient_detector.monitor_training
"""

import json
import re
import sys
import time
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data"
STATE_FILE = DATA_DIR / "pipeline_state.json"
MODELS_DIR = DATA_DIR / "models" / "transient_detector"

def get_current_epoch():
    """Check training output files to determine current epoch."""
    # Check results.csv if it exists
    results_file = MODELS_DIR / "results.csv"
    if results_file.exists():
        try:
            lines = results_file.read_text().strip().split('\n')
            # Header is first line, data follows
            return len(lines) - 1  # Number of completed epochs
        except:
            pass
    return 0

def check_training_complete():
    """Check if training has completed."""
    # Check for best.pt model
    best_model = MODELS_DIR / "weights" / "best.pt"
    last_model = MODELS_DIR / "weights" / "last.pt"
    return best_model.exists() and last_model.exists()

def update_state(epoch: int, is_complete: bool = False):
    """Update the pipeline state file."""
    if not STATE_FILE.exists():
        print("State file not found")
        return
    
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        
        for phase in state["phases"]:
            if phase["id"] == "phase2_yolo_training":
                for st in phase["subtasks"]:
                    if st["id"] == "train_yolo":
                        st["current_count"] = epoch
                        st["progress"] = min(epoch, 100)
                        if is_complete:
                            st["status"] = "completed"
                            phase["status"] = "completed"
                        else:
                            st["status"] = "in_progress"
                            phase["status"] = "in_progress"
        
        state["is_running"] = not is_complete
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error updating state: {e}")
        return False

def main():
    print("Training Monitor Started")
    print(f"Watching: {MODELS_DIR}")
    print("Press Ctrl+C to stop\n")
    
    last_epoch = 0
    
    while True:
        try:
            current_epoch = get_current_epoch()
            is_complete = check_training_complete()
            
            if current_epoch != last_epoch or is_complete:
                update_state(current_epoch, is_complete)
                print(f"Epoch {current_epoch}/100 {'(complete)' if is_complete else ''}")
                last_epoch = current_epoch
            
            if is_complete:
                print("\nTraining complete!")
                break
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\nMonitor stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
