#!/usr/bin/env python3
"""
Run the transient detection pipeline as a standalone process.

This runs independently from the UI to avoid crashes when UI closes.
Progress is written to pipeline_state.json which the UI reads.

Usage:
    python -m transient_detector.run_pipeline          # Run full pipeline
    python -m transient_detector.run_pipeline --resume # Resume from where stopped
    python -m transient_detector.run_pipeline --reset  # Reset and start fresh
"""

import argparse
import sys
import signal
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transient_detector.pipeline import TransientPipeline

# Global for signal handling
pipeline = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\nInterrupt received, stopping pipeline...")
    if pipeline:
        pipeline.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Run transient detection pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    args = parser.parse_args()
    
    global pipeline
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("Transient Detection Pipeline")
    print("=" * 60)
    
    # Reset if requested
    if args.reset:
        state_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "pipeline_state.json"
        if state_file.exists():
            state_file.unlink()
            print("Pipeline state reset.")
    
    # Initialize pipeline
    pipeline = TransientPipeline()
    
    print(f"Pipeline ID: {pipeline.state.pipeline_id}")
    print(f"Current phase: {pipeline.state.current_phase}")
    print(f"Progress: {pipeline.state.total_progress:.1f}%")
    
    if pipeline.state.is_complete:
        print("\nPipeline already complete!")
        return 0
    
    if args.resume:
        print("\nResuming from previous state...")
        print("(Will skip already downloaded files)")
    else:
        print("\nStarting pipeline...")
    
    print("-" * 60)
    
    # Mark as running
    pipeline.state.is_running = True
    pipeline._save_state()
    
    try:
        # Run the pipeline
        pipeline._run_pipeline()
        
        print("-" * 60)
        print("Pipeline finished!")
        print(f"Final progress: {pipeline.state.total_progress:.1f}%")
        
        if pipeline.state.is_complete:
            print("\n✓ All phases completed successfully!")
        else:
            print("\n⚠ Some phases may have failed. Check state file for details.")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        pipeline.state.is_running = False
        pipeline._save_state()


if __name__ == "__main__":
    sys.exit(main())
