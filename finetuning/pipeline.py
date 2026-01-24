#!/usr/bin/env python3
"""
Automated fine-tuning pipeline.

Runs the complete pipeline:
1. Download/update datasets
2. Fine-tune model
3. Evaluate performance
4. Save if improved

Usage:
    python finetuning/pipeline.py --auto
    python finetuning/pipeline.py --dataset galaxy10 --epochs 10
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATASETS_DIR, WEIGHTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str, verbose: bool = True):
    """Run a command with real-time output streaming for verbose feedback."""
    logger.info(f"🔄 {description}...")
    
    if verbose:
        # Stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                output_lines.append(line)
                # Log training progress lines
                if any(kw in line.lower() for kw in ['epoch', 'loss', 'accuracy', 'step', '%', 'training', 'evaluating']):
                    logger.info(f"  {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"✗ {description} failed (exit code {process.returncode})")
            raise subprocess.CalledProcessError(process.returncode, cmd, '\n'.join(output_lines))
        
        logger.info(f"✓ {description} completed")
        return '\n'.join(output_lines)
    else:
        # Original behavior - capture output
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"✓ {description} completed")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed: {e.stderr}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Automated fine-tuning pipeline")
    parser.add_argument("--auto", action="store_true", help="Run full auto pipeline")
    parser.add_argument("--dataset", default="galaxy10", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: timestamped)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip dataset download",
    )
    parser.add_argument(
        "--skip_evaluate",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show training progress in real-time (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress training output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (uses consistent directory per dataset)",
    )
    
    args = parser.parse_args()
    
    # Verbose mode (default on, unless --quiet)
    verbose = args.verbose and not args.quiet
    
    # Always create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use consistent directory per dataset for resumption, or timestamped for fresh training
    if args.resume:
        # Use consistent directory so checkpoints can be resumed
        output_dir = args.output_dir or str(WEIGHTS_DIR / f"vit_astrolens_{args.dataset}")
    else:
        # Fresh training with timestamp
        output_dir = args.output_dir or str(WEIGHTS_DIR / f"vit_astrolens_{timestamp}")
    
    logger.info("=" * 60)
    logger.info("🚀 ASTROLENS FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Verbose: {verbose}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📋 Training will run until completion (no timeout)")
    logger.info("   Progress will be logged in real-time below.")
    logger.info("")
    
    # Step 1: Download dataset
    if not args.skip_download:
        run_command(
            [sys.executable, "finetuning/download_datasets.py", "--dataset", args.dataset],
            f"Downloading {args.dataset} dataset",
            verbose=verbose,
        )
    else:
        logger.info("⏭ Skipping dataset download")
    
    # Step 2: Fine-tune (this is the long-running step)
    logger.info("")
    logger.info("🎓 Starting model fine-tuning...")
    logger.info("   This may take 30-60+ minutes depending on dataset size and hardware.")
    logger.info("")
    
    data_dir = str(DATASETS_DIR / args.dataset)
    train_cmd = [
        sys.executable, "finetuning/train.py",
        "--data_dir", data_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", output_dir,
    ]
    if args.resume:
        train_cmd.append("--resume_from_checkpoint")
        logger.info(f"📂 Resume mode: will continue from last checkpoint if available")
    
    run_command(train_cmd, "Fine-tuning model", verbose=verbose)
    
    # Step 3: Evaluate
    if not args.skip_evaluate:
        test_dir = f"{data_dir}/test"
        if Path(test_dir).exists():
            run_command(
                [
                    sys.executable, "finetuning/evaluate.py",
                    "--model", output_dir,
                    "--test_dir", test_dir,
                    "--output", f"{output_dir}/evaluation_report.json",
                ],
                "Evaluating model",
                verbose=verbose,
            )
        else:
            logger.warning(f"⚠ No test directory at {test_dir}, skipping evaluation")
    else:
        logger.info("⏭ Skipping evaluation")
    
    # Step 4: Update symlink to latest
    latest_link = WEIGHTS_DIR / "vit_astrolens_latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(Path(output_dir).name)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved: {output_dir}")
    logger.info(f"Latest link: {latest_link}")
    logger.info("\nTo use the new model:")
    logger.info(f"  export WEIGHTS_PATH={output_dir}")
    logger.info(f"  python -m ui.main")
    
    # Log run info
    run_log = Path("finetuning/runs.log")
    with open(run_log, "a") as f:
        f.write(f"{timestamp} | {args.dataset} | {args.epochs} epochs | {output_dir}\n")


if __name__ == "__main__":
    main()

