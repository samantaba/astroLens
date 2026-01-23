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


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    logger.info(f"🔄 {description}...")
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
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(WEIGHTS_DIR / f"vit_astrolens_{timestamp}")
    
    logger.info("=" * 60)
    logger.info("🚀 ASTROLENS FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)
    
    # Step 1: Download dataset
    if not args.skip_download:
        run_command(
            [sys.executable, "finetuning/download_datasets.py", "--dataset", args.dataset],
            f"Downloading {args.dataset} dataset",
        )
    else:
        logger.info("⏭ Skipping dataset download")
    
    # Step 2: Fine-tune
    data_dir = str(DATASETS_DIR / args.dataset)
    run_command(
        [
            sys.executable, "finetuning/train.py",
            "--data_dir", data_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--output_dir", output_dir,
        ],
        "Fine-tuning model",
    )
    
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

