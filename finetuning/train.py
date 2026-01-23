#!/usr/bin/env python3
"""
Fine-tune ViT model on labeled astronomical data.

Uses Hugging Face Transformers Trainer for efficient training with:
- Mixed precision (fp16)
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Checkpoint saving

Usage:
    python finetuning/train.py
    python finetuning/train.py --epochs 20 --batch_size 64
    python finetuning/train.py --data_dir galaxy10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image

from paths import DATASETS_DIR, WEIGHTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on astronomical data")
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Base model from Hugging Face",
    )
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATASETS_DIR / "galaxy10"),
        help="Dataset directory (ImageFolder format)",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(WEIGHTS_DIR / "vit_astrolens"),
        help="Output directory for model",
    )
    
    # Hardware
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    return parser.parse_args()


def load_dataset(data_dir: str):
    """Load dataset from ImageFolder format."""
    from datasets import load_dataset
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            f"Run: python finetuning/download_datasets.py first"
        )
    
    # Load as ImageFolder dataset
    dataset = load_dataset("imagefolder", data_dir=str(data_path))
    
    logger.info(f"Loaded dataset from {data_dir}")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    if "test" in dataset:
        logger.info(f"  Test: {len(dataset['test'])} samples")
    
    return dataset


def get_class_names(data_dir: str):
    """Extract class names from folder structure."""
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        train_dir = Path(data_dir)
    
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(classes)} classes: {classes}")
    return classes


def create_transforms(processor):
    """Create data transforms using the processor."""
    def transform(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    
    return transform


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def main():
    args = parse_args()
    
    # Import Hugging Face libraries
    try:
        from transformers import (
            ViTForImageClassification,
            ViTImageProcessor,
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_labels = len(class_names)
    
    # Create label mappings
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for i, name in enumerate(class_names)}
    
    # Load model and processor
    logger.info(f"Loading model: {args.model_name}")
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    
    # Load dataset
    dataset = load_dataset(args.data_dir)
    
    # Apply transforms
    transform = create_transforms(processor)
    dataset = dataset.with_transform(transform)
    
    # Split if no test set
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    # Detect best device (CUDA > MPS > CPU)
    use_fp16 = False
    use_mps = False
    
    if torch.cuda.is_available():
        use_fp16 = args.fp16
        logger.info("🚀 Using CUDA GPU for training")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_mps = True
        logger.info("🍎 Using Apple Metal (MPS) GPU for training")
    else:
        logger.info("💻 Using CPU for training")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        use_mps_device=use_mps,  # Enable MPS on Mac
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["labels"] for ex in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    logger.info("🚀 Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("📊 Evaluating...")
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")
    
    # Save model
    logger.info(f"💾 Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save class names
    import json
    with open(Path(args.output_dir) / "class_names.json", "w") as f:
        json.dump(class_names, f)
    
    logger.info("✅ Training complete!")
    logger.info(f"\nTo use the fine-tuned model:")
    logger.info(f"  export WEIGHTS_PATH={args.output_dir}")
    logger.info(f"  python -m ui.main")


if __name__ == "__main__":
    main()

