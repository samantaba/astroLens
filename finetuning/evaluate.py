#!/usr/bin/env python3
"""
Evaluate a fine-tuned model on test data.

Outputs:
- Per-class accuracy
- Confusion matrix
- Classification report
- Sample predictions

Usage:
    python finetuning/evaluate.py --model weights/vit_astrolens
    python finetuning/evaluate.py --model weights/vit_astrolens --test_dir data/test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from paths import DATASETS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Test data directory (ImageFolder format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetuning/evaluation_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of example predictions to show",
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load fine-tuned model."""
    from transformers import ViTForImageClassification, ViTImageProcessor
    import torch
    
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load class names
    class_names_path = Path(model_path) / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = list(model.config.id2label.values())
    
    return model, processor, class_names, device


def load_test_data(test_dir: str):
    """Load test data from ImageFolder format."""
    test_path = Path(test_dir)
    
    data = []
    classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    
    for class_idx, class_name in enumerate(classes):
        class_dir = test_path / class_name
        for img_path in class_dir.glob("*.png"):
            data.append({
                "path": str(img_path),
                "label": class_idx,
                "class_name": class_name,
            })
        for img_path in class_dir.glob("*.jpg"):
            data.append({
                "path": str(img_path),
                "label": class_idx,
                "class_name": class_name,
            })
    
    return data, classes


def predict(model, processor, image_path: str, device: str):
    """Predict class for a single image."""
    import torch
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax().item()
        confidence = probs[0, pred_idx].item()
    
    return pred_idx, confidence


def main():
    args = parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model, processor, class_names, device = load_model(args.model)
    
    # Find test data
    if args.test_dir:
        test_dir = args.test_dir
    else:
        # Try common locations
        candidates = [
            str(DATASETS_DIR / "galaxy10" / "test"),
            str(DATASETS_DIR / "custom" / "test"),
            str(DATA_DIR / "test"),
        ]
        test_dir = None
        for c in candidates:
            if Path(c).exists():
                test_dir = c
                break
        
        if not test_dir:
            logger.error("No test directory found. Specify with --test_dir")
            return
    
    # Load test data
    logger.info(f"Loading test data from {test_dir}")
    test_data, test_classes = load_test_data(test_dir)
    logger.info(f"Found {len(test_data)} test samples across {len(test_classes)} classes")
    
    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    labels = []
    
    for i, item in enumerate(test_data):
        pred_idx, confidence = predict(model, processor, item["path"], device)
        predictions.append(pred_idx)
        labels.append(item["label"])
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_data)}")
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print("\nPer-class metrics:")
    print("-" * 60)
    
    for class_name in class_names:
        if class_name in report:
            r = report[class_name]
            print(f"  {class_name:25} precision={r['precision']:.2f}  recall={r['recall']:.2f}  f1={r['f1-score']:.2f}")
    
    # Confusion matrix (simplified)
    print("\nConfusion Matrix (diagonal = correct predictions):")
    print("-" * 60)
    print(f"  Diagonal sum: {np.diag(cm).sum()} / {cm.sum()} ({np.diag(cm).sum() / cm.sum():.1%})")
    
    # Show example predictions
    if args.show_examples > 0:
        print(f"\nExample predictions (first {args.show_examples}):")
        print("-" * 60)
        for i, item in enumerate(test_data[:args.show_examples]):
            pred_idx, conf = predict(model, processor, item["path"], device)
            pred_name = class_names[pred_idx]
            actual_name = item["class_name"]
            status = "✓" if pred_name == actual_name else "✗"
            print(f"  {status} {Path(item['path']).name}: predicted={pred_name} ({conf:.1%}), actual={actual_name}")
    
    # Save report
    report_data = {
        "accuracy": accuracy,
        "class_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "num_test_samples": len(test_data),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"\n✅ Report saved to {output_path}")


if __name__ == "__main__":
    main()

