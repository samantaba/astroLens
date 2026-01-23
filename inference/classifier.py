"""
Vision Transformer classifier using Hugging Face Transformers.

Uses google/vit-base-patch16-224 with optional fine-tuned weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Default class labels for astronomical images (11 generic classes)
DEFAULT_CLASSES = [
    "galaxy_spiral",
    "galaxy_elliptical",
    "galaxy_irregular",
    "galaxy_merger",
    "supernova",
    "transient",
    "star",
    "nebula",
    "asteroid",
    "artifact",
    "unknown",
]

# Classes that should be flagged as anomalies when detected
ANOMALY_CLASSES = {
    "supernova",
    "supernova_candidate",
    "transient",
    "gravitational_lens",
    "artifact_streak",
    "unusual_morphology",
    "anomaly",
    "unknown",
}

# Galaxy10 classes (used when fine-tuned on Galaxy10 dataset)
GALAXY10_CLASSES = [
    "disturbed",
    "merging", 
    "round_smooth",
    "in_between_smooth",
    "cigar_shaped",
    "barred_spiral",
    "unbarred_tight_spiral",
    "unbarred_loose_spiral",
    "edge_on_without_bulge",
    "edge_on_with_bulge",
]


@dataclass
class ClassificationOutput:
    """Output of image classification."""
    class_label: str
    confidence: float
    probabilities: Dict[str, float]
    embedding: np.ndarray
    logits: np.ndarray


class AstroClassifier:
    """
    Vision Transformer classifier using Hugging Face.
    
    Uses google/vit-base-patch16-224 as base model.
    Can load fine-tuned weights from local directory.
    
    Outputs:
    - Class probabilities (11 classes)
    - 768-dim embedding for similarity search
    - Raw logits for OOD detection
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        weights_path: Optional[str] = None,
        num_classes: int = len(DEFAULT_CLASSES),
        class_names: Optional[List[str]] = None,
        device: str = None,
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Hugging Face model ID (default: google/vit-base-patch16-224)
            weights_path: Path to fine-tuned weights directory (optional)
            num_classes: Number of output classes
            class_names: List of class names
            device: "cpu", "cuda", or "mps" (auto-detected if None)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_CLASSES[:num_classes]
        self.device = device or self._detect_best_device()
        self.embedding_dim = 768

        # Load model and processor
        self._load_model(weights_path)

    def _detect_best_device(self) -> str:
        """
        Detect the best available device for inference.
        
        Priority: CUDA > MPS (Mac) > CPU
        MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon.
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("🚀 Using CUDA GPU acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon GPU (M1/M2/M3)
            device = "mps"
            logger.info("🍎 Using Apple Metal (MPS) GPU acceleration")
        else:
            device = "cpu"
            logger.info("💻 Using CPU (no GPU acceleration available)")
        
        return device

    def _load_model(self, weights_path: Optional[str]):
        """Load the ViT model from Hugging Face or local weights."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import json
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        # Check for fine-tuned weights first
        if weights_path and Path(weights_path).exists():
            logger.info(f"Loading fine-tuned model from {weights_path}")
            
            # Read config to get correct num_labels and class names
            config_path = Path(weights_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    
                    # Get class labels from id2label if available
                    id2label = config.get("id2label", {})
                    if id2label:
                        # Sort by id and extract labels
                        self.num_classes = len(id2label)
                        self.class_names = [id2label[str(i)] for i in range(self.num_classes)]
                        logger.info(f"Fine-tuned model has {self.num_classes} classes: {self.class_names[:3]}...")
                    else:
                        saved_num_labels = config.get("num_labels", self.num_classes)
                        self.num_classes = saved_num_labels
                        self.class_names = DEFAULT_CLASSES[:saved_num_labels]
            
            # Load WITHOUT ignore_mismatched_sizes to use exact saved weights
            self.model = ViTForImageClassification.from_pretrained(weights_path)
            self.processor = ViTImageProcessor.from_pretrained(weights_path)
        else:
            # Load from Hugging Face Hub
            logger.info(f"Loading pre-trained model: {self.model_name}")
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,  # Allows different num_labels
            )
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device} with {self.num_classes} classes")

    def classify(self, image_path: str) -> ClassificationOutput:
        """
        Classify a single image.
        
        Args:
            image_path: Path to image file (PNG, JPEG, FITS)
        
        Returns:
            ClassificationOutput with label, confidence, probs, embedding, logits
        """
        # Load and preprocess image
        image = self._load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]
            
            # Get embedding from last hidden state (CLS token)
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states[0, 0, :]  # CLS token is first

            probs = F.softmax(logits, dim=-1)

        # Extract results
        prob_np = probs.cpu().numpy()
        top_idx = int(prob_np.argmax())
        top_conf = float(prob_np[top_idx])

        return ClassificationOutput(
            class_label=self.class_names[top_idx],
            confidence=top_conf,
            probabilities={
                name: float(prob_np[i]) for i, name in enumerate(self.class_names)
            },
            embedding=embedding.cpu().numpy(),
            logits=logits.cpu().numpy(),
        )

    def classify_batch(self, image_paths: List[str]) -> List[ClassificationOutput]:
        """
        Classify multiple images in a batch (more efficient).
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            List of ClassificationOutput
        """
        images = [self._load_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states[:, 0, :]
            probs = F.softmax(logits, dim=-1)

        results = []
        for i in range(len(image_paths)):
            prob_np = probs[i].cpu().numpy()
            top_idx = int(prob_np.argmax())
            
            results.append(ClassificationOutput(
                class_label=self.class_names[top_idx],
                confidence=float(prob_np[top_idx]),
                probabilities={
                    name: float(prob_np[j]) for j, name in enumerate(self.class_names)
                },
                embedding=embeddings[i].cpu().numpy(),
                logits=logits[i].cpu().numpy(),
            ))

        return results

    def _load_image(self, path: str) -> Image.Image:
        """Load image and convert to RGB PIL Image."""
        path = Path(path)
        
        # Handle FITS files
        if path.suffix.lower() == ".fits":
            return self._load_fits(path)
        
        # Standard image formats
        image = Image.open(path)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image

    def _load_fits(self, path: Path) -> Image.Image:
        """Load FITS file and convert to PIL Image."""
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy not installed. Run: pip install astropy")

        with fits.open(path) as hdu:
            data = hdu[0].data
            if data is None and len(hdu) > 1:
                data = hdu[1].data

        if data is None:
            raise ValueError(f"No image data in FITS file: {path}")

        # Normalize to 0-255
        data = data.astype(np.float32)
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min) * 255
        data = data.astype(np.uint8)

        # Convert to RGB
        if data.ndim == 2:
            image = Image.fromarray(data, mode="L").convert("RGB")
        else:
            image = Image.fromarray(data).convert("RGB")

        return image

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Get just the embedding for an image (faster if you don't need classification)."""
        result = self.classify(image_path)
        return result.embedding

    def is_anomaly_class(self, class_label: str) -> bool:
        """Check if the predicted class is an anomaly class."""
        return class_label.lower() in ANOMALY_CLASSES or any(
            anomaly in class_label.lower() 
            for anomaly in ["supernova", "lens", "transient", "unusual", "artifact", "streak"]
        )

    def save(self, output_dir: str):
        """Save model and processor for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
