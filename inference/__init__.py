"""
AstroLens Inference Layer

ML models for image classification, anomaly detection, and embeddings.
"""

from .classifier import AstroClassifier
from .ood import OODDetector
from .embeddings import EmbeddingStore

__all__ = ["AstroClassifier", "OODDetector", "EmbeddingStore"]
