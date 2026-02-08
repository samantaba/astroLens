"""
Reconstruction-based Anomaly Detection

Uses image reconstruction error as an anomaly indicator.
Normal images can be reconstructed well; anomalies have high error.

Methods:
- PCA reconstruction error
- Autoencoder reconstruction (if model available)
- Feature space reconstruction

Useful for finding images that don't fit the learned manifold.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Result of reconstruction-based anomaly detection."""
    
    reconstruction_error: float  # Mean squared error
    normalized_error: float      # Error normalized to 0-1 range
    is_anomaly: bool
    
    # PCA-specific
    explained_variance: float    # How much variance is captured
    residual_variance: float     # Unexplained variance
    
    def to_dict(self) -> dict:
        return {
            "reconstruction_error": round(self.reconstruction_error, 4),
            "normalized_error": round(self.normalized_error, 4),
            "is_anomaly": self.is_anomaly,
            "explained_variance": round(self.explained_variance, 4),
            "residual_variance": round(self.residual_variance, 4),
        }


class PCAReconstructor:
    """
    PCA-based reconstruction for anomaly detection.
    
    Fits PCA on normal images, then measures reconstruction error
    on new images. High error = anomaly.
    """
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.image_size = 64  # Smaller for PCA
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None
        self.fitted = False
        self.error_threshold = 0.1  # Default threshold
        
    def fit(self, image_paths: List[str], n_samples: int = 1000):
        """
        Fit PCA on a sample of normal images.
        
        Args:
            image_paths: List of paths to normal images
            n_samples: Number of images to use for fitting
        """
        logger.info(f"Fitting PCA reconstructor on {min(n_samples, len(image_paths))} images...")
        
        # Load images
        images = []
        for path in image_paths[:n_samples]:
            img = self._load_image(path)
            if img is not None:
                images.append(img.flatten())
        
        if len(images) < 50:
            logger.warning(f"Not enough images for PCA: {len(images)}")
            return False
        
        X = np.array(images)
        
        # Compute mean
        self.pca_mean = np.mean(X, axis=0)
        X_centered = X - self.pca_mean
        
        # SVD for PCA
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top components
        self.pca_components = Vt[:self.n_components]
        
        # Compute reconstruction errors for calibration
        errors = []
        for x in X_centered:
            proj = self.pca_components.T @ (self.pca_components @ x)
            error = np.mean((x - proj) ** 2)
            errors.append(error)
        
        # Set threshold at 95th percentile
        self.error_threshold = np.percentile(errors, 95)
        self.fitted = True
        
        logger.info(f"PCA fitted with {self.n_components} components")
        logger.info(f"Reconstruction error threshold: {self.error_threshold:.4f}")
        
        return True
    
    def detect(self, image_path: str) -> Optional[ReconstructionResult]:
        """
        Detect anomaly using reconstruction error.
        """
        if not self.fitted:
            logger.warning("PCA not fitted - using default detection")
            return self._default_detection(image_path)
        
        img = self._load_image(image_path)
        if img is None:
            return None
        
        x = img.flatten() - self.pca_mean
        
        # Project and reconstruct
        projection = self.pca_components @ x
        reconstruction = self.pca_components.T @ projection
        
        # Reconstruction error
        error = np.mean((x - reconstruction) ** 2)
        
        # Explained variance
        total_var = np.var(x)
        residual_var = np.var(x - reconstruction)
        explained = 1 - (residual_var / (total_var + 1e-8))
        
        # Normalize error
        normalized = min(error / (self.error_threshold * 2), 1.0)
        
        return ReconstructionResult(
            reconstruction_error=error,
            normalized_error=normalized,
            is_anomaly=error > self.error_threshold,
            explained_variance=explained,
            residual_variance=residual_var,
        )
    
    def _default_detection(self, image_path: str) -> Optional[ReconstructionResult]:
        """Fallback detection without fitted PCA."""
        img = self._load_image(image_path)
        if img is None:
            return None
        
        # Use simple variance-based detection
        variance = np.var(img)
        
        # High variance often indicates complex/unusual structure
        normalized = min(variance / 0.1, 1.0)
        
        return ReconstructionResult(
            reconstruction_error=variance,
            normalized_error=normalized,
            is_anomaly=variance > 0.05,
            explained_variance=0.0,
            residual_variance=variance,
        )
    
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load and preprocess image."""
        try:
            img = Image.open(path).convert("L")
            img = img.resize((self.image_size, self.image_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None


class FeatureReconstructor:
    """
    Feature-space reconstruction anomaly detection.
    
    Uses extracted features (from ViT/CNN) instead of raw pixels.
    More robust to small visual variations.
    """
    
    def __init__(self):
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit on feature vectors from normal images.
        
        Args:
            features: Array of shape (n_samples, n_features)
        """
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        self.fitted = True
        logger.info(f"Feature reconstructor fitted on {len(features)} samples")
    
    def detect(self, features: np.ndarray) -> float:
        """
        Compute anomaly score for feature vector.
        
        Args:
            features: Single feature vector
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        if not self.fitted:
            return 0.5
        
        # Z-score
        z = (features - self.feature_mean) / self.feature_std
        
        # Mean absolute z-score as anomaly measure
        score = np.mean(np.abs(z))
        
        # Normalize to 0-1
        return min(score / 5.0, 1.0)


# Convenience functions
def create_pca_detector(n_components: int = 50) -> PCAReconstructor:
    """Create PCA-based anomaly detector."""
    return PCAReconstructor(n_components)


def fit_and_detect(
    normal_paths: List[str],
    test_path: str,
    n_components: int = 50,
) -> Optional[ReconstructionResult]:
    """
    One-shot fit and detect.
    
    Fits PCA on normal images and detects anomaly in test image.
    """
    detector = PCAReconstructor(n_components)
    detector.fit(normal_paths)
    return detector.detect(test_path)
