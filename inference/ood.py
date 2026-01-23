"""
Out-of-Distribution (OOD) Detector

Ensemble detection using multiple methods for robust anomaly detection:
- MSP (Maximum Softmax Probability)
- Energy-based detection
- Mahalanobis distance

References:
- MSP: "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" (Hendrycks & Gimpel, 2017)
- Energy: "Energy-based Out-of-distribution Detection" (Liu et al., NeurIPS 2020)  
- Mahalanobis: "A Simple Unified Framework for Detecting OOD Samples" (Lee et al., NeurIPS 2018)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODOutput:
    """Result of OOD detection."""
    ood_score: float
    is_anomaly: bool
    threshold: float
    # Ensemble details
    method_scores: Dict[str, float] = field(default_factory=dict)
    votes: int = 0  # How many methods voted "anomaly"


class OODDetector:
    """
    Ensemble out-of-distribution detector.
    
    Combines multiple OOD detection methods:
    - MSP: Low max probability = uncertain = possibly OOD
    - Energy: High energy = low confidence = possibly OOD
    - Mahalanobis: Far from class centroids = possibly OOD
    
    Final decision uses voting: if 2+ methods flag as OOD, it's an anomaly.
    """

    def __init__(
        self,
        threshold: float = 2.5,
        temperature: float = 1.0,
        msp_threshold: float = 0.5,  # Below this confidence = suspicious
        energy_threshold: float = 2.5,  # Above this energy = suspicious
        mahal_threshold: float = 50.0,  # Above this distance = suspicious
        use_ensemble: bool = True,
        voting_threshold: int = 2,  # Number of methods that must agree
    ):
        """
        Args:
            threshold: Legacy threshold for backward compatibility
            temperature: Softmax temperature
            msp_threshold: MSP threshold (lower = more sensitive)
            energy_threshold: Energy threshold (lower = more sensitive)
            mahal_threshold: Mahalanobis threshold (lower = more sensitive)
            use_ensemble: Whether to use ensemble or just energy
            voting_threshold: Minimum votes to flag as anomaly (1-3)
        """
        self.threshold = threshold
        self.temperature = temperature
        self.msp_threshold = msp_threshold
        self.energy_threshold = energy_threshold
        self.mahal_threshold = mahal_threshold
        self.use_ensemble = use_ensemble
        self.voting_threshold = voting_threshold
        
        # Class statistics for Mahalanobis (computed during calibration)
        self.class_means: Optional[np.ndarray] = None
        self.shared_cov_inv: Optional[np.ndarray] = None

    def compute_msp(self, logits: np.ndarray) -> float:
        """
        Maximum Softmax Probability.
        
        Lower MSP = less confident = more likely OOD.
        Returns: 1 - max_prob (so higher = more anomalous, like other methods)
        """
        scaled = logits / self.temperature
        # Numerically stable softmax
        exp_logits = np.exp(scaled - np.max(scaled))
        probs = exp_logits / np.sum(exp_logits)
        max_prob = np.max(probs)
        
        # Invert so higher = more anomalous
        return 1.0 - max_prob

    def compute_energy(self, logits: np.ndarray) -> float:
        """
        Energy-based score.
        
        E(x) = -T * log(sum(exp(logit_i / T)))
        Higher energy = lower confidence = more likely OOD.
        """
        scaled = logits / self.temperature
        max_logit = np.max(scaled)
        energy = -self.temperature * (max_logit + np.log(np.sum(np.exp(scaled - max_logit))))
        return float(energy)

    def compute_mahalanobis(self, embedding: np.ndarray, logits: np.ndarray) -> float:
        """
        Mahalanobis distance to nearest class centroid.
        
        Uses the predicted class centroid for distance computation.
        Higher distance = further from known patterns = more likely OOD.
        """
        if self.class_means is None or self.shared_cov_inv is None:
            # Not calibrated yet, return dummy score
            return 0.0
        
        # Get predicted class
        predicted_class = np.argmax(logits)
        
        if predicted_class >= len(self.class_means):
            return 0.0
        
        # Distance to predicted class centroid
        diff = embedding - self.class_means[predicted_class]
        distance = np.sqrt(diff @ self.shared_cov_inv @ diff)
        
        return float(distance)

    def detect(
        self, 
        logits: np.ndarray, 
        embedding: Optional[np.ndarray] = None
    ) -> OODOutput:
        """
        Detect if an image is OOD using ensemble methods.
        
        Args:
            logits: Raw classifier output (num_classes,)
            embedding: Optional feature embedding for Mahalanobis
        
        Returns:
            OODOutput with ensemble score and voting result
        """
        # Compute all scores
        msp_score = self.compute_msp(logits)
        energy_score = self.compute_energy(logits)
        mahal_score = self.compute_mahalanobis(embedding, logits) if embedding is not None else 0.0
        
        method_scores = {
            "msp": msp_score,
            "energy": energy_score,
            "mahalanobis": mahal_score,
        }
        
        if self.use_ensemble:
            # Voting: count how many methods flag as anomaly
            votes = 0
            
            if msp_score > (1.0 - self.msp_threshold):  # Inverted: low confidence
                votes += 1
            
            if energy_score > self.energy_threshold:
                votes += 1
            
            if embedding is not None and mahal_score > self.mahal_threshold:
                votes += 1
            
            is_anomaly = votes >= self.voting_threshold
            
            # Combined score (weighted average, normalized)
            combined_score = (
                msp_score * 2.0 +  # MSP on [0,1], scale up
                energy_score * 0.5 +  # Energy typically [-10, 5]
                (mahal_score / 100.0 if embedding is not None else 0)  # Mahal can be large
            ) / (2.5 if embedding is not None else 2.0)
            
        else:
            # Legacy mode: just energy
            votes = 1 if energy_score > self.threshold else 0
            is_anomaly = energy_score > self.threshold
            combined_score = energy_score

        return OODOutput(
            ood_score=combined_score,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            method_scores=method_scores,
            votes=votes,
        )

    def calibrate(
        self,
        embeddings: np.ndarray,
        logits: np.ndarray,
        labels: np.ndarray,
        target_fpr: float = 0.05,
    ):
        """
        Calibrate detector on in-distribution data.
        
        Computes class centroids and covariance for Mahalanobis,
        and sets thresholds for each method.
        
        Args:
            embeddings: Feature embeddings (N, D)
            logits: Classifier logits (N, C)
            labels: True class labels (N,)
            target_fpr: Target false positive rate
        """
        n_classes = logits.shape[1]
        embed_dim = embeddings.shape[1]
        
        # Compute class means
        self.class_means = np.zeros((n_classes, embed_dim))
        class_samples = [[] for _ in range(n_classes)]
        
        for i, label in enumerate(labels):
            if label < n_classes:
                class_samples[label].append(embeddings[i])
        
        for c in range(n_classes):
            if class_samples[c]:
                self.class_means[c] = np.mean(class_samples[c], axis=0)
        
        # Compute shared covariance
        all_centered = []
        for c in range(n_classes):
            for emb in class_samples[c]:
                all_centered.append(emb - self.class_means[c])
        
        if all_centered:
            all_centered = np.stack(all_centered)
            cov = np.cov(all_centered, rowvar=False)
            # Regularize for numerical stability
            cov += np.eye(embed_dim) * 1e-6
            self.shared_cov_inv = np.linalg.inv(cov)
        
        # Calibrate individual thresholds
        msp_scores = [self.compute_msp(l) for l in logits]
        energy_scores = [self.compute_energy(l) for l in logits]
        
        # Set thresholds at (1 - target_fpr) percentile
        self.msp_threshold = 1.0 - np.percentile(msp_scores, 100 * (1 - target_fpr))
        self.energy_threshold = np.percentile(energy_scores, 100 * (1 - target_fpr))
        
        if self.class_means is not None:
            mahal_scores = [
                self.compute_mahalanobis(embeddings[i], logits[i]) 
                for i in range(len(logits))
            ]
            self.mahal_threshold = np.percentile(mahal_scores, 100 * (1 - target_fpr))
        
        logger.info(f"Calibrated OOD detector:")
        logger.info(f"  MSP threshold: {self.msp_threshold:.3f}")
        logger.info(f"  Energy threshold: {self.energy_threshold:.3f}")
        logger.info(f"  Mahalanobis threshold: {self.mahal_threshold:.3f}")

    def calibrate_threshold(
        self,
        in_dist_logits: np.ndarray,
        target_fpr: float = 0.05,
    ) -> float:
        """
        Legacy method: Calibrate energy threshold only.
        """
        energies = [self.compute_energy(logit) for logit in in_dist_logits]
        self.threshold = float(np.percentile(energies, 100 * (1 - target_fpr)))
        self.energy_threshold = self.threshold
        logger.info(f"Calibrated OOD threshold to {self.threshold:.2f} for {target_fpr:.1%} FPR")
        return self.threshold


# Convenience function for quick detection
def is_anomaly(logits: np.ndarray, threshold: float = 2.5) -> bool:
    """Quick check if logits indicate an anomaly."""
    detector = OODDetector(threshold=threshold)
    return detector.detect(logits).is_anomaly
