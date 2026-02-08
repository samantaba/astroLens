"""
Galaxy Morphology Features

Computes standard morphological parameters for galaxy images:
- CAS: Concentration, Asymmetry, Smoothness/Clumpiness
- Gini-M20: Gini coefficient and M20 moment
- Additional: Ellipticity, Sersic index approximation

These features help identify unusual galaxy morphologies like:
- Mergers and interactions
- Tidal features
- Irregular/disturbed galaxies
- Edge-on disks
- Compact/ultra-diffuse objects

Reference: Conselice (2003), Lotz et al. (2004)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MorphologyResult:
    """Result of morphology analysis."""
    
    # CAS parameters
    concentration: float  # C: ratio of radii containing 80% vs 20% of light
    asymmetry: float      # A: rotational asymmetry
    smoothness: float     # S: clumpiness/smoothness
    
    # Gini-M20
    gini: float           # Gini coefficient (inequality of light distribution)
    m20: float            # Second-order moment of brightest 20%
    
    # Additional
    ellipticity: float    # 1 - b/a (axis ratio)
    
    # Flags
    is_irregular: bool    # High asymmetry/clumpiness
    is_merger: bool       # Gini-M20 suggests merger
    is_compact: bool      # High concentration
    
    # Overall anomaly score from morphology
    morph_score: float    # Combined score 0-1
    
    def to_dict(self) -> dict:
        return {
            "concentration": round(self.concentration, 3),
            "asymmetry": round(self.asymmetry, 3),
            "smoothness": round(self.smoothness, 3),
            "gini": round(self.gini, 3),
            "m20": round(self.m20, 3),
            "ellipticity": round(self.ellipticity, 3),
            "is_irregular": self.is_irregular,
            "is_merger": self.is_merger,
            "is_compact": self.is_compact,
            "morph_score": round(self.morph_score, 3),
        }


class GalaxyMorphology:
    """
    Compute galaxy morphology features from images.
    
    Usage:
        morph = GalaxyMorphology()
        result = morph.analyze("galaxy.jpg")
        print(f"Asymmetry: {result.asymmetry}")
    """
    
    def __init__(self):
        self.image_size = 128  # Resize for consistent analysis
        
    def analyze(self, image_path: str) -> Optional[MorphologyResult]:
        """
        Analyze a galaxy image and compute morphology features.
        
        Args:
            image_path: Path to galaxy image
            
        Returns:
            MorphologyResult with all computed features
        """
        try:
            # Load and preprocess image
            img = self._load_image(image_path)
            if img is None:
                return None
            
            # Compute features
            concentration = self._compute_concentration(img)
            asymmetry = self._compute_asymmetry(img)
            smoothness = self._compute_smoothness(img)
            gini = self._compute_gini(img)
            m20 = self._compute_m20(img)
            ellipticity = self._compute_ellipticity(img)
            
            # Classification based on thresholds
            is_irregular = asymmetry > 0.35 or smoothness > 0.3
            is_merger = gini > 0.55 and m20 > -1.6
            is_compact = concentration > 4.0
            
            # Combined morphology score (higher = more unusual)
            morph_score = self._compute_morph_score(
                concentration, asymmetry, smoothness, gini, m20
            )
            
            return MorphologyResult(
                concentration=concentration,
                asymmetry=asymmetry,
                smoothness=smoothness,
                gini=gini,
                m20=m20,
                ellipticity=ellipticity,
                is_irregular=is_irregular,
                is_merger=is_merger,
                is_compact=is_compact,
                morph_score=morph_score,
            )
            
        except Exception as e:
            logger.error(f"Morphology analysis failed for {image_path}: {e}")
            return None
    
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load and preprocess image."""
        try:
            img = Image.open(path).convert("L")  # Grayscale
            img = img.resize((self.image_size, self.image_size))
            arr = np.array(img, dtype=np.float32)
            
            # Normalize to 0-1
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            
            return arr
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def _compute_concentration(self, img: np.ndarray) -> float:
        """
        Compute concentration index C = 5 * log10(r80/r20)
        
        r80, r20 are radii containing 80% and 20% of total light.
        High C = compact/concentrated, Low C = diffuse
        """
        center = img.shape[0] // 2
        total_flux = img.sum()
        
        # Create radial distance array
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Find r20 and r80
        r20, r80 = 1, 1
        cumsum = 0
        for radius in range(1, center):
            mask = r <= radius
            cumsum = img[mask].sum()
            if cumsum >= 0.2 * total_flux and r20 == 1:
                r20 = radius
            if cumsum >= 0.8 * total_flux:
                r80 = radius
                break
        
        if r20 < 1:
            r20 = 1
        
        return 5 * np.log10(max(r80 / r20, 1.01))
    
    def _compute_asymmetry(self, img: np.ndarray) -> float:
        """
        Compute asymmetry A = sum(|I - I_180|) / (2 * sum(|I|))
        
        I_180 is image rotated 180 degrees.
        High A = asymmetric/disturbed, Low A = symmetric
        """
        img_rot = np.rot90(img, 2)  # Rotate 180 degrees
        
        diff = np.abs(img - img_rot)
        asymmetry = diff.sum() / (2 * np.abs(img).sum() + 1e-8)
        
        return min(asymmetry, 1.0)
    
    def _compute_smoothness(self, img: np.ndarray) -> float:
        """
        Compute smoothness/clumpiness S = sum(|I - I_s|) / sum(|I|)
        
        I_s is smoothed image. High S = clumpy, Low S = smooth
        """
        from scipy.ndimage import gaussian_filter
        
        # Smooth with sigma = 0.25 * Petrosian radius (approximate with 5 pixels)
        img_smooth = gaussian_filter(img, sigma=5)
        
        diff = np.abs(img - img_smooth)
        smoothness = diff.sum() / (np.abs(img).sum() + 1e-8)
        
        return min(smoothness, 1.0)
    
    def _compute_gini(self, img: np.ndarray) -> float:
        """
        Compute Gini coefficient of light distribution.
        
        G = 1/(2*n*mean) * sum(|xi - xj|)
        High G = light concentrated in few pixels
        Low G = uniform distribution
        """
        flat = img.flatten()
        flat = flat[flat > 0.01]  # Remove background
        
        if len(flat) < 10:
            return 0.5
        
        flat = np.sort(flat)
        n = len(flat)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n
        
        return max(0, min(gini, 1.0))
    
    def _compute_m20(self, img: np.ndarray) -> float:
        """
        Compute M20 - second-order moment of brightest 20%.
        
        M20 = log10(sum_i(M_i) / M_tot) for brightest 20%
        Low M20 = concentrated, High M20 = multiple bright regions (mergers)
        """
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
        
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        r2 = (x - center_x)**2 + (y - center_y)**2
        
        # Total second moment
        m_tot = np.sum(img * r2) + 1e-8
        
        # Find brightest 20%
        flat = img.flatten()
        threshold = np.percentile(flat, 80)
        bright_mask = img >= threshold
        
        # M20
        m20_sum = np.sum(img[bright_mask] * r2[bright_mask])
        m20 = np.log10(m20_sum / m_tot + 1e-8)
        
        return max(-3.0, min(m20, 0.0))
    
    def _compute_ellipticity(self, img: np.ndarray) -> float:
        """
        Compute ellipticity e = 1 - b/a from image moments.
        
        High e = elongated, Low e = round
        """
        # Compute image moments
        y, x = np.mgrid[:img.shape[0], :img.shape[1]]
        
        total = img.sum() + 1e-8
        cx = (x * img).sum() / total
        cy = (y * img).sum() / total
        
        # Second moments
        xx = ((x - cx)**2 * img).sum() / total
        yy = ((y - cy)**2 * img).sum() / total
        xy = ((x - cx) * (y - cy) * img).sum() / total
        
        # Eigenvalues of covariance matrix
        trace = xx + yy
        det = xx * yy - xy**2
        
        if trace**2 < 4 * det:
            return 0.0
        
        lambda1 = (trace + np.sqrt(trace**2 - 4 * det)) / 2
        lambda2 = (trace - np.sqrt(trace**2 - 4 * det)) / 2
        
        if lambda1 < 1e-8:
            return 0.0
        
        ellipticity = 1 - np.sqrt(max(0, lambda2) / lambda1)
        
        return max(0, min(ellipticity, 1.0))
    
    def _compute_morph_score(
        self, 
        concentration: float,
        asymmetry: float,
        smoothness: float,
        gini: float,
        m20: float,
    ) -> float:
        """
        Compute combined morphology anomaly score.
        
        Higher score = more unusual morphology
        """
        # Normalize each parameter to 0-1 based on typical ranges
        c_norm = min(concentration / 5.0, 1.0)  # C typically 2-5
        a_norm = min(asymmetry / 0.5, 1.0)      # A typically 0-0.35
        s_norm = min(smoothness / 0.4, 1.0)     # S typically 0-0.3
        g_norm = gini                            # Already 0-1
        m_norm = (m20 + 2.5) / 2.5               # M20 typically -2.5 to 0
        
        # Weighted combination (asymmetry and smoothness are stronger indicators)
        score = (
            0.15 * c_norm +
            0.30 * a_norm +
            0.25 * s_norm +
            0.15 * g_norm +
            0.15 * m_norm
        )
        
        return max(0, min(score, 1.0))


# Convenience function
def analyze_morphology(image_path: str) -> Optional[MorphologyResult]:
    """Analyze galaxy morphology from image path."""
    analyzer = GalaxyMorphology()
    return analyzer.analyze(image_path)
