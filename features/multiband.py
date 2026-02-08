"""
Multi-Band Photometry Module

Handles multiple filter bands (g, r, i, z) for astronomical observations.
Different wavelengths reveal different physics:

- g-band (blue): Hot objects, recent star formation
- r-band (red): General stellar light
- i-band (near-IR): Cooler/older stars, dust penetration  
- z-band (far-IR): Even cooler objects, high redshift

Color Analysis:
- Blue (g-r < 0): Hot, young, supernovae
- Red (g-r > 1): Cool, old, high redshift
- Variable color: Evolution happening

Usage:
    from features.multiband import MultiBandAnalyzer
    
    analyzer = MultiBandAnalyzer()
    colors = analyzer.compute_colors(g_mag=20.5, r_mag=20.2, i_mag=19.8)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Standard filter wavelengths (nm)
FILTER_WAVELENGTHS = {
    "u": 355,  # Ultraviolet
    "g": 477,  # Green/Blue
    "r": 623,  # Red
    "i": 762,  # Near-infrared
    "z": 905,  # Far-infrared
    "y": 1004, # Y-band
}


@dataclass
class MultiBandObservation:
    """Observation with multiple filter bands."""
    mjd: float
    magnitudes: Dict[str, float]  # band -> magnitude
    errors: Dict[str, float] = field(default_factory=dict)
    
    def get_mag(self, band: str) -> Optional[float]:
        return self.magnitudes.get(band)
    
    @property
    def bands(self) -> List[str]:
        return list(self.magnitudes.keys())


@dataclass  
class ColorFeatures:
    """Computed color features."""
    # Standard colors
    g_minus_r: Optional[float] = None
    r_minus_i: Optional[float] = None
    i_minus_z: Optional[float] = None
    g_minus_i: Optional[float] = None  # Wide color baseline
    
    # Classification hints
    is_blue: bool = False  # g-r < 0.3
    is_red: bool = False   # g-r > 1.0
    is_neutral: bool = True
    
    # Spectral type estimates
    spectral_type_hint: str = ""
    temperature_hint: str = ""
    
    # Metadata
    bands_used: List[str] = field(default_factory=list)
    computed_at: str = ""


class MultiBandAnalyzer:
    """
    Analyzer for multi-band photometry.
    
    Computes colors, spectral hints, and evolutionary features
    from multi-filter observations.
    """
    
    def __init__(self):
        self.observations: List[MultiBandObservation] = []
    
    def compute_colors(
        self,
        g_mag: Optional[float] = None,
        r_mag: Optional[float] = None,
        i_mag: Optional[float] = None,
        z_mag: Optional[float] = None,
    ) -> ColorFeatures:
        """
        Compute color features from single-epoch magnitudes.
        
        Args:
            g_mag, r_mag, i_mag, z_mag: Magnitudes in each band.
        
        Returns:
            ColorFeatures with computed values.
        """
        bands_used = []
        
        # Compute colors (difference between bands)
        g_minus_r = None
        if g_mag is not None and r_mag is not None:
            g_minus_r = g_mag - r_mag
            bands_used.extend(["g", "r"])
        
        r_minus_i = None
        if r_mag is not None and i_mag is not None:
            r_minus_i = r_mag - i_mag
            bands_used.extend(["i"] if "i" not in bands_used else [])
        
        i_minus_z = None
        if i_mag is not None and z_mag is not None:
            i_minus_z = i_mag - z_mag
            bands_used.extend(["z"] if "z" not in bands_used else [])
        
        g_minus_i = None
        if g_mag is not None and i_mag is not None:
            g_minus_i = g_mag - i_mag
        
        # Classify color
        is_blue = g_minus_r is not None and g_minus_r < 0.3
        is_red = g_minus_r is not None and g_minus_r > 1.0
        is_neutral = not is_blue and not is_red
        
        # Spectral hints
        spectral_hint = ""
        temp_hint = ""
        
        if is_blue:
            spectral_hint = "O/B/A type (hot)"
            temp_hint = ">7500K"
        elif is_red:
            spectral_hint = "K/M type (cool)"
            temp_hint = "<4500K"
        else:
            spectral_hint = "F/G type (solar-like)"
            temp_hint = "4500-7500K"
        
        return ColorFeatures(
            g_minus_r=g_minus_r,
            r_minus_i=r_minus_i,
            i_minus_z=i_minus_z,
            g_minus_i=g_minus_i,
            is_blue=is_blue,
            is_red=is_red,
            is_neutral=is_neutral,
            spectral_type_hint=spectral_hint,
            temperature_hint=temp_hint,
            bands_used=list(set(bands_used)),
            computed_at=datetime.now().isoformat(),
        )
    
    def compute_color_evolution(
        self,
        observations: List[MultiBandObservation]
    ) -> Dict[str, float]:
        """
        Compute how colors change over time.
        
        Returns:
            Dict with color evolution rates.
        """
        if len(observations) < 2:
            return {"color_evolution": 0.0}
        
        # Sort by time
        obs_sorted = sorted(observations, key=lambda x: x.mjd)
        
        # Track g-r color over time
        g_r_colors = []
        times = []
        
        for obs in obs_sorted:
            g = obs.get_mag("g")
            r = obs.get_mag("r")
            if g is not None and r is not None:
                g_r_colors.append(g - r)
                times.append(obs.mjd)
        
        if len(g_r_colors) < 2:
            return {"color_evolution": 0.0}
        
        # Calculate color change rate
        time_span = times[-1] - times[0]
        color_change = g_r_colors[-1] - g_r_colors[0]
        
        rate = color_change / time_span if time_span > 0 else 0.0
        
        return {
            "color_evolution_rate": rate,  # mag/day
            "color_start": g_r_colors[0],
            "color_end": g_r_colors[-1],
            "becoming_redder": rate > 0.01,
            "becoming_bluer": rate < -0.01,
        }
    
    def classify_by_color(self, colors: ColorFeatures) -> Tuple[str, float]:
        """
        Classify object type based on colors.
        
        Returns:
            Tuple of (classification, confidence)
        """
        if colors.g_minus_r is None:
            return "unknown", 0.0
        
        g_r = colors.g_minus_r
        
        # Supernova candidates: Often blue early, reddening over time
        if g_r < 0.0:
            return "young_supernova_candidate", 0.6
        
        # Normal galaxies: Moderate colors
        if 0.3 < g_r < 0.8:
            return "normal_galaxy", 0.5
        
        # Red galaxies: Older, less active
        if g_r > 1.2:
            return "red_galaxy", 0.5
        
        # AGN can have various colors
        return "unclassified", 0.3


# Convenience function
def analyze_colors(
    g_mag: float = None,
    r_mag: float = None, 
    i_mag: float = None
) -> ColorFeatures:
    """Quick color analysis."""
    analyzer = MultiBandAnalyzer()
    return analyzer.compute_colors(g_mag=g_mag, r_mag=r_mag, i_mag=i_mag)
