"""
Features Module

Provides feature extraction for astronomical analysis:

1. Galaxy Morphology (morphology.py)
   - CAS: Concentration, Asymmetry, Smoothness
   - Gini-M20 coefficients
   - Ellipticity
   - Merger/Irregular detection

2. Reconstruction-based Anomaly Detection (reconstruction.py)
   - PCA reconstruction error
   - Feature-space reconstruction

3. Time-Series Features (time_series.py)
   - Amplitude, rise time, fade rate
   - Periodicity detection

4. Multi-Band Features (multiband.py)
   - Color analysis (g-r, r-i, etc.)
   - Spectral type hints

5. Export (export.py)
   - CSV, JSON, HTML, VOTable export formats
"""

from .morphology import GalaxyMorphology, MorphologyResult, analyze_morphology
from .reconstruction import PCAReconstructor, ReconstructionResult
from .time_series import TimeSeriesFeatures, Observation, compute_time_features
from .multiband import MultiBandAnalyzer, ColorFeatures, analyze_colors
from .export import ResultsExporter

__all__ = [
    # Morphology
    "GalaxyMorphology",
    "MorphologyResult", 
    "analyze_morphology",
    # Reconstruction
    "PCAReconstructor",
    "ReconstructionResult",
    # Time series
    "TimeSeriesFeatures",
    "Observation",
    "compute_time_features",
    # Multi-band
    "MultiBandAnalyzer", 
    "ColorFeatures",
    "analyze_colors",
    # Export
    "ResultsExporter",
]
